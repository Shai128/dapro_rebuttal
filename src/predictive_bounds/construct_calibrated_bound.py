import re
import os
import traceback
import hashlib
import time
from pathlib import Path
import pandas as pd
import torch
import tqdm
import numpy as np
from typing import List
import argparse

from src.predictive_bounds.budget_allocators.adaptive_optimized_allocator import AdaptiveOptimizedBudgetAllocator
from src.predictive_bounds.budget_allocators.basic_allocator import BasicBudgetAllocator
from src.predictive_bounds.budget_allocators.budget_allocator import BudgetAllocator
from src.predictive_bounds.budget_allocators.optimized_allocators import OptimizedBudgetAllocator
from src.predictive_bounds.budget_allocators.oracle_dapro_allocator import (
    CRCOracleTargetADAPRO,
    GlobalOracleTargetADAPRO,
    SplitOracleTargetADAPRO,
)
from src.predictive_bounds.budget_allocators.DAPRO import (
    DAPRO,
    DefinitiveCRCDAPRO,
    DefinitiveCRCUPBDAPRO,
    DefinitiveDAPRO,
    LegacyMeanWeightDAPRO,
    TargetAWeightedDAPRO, BandRegularizedTargetAWeightedDAPRO, RegularizedTargetAWeightedDAPRO,
    RobustTargetAWeightedDAPRO, RandomAnchoredTargetAWeightedDAPRO, AWeightedDAPRO,
)
from src.predictive_bounds.budget_allocators.random_adaptive_optimized_allocator import (
    ConstantCRCBudgetAllocator,
    RandomAdaptiveOptimizedBudgetAllocator,
)
from src.predictive_bounds.budget_allocators.trimmed_allocator import TrimmedBudgetAllocator

# LPB Calibrations
from src.predictive_bounds.calibration.abstract_calibration import SurvivalLPBCalibration
from src.predictive_bounds.calibration.dummy_calibration import UncalibratedLPBSurvivalCalibration
from src.predictive_bounds.calibration.oracle_survival_calibration import (
    OracleSurvivalCalibration,
    OracleSurvivalUPBCalibration,
)
from src.predictive_bounds.calibration.calibration_utils import (
    indexed_tensor_metrics,
)
from src.predictive_bounds.calibration.survival_calibration_with_known_weights import get_gamma, \
    SurvivalCalibrationWithKnownWeights

# UPB Calibrations
from src.predictive_bounds.calibration.abstract_calibration import SurvivalUPBCalibration
from src.predictive_bounds.calibration.dummy_calibration import UncalibratedUPBSurvivalCalibration
from src.predictive_bounds.calibration.survival_upb_calibration_with_known_weights import \
    SurvivalUPBCalibrationWithKnownWeights

from src.train_model.models.utils import SurvivalModelPrediction

from src.predictive_bounds.utils.get_calibration_methods_utils import (
    get_baseline_calibrations as get_registered_baseline_calibrations, is_budget_sufficient_for_split,
)
from src.predictive_bounds.utils.utils import (
    split_data,
    get_tmp_calibration_result_path,
    get_tmp_upb_calibration_result_path,
    get_calibration_experiment_name,
    make_lpb_tau_grid,
    resolve_m_upper_bound,
    setup_experiment_data
)
from src.utils.utils import set_seeds


def _predictive_bounds_source_fingerprint() -> str:
    """Hash the exact Python source tree used to generate result rows."""
    source_root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(source_root.rglob("*.py")):
        digest.update(path.relative_to(source_root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _index_fingerprint(indices) -> str:
    values = (
        indices.detach().cpu().numpy()
        if torch.is_tensor(indices)
        else np.asarray(indices)
    )
    return hashlib.sha256(
        np.asarray(values, dtype=np.int64).tobytes()
    ).hexdigest()


def _make_common_acquisition_uniforms(
        seed: int,
        population_size: int,
        n_times: int,
        selected_indices=None,
) -> np.ndarray:
    """Build device-invariant CRNs, optionally mapped by original row index.

    Generating the population-level table before selecting calibration rows
    ensures that original sample ``i`` receives the same time-indexed uniforms
    even when the calibration split, Phase-I size, or allocator changes.
    """
    if population_size < 0 or n_times < 0:
        raise ValueError("Population size and time width must be nonnegative.")
    population_uniforms = np.random.default_rng(int(seed)).random(
        (population_size, n_times)
    )
    if selected_indices is None:
        population_uniforms.setflags(write=False)
        return population_uniforms
    indices = (
        selected_indices.detach().cpu().numpy()
        if torch.is_tensor(selected_indices)
        else np.asarray(selected_indices)
    )
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if np.any(indices < 0) or np.any(indices >= population_size):
        raise ValueError("Selected acquisition-uniform indices are out of range.")
    selected = np.ascontiguousarray(population_uniforms[indices])
    selected.setflags(write=False)
    return selected


def compute_metrics_bound(bound: torch.Tensor, t_tilde: torch.Tensor, bound_type: str):
    if bound_type == 'lpb':
        coverage_rate = (t_tilde.unsqueeze(1) >= bound).float().mean(dim=0)
    elif bound_type == 'upb':
        coverage_rate = ((t_tilde.unsqueeze(1) <= bound) | (bound == 200)).float().mean(dim=0)
    else:
        raise ValueError("Invalid bound_type. Must be 'lpb' or 'upb'")

    length = bound.mean(dim=0)
    return coverage_rate, length


def preprocess_df(df, target_taus_list, defaults=None):
    if defaults is None:
        defaults = {
            'budget_used': 0, 'mean_weight': 0, 'n_observed_events': 0,
            'n_achieved_q_prior1': 0, 'n_achieved_q_prior2': 0
        }

    row_series = df.rename(columns={"Unnamed: 0": "seed"}, errors='ignore').iloc[0]
    row_dict = row_series.to_dict()

    for key, default_val in defaults.items():
        if key not in row_dict:
            row_dict[key] = default_val

    static_data = {}
    dynamic_data = {}
    n_taus = len(target_taus_list)

    for key, val in row_dict.items():
        match = re.search(r'^(.*)_(\d+)$', str(key))
        if match:
            base_name, idx_str = match.groups()
            idx = int(idx_str)

            if 0 <= idx < n_taus:
                if base_name not in dynamic_data:
                    dynamic_data[base_name] = {}
                dynamic_data[base_name][idx] = val
        else:
            static_data[key] = val

    rows = []
    for i in range(n_taus):
        new_row = static_data.copy()
        for base_name, values in dynamic_data.items():
            if i in values:
                new_row[base_name] = values[i]
        rows.append(new_row)

    return pd.DataFrame(rows)


def run_one_experiment(experiments_name, seed, calibration, x_cal, t_tilde_cal, cal_model_prediction, x_test,
                       t_tilde_test, test_model_prediction, target_taus_list, bound_type, skip_existing=True,
                       experiment_metadata=None, policy_seed=None,
                       acquisition_seed=None, acquisition_uniforms=None):
    try:
        if bound_type == 'lpb':
            dir_path = get_tmp_calibration_result_path(experiments_name, calibration.name)
        else:
            dir_path = get_tmp_upb_calibration_result_path(experiments_name, calibration.name)

        save_path = os.path.join(f"{dir_path}", f"seed={seed}.csv")
        if os.path.exists(save_path) and skip_existing:
            return

        effective_policy_seed = (
            seed if policy_seed is None else int(policy_seed)
        )
        set_seeds(effective_policy_seed)
        if hasattr(calibration, "budget_allocator"):
            allocator = calibration.budget_allocator
            effective_acquisition_seed = (
                seed if acquisition_seed is None else int(acquisition_seed)
            )
            if acquisition_uniforms is None:
                conditional_grid = getattr(
                    allocator,
                    "conditional_grid",
                    None,
                )
                n_times = (
                    int(conditional_grid.shape[1])
                    if conditional_grid is not None
                    else 1
                )
                acquisition_uniforms = _make_common_acquisition_uniforms(
                    effective_acquisition_seed,
                    len(t_tilde_cal),
                    n_times,
                )
            allocator.set_acquisition_randomness(
                seed=effective_acquisition_seed,
                uniforms=acquisition_uniforms,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        method_started = time.perf_counter()
        calibration.calibrate(x_cal, t_tilde_cal, cal_model_prediction)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        calibration_seconds = time.perf_counter() - method_started
        target_taus = torch.Tensor(target_taus_list)

        bound_started = time.perf_counter()
        with torch.no_grad():
            if bound_type == 'lpb':
                calibrated_test_bound = calibration.get_calibrated_lpb(target_taus, x_test, test_model_prediction)
            else:
                calibrated_test_bound = calibration.get_calibrated_upb(target_taus, x_test, test_model_prediction)

        coverage_rate, length = compute_metrics_bound(calibrated_test_bound, t_tilde_test, bound_type)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        bound_seconds = time.perf_counter() - bound_started
        metrics_started = time.perf_counter()
        calibration_metrics = calibration.compute_metrics(cal_model_prediction, target_taus)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        metrics_seconds = time.perf_counter() - metrics_started
        method_seconds = time.perf_counter() - method_started

        if bound_type == 'lpb':
            target_coverage = {f'target_coverage_{i}': 1 - target_taus_list[i] for i in range(len(target_taus_list))}
        else:
            target_coverage = {f'target_coverage_{i}': target_taus_list[i] for i in range(len(target_taus_list))}

        bound_metrics = indexed_tensor_metrics({
            "coverage": coverage_rate,
            "size": length,
        })
        all_metrics = {
            'seed': seed,
            'calibration_name': calibration.name,
            'calibration_runtime_seconds': calibration_seconds,
            'bound_runtime_seconds': bound_seconds,
            'metrics_runtime_seconds': metrics_seconds,
            'method_runtime_seconds': method_seconds,
            **(experiment_metadata or {}),
            **bound_metrics,
            **target_coverage,
            **calibration_metrics,
        }

        df = pd.DataFrame(all_metrics, index=[seed])
        processed_df = preprocess_df(df, target_taus_list)
        abs_path = os.path.abspath(save_path)

        if os.name == 'nt' and not abs_path.startswith('\\\\?\\'):
            abs_path = f"\\\\?\\{abs_path}"
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)

        processed_df.to_csv(abs_path)

    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(
            f"Calibration {calibration.name} failed for seed {seed}."
        ) from e


def store(method_name, lengths):
    data_to_save = {
        'lengths': lengths.cpu(),
    }
    torch.save(data_to_save, f'{method_name}_lengths.pt')


def run_experiments(cal_size, is_real, device, dataset_name, data_setup, experiments_name, seeds,
                    taus_range, budget_per_sample, tau_prior, m_upper_bound, target_taus_list, skip_existing,
                    allocations, bound_type, calibration_names=None,
                    evaluate_dapro_projection=False, fixed_data_seed=None,
                    fixed_policy_seed=None, fixed_acquisition_seed=None,
                    dapro_n1_values=(200,100, 50),
                    definitive_dapro_margins=(1.0,)):
    max_time, t_tilde_cal_test, quantile_est_cal_test, probability_est, conditional_grid, test_size = setup_experiment_data(
        cal_size, is_real, device, dataset_name, data_setup, taus_range, m_upper_bound
    )
    source_fingerprint = _predictive_bounds_source_fingerprint()
    execution_device = str(conditional_grid.device)
    cuda_device_name = (
        torch.cuda.get_device_name(conditional_grid.device)
        if conditional_grid.is_cuda
        else "cpu"
    )
    taus_range = taus_range.detach()
    for seed in tqdm.tqdm(range(seeds[0], seeds[1]), desc="running calibration algorithms"):
        data_seed = seed if fixed_data_seed is None else int(fixed_data_seed)
        policy_seed = (
            seed if fixed_policy_seed is None else int(fixed_policy_seed)
        )
        # Acquisition randomness is a dedicated stream.  By default it follows
        # the outer seed; fixing it permits one-factor variance decompositions
        # without changing normal experiment semantics.
        acquisition_seed = (
            seed
            if fixed_acquisition_seed is None
            else int(fixed_acquisition_seed)
        )
        x_cal, x_test, t_tilde_cal, probability_est_cal, quantile_est_cal, t_tilde_test, quantile_est_test, \
            probability_est_test, cal_idx, test_idx = split_data(data_seed, cal_size, test_size, None, t_tilde_cal_test,
                                                                 probability_est, quantile_est_cal_test)

        curr_conditional_grid = conditional_grid[cal_idx]
        acquisition_uniforms = _make_common_acquisition_uniforms(
            acquisition_seed,
            len(conditional_grid),
            int(conditional_grid.shape[1]),
            selected_indices=cal_idx,
        )
        acquisition_uniforms_fingerprint = hashlib.sha256(
            acquisition_uniforms.tobytes()
        ).hexdigest()
        # Split and provenance metadata are identical for every method in this
        # seed.  Hash each index vector once instead of once per calibration.
        experiment_metadata = {
            "experiment_name": experiments_name,
            "configured_cal_size": cal_size,
            "configured_budget_per_sample": budget_per_sample,
            "configured_tau_prior": tau_prior,
            "configured_m_upper_bound": m_upper_bound,
            "execution_device": execution_device,
            "cuda_device_name": cuda_device_name,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "predictive_bounds_source_sha256": source_fingerprint,
            "calibration_split_sha256": _index_fingerprint(cal_idx),
            "test_split_sha256": _index_fingerprint(test_idx),
            "data_split_seed": data_seed,
            "policy_rng_seed": policy_seed,
            "acquisition_rng_seed": acquisition_seed,
            "acquisition_rng_reseeded": 1,
            "acquisition_rng_backend": "numpy.default_rng",
            "acquisition_common_random_numbers": 1,
            "acquisition_uniforms_sha256": (
                acquisition_uniforms_fingerprint
            ),
        }

        if bound_type == 'lpb':
            quantile_est_cal = quantile_est_cal.clip(max=max_time)

        cal_model_prediction = SurvivalModelPrediction(quantile_est_cal, probability_est_cal)
        test_model_prediction = SurvivalModelPrediction(quantile_est_test, probability_est_test)

        all_calibrations = get_calibration_methods(
            curr_conditional_grid, budget_per_sample, taus_range, tau_prior,
            m_upper_bound, allocations, cal_model_prediction, t_tilde_cal,
            device, bound_type, evaluate_dapro_projection,
            dapro_n1_values=dapro_n1_values,
            definitive_dapro_margins=definitive_dapro_margins,
        )
        if calibration_names:
            available = {calibration.name for calibration in all_calibrations}
            missing = sorted(set(calibration_names) - available)
            if missing:
                raise ValueError(
                    "Unknown calibration names: "
                    f"{missing}. Available names: {sorted(available)}"
                )
            selected_names = set(calibration_names)
            all_calibrations = [
                calibration
                for calibration in all_calibrations
                if calibration.name in selected_names
            ]

        # These allocators draw from NumPy and Torch's process-global RNGs.
        # Running them in threads races the repeated set_seeds(seed) calls and
        # makes identical commands non-reproducible.  Execute methods
        # sequentially; each receives the same deterministic seed.
        for calibration in all_calibrations:
            run_one_experiment(
                experiments_name,
                seed,
                calibration,
                x_cal,
                t_tilde_cal,
                cal_model_prediction,
                x_test,
                t_tilde_test,
                test_model_prediction,
                target_taus_list,
                bound_type,
                skip_existing,
                policy_seed=policy_seed,
                acquisition_seed=acquisition_seed,
                acquisition_uniforms=acquisition_uniforms,
                experiment_metadata=experiment_metadata,
            )


def set_m_upper_bound(gamma: float, budget_per_sample: float):
    m_upper_bound = gamma * budget_per_sample
    if m_upper_bound < budget_per_sample:
        print(f"warning, m_upper_bound = {m_upper_bound} which leads to gamma lower than 1")
    if abs(gamma - (m_upper_bound / budget_per_sample)) > 0.01:
        print(f"warning, gamma is: {gamma} but the bound leads to {(m_upper_bound / budget_per_sample)}")
    return m_upper_bound


def get_baseline_calibrations(conditional_grid, budget_per_sample, taus_range, tau_prior, m_upper_bound,
                              cal_model_prediction, t_tilde_cal, bound_type,
                              evaluate_dapro_projection=False,
                              dapro_n1_values=(200,100, 50),
                              definitive_dapro_margins=(1.0,)):
    # Construction and merging must enumerate methods from the same registry.
    # Keeping this compatibility wrapper avoids exact-name drift while older
    # experiment callers continue to pass prediction and event-time arguments.
    if bound_type == "lpb":
        calibrations = get_registered_baseline_calibrations(
            conditional_grid,
            budget_per_sample,
            taus_range,
            tau_prior,
            m_upper_bound,
            include_a_weighted=True,
            evaluate_dapro_projection=evaluate_dapro_projection,
            dapro_n1_values=dapro_n1_values,
            definitive_dapro_margins=definitive_dapro_margins,
        )
        calibrations.insert(
            1,
            OracleSurvivalCalibration(taus_range, tau_prior),
        )
        return calibrations

    basic_allocation = BasicBudgetAllocator(budget_per_sample, taus_range, tau_prior)
    trimmed_allocation = TrimmedBudgetAllocator(budget_per_sample, taus_range, tau_prior, m_upper_bound)
    optimized_allocation = OptimizedBudgetAllocator(budget_per_sample, taus_range, tau_prior, m_upper_bound)

    alloc_kwargs = {}
    if bound_type == 'upb':
        alloc_kwargs['reach_t_max_is_success'] = True

    adaptive_allocation = AdaptiveOptimizedBudgetAllocator(
        conditional_grid, budget_per_sample, taus_range, tau_prior, m_upper_bound, **alloc_kwargs
    )
    adaptive_mixture_floor_allocation = AdaptiveOptimizedBudgetAllocator(
        conditional_grid,
        budget_per_sample,
        taus_range,
        tau_prior,
        m_upper_bound,
        terminal_pi_min=1.0 / float(m_upper_bound),
        terminal_floor_mode="mixture",
        **alloc_kwargs,
    )
    adaptive_no_floor_allocation = AdaptiveOptimizedBudgetAllocator(
        conditional_grid,
        budget_per_sample,
        taus_range,
        tau_prior,
        m_upper_bound,
        terminal_pi_min=None,
        terminal_floor_mode="none",
        **alloc_kwargs,
    )
    adaptive_crc_allocation = AdaptiveOptimizedBudgetAllocator(
        conditional_grid,
        budget_per_sample,
        taus_range,
        tau_prior,
        m_upper_bound,
        budget_control_mode="crc",
        **alloc_kwargs,
    )

    all_allocations: List[BudgetAllocator] = [
        basic_allocation,
        trimmed_allocation,
        optimized_allocation,
        adaptive_allocation,
        adaptive_mixture_floor_allocation,
        adaptive_no_floor_allocation,
        adaptive_crc_allocation,
    ]

    all_allocations.extend([
        RandomAdaptiveOptimizedBudgetAllocator(
            conditional_grid,
            budget_per_sample,
            taus_range,
            tau_prior,
            m_upper_bound,
            **alloc_kwargs,
        ),
        ConstantCRCBudgetAllocator(
            conditional_grid,
            budget_per_sample,
            taus_range,
            tau_prior,
            m_upper_bound,
            **alloc_kwargs,
        ),
        RandomAdaptiveOptimizedBudgetAllocator(
            conditional_grid,
            budget_per_sample,
            taus_range,
            tau_prior,
            m_upper_bound,
            schedule_family="power_reach",
            schedule_alpha=2.0,
            budget_control_mode="crc",
            **alloc_kwargs,
        ),
    ])

    if bound_type == "lpb":
        for n1 in dapro_n1_values:
            for margin in definitive_dapro_margins:
                all_allocations.append(DefinitiveDAPRO(
                    conditional_grid,
                    budget_per_sample,
                    taus_range,
                    tau_prior,
                    m_upper_bound,
                    n1=n1,
                    projection_budget_margin=margin,
                    **alloc_kwargs,
                ))
            all_allocations.extend([
                LegacyMeanWeightDAPRO(
                    conditional_grid,
                    budget_per_sample,
                    taus_range,
                    tau_prior,
                    m_upper_bound,
                    projection="direct_bins_2",
                    score="prob",
                    n1=n1,
                    **alloc_kwargs,
                ),
                TargetAWeightedDAPRO(
                    conditional_grid,
                    budget_per_sample,
                    taus_range,
                    tau_prior,
                    m_upper_bound,
                    projection="direct_bins_2",
                    score="prob",
                    n1=n1,
                    anchor_kind="raw_alpha",
                    target_alpha=0.10,
                    **alloc_kwargs,
                ),
            ])
            if n1 >= 50:
                control_size = min(100, n1 // 2)
                crc_kwargs = {
                    "budget_control_mode": "crc",
                    "budget_control_size": control_size,
                    "risk_candidate_row_cost_cap": 2.0 * budget_per_sample,
                }
                all_allocations.extend([
                    LegacyMeanWeightDAPRO(
                        conditional_grid,
                        budget_per_sample,
                        taus_range,
                        tau_prior,
                        m_upper_bound,
                        projection="direct_bins_2",
                        score="prob",
                        n1=n1,
                        **crc_kwargs,
                        **alloc_kwargs,
                    ),
                    TargetAWeightedDAPRO(
                        conditional_grid,
                        budget_per_sample,
                        taus_range,
                        tau_prior,
                        m_upper_bound,
                        projection="direct_bins_2",
                        score="prob",
                        n1=n1,
                        anchor_kind="raw_alpha",
                        target_alpha=0.10,
                        **crc_kwargs,
                        **alloc_kwargs,
                    ),
                    DefinitiveCRCDAPRO(
                        conditional_grid,
                        budget_per_sample,
                        taus_range,
                        tau_prior,
                        m_upper_bound,
                        n1=n1,
                        budget_control_size=control_size,
                        row_cost_cap_multiplier=2.0,
                        **alloc_kwargs,
                    ),
                ])
        for n1 in dapro_n1_values:
            all_allocations.append(SplitOracleTargetADAPRO(
                conditional_grid,
                budget_per_sample,
                taus_range,
                tau_prior,
                m_upper_bound,
                n1=n1,
                **alloc_kwargs,
            ))
            if n1 >= 2:
                all_allocations.append(CRCOracleTargetADAPRO(
                    conditional_grid,
                    budget_per_sample,
                    taus_range,
                    tau_prior,
                    m_upper_bound,
                    n1=n1,
                    budget_control_size=min(100, n1 // 2),
                    **alloc_kwargs,
                ))
        all_allocations.append(GlobalOracleTargetADAPRO(
            conditional_grid,
            budget_per_sample,
            taus_range,
            tau_prior,
            m_upper_bound,
            **alloc_kwargs,
        ))
    else:
        for n1 in dapro_n1_values:
            if n1 >= 100:
                all_allocations.append(DefinitiveCRCUPBDAPRO(
                    conditional_grid,
                    budget_per_sample,
                    taus_range,
                    tau_prior,
                    m_upper_bound,
                    n1=n1,
                    budget_control_size=min(100, n1 // 2),
                    row_cost_cap_multiplier=2.0,
                ))
    # if bound_type == "lpb":
    #     for n1 in dapro_n1_values:
    #         if is_budget_sufficient_for_split(
    #                 len(conditional_grid),
    #                 n1,
    #                 total_budget,
    #                 t_tilde_cal,
    #                 prior_q,
    #         ):
    #             for margin in definitive_dapro_margins:
    #                 all_allocations.append(DefinitiveDAPRO(
    #                     conditional_grid,
    #                     budget_per_sample,
    #                     taus_range,
    #                     tau_prior,
    #                     m_upper_bound,
    #                     n1=n1,
    #                     projection_budget_margin=margin,
    #                     **alloc_kwargs,
    #                 ))
    #             if n1 >= 100:
    #                 all_allocations.append(DefinitiveCRCDAPRO(
    #                     conditional_grid,
    #                     budget_per_sample,
    #                     taus_range,
    #                     tau_prior,
    #                     m_upper_bound,
    #                     n1=n1,
    #                     budget_control_size=min(100, n1 // 2),
    #                     row_cost_cap_multiplier=None,
    #                     **alloc_kwargs,
    #                 ))
    #                 for row_cost_cap_multiplier in [1.0, 2.0]:
    #                     all_allocations.append(DefinitiveCRCDAPRO(
    #                         conditional_grid,
    #                         budget_per_sample,
    #                         taus_range,
    #                         tau_prior,
    #                         m_upper_bound,
    #                         n1=n1,
    #                         budget_control_size=min(100, n1 // 2),
    #                         row_cost_cap_multiplier=(
    #                             row_cost_cap_multiplier
    #                         ),
    #                         **alloc_kwargs,
    #                     ))
    # else:
    #     for n1 in dapro_n1_values:
    #         if (
    #                 n1 >= 100
    #                 and is_budget_sufficient_for_split(
    #                     len(conditional_grid),
    #                     n1,
    #                     total_budget,
    #                     t_tilde_cal,
    #                     prior_q,
    #                 )
    #         ):
    #             all_allocations.append(DefinitiveCRCUPBDAPRO(
    #                 conditional_grid,
    #                 budget_per_sample,
    #                 taus_range,
    #                 tau_prior,
    #                 m_upper_bound,
    #                 n1=n1,
    #                 budget_control_size=min(100, n1 // 2),
    #                 row_cost_cap_multiplier=2.0,
    #             ))

    # for projection in ['platt', 'beta']:
    #     for score in ['prob', 'quantile']:
    # for projection in [
    #     'platt',
    #     'cumulative_platt',
    #     'direct_time',
    #     'direct_bins_2',
    #     'direct_bins_4',
    # ]:
    #     for score in ['prob', ]:
    #         for n1 in dapro_n1_values:
    #             if is_budget_sufficient_for_split(N, n1, total_budget, censored_event_time, prior_q):
    #                 all_allocations.append(LegacyMeanWeightDAPRO(conditional_grid, budget_per_sample, taus_range,
    #                                              tau_prior, m_upper_bound, projection=projection,
    #                                              evaluate_projection=evaluate_dapro_projection,
    #                                              score=score, n1=n1, **alloc_kwargs))
    #                 if bound_type == 'lpb':
    #                     all_allocations.append(
    #                         AWeightedDAPRO(
    #                             conditional_grid,
    #                             budget_per_sample,
    #                             taus_range,
    #                             tau_prior,
    #                             m_upper_bound,
    #                             projection=projection,
    #                             score=score,
    #                             n1=n1,
    #                             **alloc_kwargs,
    #                         )
    #                     )
    #                     for anchor_kind in [
    #                         "raw_alpha",
    #                         "phase1_unweighted",
    #                     ]:
    #                         all_allocations.append(
    #                             TargetAWeightedDAPRO(
    #                                 conditional_grid,
    #                                 budget_per_sample,
    #                                 taus_range,
    #                                 tau_prior,
    #                                 m_upper_bound,
    #                                 projection=projection,
    #                                 score=score,
    #                                 n1=n1,
    #                                 anchor_kind=anchor_kind,
    #                                 target_alpha=0.10,
    #                                 **alloc_kwargs,
    #                             )
    #                         )
    #                     if projection == "direct_time":
    #                         if n1 >= 100:
    #                             for target_policy_fraction in [
    #                                 0.00,
    #                                 0.25,
    #                                 0.50,
    #                                 0.75,
    #                                 1.00,
    #                             ]:
    #                                 all_allocations.append(
    #                                     RandomAnchoredTargetAWeightedDAPRO(
    #                                         conditional_grid,
    #                                         budget_per_sample,
    #                                         taus_range,
    #                                         tau_prior,
    #                                         m_upper_bound,
    #                                         projection=projection,
    #                                         score=score,
    #                                         n1=n1,
    #                                         target_alpha=0.10,
    #                                         target_policy_fraction=(
    #                                             target_policy_fraction
    #                                         ),
    #                                         budget_control_mode="crc",
    #                                         budget_control_size=n1 // 2,
    #                                         **alloc_kwargs,
    #                                     )
    #                                 )
    #                                 if target_policy_fraction in {0.50, 0.75}:
    #                                     all_allocations.append(
    #                                         RandomAnchoredTargetAWeightedDAPRO(
    #                                             conditional_grid,
    #                                             budget_per_sample,
    #                                             taus_range,
    #                                             tau_prior,
    #                                             m_upper_bound,
    #                                             projection=projection,
    #                                             score=score,
    #                                             n1=n1,
    #                                             target_alpha=0.10,
    #                                             target_policy_fraction=(
    #                                                 target_policy_fraction
    #                                             ),
    #                                             fill_random_slack=True,
    #                                             budget_control_mode="crc",
    #                                             budget_control_size=n1 // 2,
    #                                             **alloc_kwargs,
    #                                         )
    #                                     )
    #                         for robustness_weight in [0.10, 0.50]:
    #                             all_allocations.append(
    #                                 RobustTargetAWeightedDAPRO(
    #                                     conditional_grid,
    #                                     budget_per_sample,
    #                                     taus_range,
    #                                     tau_prior,
    #                                     m_upper_bound,
    #                                     projection=projection,
    #                                     score=score,
    #                                     n1=n1,
    #                                     target_alpha=0.10,
    #                                     robustness_weight=robustness_weight,
    #                                     **alloc_kwargs,
    #                                 )
    #                             )
    #                         for global_regularization in [0.001, 0.01, 0.05]:
    #                             all_allocations.append(
    #                                 RegularizedTargetAWeightedDAPRO(
    #                                     conditional_grid,
    #                                     budget_per_sample,
    #                                     taus_range,
    #                                     tau_prior,
    #                                     m_upper_bound,
    #                                     projection=projection,
    #                                     score=score,
    #                                     n1=n1,
    #                                     target_alpha=0.10,
    #                                     global_regularization=(
    #                                         global_regularization
    #                                     ),
    #                                     **alloc_kwargs,
    #                                 )
    #                             )
    #                             all_allocations.append(
    #                                 RegularizedTargetAWeightedDAPRO(
    #                                     conditional_grid,
    #                                     budget_per_sample,
    #                                     taus_range,
    #                                     tau_prior,
    #                                     m_upper_bound,
    #                                     projection=projection,
    #                                     score=score,
    #                                     n1=n1,
    #                                     anchor_kind="phase1_unweighted",
    #                                     target_alpha=0.10,
    #                                     global_regularization=(
    #                                         global_regularization
    #                                     ),
    #                                     **alloc_kwargs,
    #                                 )
    #                             )
    #                         all_allocations.append(
    #                             BandRegularizedTargetAWeightedDAPRO(
    #                                 conditional_grid,
    #                                 budget_per_sample,
    #                                 taus_range,
    #                                 tau_prior,
    #                                 m_upper_bound,
    #                                 projection=projection,
    #                                 score=score,
    #                                 n1=n1,
    #                                 target_alphas=tuple(
    #                                     0.07 + 0.01 * offset
    #                                     for offset in range(7)
    #                                 ),
    #                                 global_regularization=0.01,
    #                                 **alloc_kwargs,
    #                             )
    #                         )
    # all_allocations.extend([
    #     RandomAdaptiveOptimizedBudgetAllocator(
    #         conditional_grid,
    #         budget_per_sample,
    #         taus_range,
    #         tau_prior,
    #         m_upper_bound,
    #         **alloc_kwargs,
    #     ),
    #     RandomAdaptiveOptimizedBudgetAllocator(
    #         conditional_grid,
    #         budget_per_sample,
    #         taus_range,
    #         tau_prior,
    #         m_upper_bound,
    #         terminal_pi_min=1.0 / float(m_upper_bound),
    #         terminal_floor_mode="hard",
    #         **alloc_kwargs,
    #     ),
    #     RandomAdaptiveOptimizedBudgetAllocator(
    #         conditional_grid,
    #         budget_per_sample,
    #         taus_range,
    #         tau_prior,
    #         m_upper_bound,
    #         terminal_pi_min=None,
    #         terminal_floor_mode="none",
    #         **alloc_kwargs,
    #     ),
    #     RandomAdaptiveOptimizedBudgetAllocator(
    #         conditional_grid,
    #         budget_per_sample,
    #         taus_range,
    #         tau_prior,
    #         m_upper_bound,
    #         budget_control_mode="crc",
    #         **alloc_kwargs,
    #     ),
    #     RandomAdaptiveOptimizedBudgetAllocator(
    #         conditional_grid,
    #         budget_per_sample,
    #         taus_range,
    #         tau_prior,
    #         m_upper_bound,
    #         terminal_pi_min=1.0 / float(m_upper_bound),
    #         terminal_floor_mode="hard",
    #         budget_control_mode="crc",
    #         **alloc_kwargs,
    #     ),
    # ])
    # for schedule_family in ["complement_power", "power_reach"]:
    #     for schedule_alpha in [0.5, 1.0, 2.0]:
    #         all_allocations.append(
    #             RandomAdaptiveOptimizedBudgetAllocator(
    #                 conditional_grid,
    #                 budget_per_sample,
    #                 taus_range,
    #                 tau_prior,
    #                 m_upper_bound,
    #                 schedule_family=schedule_family,
    #                 schedule_alpha=schedule_alpha,
    #                 budget_control_mode="crc",
    #                 **alloc_kwargs,
    #             )
    #         )

    if bound_type == 'lpb':
        dummy_calibration = UncalibratedLPBSurvivalCalibration(taus_range)
        oracle_calibration = OracleSurvivalCalibration(taus_range, tau_prior)
        all_calibrations: List[SurvivalLPBCalibration] = [dummy_calibration, oracle_calibration]
        all_calibrations.extend([SurvivalCalibrationWithKnownWeights(allocation, taus_range, tau_prior) for
                                 allocation in all_allocations])
    else:
        dummy_calibration = UncalibratedUPBSurvivalCalibration(taus_range)
        oracle_calibration = OracleSurvivalUPBCalibration(
            taus_range, tau_prior
        )
        all_calibrations: List[SurvivalUPBCalibration] = [
            dummy_calibration,
            oracle_calibration,
        ]
        all_calibrations.extend([SurvivalUPBCalibrationWithKnownWeights(allocation, taus_range, tau_prior) for
                                 allocation in all_allocations])

    return all_calibrations


def get_calibration_methods(conditional_grid, budget_per_sample, taus_range, tau_prior, m_upper_bound, allocations,
                            cal_model_prediction, t_tilde_cal, device, bound_type,
                            evaluate_dapro_projection=False,
                            dapro_n1_values=(200,100, 50),
                            definitive_dapro_margins=(1.0,)):
    baseline_calibrations = get_baseline_calibrations(
        conditional_grid, budget_per_sample, taus_range, tau_prior,
        m_upper_bound, cal_model_prediction, t_tilde_cal, bound_type,
        evaluate_dapro_projection,
        dapro_n1_values=dapro_n1_values,
        definitive_dapro_margins=definitive_dapro_margins,
    )
    return baseline_calibrations


def main():
    parser = argparse.ArgumentParser(description="Construct Calibrated LPB or UPB")
    parser.add_argument('--bound-type', type=str, choices=['lpb', 'upb'], default='lpb',
                        help="Whether to compute LPB or UPB")
    parser.add_argument('--seed-start', type=int, default=0)
    parser.add_argument('--seed-end', type=int, default=1)
    parser.add_argument('--allocations', type=str, default='all')
    parser.add_argument('--data-type', type=str, default='real')
    parser.add_argument('--dataset-name', type=str, default='')
    parser.add_argument('--dataset-setup', type=str, default='')
    parser.add_argument('--budget-per-sample', type=float, default=1)
    parser.add_argument('--cal-size', type=int, default=4000)
    parser.add_argument('--tau-prior', type=float, default=None, help="Prior for tau (defaults depend on bound-type)")
    parser.add_argument(
        '--gamma',
        type=float,
        default=None,
        help="Optional ratio m_upper_bound / budget_per_sample.",
    )
    parser.add_argument(
        '--m-upper-bound',
        type=float,
        default=None,
        help="Optional explicit interaction horizon.",
    )
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument(
        '--experiment-suffix',
        type=str,
        default='',
        help="Optional suffix used to isolate this run from existing results.",
    )
    parser.add_argument(
        '--calibration-names',
        type=str,
        default='',
        help=(
            "Optional comma-separated exact calibration names. This permits "
            "running only newly added methods without overwriting other "
            "method/seed files in the same versioned experiment."
        ),
    )
    parser.add_argument(
        '--evaluate-dapro-projection',
        action='store_true',
        help=(
            "Run the expensive full-data DAPRO oracle diagnostic. This does "
            "not affect the deployed allocation."
        ),
    )
    parser.add_argument(
        '--dapro-n1-values',
        type=int,
        nargs='+',
        default=[DAPRO.DEFAULT_N1],
        help=(
            "Phase-I sample sizes to instantiate for every DAPRO-family "
            "method. The definitive DAPRO default is N1=200."
        ),
    )
    parser.add_argument(
        '--definitive-dapro-margins',
        type=float,
        nargs='+',
        default=[DAPRO.DEFAULT_PROJECTION_BUDGET_MARGIN],
        help=(
            "Projection-error reserves, in expected interactions per "
            "Phase-II sample, used by the assumption-based projection-DAPRO "
            "ablation. The public DAPRO method uses an independent CRC fold."
        ),
    )
    parser.add_argument(
        '--fixed-data-seed',
        type=int,
        default=None,
        help=(
            "Keep the calibration/test split fixed while seed-start:seed-end "
            "index independent acquisition replicates."
        ),
    )
    parser.add_argument(
        '--fixed-policy-seed',
        type=int,
        default=None,
        help=(
            "Keep Phase-I sampling and policy fitting fixed while the outer "
            "seed controls only stochastic acquisition. Use together with "
            "--fixed-data-seed to estimate conditional acquisition variance."
        ),
    )
    parser.add_argument(
        '--fixed-acquisition-seed',
        type=int,
        default=None,
        help=(
            "Keep stochastic acquisition fixed while the outer seed varies "
            "the data split and/or Phase-I policy fit. Use with "
            "--fixed-data-seed to isolate policy-fit variance."
        ),
    )

    args = parser.parse_args()
    args.is_real = True if args.data_type.lower() == 'real' else False
    bound_type = args.bound_type

    seed_start = args.seed_start
    seed_end = args.seed_end
    dataset_name = args.dataset_name
    data_setup = args.dataset_setup
    is_real = args.is_real
    cal_size = args.cal_size
    if (
        len(set(args.dapro_n1_values)) != len(args.dapro_n1_values)
        or any(
            n1 <= 0 or n1 >= cal_size
            for n1 in args.dapro_n1_values
        )
    ):
        parser.error(
            "--dapro-n1-values must contain distinct integers between 1 "
            "and cal-size - 1."
        )
    if (
            len(set(args.definitive_dapro_margins))
            != len(args.definitive_dapro_margins)
            or any(
                not np.isfinite(margin) or margin < 0
                for margin in args.definitive_dapro_margins
            )
    ):
        parser.error(
            "--definitive-dapro-margins must contain distinct finite "
            "nonnegative values."
        )

    # Contextual initialization parameters dependent on bound-type choice
    if bound_type == 'lpb':
        tau_prior = args.tau_prior if args.tau_prior is not None else 0.56
        target_taus_list = np.arange(0.01, 0.5, 0.01)
    else:
        tau_prior = args.tau_prior if args.tau_prior is not None else 0.98
        target_taus_list = 1 - np.arange(0.01, 0.5, 0.01)
        num_taus = 3000

    device = (
        args.device
        if torch.cuda.is_available() and 'cuda' in args.device
        else 'cpu'
    )
    set_seeds(0)

    seeds = (seed_start, seed_end)

    if bound_type == 'lpb':
        taus_range = make_lpb_tau_grid(device=device)
    else:
        taus_range = torch.tensor(np.linspace(0.5, 0.95, num_taus)).to(device)

    budget_per_sample = args.budget_per_sample

    try:
        m_upper_bound = resolve_m_upper_bound(
            is_real,
            budget_per_sample,
            gamma=args.gamma,
            m_upper_bound=args.m_upper_bound,
        )
    except ValueError as error:
        parser.error(str(error))

    gamma = get_gamma(m_upper_bound, budget_per_sample)
    allocations = args.allocations
    calibration_names = [
        name.strip()
        for name in args.calibration_names.split(',')
        if name.strip()
    ]

    print(f"Executing for Bound Type: {bound_type.upper()}")
    print(f"budget_per_sample: {budget_per_sample}, gamma: {gamma}, m_upper_bound: {m_upper_bound}")

    try:
        experiments_name = get_calibration_experiment_name(
            dataset_name,
            data_setup,
            budget_per_sample,
            cal_size,
            tau_prior,
            gamma,
            args.experiment_suffix,
        )
    except ValueError as error:
        parser.error(str(error))

    run_experiments(cal_size, is_real, device, dataset_name, data_setup, experiments_name, seeds,
                    taus_range, budget_per_sample, tau_prior, m_upper_bound, target_taus_list,
                    skip_existing=True, allocations=allocations, bound_type=bound_type,
                    calibration_names=calibration_names,
                    evaluate_dapro_projection=args.evaluate_dapro_projection,
                    fixed_data_seed=args.fixed_data_seed,
                    fixed_policy_seed=args.fixed_policy_seed,
                    fixed_acquisition_seed=args.fixed_acquisition_seed,
                    dapro_n1_values=tuple(args.dapro_n1_values),
                    definitive_dapro_margins=tuple(
                        args.definitive_dapro_margins
                    ))

    print("Finished")


if __name__ == '__main__':
    main()
