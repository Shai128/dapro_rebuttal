"""Run only the Random terminal-floor variants on an existing data cache.

This deliberately lives outside the main method registry so a focused floor
ablation does not require constructing every DAPRO projection.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.safety_evaluation.budget_allocators.random_adaptive_optimized_allocator import (
    RandomAdaptiveOptimizedBudgetAllocator,
)
from src.safety_evaluation.calibration.survival_calibration_with_known_weights import (
    SurvivalCalibrationWithKnownWeights,
)
from src.safety_evaluation.construct_calibrated_bound import (
    _index_fingerprint,
    _safety_evaluation_source_fingerprint,
    run_one_experiment,
)
from src.safety_evaluation.utils.utils import (
    get_calibration_experiment_name,
    get_tmp_calibration_result_path,
    setup_experiment_data,
    split_data,
)
from src.train_model.models.utils import SurvivalModelPrediction
from src.utils.utils import set_seeds


def _absolute_io_path(path):
    absolute = os.path.abspath(path)
    if os.name == "nt" and not absolute.startswith("\\\\?\\"):
        return f"\\\\?\\{absolute}"
    return absolute


def _allocator_variants(
        conditional_grid,
        budget_per_sample,
        taus_range,
        tau_prior,
        m_upper_bound):
    common = (
        conditional_grid,
        budget_per_sample,
        taus_range,
        tau_prior,
        m_upper_bound,
    )
    return [
        RandomAdaptiveOptimizedBudgetAllocator(*common),
        RandomAdaptiveOptimizedBudgetAllocator(
            *common,
            terminal_pi_min=1 / float(m_upper_bound),
            terminal_floor_mode="hard",
        ),
        RandomAdaptiveOptimizedBudgetAllocator(
            *common,
            terminal_pi_min=None,
            terminal_floor_mode="none",
        ),
    ]


def _load_and_summarize(experiment_name, methods, seeds):
    frames = []
    for method in methods:
        directory = get_tmp_calibration_result_path(
            experiment_name,
            f"calibration_{method.name}_allocation",
        )
        for seed in seeds:
            frame = pd.read_csv(
                _absolute_io_path(
                    os.path.join(directory, f"seed={seed}.csv")
                )
            )
            frame["floor_ablation_seed"] = seed
            frames.append(frame)
    all_df = pd.concat(frames, ignore_index=True)
    target = all_df.loc[
        np.isclose(all_df["target_coverage"], 0.90)
    ].copy()
    grouped = target.groupby("calibration_name", sort=True)
    summary = grouped.agg(
        n_seeds=("floor_ablation_seed", "nunique"),
        coverage_mean=("coverage", "mean"),
        coverage_variance=("coverage", "var"),
        lpb_mean=("size", "mean"),
        mean_weight=("mean_weight", "mean"),
        max_weight=("max_weight", "max"),
        mean_a_over_pi=(
            "mean_a_weighted_inverse_probability",
            "mean",
        ),
        mean_a_inverse_excess=(
            "mean_a_weighted_inverse_probability_minus_one",
            "mean",
        ),
        expected_budget_per_sample=(
            "total_expected_budget_per_sample",
            "mean",
        ),
        expected_budget_gap_per_sample=(
            "total_expected_budget_gap_per_sample",
            "mean",
        ),
        expected_budget_violation_rate=(
            "total_expected_budget_valid",
            lambda values: 1 - values.mean(),
        ),
        constant_probability=(
            "random_constant_probability",
            "mean",
        ),
        phase2_objective_transfer_gap=(
            "phase2_expected_budget_gap_per_sample",
            "mean",
        ),
    ).reset_index()

    output_dir = Path(_absolute_io_path(os.path.join(
        "results",
        "merged_calibration_dfs",
        experiment_name,
    )))
    output_dir.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(output_dir / "random_floor_all_df.csv", index=False)
    summary.to_csv(
        output_dir / "random_floor_summary.csv",
        index=False,
    )
    print(summary.to_string(index=False))
    print(f"Stored floor ablation at {output_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-setup", required=True)
    parser.add_argument("--cal-size", type=int, default=3000)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=50)
    parser.add_argument("--budget-per-sample", type=float, default=20)
    parser.add_argument("--tau-prior", type=float, default=0.56)
    parser.add_argument("--m-upper-bound", type=float, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--experiment-suffix",
        default="random_floor_ablation",
    )
    args = parser.parse_args()

    device = (
        args.device
        if torch.cuda.is_available() and "cuda" in args.device
        else "cpu"
    )
    set_seeds(0)
    taus_range = torch.tensor(
        np.logspace(-3, -0.01, 1000),
        device=device,
    )
    target_taus = np.arange(0.01, 0.5, 0.01)
    gamma = args.m_upper_bound / args.budget_per_sample
    experiment_name = get_calibration_experiment_name(
        args.dataset_name,
        args.dataset_setup,
        args.budget_per_sample,
        args.cal_size,
        args.tau_prior,
        gamma,
        args.experiment_suffix,
    )
    (
        max_time,
        t_cal_test,
        q_cal_test,
        probability_est,
        conditional_grid,
        test_size,
    ) = setup_experiment_data(
        args.cal_size,
        True,
        device,
        args.dataset_name,
        args.dataset_setup,
        taus_range,
        args.m_upper_bound,
    )
    source_sha = _safety_evaluation_source_fingerprint()
    method_prototypes = _allocator_variants(
        conditional_grid[:1],
        args.budget_per_sample,
        taus_range,
        args.tau_prior,
        args.m_upper_bound,
    )

    for seed in range(args.seed_start, args.seed_end):
        (
            x_cal,
            x_test,
            t_cal,
            probability_cal,
            q_cal,
            t_test,
            q_test,
            probability_test,
            cal_idx,
            test_idx,
        ) = split_data(
            seed,
            args.cal_size,
            test_size,
            None,
            t_cal_test,
            probability_est,
            q_cal_test,
        )
        q_cal = q_cal.clip(max=max_time)
        cal_prediction = SurvivalModelPrediction(q_cal, probability_cal)
        test_prediction = SurvivalModelPrediction(q_test, probability_test)
        methods = _allocator_variants(
            conditional_grid[cal_idx],
            args.budget_per_sample,
            taus_range,
            args.tau_prior,
            args.m_upper_bound,
        )
        for allocator in methods:
            calibration = SurvivalCalibrationWithKnownWeights(
                allocator,
                taus_range,
                args.tau_prior,
            )
            run_one_experiment(
                experiment_name,
                seed,
                calibration,
                x_cal,
                t_cal,
                cal_prediction,
                x_test,
                t_test,
                test_prediction,
                target_taus,
                "lpb",
                skip_existing=False,
                policy_seed=seed,
                acquisition_seed=seed,
                experiment_metadata={
                    "experiment_name": experiment_name,
                    "configured_cal_size": args.cal_size,
                    "configured_budget_per_sample": (
                        args.budget_per_sample
                    ),
                    "configured_tau_prior": args.tau_prior,
                    "configured_m_upper_bound": args.m_upper_bound,
                    "execution_device": str(conditional_grid.device),
                    "safety_evaluation_source_sha256": source_sha,
                    "calibration_split_sha256": _index_fingerprint(cal_idx),
                    "test_split_sha256": _index_fingerprint(test_idx),
                    "data_split_seed": seed,
                    "policy_rng_seed": seed,
                    "acquisition_rng_seed": seed,
                    "acquisition_rng_reseeded": 1,
                },
            )

    _load_and_summarize(
        experiment_name,
        method_prototypes,
        range(args.seed_start, args.seed_end),
    )


if __name__ == "__main__":
    main()
