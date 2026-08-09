"""Estimate allocation metrics from cached trajectories and model outputs."""

import os
import traceback
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Union

import numpy as np
import pandas as pd
import torch
import tqdm

from src.predictive_bounds.budget_allocators.budget_allocator import BudgetAllocator
from src.predictive_bounds.utils.get_calibration_methods_utils import get_metric_allocators
from src.predictive_bounds.utils.utils import (
    get_tmp_metric_calibration_result_path,
    setup_experiment_data,
    split_data,
)
from src.utils.utils import set_seeds
from src.train_model.models.utils import SurvivalModelPrediction


# ==========================================
# 1. DATA STRUCTURES
# ==========================================

@dataclass
class TrajectoryData:
    """Data Transfer Object (DTO) holding simulated trajectory outcomes."""
    N: int
    max_time: int
    t_tilde: torch.Tensor
    C_i: torch.Tensor  # Censoring time (when algorithm halted)
    Y_i: torch.Tensor  # Observed time min(T_i, C_i)
    total_budget_utilized: int  # Observed time min(T_i, C_i)
    Delta_i: torch.Tensor  # Event indicator I(T_i <= C_i)
    W_i: torch.Tensor  # IPCW weights (1 / prod P_i(k))
    device: torch.device
    allocation_metrics: Dict[str, Any]


# ==========================================
# 2. SIMULATOR
# ==========================================

class IPCWTrajectorySimulator:
    """Simulates or extracts right-censored trajectories given an allocator's probabilities."""

    @staticmethod
    def simulate(allocator: BudgetAllocator, x: torch.Tensor, model_prediction: SurvivalModelPrediction,
                 t_tilde: torch.Tensor, max_time: int) -> TrajectoryData:
        N = len(t_tilde)
        device = t_tilde.device
        result = allocator.allocate_budget(
            model_prediction.probability_est, x, t_tilde, model_prediction.quantile_est)
        C_i = result.C.squeeze()
        Y_i = torch.minimum(t_tilde.long(), C_i.long())
        # C_i is a number of acquired turns.  An event on the final acquired
        # turn is observed, hence the non-strict comparison.
        Delta_i = t_tilde.long() <= C_i.long()
        W_i = result.C_probs.squeeze()
        # Enforce strict positivity for IPCW stability if needed
        # W_i = torch.clamp(W_i, min=1e-5)
        total_budget_used = result.total_budget_used if result.total_budget_used is not None else Y_i.sum().item()
        return TrajectoryData(
            N=N, max_time=max_time, t_tilde=t_tilde,
            C_i=C_i, Y_i=Y_i, Delta_i=Delta_i, W_i=W_i, device=device,
            total_budget_utilized=total_budget_used,
            allocation_metrics=result.additional_metrics or {},
        )


# ==========================================
# 3. METRICS DOMAIN (OCP & ISP)
# ==========================================

class SafetyMetric(ABC):
    """Abstract Base Class for all safety evaluation metrics."""

    @abstractmethod
    def compute(self, data: TrajectoryData) -> Dict[str, Any]:
        pass


def _observed_event_ipcw(data: TrajectoryData) -> torch.Tensor:
    """Return safe IPCW contributions without producing ``0 / 0`` NaNs."""
    propensities = data.W_i.reshape(-1).to(torch.float64)
    observed = data.Delta_i.reshape(-1).bool()
    if bool((propensities[observed] <= 0).any()):
        raise ValueError(
            "observed events must have strictly positive inclusion propensities")
    return torch.where(
        observed,
        propensities.reciprocal(),
        torch.zeros_like(propensities),
    )


class CumulativeJailbreakRateMetric(SafetyMetric):
    def __init__(self, oracle_cjr: float):
        self.oracle_cjr = oracle_cjr

    def compute(self, data: TrajectoryData) -> Dict[str, Any]:
        observed_event = data.Y_i <= data.max_time
        est_cjr = (
            _observed_event_ipcw(data) * observed_event.to(torch.float64)
        ).mean().item()

        return {
            'oracle_cjr': self.oracle_cjr * 100,
            'estimated_cjr': est_cjr * 100,
            'abs_diff_cjr': abs(est_cjr - self.oracle_cjr) * 100
        }


class TotalBudgetUsed(SafetyMetric):
    def __init__(self):
        pass

    def compute(self, data: TrajectoryData) -> dict:
        return {
            'total_budget_utilized': data.total_budget_utilized,
            'budget_per_sample': data.total_budget_utilized / data.N,
        }


class ObservedJailbreaks(SafetyMetric):
    def __init__(self):
        pass

    def compute(self, data: TrajectoryData) -> dict:
        true_event = data.t_tilde <= data.max_time
        observed = true_event & data.Delta_i
        return {
            'observed_jailbreaks': observed.float().sum().item(),
            'num_events_observed': observed.float().sum().item(),
        }


class AllocationEstimationDiagnostics(SafetyMetric):
    """Diagnostics for the Horvitz-Thompson unsafe-event-rate estimator."""

    def compute(self, data: TrajectoryData) -> Dict[str, Any]:
        propensities = data.W_i.reshape(-1).to(torch.float64)
        if bool((propensities <= 0).any()):
            raise ValueError("all inclusion propensities must be positive")
        inverse = propensities.reciprocal()
        target_a = (data.t_tilde.reshape(-1) <= data.max_time).to(
            torch.float64
        )
        observed_target_a = (
            data.Delta_i.reshape(-1).bool()
            & (data.Y_i.reshape(-1) <= data.max_time)
        ).to(torch.float64)
        a_weight = target_a * inverse
        sum_a_weight = a_weight.sum()
        ess = (
            (sum_a_weight.square() / a_weight.square().sum()).item()
            if bool((a_weight != 0).any())
            else 0.0
        )
        resolved = data.Delta_i.reshape(-1) | (
            data.C_i.reshape(-1) >= data.max_time
        )
        results = {
            "mean_weight": inverse.mean().item(),
            "mean_inverse_probability": inverse.mean().item(),
            "mean_inverse_probability_minus_one": (inverse - 1).mean().item(),
            "variance_weight": inverse.var(unbiased=False).item(),
            "max_weight": inverse.max().item(),
            "mean_a_weighted_weight": a_weight.mean().item(),
            "mean_metric_a_weighted_inverse_probability": (
                a_weight.mean().item()
            ),
            "mean_metric_a_weighted_inverse_probability_minus_one": (
                (target_a * (inverse - 1)).mean().item()
            ),
            "metric_a_weighted_effective_sample_size": ess,
            "metric_target_a_rate": target_a.mean().item(),
            "metric_target_a_count": target_a.sum().item(),
            "num_trajectories_fully_resolved": resolved.sum().item(),
            "fraction_trajectories_fully_resolved": resolved.float().mean().item(),
            # Conditional allocation variance of the rate estimator, given
            # the fixed benchmark rows.  Percent-scale plots can multiply by
            # 100^2 if desired.
            "conditional_variance_unsafe_event_rate_estimator": (
                (target_a * (inverse - 1)).sum().item() / data.N ** 2
            ),
            # Design-unbiased estimate of the same conditional acquisition
            # variance.  Unlike the preceding offline diagnostic, this uses
            # only unsafe events actually observed under the policy:
            # E[R_i A_i (1-pi_i)/pi_i^2] = A_i(1/pi_i-1).
            "estimated_conditional_variance_unsafe_event_rate_estimator": (
                (
                    observed_target_a
                    * (1 - propensities)
                    / propensities.square()
                ).sum().item()
                / data.N ** 2
            ),
        }
        for key, value in data.allocation_metrics.items():
            if isinstance(value, (str, int, float, bool, np.number)):
                results[key] = value
            elif torch.is_tensor(value) and value.numel() == 1:
                results[key] = value.item()
        return results


class RestrictedMeanTimeToUnsafeMetric(SafetyMetric):
    def __init__(self, oracle_rmttu: float):
        self.oracle_rmttu = oracle_rmttu

    def compute(self, data: TrajectoryData) -> Dict[str, Any]:
        valid_observed_event = (data.Y_i <= data.max_time).to(torch.float64)
        event_ipcw = _observed_event_ipcw(data) * valid_observed_event

        # Weighted sum of times for observed failures
        estimated_sum_times = (
            event_ipcw * data.Y_i.to(torch.float64)).sum().item()

        # Weighted count MUST also use the exact same filter
        estimated_count_failures = event_ipcw.sum().item()

        if estimated_count_failures > 0:
            est_rmttu = estimated_sum_times / estimated_count_failures
        else:
            est_rmttu = float('nan')

        return {
            'oracle_rmttu': self.oracle_rmttu,
            'estimated_rmttu': est_rmttu,
            'abs_diff_rmttu': abs(est_rmttu - self.oracle_rmttu) if not np.isnan(est_rmttu) else float('nan')
        }


class CostPerJailbreakMetric(SafetyMetric):
    def compute(self, data: TrajectoryData) -> Dict[str, Any]:
        total_compute = data.Y_i.float().sum().item()

        observed_event = data.Y_i <= data.max_time
        est_cjr = (
            _observed_event_ipcw(data) * observed_event.to(torch.float64)
        ).mean().item()
        estimated_total_jailbreaks = data.N * est_cjr

        cpj = total_compute / \
            estimated_total_jailbreaks if estimated_total_jailbreaks > 0 else float(
                'nan')
        return {
            'total_compute_iterations': total_compute,
            'cost_per_jailbreak': cpj
        }


class IPCWHazardFunctionMetric(SafetyMetric):
    def compute(self, data: TrajectoryData) -> Dict[str, Any]:
        hazard_rates = []
        for t in range(1, data.max_time + 1):
            events_at_t = (data.Y_i == t) & (data.Delta_i == 1)
            at_risk_at_t = (data.Y_i >= t)

            weighted_events = (events_at_t.float() / data.W_i).sum()
            weighted_risk_set = (at_risk_at_t.float() / data.W_i).sum()

            if weighted_risk_set > 1e-5:
                hazard = (weighted_events / weighted_risk_set).item()
            else:
                hazard = 0.0
            hazard_rates.append(hazard)

        return {
            'hazard_function_array': json.dumps(hazard_rates)
        }


class SurvivalQuantilesMetric(SafetyMetric):
    def __init__(self, oracle_quantiles: Dict[str, float], quantiles: Union[List[float],None ]= None):
        self.quantiles = [0.25, 0.50, 0.75] if quantiles is None else quantiles
        self.oracle_quantiles = oracle_quantiles

    def compute(self, data: TrajectoryData) -> Dict[str, Any]:
        time_range = torch.arange(
            1, data.max_time + 1, device=data.device).unsqueeze(0)
        y_expanded = data.Y_i.unsqueeze(1)
        delta_expanded = data.Delta_i.unsqueeze(1)

        events_occurred = (y_expanded <= time_range) & (delta_expanded == 1)
        weighted_events = (
            events_occurred.to(torch.float64)
            * _observed_event_ipcw(data).unsqueeze(1)
        )
        S_t = 1.0 - weighted_events.mean(dim=0)

        results = {}
        for q in self.quantiles:
            target_prob = 1.0 - q
            below_target = (S_t <= target_prob)
            if below_target.any():
                t_q = torch.argmax(below_target.float()).item() + 1
            else:
                t_q = float('nan')
            results[f'estimated_quantile_{int(q * 100)}'] = t_q

        # Add pre-computed oracle quantiles
        for k, v in self.oracle_quantiles.items():
            results[k] = v

        return results


# ==========================================
# 4. NEW BUDGET ALLOCATORS
# ==========================================


# ==========================================
# 5. ENGINE & FACTORIES
# ==========================================

class MetricsEngine:
    """Executes a suite of SafetyMetrics on TrajectoryData."""

    def __init__(self, metrics: List[SafetyMetric]):
        self.metrics = metrics

    def evaluate(self, data: TrajectoryData) -> Dict[str, Any]:
        results = {}
        for metric in self.metrics:
            results.update(metric.compute(data))
        return results


# ==========================================
# 6. EXPERIMENT RUNNER
# ==========================================

def run_one_experiment(experiments_name, seed, allocator: BudgetAllocator, x, t_tilde, model_prediction, oracle_metrics,
                       max_time=200,
                       skip_existing=True):
    try:
        allocator_name = allocator.name
        dir_path = get_tmp_metric_calibration_result_path(
            experiments_name, allocator_name)
        save_path = os.path.join(f"{dir_path}", f"seed={seed}.csv")

        if os.path.exists(save_path) and skip_existing:
            return

        set_seeds(seed)

        with torch.no_grad():
            # 1. Simulate Trajectories
            traj_data = IPCWTrajectorySimulator.simulate(
                allocator, x, model_prediction, t_tilde, max_time
            )

            # 2. Setup and Run Metrics Engine
            engine = MetricsEngine([
                CumulativeJailbreakRateMetric(
                    oracle_cjr=oracle_metrics['cjr']),
                RestrictedMeanTimeToUnsafeMetric(
                    oracle_rmttu=oracle_metrics['rmttu']),
                TotalBudgetUsed(),
                ObservedJailbreaks(),
                AllocationEstimationDiagnostics(),
                CostPerJailbreakMetric(),
                # IPCWHazardFunctionMetric(),
                SurvivalQuantilesMetric(
                    oracle_quantiles=oracle_metrics['quantiles'], quantiles=[0.25, 0.50, 0.75])
            ])

            metrics_results = engine.evaluate(traj_data)

        # 3. Format and Save
        all_metrics = {
            'seed': seed,
            'allocator_name': allocator_name,
            'calibration_name': (
                allocator_name
                if allocator_name == 'oracle_full_budget'
                else f'calibration_{allocator_name}_allocation'
            ),
            'evaluation_sample_size': traj_data.N,
            'evaluation_scope': (
                'full_calibration_test_benchmark'
                if allocator_name == 'oracle_full_budget'
                else 'calibration_split'
            ),
            'full_benchmark_sample_size': oracle_metrics.get(
                'sample_size', np.nan
            ),
            'full_benchmark_cjr': oracle_metrics['cjr'] * 100,
            'full_benchmark_rmttu': oracle_metrics['rmttu'],
            **metrics_results
        }

        df = pd.DataFrame(all_metrics, index=[seed])

        abs_path = os.path.abspath(save_path)
        if os.name == 'nt' and not abs_path.startswith('\\\\?\\'):
            abs_path = f"\\\\?\\{abs_path}"
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)

        df.to_csv(abs_path, index=False)

    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(
            f"metric experiment failed for seed {seed} with allocator {allocator.name}"
        ) from e


def compute_oracle_metric(t_tilde_cal_test, max_time):
    # --- SINGLE ORACLE COMPUTATION ACROSS CAL+TEST ---
    full_times = t_tilde_cal_test.reshape(-1).to(torch.float64)
    true_event_all = (full_times <= max_time).to(torch.float64)
    global_oracle_cjr = true_event_all.mean().item()
    global_oracle_rmttu = full_times[full_times <= max_time].mean().item()
    global_oracle_quantiles = {}
    for q in [0.25, 0.50, 0.75]:
        global_oracle_quantiles[f'oracle_quantile_{int(q * 100)}'] = torch.quantile(
            full_times, q).item()

    oracle_metrics = {
        'cjr': global_oracle_cjr,
        'rmttu': global_oracle_rmttu,
        'quantiles': global_oracle_quantiles,
        'sample_size': len(full_times),
        'horizon': int(max_time),
    }
    return oracle_metrics


def metric_experiment_name(
        dataset_name,
        data_setup,
        budget_per_sample,
        dapro_n1=None,
        crc_control_size=None,
        experiment_suffix="",
):
    """Return the shared, compact directory name for metric experiments.

    N1 and CRC identify allocator methods inside the experiment and therefore
    do not belong in the experiment directory name.  The parameters remain in
    the signature for compatibility with existing callers.
    """
    budget_label = f"{float(budget_per_sample):g}"
    name = f"{dataset_name}_{data_setup}_{budget_label}_metric_estimation"
    return f"{name}_{experiment_suffix}" if experiment_suffix else name


REQUIRED_RESULT_COLUMNS = {
    "seed",
    "allocator_name",
    "calibration_name",
    "total_budget_utilized",
    "budget_per_sample",
    "estimated_cjr",
    "abs_diff_cjr",
    "mean_weight",
    "mean_a_weighted_weight",
    "num_events_observed",
    "conditional_variance_unsafe_event_rate_estimator",
    "estimated_conditional_variance_unsafe_event_rate_estimator",
}


def _completed_result_exists(path):
    if not os.path.exists(path):
        return False
    try:
        return REQUIRED_RESULT_COLUMNS.issubset(pd.read_csv(path, nrows=1).columns)
    except Exception:
        return False


def run_experiments(
        cal_size,
        is_real,
        device,
        dataset_name,
        data_setup,
        experiments_name,
        seeds,
        budget_per_sample,
        skip_existing,
        dapro_n1=200,
        crc_control_size=100,
        tau_prior=0.56,
        exclude_legacy_dapro=False,
        exclude_locally_adaptive=False,
        allocator_names=None,
):
    taus_range = torch.tensor(np.arange(0.01, 1.0, 0.01)).to(device)
    m_upper_bound = 200 if is_real else 20

    max_time, t_tilde_cal_test, quantile_est_cal_test, probability_est, conditional_grid, test_size = setup_experiment_data(
        cal_size, is_real, device, dataset_name, data_setup, taus_range, m_upper_bound
    )

    oracle_metrics = compute_oracle_metric(t_tilde_cal_test, max_time)

    for seed in tqdm.tqdm(
            range(seeds[0], seeds[1]),
            desc="running metric-estimation algorithms",
    ):
        x_cal, x_test, t_tilde_cal, probability_est_cal, quantile_est_cal, t_tilde_test, quantile_est_test, \
            probability_est_test, cal_idx, test_idx = split_data(seed, cal_size, test_size, None, t_tilde_cal_test,
                                                                 probability_est,
                                                                 quantile_est_cal_test)

        curr_conditional_grid = conditional_grid[cal_idx]
        quantile_est_cal = quantile_est_cal.clip(max=max_time)
        quantile_est_cal[:] = max_time

        cal_model_prediction = SurvivalModelPrediction(
            quantile_est_cal, probability_est_cal)
        allocators = get_metric_allocators(
            curr_conditional_grid,
            budget_per_sample,
            m_upper_bound,
            taus_range,
            tau_prior,
            device,
            t_tilde_cal,
            cal_model_prediction,
            dapro_n1=dapro_n1,
            crc_control_size=crc_control_size,
            include_legacy_dapro=not exclude_legacy_dapro,
            include_locally_adaptive=not exclude_locally_adaptive,
        )
        if allocator_names is not None:
            requested = set(allocator_names)
            available = {allocator.name for allocator in allocators}
            missing = requested - available
            if missing:
                raise ValueError(
                    "Unknown metric allocator name(s): "
                    + ", ".join(sorted(missing))
                )
            allocators = [
                allocator
                for allocator in allocators
                if allocator.name in requested
            ]
        common_uniforms = np.random.default_rng(seed).random(
            (cal_size, curr_conditional_grid.shape[1])
        )
        full_model_prediction = SurvivalModelPrediction(
            quantile_est_cal_test.clip(max=max_time),
            probability_est,
        )

        # Allocators use NumPy for their Phase-I splits.  Running them in
        # sequence and resetting the same seed gives every method the
        # same split without process-global RNG races between threads.
        for allocator in allocators:
            result_path = os.path.join(
                get_tmp_metric_calibration_result_path(
                    experiments_name, allocator.name
                ),
                f"seed={seed}.csv",
            )
            if skip_existing and _completed_result_exists(result_path):
                continue
            if not getattr(allocator, "uses_full_benchmark", False):
                allocator.set_acquisition_randomness(
                    seed=seed,
                    uniforms=common_uniforms,
                )
                run_x = x_cal
                run_t = t_tilde_cal
                run_prediction = cal_model_prediction
            else:
                run_x = None
                run_t = t_tilde_cal_test
                run_prediction = full_model_prediction
            run_one_experiment(
                experiments_name,
                seed,
                allocator,
                run_x,
                run_t,
                run_prediction,
                oracle_metrics,
                max_time,
                skip_existing=False,
            )

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Run advanced safety evaluation metrics.")
    parser.add_argument('--seed-start', type=int, default=0)
    parser.add_argument('--seed-end', type=int, default=50)
    parser.add_argument('--data-type', type=str)
    parser.add_argument('--dataset-name', type=str, default='')
    parser.add_argument('--dataset-setup', type=str, default='')
    parser.add_argument('--budget-per-sample', type=float, default=40)
    parser.add_argument('--cal-size', type=int, default=4000)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--tau-prior', type=float, default=0.56)
    parser.add_argument('--dapro-n1', type=int, default=200)
    parser.add_argument('--crc-control-size', type=int, default=100)
    parser.add_argument('--experiment-suffix', type=str, default='')
    parser.add_argument('--exclude-legacy-dapro', action='store_true')
    parser.add_argument('--exclude-locally-adaptive', action='store_true')
    parser.add_argument(
        '--allocator-name',
        action='append',
        dest='allocator_names',
        help=(
            'Run only the named allocator; repeat the option for multiple '
            'methods.'
        ),
    )
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    args.is_real = True if args.data_type.lower() == 'real' else False

    experiments_name = metric_experiment_name(
        args.dataset_name,
        args.dataset_setup,
        args.budget_per_sample,
        args.dapro_n1,
        args.crc_control_size,
        args.experiment_suffix,
    )

    run_experiments(cal_size=args.cal_size,
                    is_real=args.is_real,
                    device='cuda:0' if torch.cuda.is_available() and 'cuda' in args.device else 'cpu',
                    dataset_name=args.dataset_name,
                    data_setup=args.dataset_setup,
                    experiments_name=experiments_name,
                    seeds=(args.seed_start, args.seed_end),
                    budget_per_sample=args.budget_per_sample,
                    skip_existing=not args.overwrite,
                    dapro_n1=args.dapro_n1,
                    crc_control_size=args.crc_control_size,
                    tau_prior=args.tau_prior,
                    exclude_legacy_dapro=args.exclude_legacy_dapro,
                    exclude_locally_adaptive=args.exclude_locally_adaptive,
                    allocator_names=args.allocator_names)
    print("Finished Metrics Evaluation Suite.")


if __name__ == '__main__':
    main()
