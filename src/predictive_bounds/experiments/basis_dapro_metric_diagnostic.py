"""Paired real-data diagnostic for the isolated Basis-DAPRO prototype.

This driver is deliberately separate from the production allocator registry.
It reproduces metric-estimation Phase-I accounting, the cumulative-probability
budget correction, and the terminal exploration floor, then compares:

* current hazard-score K=2 deployable DAPRO;
* continuous Basis-DAPRO using hazard or causal target-value scores;
* sequential-AHT Basis-DAPRO using realized information increments or robust
  residual-tail masses.

The default command screens 10 paired toxicity splits at B=20, N1=50, chooses
the best ordinary-HT and sequential-AHT policies by exact conditional design
variance, and extends those winners plus K=2 to 50 paired splits.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.dataset_utils.data_utils import get_data
from src.predictive_bounds.budget_allocators.optimization_solver_utils import (
    solve_binned_deployable_policy,
)
from src.predictive_bounds.budget_allocators.projected_optimization_utils import (
    correct_projected_cumulative_probabilities_to_budget,
)
from src.predictive_bounds.experiments.basis_dapro_prototype import (
    fit_basis_dapro,
)


DEFAULT_DATASET = "dataset_toxicity"
DEFAULT_SETUP = (
    "attack_toxic_attack_qwen25_14b_instruct_lm_target_"
    "qwen25_14b_instruct_judge_detoxify"
)


@dataclass(frozen=True)
class MethodSpec:
    name: str
    family: str
    score: str
    time_basis_size: int
    score_basis_size: int

    @property
    def parameter_count(self) -> int:
        if self.family == "current_k2":
            return 2 * 200
        return self.time_basis_size * self.score_basis_size


def method_specs(
        time_sizes=(4, 6, 8),
        score_sizes=(2, 4),
        *,
        include_sequential: bool = True,
) -> list[MethodSpec]:
    result = [MethodSpec("current_hazard_k2", "current_k2", "hazard", 200, 2)]
    for score in ["hazard", "target_value"]:
        for time_size in time_sizes:
            for score_size in score_sizes:
                result.append(MethodSpec(
                    f"basis_ht_{score}_a{time_size}_k{score_size}",
                    "basis_ht",
                    score,
                    time_size,
                    score_size,
                ))
    if include_sequential:
        for family, score in [
            ("basis_seq_increment", "information_gain"),
            ("basis_seq_residual", "target_value"),
        ]:
            for time_size in time_sizes:
                for score_size in score_sizes:
                    result.append(MethodSpec(
                        f"{family}_{score}_a{time_size}_k{score_size}",
                        family,
                        score,
                        time_size,
                        score_size,
                    ))
    return result


def _load_labels(dataset: str, setup: str) -> torch.Tensor:
    data = get_data(True, "cpu", dataset, setup, load_x=False)
    return torch.cat([data[10], data[11]]).to(torch.long)


def _prediction_cache(dataset: str, setup: str) -> Path:
    return Path(
        f"alg_playground_model/is_real_True_dataset_{dataset}_dataset_{setup}"
    ) / "probability_est_cal_test.pt"


def prefix_metric_predictions_and_scores(
        conditional_grid: torch.Tensor,
        horizon: int,
        *,
        chunk_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract hazard, event prediction, target-value, and information scores.

    Computation is chunked and stays in float32 because the real conditional
    grid is approximately one GiB.  Every prefix column uses only its own
    conditional PMF row.
    """
    if conditional_grid.ndim != 3:
        raise ValueError("Conditional grid must have three dimensions.")
    n, current_count, outcome_count = conditional_grid.shape
    if current_count < horizon or outcome_count < horizon:
        raise ValueError("Conditional grid does not cover the metric horizon.")
    device = conditional_grid.device
    current = torch.arange(horizon, device=device).view(horizon, 1)
    outcome = torch.arange(outcome_count, device=device).view(1, outcome_count)
    future = outcome >= current
    target = future & (outcome < horizon)
    remaining_capacity = horizon - current
    raw_steps = (outcome - current + 1).clamp_min(1)
    cost_weight = torch.where(
        future,
        torch.minimum(raw_steps, remaining_capacity),
        torch.zeros((), device=device, dtype=raw_steps.dtype),
    ).to(torch.float32)
    future_weight = future.to(torch.float32)
    target_weight = target.to(torch.float32)

    hazard = np.empty((n, horizon), dtype=np.float32)
    prediction = np.empty_like(hazard)
    value = np.empty_like(hazard)
    information = np.empty_like(hazard)
    diagonal = torch.arange(horizon, device=device)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        grid = conditional_grid[start:stop, :horizon].to(torch.float32)
        valid_mass = torch.einsum("bto,to->bt", grid, future_weight)
        event_mass = torch.einsum("bto,to->bt", grid, target_weight)
        remaining_cost_mass = torch.einsum("bto,to->bt", grid, cost_weight)
        valid = valid_mass.clamp_min(torch.finfo(torch.float32).tiny)
        event_probability = (event_mass / valid).clamp(0.0, 1.0)
        remaining_cost = (remaining_cost_mass / valid).clamp_min(
            torch.finfo(torch.float32).tiny
        )
        hazard[start:stop] = grid[:, diagonal, diagonal].cpu().numpy()
        prediction[start:stop] = event_probability.cpu().numpy()
        value[start:stop] = torch.sqrt(
            event_probability / remaining_cost
        ).cpu().numpy()
        information[start:stop] = torch.sqrt(
            event_probability * (1.0 - event_probability) / remaining_cost
        ).cpu().numpy()
    return hazard, prediction, value, information


def sequential_aht_components(
        event_predictions: np.ndarray,
        event_times: np.ndarray,
        horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return squared update masses, residual masses, increments, and activity."""
    predictions = np.asarray(event_predictions, dtype=np.float64)
    times = np.asarray(event_times, dtype=np.int64).reshape(-1)
    if predictions.shape != (len(times), horizon):
        raise ValueError("Prediction paths and event times do not align.")
    lengths = np.minimum(times, horizon)
    active = np.arange(horizon)[None, :] < lengths[:, None]
    target = (times <= horizon).astype(np.float64)
    next_prediction = np.concatenate(
        [predictions[:, 1:], target[:, None]],
        axis=1,
    )
    terminal_step = np.arange(horizon)[None, :] == (lengths - 1)[:, None]
    revealed = np.where(terminal_step, target[:, None], next_prediction)
    increments = np.where(active, revealed - predictions, 0.0)
    residual = np.where(active, target[:, None] - predictions, 0.0)
    return np.square(increments), np.square(residual), increments, active


def exact_ht_variance(
        event_times: np.ndarray,
        cumulative_reach: np.ndarray,
        horizon: int,
        total_sample_count: int,
) -> float:
    """Conditional allocation variance of the ordinary HT rate estimator."""
    times = np.asarray(event_times, dtype=np.int64).reshape(-1)
    lengths = np.minimum(times, horizon)
    terminal = cumulative_reach[np.arange(len(times)), lengths - 1]
    target = times <= horizon
    return float(np.sum(np.where(
        target,
        1.0 / terminal - 1.0,
        0.0,
    )) / total_sample_count**2)


def exact_sequential_aht_variance(
        squared_residuals: np.ndarray,
        active: np.ndarray,
        cumulative_reach: np.ndarray,
        total_sample_count: int,
) -> float:
    """Exact pathwise conditional variance of nested sequential AHT.

    If ``r_t=A-m_t`` is the residual before acquisition ``t+1``, then

        Var(AHT | full path) = sum_t r_t^2 (1/rho_t - 1/rho_{t-1}),

    with ``rho_{-1}=1``.  This retains all covariance terms induced by nested
    reach indicators; it is not the ordinary terminal HT proxy.
    """
    residual = np.asarray(squared_residuals, dtype=np.float64)
    mask = np.asarray(active, dtype=bool)
    reach = np.asarray(cumulative_reach, dtype=np.float64)
    previous = np.concatenate(
        [np.ones((len(reach), 1)), reach[:, :-1]],
        axis=1,
    )
    increments = 1.0 / reach - 1.0 / previous
    return float(np.sum(np.where(mask, residual * increments, 0.0))
                 / total_sample_count**2)


def _realized_reach(
        conditionals: np.ndarray,
        uniforms: np.ndarray,
        active: np.ndarray,
) -> np.ndarray:
    keep = np.cumprod((uniforms < conditionals).astype(np.int8), axis=1)
    return keep.astype(bool) & active


def evaluate_policy(
        *,
        phase1_targets: np.ndarray,
        phase1_cost: float,
        deployment_times: np.ndarray,
        deployment_predictions: np.ndarray,
        deployment_increments: np.ndarray,
        deployment_residual_squared: np.ndarray,
        deployment_active: np.ndarray,
        deployment_conditionals: np.ndarray,
        deployment_uniforms: np.ndarray,
        horizon: int,
        total_sample_count: int,
) -> dict[str, float]:
    cumulative = np.cumprod(deployment_conditionals, axis=1)
    active_float = deployment_active.astype(np.float64)
    expected_deployment_cost = float(np.sum(cumulative * active_float))
    reached = _realized_reach(
        deployment_conditionals,
        deployment_uniforms,
        deployment_active,
    )
    realized_deployment_cost = float(reached.sum())
    lengths = np.minimum(deployment_times, horizon)
    terminal_reach = cumulative[np.arange(len(lengths)), lengths - 1]
    terminal_observed = reached[np.arange(len(lengths)), lengths - 1]
    target = (deployment_times <= horizon).astype(np.float64)
    ht_contributions = target * terminal_observed / terminal_reach

    weighted_updates = np.where(
        reached,
        deployment_increments / cumulative,
        0.0,
    ).sum(axis=1)
    sequential_contributions = deployment_predictions[:, 0] + weighted_updates
    all_ht = np.concatenate([phase1_targets, ht_contributions])
    all_sequential = np.concatenate([phase1_targets, sequential_contributions])
    return {
        "exact_ht_conditional_variance": exact_ht_variance(
            deployment_times,
            cumulative,
            horizon,
            total_sample_count,
        ),
        "exact_sequential_aht_conditional_variance": (
            exact_sequential_aht_variance(
                deployment_residual_squared,
                deployment_active,
                cumulative,
                total_sample_count,
            )
        ),
        "expected_budget_per_sample": (
            phase1_cost + expected_deployment_cost
        ) / total_sample_count,
        "realized_budget_per_sample": (
            phase1_cost + realized_deployment_cost
        ) / total_sample_count,
        "estimated_cjr": float(all_ht.mean()),
        "estimated_cjr_sequential_aht": float(all_sequential.mean()),
    }


def _correct_cumulative_policy(
        fit_conditionals: np.ndarray,
        deployment_conditionals: np.ndarray,
        fit_lengths: np.ndarray,
        target_policy_budget: float,
        horizon: int,
        terminal_floor: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    fit_q = torch.full((len(fit_conditionals),), horizon)
    deployment_q = torch.full((len(deployment_conditionals),), horizon)
    corrected_fit, corrected_deployment, diagnostics = (
        correct_projected_cumulative_probabilities_to_budget(
            torch.as_tensor(np.cumprod(fit_conditionals, axis=1)),
            torch.as_tensor(np.cumprod(deployment_conditionals, axis=1)),
            torch.as_tensor(fit_lengths),
            fit_q,
            deployment_q,
            target_policy_budget,
            terminal_pi_min=terminal_floor,
        )
    )
    return (
        corrected_fit.numpy(),
        corrected_deployment.numpy(),
        diagnostics,
    )


def fit_method(
        spec: MethodSpec,
        *,
        fit_scores: dict[str, np.ndarray],
        deployment_scores: dict[str, np.ndarray],
        fit_lengths: np.ndarray,
        ht_masses: np.ndarray,
        increment_masses: np.ndarray,
        residual_masses: np.ndarray,
        target_policy_budget: float,
        horizon: int,
        terminal_floor: float,
) -> tuple[np.ndarray, dict]:
    start = time.perf_counter()
    score_fit = fit_scores[spec.score]
    score_deployment = deployment_scores[spec.score]
    active = np.arange(horizon)[None, :] < fit_lengths[:, None]
    cost_masses = active.astype(np.float64)
    if spec.family == "current_k2":
        _, raw_fit, raw_deployment, _ = solve_binned_deployable_policy(
            torch.as_tensor(score_fit),
            torch.as_tensor(score_deployment),
            torch.as_tensor(fit_lengths),
            target_policy_budget,
            objective_weights=None,
            n_bins=2,
            objective_masses=ht_masses,
        )
        raw_fit = raw_fit.numpy()
        raw_deployment = raw_deployment.numpy()
        raw_objective = np.nan
        iterations = np.nan
    else:
        if spec.family == "basis_ht":
            masses = ht_masses
        elif spec.family == "basis_seq_increment":
            masses = increment_masses
        elif spec.family == "basis_seq_residual":
            masses = residual_masses
        else:
            raise ValueError(f"Unknown method family {spec.family!r}.")
        model = fit_basis_dapro(
            score_fit,
            fit_lengths,
            masses,
            cost_masses,
            target_policy_budget,
            score_basis_kind="linear_rank",
            score_basis_size=spec.score_basis_size,
            time_basis_size=spec.time_basis_size,
            # The production projection imposes this same cumulative floor.
            # Including it in the convex fit avoids optimizing policies that
            # the projection would subsequently and materially overwrite.
            terminal_reach_floor=terminal_floor,
        )
        raw_fit = model.conditionals(score_fit)
        raw_deployment = model.conditionals(score_deployment)
        raw_objective = model.objective_value
        iterations = model.optimizer_iterations
    _, corrected_deployment, correction = _correct_cumulative_policy(
        raw_fit,
        raw_deployment,
        fit_lengths,
        target_policy_budget,
        horizon,
        terminal_floor,
    )
    return corrected_deployment, {
        "fit_runtime_seconds": time.perf_counter() - start,
        "parameter_count": spec.parameter_count,
        "raw_fit_objective": raw_objective,
        "optimizer_iterations": iterations,
        "phase1_corrected_policy_cost": correction[
            "projection_corrected_phase1_expected_cost"
        ],
        "cumulative_logit_shift": correction[
            "projection_budget_logit_shift"
        ],
    }


def run_seed(
        seed: int,
        specs: list[MethodSpec],
        *,
        labels: np.ndarray,
        hazard: np.ndarray,
        predictions: np.ndarray,
        target_value: np.ndarray,
        information_gain: np.ndarray,
        cal_size: int,
        n1: int,
        budget_per_sample: float,
        horizon: int,
        projection_margin: float,
        terminal_floor: float,
) -> list[dict]:
    outer = np.random.RandomState(seed).permutation(len(labels))[:cal_size]
    phase = np.random.RandomState(seed).permutation(cal_size)
    fit_local = phase[:n1]
    deployment_local = phase[n1:]
    fit = outer[fit_local]
    deployment = outer[deployment_local]
    fit_times = labels[fit]
    deployment_times = labels[deployment]
    fit_lengths = np.minimum(fit_times, horizon)
    phase1_cost = float(fit_lengths.sum())
    remaining_budget = (
        budget_per_sample * cal_size - phase1_cost
    ) / (cal_size - n1)
    target_policy_budget = remaining_budget - projection_margin
    if target_policy_budget <= 0:
        raise ValueError("Phase-I cost leaves no positive policy budget.")

    fit_prediction = predictions[fit]
    deployment_prediction = predictions[deployment]
    fit_sq_increment, fit_sq_residual, _, fit_active = (
        sequential_aht_components(fit_prediction, fit_times, horizon)
    )
    _, deployment_sq_residual, deployment_increment, deployment_active = (
        sequential_aht_components(
            deployment_prediction,
            deployment_times,
            horizon,
        )
    )
    ht_masses = hazard[fit].astype(np.float64) * fit_active
    phase1_targets = (fit_times <= horizon).astype(np.float64)
    score_fit = {
        "hazard": hazard[fit],
        "target_value": target_value[fit],
        "information_gain": information_gain[fit],
    }
    score_deployment = {
        "hazard": hazard[deployment],
        "target_value": target_value[deployment],
        "information_gain": information_gain[deployment],
    }
    common_uniforms = np.random.default_rng(seed).random((cal_size, horizon))
    deployment_uniforms = common_uniforms[deployment_local]

    rows = []
    for spec in specs:
        common = {
            "seed": seed,
            "method": spec.name,
            "family": spec.family,
            "score": spec.score,
            "time_basis_size": spec.time_basis_size,
            "score_basis_size": spec.score_basis_size,
            "phase1_size": n1,
            "phase1_realized_cost": phase1_cost,
            "deployment_budget_before_margin": remaining_budget,
            "target_policy_budget": target_policy_budget,
            "parameter_count": spec.parameter_count,
        }
        try:
            conditionals, fit_diagnostics = fit_method(
                spec,
                fit_scores=score_fit,
                deployment_scores=score_deployment,
                fit_lengths=fit_lengths,
                ht_masses=ht_masses,
                increment_masses=fit_sq_increment,
                residual_masses=fit_sq_residual,
                target_policy_budget=target_policy_budget,
                horizon=horizon,
                terminal_floor=terminal_floor,
            )
            metrics = evaluate_policy(
                phase1_targets=phase1_targets,
                phase1_cost=phase1_cost,
                deployment_times=deployment_times,
                deployment_predictions=deployment_prediction,
                deployment_increments=deployment_increment,
                deployment_residual_squared=deployment_sq_residual,
                deployment_active=deployment_active,
                deployment_conditionals=conditionals,
                deployment_uniforms=deployment_uniforms,
                horizon=horizon,
                total_sample_count=cal_size,
            )
            rows.append({
                **common,
                "fit_error": "",
                **fit_diagnostics,
                **metrics,
            })
        except Exception as error:  # keep a long screening run alive
            print(
                f"  {spec.name} failed on seed {seed}: {error}",
                flush=True,
            )
            rows.append({
                **common,
                "fit_error": f"{type(error).__name__}: {error}",
                "fit_runtime_seconds": np.nan,
                "raw_fit_objective": np.nan,
                "optimizer_iterations": np.nan,
                "phase1_corrected_policy_cost": np.nan,
                "cumulative_logit_shift": np.nan,
                "exact_ht_conditional_variance": np.nan,
                "exact_sequential_aht_conditional_variance": np.nan,
                "expected_budget_per_sample": np.nan,
                "realized_budget_per_sample": np.nan,
                "estimated_cjr": np.nan,
                "estimated_cjr_sequential_aht": np.nan,
            })
    return rows


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for method, group in rows.groupby("method"):
        errors = group["fit_error"].fillna("").astype(str)
        summaries.append({
            "method": method,
            "family": group["family"].iloc[0],
            "score": group["score"].iloc[0],
            "n_splits": len(group),
            "successful_splits": int(errors.eq("").sum()),
            "parameter_count": group["parameter_count"].iloc[0],
            "mean_exact_ht_variance_pp2": (
                10000 * group["exact_ht_conditional_variance"].mean()
            ),
            "mean_exact_sequential_aht_variance_pp2": (
                10000
                * group["exact_sequential_aht_conditional_variance"].mean()
            ),
            "across_split_ht_variance_pp2": float(np.var(
                100 * group["estimated_cjr"], ddof=1
            )) if len(group) > 1 else np.nan,
            "across_split_sequential_aht_variance_pp2": float(np.var(
                100 * group["estimated_cjr_sequential_aht"], ddof=1
            )) if len(group) > 1 else np.nan,
            "mean_expected_budget_per_sample": (
                group["expected_budget_per_sample"].mean()
            ),
            "mean_realized_budget_per_sample": (
                group["realized_budget_per_sample"].mean()
            ),
            "mean_fit_runtime_seconds": group["fit_runtime_seconds"].mean(),
            "fit_failure_count": int(errors.ne("").sum()),
        })
    return pd.DataFrame(summaries).sort_values(
        ["mean_exact_ht_variance_pp2", "method"]
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--setup", default=DEFAULT_SETUP)
    parser.add_argument("--prediction-cache", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(
        "outputs/basis_dapro_metric_toxicity_b20_n1_50"
    ))
    parser.add_argument("--budget", type=float, default=20.0)
    parser.add_argument("--cal-size", type=int, default=3000)
    parser.add_argument("--n1", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--projection-margin", type=float, default=1.0)
    parser.add_argument("--terminal-floor", type=float, default=0.005)
    parser.add_argument("--screen-splits", type=int, default=10)
    parser.add_argument("--final-splits", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument(
        "--methods",
        nargs="*",
        help="Optional exact method names for targeted/resume diagnostics.",
    )
    parser.add_argument("--no-sequential", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cache = args.prediction_cache or _prediction_cache(args.dataset, args.setup)
    grid = torch.load(cache, map_location="cpu", weights_only=True)
    labels = np.asarray(_load_labels(args.dataset, args.setup), dtype=np.int64)
    if len(grid) != len(labels):
        raise ValueError("Prediction cache and labels are not row-aligned.")
    hazard, predictions, target_value, information = (
        prefix_metric_predictions_and_scores(
            grid,
            args.horizon,
            chunk_size=args.chunk_size,
        )
    )
    del grid

    all_specs = method_specs(include_sequential=not args.no_sequential)
    if args.methods:
        requested = set(args.methods)
        known = {spec.name for spec in all_specs}
        unknown = requested - known
        if unknown:
            raise ValueError(f"Unknown requested methods: {sorted(unknown)}")
        all_specs = [spec for spec in all_specs if spec.name in requested]
    rows = []
    screen_stop = args.seed_start + args.screen_splits
    final_stop = args.seed_start + args.final_splits
    for seed in range(args.seed_start, screen_stop):
        print(
            f"screen seed {seed} "
            f"({seed - args.seed_start + 1}/{args.screen_splits})",
            flush=True,
        )
        rows.extend(run_seed(
            seed,
            all_specs,
            labels=labels,
            hazard=hazard,
            predictions=predictions,
            target_value=target_value,
            information_gain=information,
            cal_size=args.cal_size,
            n1=args.n1,
            budget_per_sample=args.budget,
            horizon=args.horizon,
            projection_margin=args.projection_margin,
            terminal_floor=args.terminal_floor,
        ))
    screen = pd.DataFrame(rows)
    screen_summary = summarize(screen)
    best_ht = screen_summary.iloc[0]["method"]
    best_sequential = screen_summary.sort_values(
        ["mean_exact_sequential_aht_variance_pp2", "method"]
    ).iloc[0]["method"]
    extension_names = {
        "current_hazard_k2",
        str(best_ht),
        str(best_sequential),
    }
    extension_specs = [
        spec for spec in all_specs if spec.name in extension_names
    ]
    for seed in range(screen_stop, final_stop):
        print(
            f"extension seed {seed} "
            f"({seed - args.seed_start + 1}/{args.final_splits})",
            flush=True,
        )
        rows.extend(run_seed(
            seed,
            extension_specs,
            labels=labels,
            hazard=hazard,
            predictions=predictions,
            target_value=target_value,
            information_gain=information,
            cal_size=args.cal_size,
            n1=args.n1,
            budget_per_sample=args.budget,
            horizon=args.horizon,
            projection_margin=args.projection_margin,
            terminal_floor=args.terminal_floor,
        ))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    final_summary = summarize(result)
    screen.to_csv(args.output_dir / "screen_per_split.csv", index=False)
    screen_summary.to_csv(args.output_dir / "screen_summary.csv", index=False)
    result.to_csv(args.output_dir / "all_per_split.csv", index=False)
    final_summary.to_csv(args.output_dir / "final_summary.csv", index=False)
    metadata = {
        "dataset": args.dataset,
        "setup": args.setup,
        "budget": args.budget,
        "cal_size": args.cal_size,
        "n1": args.n1,
        "horizon": args.horizon,
        "projection_margin": args.projection_margin,
        "terminal_floor": args.terminal_floor,
        "screen_splits": args.screen_splits,
        "final_splits": args.final_splits,
        "seed_start": args.seed_start,
        "best_ht": best_ht,
        "best_sequential_aht": best_sequential,
        "extension_methods": sorted(extension_names),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print("\nScreen summary:\n", screen_summary.to_string(index=False))
    print("\nFinal summary:\n", final_summary.to_string(index=False))


if __name__ == "__main__":
    main()
