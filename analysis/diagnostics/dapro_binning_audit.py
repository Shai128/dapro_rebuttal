"""Reproducible offline DAPRO score/bin audit; not production allocation code."""

from __future__ import annotations

import argparse
import json
import time
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


SETUPS = {
    "toxicity_qwen": (
        "dataset_toxicity",
        "attack_toxic_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_detoxify",
    ),
    "red_qwen": (
        "dataset_red_team",
        "attack_default_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_llm-judge_qwen25_14b_instruct",
    ),
    "toxicity_phi": (
        "dataset_toxicity",
        "attack_toxic_attack_qwen25_14b_instruct_lm_target_mini_phi_4_instruct_judge_detoxify",
    ),
    "red_llamaguard": (
        "dataset_red_team",
        "attack_default_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_llama_guard",
    ),
    "hallucination_qwen": (
        "dataset_hallucination3",
        "attack_hallucination_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_llm-judge_qwen25_14b_instruct",
    ),
    "hallucination_llama": (
        "dataset_hallucination3",
        "attack_hallucination_attack_qwen25_14b_instruct_lm_target_llama_31_8B_instruct_judge_llm-judge_qwen25_14b_instruct",
    ),
    "hallucination_phi": (
        "dataset_hallucination3",
        "attack_hallucination_attack_qwen25_14b_instruct_lm_target_mini_phi_4_instruct_judge_llm-judge_qwen25_14b_instruct",
    ),
    "autoif_qwen": (
        "dataset_autoif",
        "attack_autoif_helper_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_autoif",
    ),
}


def cache_path(dataset: str, setup: str) -> Path:
    return Path(
        f"alg_playground_model/is_real_True_dataset_{dataset}_dataset_{setup}"
    ) / "probability_est_cal_test.pt"


def load_setup(key: str) -> tuple[torch.Tensor, torch.Tensor, str, str]:
    dataset, setup = SETUPS[key]
    grid = torch.load(cache_path(dataset, setup), map_location="cpu", weights_only=False)
    loaded = get_data(True, torch.device("cpu"), dataset, setup, load_x=False)
    times = torch.cat([loaded[10], loaded[11]]).to(torch.long)
    if len(grid) != len(times):
        raise RuntimeError("Prediction cache and event-time rows are misaligned.")
    return grid, times, dataset, setup


def prefix_scores(grid: torch.Tensor, width: int) -> dict[str, np.ndarray]:
    """Compute causal one-step and suffix scores without a large float64 copy."""
    n = len(grid)
    hazard = np.empty((n, width), dtype=np.float64)
    risk = np.empty_like(hazard)
    cost = np.empty_like(hazard)
    for step in range(width):
        pmf = grid[:, step, :]
        future = pmf[:, step:]
        valid = future.sum(dim=1).clamp_min(torch.finfo(pmf.dtype).tiny)
        event = pmf[:, step:width].sum(dim=1)
        outcome_cost = torch.arange(
            1, width - step + 1, dtype=pmf.dtype, device=pmf.device
        )
        event_cost = (pmf[:, step:width] * outcome_cost).sum(dim=1)
        if pmf.shape[1] > width:
            event_cost = event_cost + pmf[:, width:].sum(dim=1) * (width - step)
        hazard[:, step] = pmf[:, step].to(torch.float64).numpy()
        risk[:, step] = (event / valid).to(torch.float64).numpy()
        cost[:, step] = (event_cost / valid).to(torch.float64).numpy()
    neyman = np.sqrt(np.divide(risk, cost, out=np.zeros_like(risk), where=cost > 0))
    return {"hazard": hazard, "future_risk": risk, "neyman": neyman}


def rank_auc(score: np.ndarray, label: np.ndarray) -> float:
    """Tie-aware AUC via average ranks, with no sklearn dependency in the loop."""
    positive = label.astype(bool)
    n1 = int(positive.sum())
    n0 = len(label) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = pd.Series(score).rank(method="average").to_numpy()
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def score_diagnostics(
    scores: dict[str, np.ndarray], times: np.ndarray, width: int
) -> pd.DataFrame:
    rows = []
    eventual = times <= width
    for name, values in scores.items():
        for step in [0, 1, 4, 9, 19, 49, 99, 149, 199]:
            active = times > step
            raw = values[active, step]
            label = eventual[active]
            edge = np.unique(np.quantile(raw, [0.5]))
            binary = np.searchsorted(edge, raw, side="right")
            rows.append(
                {
                    "score": name,
                    "step": step + 1,
                    "active": int(active.sum()),
                    "event_rate": float(label.mean()),
                    "raw_auc": rank_auc(raw, label),
                    "two_bin_auc": rank_auc(binary, label),
                    "unique_scores": int(np.unique(raw).size),
                    "low_bin_share": float((binary == 0).mean()),
                    "score_q05": float(np.quantile(raw, 0.05)),
                    "score_q50": float(np.quantile(raw, 0.50)),
                    "score_q95": float(np.quantile(raw, 0.95)),
                }
            )
    return pd.DataFrame(rows)


def bin_stats(
    fit_scores: np.ndarray,
    fit_lengths: np.ndarray,
    fit_policy: torch.Tensor,
    n_bins: int,
) -> dict[str, float]:
    width = fit_scores.shape[1]
    occupancies = []
    raw_unique = []
    retained_variance = []
    for step in range(width):
        active = fit_lengths > step
        values = fit_scores[active, step]
        if len(values) == 0:
            continue
        edges = np.unique(np.quantile(values, np.arange(1, n_bins) / n_bins))
        bins = np.searchsorted(edges, values, side="right")
        counts = np.bincount(bins, minlength=n_bins)
        occupancies.extend((counts / counts.sum()).tolist())
        raw_unique.append(np.unique(np.round(fit_policy.numpy()[active, step], 12)).size)
        if np.var(values) > 0:
            fitted = np.array([values[bins == b].mean() for b in bins])
            retained_variance.append(float(np.var(fitted) / np.var(values)))
    return {
        "mean_bin_occupancy": float(np.mean(occupancies)),
        "min_bin_occupancy": float(np.min(occupancies)),
        "mean_raw_conditional_unique": float(np.mean(raw_unique)),
        "max_raw_conditional_unique": float(np.max(raw_unique)),
        "score_variance_retained_by_bins": float(np.mean(retained_variance)),
    }


def smooth_rank_policy(
    fit_scores: np.ndarray,
    deployment_scores: np.ndarray,
    fit_lengths: np.ndarray,
    fit_binned_policy: torch.Tensor,
    n_bins: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the optimized bin table as monotone rank-interpolation knots."""
    n_fit, width = fit_scores.shape
    fit_output = np.ones((n_fit, width), dtype=np.float64)
    deployment_output = np.ones(
        (len(deployment_scores), width), dtype=np.float64
    )
    raw = fit_binned_policy.numpy()
    centers = (np.arange(n_bins, dtype=np.float64) + 0.5) / n_bins
    for step in range(width):
        active = fit_lengths > step
        values = fit_scores[active, step]
        if len(values) == 0:
            continue
        edges = np.unique(np.quantile(values, np.arange(1, n_bins) / n_bins))
        bins = np.searchsorted(edges, values, side="right")
        table = np.ones(n_bins, dtype=np.float64)
        observed = []
        for index in range(n_bins):
            selected = bins == index
            if np.any(selected):
                table[index] = raw[active, step][selected].mean()
                observed.append(index)
        for index in range(n_bins):
            if index not in observed:
                nearest = min(observed, key=lambda candidate: abs(candidate - index))
                table[index] = table[nearest]
        table = np.maximum.accumulate(table)
        sorted_values = np.sort(values)

        def interpolate(query: np.ndarray) -> np.ndarray:
            # Mid-rank empirical CDF, followed by geometric interpolation of
            # continuation probabilities.  This is monotone in score and
            # reduces exactly to the hard table when neighboring knots pool.
            rank = (
                np.searchsorted(sorted_values, query, side="left")
                + np.searchsorted(sorted_values, query, side="right")
            ) / (2 * len(sorted_values))
            return np.exp(np.interp(
                rank,
                centers,
                np.log(np.clip(table, 1e-12, 1.0)),
                left=np.log(np.clip(table[0], 1e-12, 1.0)),
                right=np.log(np.clip(table[-1], 1e-12, 1.0)),
            ))

        fit_output[:, step] = interpolate(fit_scores[:, step])
        deployment_output[:, step] = interpolate(deployment_scores[:, step])
        fit_output[~active, step] = 1.0
    return torch.as_tensor(fit_output), torch.as_tensor(deployment_output)


def evaluate_policy(
    *,
    score_name: str,
    score_all: np.ndarray,
    hazard_all: np.ndarray,
    times: np.ndarray,
    fit_idx: np.ndarray,
    deploy_idx: np.ndarray,
    n_bins: int,
    budget: float,
    projection_margin: float,
    total_n: int,
    width: int,
    objective: str,
    projection_kind: str,
    deployment_uniforms: np.ndarray | None = None,
    acquisition_seeds: list[int] | None = None,
    deployment_local: np.ndarray | None = None,
    outer_sample_count: int | None = None,
) -> tuple[dict[str, float | str], list[dict[str, float | int | None]]]:
    started = time.perf_counter()
    fit_times = times[fit_idx]
    deploy_times = times[deploy_idx]
    fit_lengths = np.minimum(fit_times, width)
    deploy_lengths = np.minimum(deploy_times, width)
    phase1_cost = float(fit_lengths.sum())
    target_budget = (
        (budget * total_n - phase1_cost) / len(deploy_idx)
        - projection_margin
    )
    if target_budget <= 0:
        raise ValueError("Phase-I rows exhaust the budget.")

    active_fit = np.arange(width)[None, :] < fit_lengths[:, None]
    objective_masses = None
    objective_weights = None
    if objective == "soft":
        objective_masses = hazard_all[fit_idx] * active_fit
    elif objective == "hard":
        objective_weights = (fit_times <= width).astype(np.float64)
    elif objective == "hard_reg":
        objective_weights = (
            (fit_times <= width).astype(np.float64) + 0.001
        ) / 1.001
    else:
        raise ValueError(objective)

    fit_score_tensor = torch.as_tensor(score_all[fit_idx], dtype=torch.float64)
    deploy_score_tensor = torch.as_tensor(score_all[deploy_idx], dtype=torch.float64)
    (
        _optimal,
        fit_raw_q,
        deploy_raw_q,
        diagnostics,
    ) = solve_binned_deployable_policy(
        fit_score_tensor,
        deploy_score_tensor,
        torch.as_tensor(fit_lengths),
        target_budget,
        None if objective_weights is None else torch.as_tensor(objective_weights),
        n_bins,
        objective_masses=(
            None if objective_masses is None else torch.as_tensor(objective_masses)
        ),
    )
    if projection_kind == "smooth_rank":
        fit_raw_q, deploy_raw_q = smooth_rank_policy(
            score_all[fit_idx],
            score_all[deploy_idx],
            fit_lengths,
            fit_raw_q,
            n_bins,
        )
    elif projection_kind != "hard_bin":
        raise ValueError(projection_kind)
    fit_raw_rho = fit_raw_q.cumprod(dim=1)
    deploy_raw_rho = deploy_raw_q.cumprod(dim=1)
    prior_fit = torch.full((len(fit_idx),), width)
    prior_deploy = torch.full((len(deploy_idx),), width)
    fit_q, deploy_q, correction = correct_projected_cumulative_probabilities_to_budget(
        fit_raw_rho,
        deploy_raw_rho,
        torch.as_tensor(fit_lengths),
        prior_fit,
        prior_deploy,
        target_budget,
        terminal_pi_min=0.005,
    )
    fit_rho = fit_q.cumprod(dim=1).numpy()
    deploy_rho = deploy_q.cumprod(dim=1).numpy()
    fit_active = np.arange(width)[None, :] < fit_lengths[:, None]
    deploy_active = np.arange(width)[None, :] < deploy_lengths[:, None]
    deploy_pi = deploy_rho[np.arange(len(deploy_idx)), deploy_lengths - 1]
    target_a = deploy_times <= width
    hard_excess = target_a * (1 / deploy_pi - 1)
    exact_variance_pp2 = float(hard_excess.sum() / total_n**2 * 10_000)
    expected_deploy_cost = float((deploy_rho * deploy_active).sum())
    expected_total_cost = phase1_cost + expected_deploy_cost

    # Reproduce the production acquisition convention exactly: the experiment
    # driver assigns one NumPy PCG64 uniform table to the outer calibration
    # rows, DAPRO selects its Phase-II rows from that table, and acquisition is
    # a sequential Bernoulli continuation process.  This diagnostic uses the
    # already corrected policy and never changes allocator behavior.
    phase1_events = float((fit_times <= width).sum())
    expected_shape = (len(deploy_idx), width)
    if (deployment_uniforms is None) == (acquisition_seeds is None):
        raise ValueError(
            "Pass exactly one of `deployment_uniforms` or `acquisition_seeds`."
        )
    if acquisition_seeds is not None and (
        deployment_local is None or outer_sample_count is None
    ):
        raise ValueError(
            "Repeated acquisition requires deployment row indices and the "
            "outer sample count."
        )

    def acquisition_tables():
        if deployment_uniforms is not None:
            yield None, np.asarray(deployment_uniforms, dtype=np.float64)
            return
        for acquisition_seed in acquisition_seeds or []:
            table = np.random.default_rng(acquisition_seed).random(
                (int(outer_sample_count), width)
            )[deployment_local]
            yield acquisition_seed, table

    acquisition_results = []
    deploy_q_numpy = deploy_q.detach().cpu().numpy()
    for acquisition_seed, uniforms in acquisition_tables():
        if uniforms.shape != expected_shape:
            raise ValueError(
                "Acquisition uniforms must have shape "
                f"{expected_shape}; got {uniforms.shape}."
            )
        sequential_keep = np.logical_and.accumulate(
            uniforms < deploy_q_numpy,
            axis=1,
        )
        acquired = sequential_keep & deploy_active
        endpoint_observed = sequential_keep[
            np.arange(len(deploy_idx)), deploy_lengths - 1
        ]
        deployment_ht = np.where(
            target_a & endpoint_observed,
            1.0 / deploy_pi,
            0.0,
        )
        estimated_cjr = 100.0 * (
            phase1_events + deployment_ht.sum()
        ) / total_n
        realized_total_cost = phase1_cost + float(acquired.sum())
        acquisition_results.append({
            "acquisition_seed": acquisition_seed,
            "realized_cost_per_sample": realized_total_cost / total_n,
            "estimated_cjr": float(estimated_cjr),
            "observed_phase2_events": int(
                (target_a & endpoint_observed).sum()
            ),
        })

    fit_soft_excess = float(
        np.mean(
            np.sum(
                hazard_all[fit_idx] * fit_active * (1 / fit_rho - 1), axis=1
            )
        )
    )
    deploy_soft_excess = float(
        np.mean(
            np.sum(
                hazard_all[deploy_idx]
                * deploy_active
                * (1 / deploy_rho - 1),
                axis=1,
            )
        )
    )
    conditional_unique = []
    cumulative_unique = []
    for step in range(width):
        active = deploy_lengths > step
        if not np.any(active):
            continue
        conditional_unique.append(
            np.unique(np.round(deploy_q.numpy()[active, step], 10)).size
        )
        cumulative_unique.append(
            np.unique(np.round(deploy_rho[active, step], 10)).size
        )
    deterministic = {
        "score": score_name,
        "bins": n_bins,
        "objective": objective,
        "projection_kind": projection_kind,
        "exact_variance_pp2": exact_variance_pp2,
        "phase2_hard_variance_proxy": float(hard_excess.mean()),
        "fit_soft_surrogate": fit_soft_excess,
        "phase2_soft_surrogate": deploy_soft_excess,
        "expected_cost_per_sample": expected_total_cost / total_n,
        "phase2_expected_cost": expected_deploy_cost / len(deploy_idx),
        "mean_endpoint_pi": float(deploy_pi.mean()),
        "min_endpoint_pi": float(deploy_pi.min()),
        "mean_corrected_conditional_unique": float(np.mean(conditional_unique)),
        "mean_corrected_cumulative_unique": float(np.mean(cumulative_unique)),
        "runtime_seconds": time.perf_counter() - started,
        **diagnostics,
        **bin_stats(score_all[fit_idx], fit_lengths, fit_raw_q, n_bins),
        **correction,
    }
    return deterministic, acquisition_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", choices=SETUPS, required=True)
    parser.add_argument(
        "--mode", choices=["outer", "acquisition"], default="outer"
    )
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--fixed-data-seed", type=int, default=0)
    parser.add_argument("--fixed-policy-seed", type=int, default=0)
    parser.add_argument("--n1", type=int, default=50)
    parser.add_argument("--budget", type=float, default=20.0)
    parser.add_argument("--projection-margin", type=float, default=1.0)
    parser.add_argument("--bins", default="1,2,4,8")
    parser.add_argument("--scores", default="hazard,future_risk,neyman")
    parser.add_argument("--objectives", default="soft")
    parser.add_argument("--projections", default="hard_bin")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    width = 200
    grid, times_tensor, dataset, setup = load_setup(args.setup)
    times = times_tensor.numpy().astype(np.int64)
    scores = prefix_scores(grid, width)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    diagnostics = score_diagnostics(scores, times, width)
    diagnostics.to_csv(args.output.with_name(args.output.stem + "_scores.csv"), index=False)

    seeds = [int(value) for value in args.seeds.split(",")]
    bins = [int(value) for value in args.bins.split(",")]
    selected_scores = args.scores.split(",")
    objectives = args.objectives.split(",")
    projections = args.projections.split(",")
    rows = []
    jobs = (
        [(seed, seed, [seed]) for seed in seeds]
        if args.mode == "outer"
        else [(args.fixed_data_seed, args.fixed_policy_seed, seeds)]
    )
    for data_seed, policy_seed, acquisition_seeds in jobs:
        np.random.seed(data_seed)
        outer = np.random.permutation(len(times))[:3000]
        # run_one_experiment resets the RNG to the seed immediately before
        # allocator.allocate_budget, so the internal DAPRO split starts from a
        # fresh stream rather than continuing after the outer split.
        inner = np.random.RandomState(policy_seed).permutation(len(outer))
        fit_local = inner[: args.n1]
        deploy_local = inner[args.n1 :]
        fit_idx = outer[fit_local]
        deploy_idx = outer[deploy_local]
        deployment_uniforms = None
        if args.mode == "outer":
            common_uniforms = np.random.default_rng(
                acquisition_seeds[0]
            ).random((len(outer), width))
            deployment_uniforms = common_uniforms[deploy_local]
        for objective in objectives:
            for score_name in selected_scores:
                for n_bins in bins:
                    for projection_kind in projections:
                        print(
                            json.dumps(
                                {
                                    "setup": args.setup,
                                    "mode": args.mode,
                                    "data_seed": data_seed,
                                    "policy_seed": policy_seed,
                                    "acquisition_seeds": acquisition_seeds,
                                    "objective": objective,
                                    "score": score_name,
                                    "bins": n_bins,
                                    "projection": projection_kind,
                                }
                            ),
                            flush=True,
                        )
                        deterministic, acquisition_results = evaluate_policy(
                            score_name=score_name,
                            score_all=scores[score_name],
                            hazard_all=scores["hazard"],
                            times=times,
                            fit_idx=fit_idx,
                            deploy_idx=deploy_idx,
                            n_bins=n_bins,
                            budget=args.budget,
                            projection_margin=args.projection_margin,
                            total_n=len(outer),
                            width=width,
                            objective=objective,
                            projection_kind=projection_kind,
                            deployment_uniforms=deployment_uniforms,
                            acquisition_seeds=(
                                acquisition_seeds
                                if args.mode == "acquisition"
                                else None
                            ),
                            deployment_local=deploy_local,
                            outer_sample_count=len(outer),
                        )
                        for acquisition_result in acquisition_results:
                            acquisition_seed = acquisition_result[
                                "acquisition_seed"
                            ]
                            if acquisition_seed is None:
                                acquisition_seed = acquisition_seeds[0]
                            acquisition_result["acquisition_seed"] = (
                                acquisition_seed
                            )
                            rows.append({
                                "setup_key": args.setup,
                                "dataset": dataset,
                                "data_setup": setup,
                                "seed": acquisition_seed,
                                "mode": args.mode,
                                "data_seed": data_seed,
                                "policy_seed": policy_seed,
                                "acquisition_seed": acquisition_seed,
                                "n1": args.n1,
                                "budget": args.budget,
                                **deterministic,
                                **acquisition_result,
                            })
                        pd.DataFrame(rows).to_csv(args.output, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
