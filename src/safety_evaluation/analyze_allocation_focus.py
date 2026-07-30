"""Allocation-focused diagnostics for DAPRO and constant Random policies.

This analyzer deliberately does not modify or call the calibration pipeline.
It has two complementary outputs:

1. Exact diagnostics derived from an existing ``per_seed_90pct.csv``.
2. Pointwise allocation diagnostics reconstructed from the *current* cached
   model outputs and source tree.

The distinction matters when an audit was produced before later source edits.
``reconstruction_check.csv`` quantifies that provenance gap instead of silently
mixing allocations from two source states.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.safety_evaluation.budget_allocators.optimization_solver_utils import (
    solve_time_only_cumulative_policy,
)
from src.safety_evaluation.survival_utils.compute_mean_time_given_pmf import (
    compute_quantiles_survival_time,
)


WIDTH = 200
CAL_SIZE = 3000
TAU_PRIOR = 0.56
TERMINAL_FLOOR = 0.005
TAUS = torch.tensor(np.logspace(-3, -0.01, 1000), dtype=torch.float64)
PRIOR_INDEX = int(torch.argmin(torch.abs(TAUS - TAU_PRIOR)).item())
RAW_INDEX = int(
    np.flatnonzero(TAUS[: PRIOR_INDEX + 1].numpy() < 0.10)[-1]
)


def _quantiles_to_interaction_counts(
        quantiles: torch.Tensor,
        width: int,
        upper_bound: float,
) -> torch.Tensor:
    return torch.clamp(
        quantiles + 1,
        max=min(float(width), float(upper_bound)),
    )


def _phase1_empirical_budget_limit(
        phase2_target_budget_per_sample: float,
        phase1_sample_count: int,
        maximum_cost_per_sample: float,
        phase2_sample_count: int,
) -> float:
    rho = (phase1_sample_count + 1) / phase2_sample_count
    envelope = (1 + rho) * maximum_cost_per_sample
    return float(
        (
            (phase1_sample_count + 1)
            * phase2_target_budget_per_sample
            - envelope
        )
        / phase1_sample_count
    )


@dataclasses.dataclass(frozen=True)
class DatasetSpec:
    label: str
    dataset_name: str
    setup: str
    budget: float

    @property
    def prediction_dir(self) -> Path:
        return Path(
            "alg_playground_model"
        ) / (
            f"is_real_True_dataset_{self.dataset_name}_dataset_{self.setup}"
        )

    @property
    def data_dir(self) -> Path:
        return Path("src/datasets/real_data") / self.dataset_name / self.setup


DATASETS = (
    DatasetSpec(
        "autoif",
        "dataset_autoif",
        (
            "attack_autoif_helper_qwen25_14b_instruct_lm_target_"
            "qwen25_14b_instruct_judge_autoif"
        ),
        20.0,
    ),
    DatasetSpec(
        "hallucination",
        "dataset_hallucination3",
        (
            "attack_hallucination_attack_qwen25_14b_instruct_lm_target_"
            "qwen25_14b_instruct_judge_llm-judge_qwen25_14b_instruct"
        ),
        10.0,
    ),
    DatasetSpec(
        "redteam_llamaguard",
        "dataset_red_team",
        (
            "attack_default_attack_qwen25_14b_instruct_lm_target_"
            "qwen25_14b_instruct_judge_llama_guard"
        ),
        10.0,
    ),
    DatasetSpec(
        "redteam_qwen",
        "dataset_red_team",
        (
            "attack_default_attack_qwen25_14b_instruct_lm_target_"
            "qwen25_14b_instruct_judge_llm-judge_qwen25_14b_instruct"
        ),
        20.0,
    ),
    DatasetSpec(
        "toxicity",
        "dataset_toxicity",
        (
            "attack_toxic_attack_qwen25_14b_instruct_lm_target_"
            "qwen25_14b_instruct_judge_detoxify"
        ),
        20.0,
    ),
)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2:
        return np.nan
    centered_x = x - x.mean()
    centered_y = y - y.mean()
    denominator = np.sqrt(
        np.square(centered_x).sum()
        * np.square(centered_y).sum()
    )
    if denominator <= np.finfo(np.float64).tiny:
        return np.nan
    return float(
        np.sum(centered_x * centered_y) / denominator
    )


def _host_path(path: Path) -> str:
    resolved = str(path.resolve())
    if os.name == "nt" and not resolved.startswith("\\\\?\\"):
        return f"\\\\?\\{resolved}"
    return resolved


def _ess(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    denominator = np.square(values).sum()
    if denominator <= np.finfo(np.float64).tiny:
        return 0.0
    return float(np.square(values.sum()) / denominator)


def _fixed_quantile_strata(values: np.ndarray, prefix: str) -> np.ndarray:
    """Return stable, population-defined quantile strata.

    Ties can collapse adjacent bins (notably for the event-time sentinel).
    Labels include the realized numeric interval so this is explicit.
    """
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    edges = np.unique(np.quantile(finite, [0, 0.25, 0.50, 0.75, 1]))
    if len(edges) == 1:
        return np.full(len(values), f"{prefix}: all", dtype=object)
    edges[0] = -np.inf
    edges[-1] = np.inf
    labels = []
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:]), 1):
        left_text = "-inf" if not np.isfinite(left) else f"{left:.4g}"
        right_text = "+inf" if not np.isfinite(right) else f"{right:.4g}"
        labels.append(f"{prefix} Q{index}: ({left_text}, {right_text}]")
    return np.asarray(
        pd.cut(
            values,
            bins=edges,
            labels=labels,
            include_lowest=True,
        ).astype(object)
    )


def _load_population(spec: DatasetSpec) -> dict[str, np.ndarray]:
    prediction_path = spec.prediction_dir / "probability_est_cal_test.pt"
    probability = torch.load(
        _host_path(prediction_path),
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    candidate_taus = TAUS[: PRIOR_INDEX + 1].to(probability.dtype)
    quantiles = compute_quantiles_survival_time(
        probability[:, 0].unsqueeze(1),
        candidate_taus,
        tail_distribution="geometric",
    ).squeeze(1)
    quantiles = _quantiles_to_interaction_counts(
        quantiles,
        width=WIDTH,
        upper_bound=WIDTH,
    ).cpu().numpy()
    event_time = np.concatenate(
        [
            np.load(_host_path(spec.data_dir / "t_tilde_cal.npy")),
            np.load(_host_path(spec.data_dir / "t_tilde_test.npy")),
        ]
    ).astype(np.float64)
    diagonal = probability[
        :,
        torch.arange(WIDTH),
        torch.arange(WIDTH),
    ].to(torch.float64)
    q_prior = quantiles[:, PRIOR_INDEX]
    active = (
        np.arange(WIDTH)[None, :]
        < q_prior.astype(np.int64)[:, None]
    )
    score_mean_active = (
        (diagonal.numpy() * active).sum(axis=1)
        / active.sum(axis=1).clip(min=1)
    )
    q_raw = quantiles[:, RAW_INDEX]
    endpoint = np.minimum(event_time, q_prior).astype(np.int64)
    raw_a = (event_time < q_raw).astype(np.float64)
    return {
        "event_time": event_time,
        "quantiles": quantiles,
        "q_prior": q_prior,
        "q_raw": q_raw,
        "endpoint": endpoint,
        "raw_a": raw_a,
        "score": score_mean_active,
        "score0": probability[:, 0, 0].to(torch.float64).numpy(),
    }


def _fit_random_schedule(
        lengths: np.ndarray,
        budget_limit: float,
        floor: float = TERMINAL_FLOOR,
) -> tuple[np.ndarray, dict]:
    lengths = np.asarray(lengths, dtype=np.int64)
    steps = np.arange(1, WIDTH + 1, dtype=np.float64)

    def schedule(probability: float) -> np.ndarray:
        return np.maximum(np.power(probability, steps), floor)

    def mean_cost(probability: float) -> float:
        cumulative_cost = np.cumsum(schedule(probability))
        return float(cumulative_cost[lengths - 1].mean())

    if mean_cost(1.0) <= budget_limit:
        fitted = 1.0
    else:
        low, high = 0.0, 1.0
        for _ in range(64):
            midpoint = (low + high) / 2
            if mean_cost(midpoint) <= budget_limit:
                low = midpoint
            else:
                high = midpoint
        fitted = low
    cumulative = schedule(fitted)
    return cumulative, {
        "constant_continuation_probability": fitted,
        "fit_expected_cost": mean_cost(fitted),
        "fit_budget_limit": budget_limit,
        "fit_budget_slack": budget_limit - mean_cost(fitted),
        "direct_time_pav_blocks": np.nan,
        "direct_time_budget_boundary": "",
    }


def _fit_direct_schedule(
        lengths: np.ndarray,
        weights: np.ndarray,
        budget_limit: float,
) -> tuple[np.ndarray, dict]:
    _, cumulative, diagnostics = solve_time_only_cumulative_policy(
        lengths,
        budget_limit,
        weights,
        WIDTH,
        TERMINAL_FLOOR,
    )
    return cumulative, {
        "constant_continuation_probability": np.nan,
        "fit_expected_cost": diagnostics["direct_time_expected_cost"],
        "fit_budget_limit": budget_limit,
        "fit_budget_slack": diagnostics["direct_time_budget_slack"],
        "direct_time_pav_blocks": diagnostics["direct_time_pav_blocks"],
        "direct_time_budget_boundary": (
            diagnostics["direct_time_budget_boundary"] or ""
        ),
    }


def _phase1_anchor(
        event_time: np.ndarray,
        candidate_quantiles: np.ndarray,
) -> int:
    miscoverage = (
        event_time[:, None] < candidate_quantiles
    ).mean(axis=0)
    infeasible = np.flatnonzero(miscoverage >= 0.10)
    if len(infeasible) == 0:
        return len(miscoverage) - 1
    if infeasible[0] == 0:
        return 0
    return int(infeasible[0] - 1)


def _inner_permutation(seed: int, n: int = CAL_SIZE) -> np.ndarray:
    state = np.random.RandomState(seed)
    return state.permutation(n)


def _outer_calibration_indices(seed: int, population_size: int) -> np.ndarray:
    state = np.random.RandomState(seed)
    return state.permutation(population_size)[:CAL_SIZE]


def _policy_fit(
        policy: str,
        seed: int,
        spec: DatasetSpec,
        population: dict[str, np.ndarray],
        cal_indices: np.ndarray,
) -> dict:
    endpoint = population["endpoint"][cal_indices]
    event_time = population["event_time"][cal_indices]
    quantiles = population["quantiles"][cal_indices]
    raw_a = population["raw_a"][cal_indices]
    n1 = 200 if "N1=200" in policy else 100
    permutation = _inner_permutation(seed)
    phase1 = permutation[:n1]
    phase2 = permutation[n1:]
    phase1_cost = float(endpoint[phase1].sum())
    target_budget = (
        spec.budget * CAL_SIZE - phase1_cost
    ) / len(phase2)
    budget_limit = target_budget
    objective_anchor = RAW_INDEX
    objective_weight_all = raw_a.copy()

    if policy.startswith("Random"):
        if "CRC" in policy:
            budget_limit = _phase1_empirical_budget_limit(
                target_budget,
                n1,
                WIDTH,
                len(phase2),
            )
        schedule, diagnostics = _fit_random_schedule(
            endpoint[phase1],
            budget_limit,
        )
        objective_weight_all = raw_a.copy()
    else:
        if policy.startswith("Unweighted"):
            objective_weight_all = np.ones(CAL_SIZE, dtype=np.float64)
            objective_anchor = -1
        elif policy.startswith("Phase-I"):
            objective_anchor = _phase1_anchor(
                event_time[phase1],
                quantiles[phase1],
            )
            objective_weight_all = (
                event_time
                < quantiles[:, objective_anchor]
            ).astype(np.float64)
            if "global 0.05" in policy:
                objective_weight_all = (
                    objective_weight_all + 0.05
                ) / 1.05
        schedule, diagnostics = _fit_direct_schedule(
            endpoint[phase1],
            objective_weight_all[phase1],
            budget_limit,
        )

    diagnostics.update(
        {
            "policy": policy,
            "n1": n1,
            "phase1_cost": phase1_cost,
            "phase2_target_budget": target_budget,
            "objective_anchor_index": objective_anchor,
            "objective_anchor_tau": (
                float(TAUS[objective_anchor])
                if objective_anchor >= 0
                else np.nan
            ),
            "objective_phase1_rate": float(
                objective_weight_all[phase1].mean()
            ),
            "objective_phase2_rate": float(
                objective_weight_all[phase2].mean()
            ),
        }
    )
    return {
        "schedule": schedule,
        "phase1": phase1,
        "phase2": phase2,
        "objective_weights": objective_weight_all,
        "diagnostics": diagnostics,
    }


def _evaluate_policy(
        spec: DatasetSpec,
        population: dict[str, np.ndarray],
        cal_indices: np.ndarray,
        fit: dict,
) -> tuple[dict, np.ndarray, np.ndarray]:
    endpoint = population["endpoint"][cal_indices]
    raw_a = population["raw_a"][cal_indices]
    schedule = fit["schedule"]
    phase1 = fit["phase1"]
    phase2 = fit["phase2"]
    cumulative_cost = np.cumsum(schedule)
    terminal_probability = schedule[endpoint - 1]
    expected_queries = cumulative_cost[endpoint - 1]
    terminal_probability = terminal_probability.copy()
    expected_queries = expected_queries.copy()
    terminal_probability[phase1] = 1.0
    expected_queries[phase1] = endpoint[phase1]
    inverse_probability = 1 / terminal_probability
    phase2_pi = terminal_probability[phase2]
    phase2_weight = inverse_probability[phase2]
    phase2_a = raw_a[phase2]
    a_weight = phase2_a * phase2_weight
    objective_weights = fit["objective_weights"]
    counterfactual_pi_phase1 = schedule[endpoint[phase1] - 1]
    counterfactual_pi_phase2 = schedule[endpoint[phase2] - 1]
    fit_objective_phase1 = np.mean(
        objective_weights[phase1]
        * (1 / counterfactual_pi_phase1 - 1)
    )
    fit_objective_phase2 = np.mean(
        objective_weights[phase2]
        * (1 / counterfactual_pi_phase2 - 1)
    )
    cov_a_pi = float(
        np.mean(phase2_a * phase2_pi)
        - phase2_a.mean() * phase2_pi.mean()
    )
    cov_a_weight = float(
        np.mean(phase2_a * phase2_weight)
        - phase2_a.mean() * phase2_weight.mean()
    )
    target_proxy = float(np.mean(phase2_a * (phase2_weight - 1)))
    total_expected_budget = float(expected_queries.mean())
    metrics = {
        **fit["diagnostics"],
        "total_expected_budget": total_expected_budget,
        "expected_budget_gap": total_expected_budget - spec.budget,
        "expected_budget_valid": int(
            total_expected_budget <= spec.budget + 1e-7
        ),
        "phase2_expected_budget": float(
            expected_queries[phase2].mean()
        ),
        "phase2_mean_pi": float(phase2_pi.mean()),
        "phase2_sd_pi": float(phase2_pi.std()),
        "phase2_cv_pi": float(
            phase2_pi.std() / max(phase2_pi.mean(), 1e-15)
        ),
        "phase2_min_pi": float(phase2_pi.min()),
        "phase2_p10_pi": float(np.quantile(phase2_pi, 0.10)),
        "phase2_median_pi": float(np.median(phase2_pi)),
        "phase2_mean_weight": float(phase2_weight.mean()),
        "phase2_sd_weight": float(phase2_weight.std()),
        "phase2_cv_weight": float(
            phase2_weight.std() / max(phase2_weight.mean(), 1e-15)
        ),
        "phase2_p90_weight": float(np.quantile(phase2_weight, 0.90)),
        "phase2_p99_weight": float(np.quantile(phase2_weight, 0.99)),
        "phase2_floor_fraction": float(
            np.isclose(phase2_pi, TERMINAL_FLOOR).mean()
        ),
        "phase2_weight_ess": _ess(phase2_weight),
        "phase2_weight_ess_fraction": _ess(phase2_weight) / len(phase2),
        "phase2_target_a_ess": _ess(a_weight),
        "phase2_target_a_rate": float(phase2_a.mean()),
        "phase2_mean_a_over_pi": float(a_weight.mean()),
        "phase2_target_variance_proxy": target_proxy,
        "conditional_variance_target_ht_mean": (
            target_proxy / len(phase2)
        ),
        "cov_target_a_pi": cov_a_pi,
        "corr_target_a_pi": _safe_corr(phase2_a, phase2_pi),
        "cov_target_a_weight": cov_a_weight,
        "corr_target_a_weight": _safe_corr(
            phase2_a,
            phase2_weight,
        ),
        "target_a_query_share": float(
            expected_queries[phase2][phase2_a > 0].sum()
            / expected_queries[phase2].sum()
        ),
        "target_a_sample_share": float(phase2_a.mean()),
        "target_a_query_lift": float(
            (
                expected_queries[phase2][phase2_a > 0].sum()
                / expected_queries[phase2].sum()
            )
            / max(phase2_a.mean(), 1e-15)
        ),
        "expected_observed_target_a_per_query": float(
            np.sum(phase2_a * phase2_pi)
            / np.sum(expected_queries[phase2])
        ),
        "fit_objective_phase1": float(fit_objective_phase1),
        "fit_objective_phase2": float(fit_objective_phase2),
        "fit_objective_transfer_gap": float(
            fit_objective_phase2 - fit_objective_phase1
        ),
        "fit_objective_transfer_ratio": (
            float(fit_objective_phase2 / fit_objective_phase1)
            if fit_objective_phase1 > 1e-12
            else np.nan
        ),
    }
    for step in (1, 5, 10, 20, 50, 100, 200):
        metrics[f"reach_probability_t{step}"] = schedule[step - 1]
    return metrics, terminal_probability, expected_queries


def _stratum_rows(
        spec: DatasetSpec,
        seed: int,
        policy: str,
        population: dict[str, np.ndarray],
        cal_indices: np.ndarray,
        fit: dict,
        terminal_probability: np.ndarray,
        expected_queries: np.ndarray,
        stratum_population: dict[str, np.ndarray],
) -> list[dict]:
    phase2 = fit["phase2"]
    raw_a = population["raw_a"][cal_indices]
    inverse_probability = 1 / terminal_probability
    rows = []
    for variable, population_labels in stratum_population.items():
        labels = population_labels[cal_indices]
        for label in sorted(set(labels[phase2])):
            selected = phase2[labels[phase2] == label]
            a = raw_a[selected]
            cost = expected_queries[selected]
            weight = inverse_probability[selected]
            pi = terminal_probability[selected]
            all_phase2_cost = expected_queries[phase2].sum()
            all_phase2_a = raw_a[phase2].sum()
            sample_share = len(selected) / len(phase2)
            query_share = cost.sum() / all_phase2_cost
            target_proxy_contribution = np.sum(a * (weight - 1))
            total_target_proxy = np.sum(
                raw_a[phase2] * (inverse_probability[phase2] - 1)
            )
            rows.append(
                {
                    "dataset": spec.label,
                    "seed": seed,
                    "policy": policy,
                    "stratum_variable": variable,
                    "stratum": label,
                    "n": len(selected),
                    "sample_share": sample_share,
                    "target_a_rate": float(a.mean()),
                    "target_a_event_share": float(
                        a.sum() / max(all_phase2_a, 1)
                    ),
                    "expected_query_share": float(query_share),
                    "query_allocation_lift": float(
                        query_share / max(sample_share, 1e-15)
                    ),
                    "mean_expected_queries": float(cost.mean()),
                    "mean_terminal_pi": float(pi.mean()),
                    "mean_weight": float(weight.mean()),
                    "mean_a_over_pi": float(np.mean(a * weight)),
                    "target_proxy_contribution_share": float(
                        target_proxy_contribution
                        / max(total_target_proxy, 1e-15)
                    ),
                }
            )
    return rows


def _exact_stored_diagnostics(input_csv: Path) -> pd.DataFrame:
    frame = pd.read_csv(input_csv)
    selected = frame[
        frame["method"].isin(
            [
                "Direct time raw-target (N1=200)",
                "Direct time Phase-I target",
                "Direct Phase-I target + global 0.05",
                "Random (hard pi>=0.005)",
                "Random (hard pi>=0.005, CRC)",
            ]
        )
    ].copy()
    a_rate = selected["all_observed_jailbreaks"] / selected[
        "configured_cal_size"
    ]
    covariance = (
        selected["mean_a_weighted_inverse_probability"]
        - a_rate * selected["mean_weight"]
    )
    denominator = np.sqrt(
        a_rate * (1 - a_rate) * selected["variance_weight"]
    )
    selected["selected_lpb_a_rate"] = a_rate
    selected["cov_selected_lpb_a_weight"] = covariance
    selected["corr_selected_lpb_a_weight"] = covariance / denominator.replace(
        0,
        np.nan,
    )
    phase2_rate = selected["phase2_objective_weight_mean"]
    phase2_covariance = (
        selected["phase2_mean_objective_inverse_probability"]
        - phase2_rate * selected["phase2_mean_inverse_probability"]
    )
    phase2_denominator = np.sqrt(
        phase2_rate
        * (1 - phase2_rate)
        * selected["phase2_variance_inverse_probability"]
    )
    binary_objective = selected["objective_kind"].fillna("").str.contains(
        "target_a_weighted"
    ) & ~selected["objective_kind"].fillna("").str.contains(
        "regularized"
    )
    selected["cov_phase2_objective_weight_inverse_probability"] = (
        phase2_covariance
    )
    selected["corr_phase2_binary_target_a_inverse_probability"] = np.where(
        binary_objective,
        phase2_covariance / phase2_denominator.replace(0, np.nan),
        np.nan,
    )
    columns = [
        "dataset",
        "seed",
        "method",
        "coverage",
        "total_expected_budget_per_sample",
        "mean_weight",
        "variance_weight",
        "effective_sample_size_weight",
        "top_1pct_weight_share",
        "median_weight",
        "p90_weight",
        "p99_weight",
        "mean_a_weighted_inverse_probability",
        "mean_a_weighted_inverse_probability_minus_one",
        "a_weighted_effective_sample_size",
        "selected_lpb_a_rate",
        "cov_selected_lpb_a_weight",
        "corr_selected_lpb_a_weight",
        "phase2_mean_inverse_probability",
        "phase2_variance_inverse_probability",
        "phase2_objective_weight_mean",
        "phase2_mean_objective_inverse_probability",
        "cov_phase2_objective_weight_inverse_probability",
        "corr_phase2_binary_target_a_inverse_probability",
        "random_constant_probability",
        "target_anchor_tau",
        "target_anchor_phase1_a_rate",
        "target_anchor_phase2_a_rate",
        "phase2_to_phase1_oracle_mean_objective_ratio",
    ]
    return selected[[column for column in columns if column in selected]]


def _reconstruction_check(
        stored: pd.DataFrame,
        reconstructed: pd.DataFrame,
) -> pd.DataFrame:
    stored_check = stored[
        stored["method"].eq("Random (hard pi>=0.005, CRC)")
    ][
        [
            "dataset",
            "seed",
            "phase1_realized_cost_total",
            "random_constant_probability",
            "total_expected_budget_per_sample",
        ]
    ].rename(
        columns={
            "phase1_realized_cost_total": "stored_phase1_cost",
            "random_constant_probability": "stored_random_probability",
            "total_expected_budget_per_sample": "stored_expected_budget",
        }
    )
    current = reconstructed[
        reconstructed["policy"].eq("Random CRC N1=100")
    ][
        [
            "dataset",
            "seed",
            "phase1_cost",
            "constant_continuation_probability",
            "total_expected_budget",
        ]
    ].rename(
        columns={
            "phase1_cost": "current_phase1_cost",
            "constant_continuation_probability": (
                "current_random_probability"
            ),
            "total_expected_budget": "current_expected_budget",
        }
    )
    merged = stored_check.merge(current, on=["dataset", "seed"], how="inner")
    for stem in ("phase1_cost", "random_probability", "expected_budget"):
        merged[f"{stem}_difference_current_minus_stored"] = (
            merged[f"current_{stem}"] - merged[f"stored_{stem}"]
        )
    return merged


def _hybrid_rows(
        spec: DatasetSpec,
        seed: int,
        population: dict[str, np.ndarray],
        cal_indices: np.ndarray,
) -> list[dict]:
    endpoint = population["endpoint"][cal_indices]
    raw_a = population["raw_a"][cal_indices]
    permutation = _inner_permutation(seed)
    phase1 = permutation[:200]
    phase2 = permutation[200:]
    phase1_cost = endpoint[phase1].sum()
    phase2_target = (
        spec.budget * CAL_SIZE - phase1_cost
    ) / len(phase2)
    crc_limit = _phase1_empirical_budget_limit(
        phase2_target,
        len(phase1),
        WIDTH,
        len(phase2),
    )
    random_schedule, _ = _fit_random_schedule(
        endpoint[phase1],
        crc_limit,
    )
    target_schedule, _ = _fit_direct_schedule(
        endpoint[phase1],
        raw_a[phase1],
        crc_limit,
    )
    eta_grid = np.linspace(0, 1, 101)
    rows = []
    for eta in eta_grid:
        schedule = (1 - eta) * random_schedule + eta * target_schedule
        phase1_schedule_cost = float(
            np.cumsum(schedule)[endpoint[phase1] - 1].mean()
        )
        phase1_maximum_cost = float(endpoint[phase1].mean())
        fill_denominator = phase1_maximum_cost - phase1_schedule_cost
        fill_fraction = (
            np.clip(
                (crc_limit - phase1_schedule_cost) / fill_denominator,
                0,
                1,
            )
            if fill_denominator > 1e-12
            else 0.0
        )
        # A scalar mixture with the always-continue schedule preserves
        # temporal monotonicity and spends any target-objective plateau slack.
        filled_schedule = (
            fill_fraction + (1 - fill_fraction) * schedule
        )
        cumulative_cost = np.cumsum(schedule)
        filled_cumulative_cost = np.cumsum(filled_schedule)
        pi = schedule[endpoint - 1].copy()
        filled_pi = filled_schedule[endpoint - 1].copy()
        expected_cost = cumulative_cost[endpoint - 1].copy()
        filled_expected_cost = filled_cumulative_cost[
            endpoint - 1
        ].copy()
        pi[phase1] = 1
        filled_pi[phase1] = 1
        expected_cost[phase1] = endpoint[phase1]
        filled_expected_cost[phase1] = endpoint[phase1]
        weight = 1 / pi
        filled_weight = 1 / filled_pi
        phase2_proxy = np.mean(
            raw_a[phase2] * (weight[phase2] - 1)
        )
        filled_phase2_proxy = np.mean(
            raw_a[phase2] * (filled_weight[phase2] - 1)
        )
        rows.append(
            {
                "dataset": spec.label,
                "seed": seed,
                "eta_target_dapro": eta,
                "phase2_target_variance_proxy": phase2_proxy,
                "total_expected_budget": float(expected_cost.mean()),
                "expected_budget_gap": float(
                    expected_cost.mean() - spec.budget
                ),
                "expected_budget_valid": int(
                    expected_cost.mean() <= spec.budget + 1e-7
                ),
                "phase2_weight_ess_fraction": (
                    _ess(weight[phase2]) / len(phase2)
                ),
                "phase2_target_a_ess": _ess(
                    raw_a[phase2] * weight[phase2]
                ),
                "phase2_mean_weight": float(weight[phase2].mean()),
                "schedule_rmse_from_random": float(
                    np.sqrt(np.mean(
                        np.square(schedule - random_schedule)
                    ))
                ),
                "phase1_crc_budget_fill_fraction": fill_fraction,
                "filled_phase2_target_variance_proxy": (
                    filled_phase2_proxy
                ),
                "filled_total_expected_budget": float(
                    filled_expected_cost.mean()
                ),
                "filled_expected_budget_gap": float(
                    filled_expected_cost.mean() - spec.budget
                ),
                "filled_expected_budget_valid": int(
                    filled_expected_cost.mean()
                    <= spec.budget + 1e-7
                ),
                "filled_phase2_weight_ess_fraction": (
                    _ess(filled_weight[phase2]) / len(phase2)
                ),
                "filled_phase2_target_a_ess": _ess(
                    raw_a[phase2] * filled_weight[phase2]
                ),
                "filled_phase2_mean_weight": float(
                    filled_weight[phase2].mean()
                ),
            }
        )
    valid = [row for row in rows if row["expected_budget_valid"]]
    oracle = min(valid, key=lambda row: row["phase2_target_variance_proxy"])
    for row in rows:
        row["offline_oracle_eta"] = oracle["eta_target_dapro"]
        row["offline_oracle_proxy"] = oracle[
            "phase2_target_variance_proxy"
        ]
        row["is_offline_oracle"] = int(
            row["eta_target_dapro"] == oracle["eta_target_dapro"]
        )
    return rows


def _write_markdown_summary(
        output_path: Path,
        exact: pd.DataFrame,
        reconstructed: pd.DataFrame,
        variability: pd.DataFrame,
        hybrid: pd.DataFrame,
        check: pd.DataFrame,
) -> None:
    exact_summary = exact.groupby(["dataset", "method"], as_index=False).agg(
        mean_weight=("mean_weight", "mean"),
        weight_ess=("effective_sample_size_weight", "mean"),
        selected_a_weight_cov=("cov_selected_lpb_a_weight", "mean"),
        selected_a_weight_corr=("corr_selected_lpb_a_weight", "mean"),
        a_weighted_ess=("a_weighted_effective_sample_size", "mean"),
    )
    reconstructed_summary = reconstructed.groupby(
        ["dataset", "policy"],
        as_index=False,
    ).agg(
        expected_budget=("total_expected_budget", "mean"),
        target_proxy=("phase2_target_variance_proxy", "mean"),
        weight_ess_fraction=("phase2_weight_ess_fraction", "mean"),
        target_a_ess=("phase2_target_a_ess", "mean"),
        cov_a_pi=("cov_target_a_pi", "mean"),
        corr_a_pi=("corr_target_a_pi", "mean"),
        cov_a_weight=("cov_target_a_weight", "mean"),
        target_query_lift=("target_a_query_lift", "mean"),
        transfer_ratio=("fit_objective_transfer_ratio", "median"),
    )
    fixed_eta = hybrid[
        hybrid["eta_target_dapro"].isin([0, 0.25, 0.5, 0.75, 1])
    ].groupby(["dataset", "eta_target_dapro"], as_index=False).agg(
        target_proxy=("phase2_target_variance_proxy", "mean"),
        expected_budget=("total_expected_budget", "mean"),
        budget_valid_rate=("expected_budget_valid", "mean"),
        weight_ess_fraction=("phase2_weight_ess_fraction", "mean"),
        filled_target_proxy=(
            "filled_phase2_target_variance_proxy",
            "mean",
        ),
        filled_expected_budget=("filled_total_expected_budget", "mean"),
        filled_budget_valid_rate=(
            "filled_expected_budget_valid",
            "mean",
        ),
        mean_fill_fraction=(
            "phase1_crc_budget_fill_fraction",
            "mean",
        ),
    )
    oracle = hybrid[hybrid["is_offline_oracle"].eq(1)].groupby(
        "dataset",
        as_index=False,
    ).agg(
        mean_oracle_eta=("offline_oracle_eta", "mean"),
        median_oracle_eta=("offline_oracle_eta", "median"),
        oracle_proxy=("offline_oracle_proxy", "mean"),
    )
    mismatch = check[
        [
            column
            for column in check
            if column.endswith("_difference_current_minus_stored")
        ]
    ].abs().mean()
    text = [
        "# Allocation-focus diagnostics",
        "",
        "The exact table below is derived from the saved audit. The pointwise",
        "tables are reconstructions under the current source and cached model",
        "outputs. They must not be treated as the exact policies from an older",
        "source hash when `reconstruction_check.csv` is nonzero.",
        "",
        "## Mean absolute reconstruction gaps",
        "",
        mismatch.to_frame("mean_absolute_gap").to_markdown(),
        "",
        "## Exact saved-table diagnostics",
        "",
        exact_summary.to_markdown(index=False, floatfmt=".5g"),
        "",
        "## Current-source pointwise diagnostics",
        "",
        reconstructed_summary.to_markdown(index=False, floatfmt=".5g"),
        "",
        "## Between-fit policy variability",
        "",
        variability.to_markdown(index=False, floatfmt=".5g"),
        "",
        "## Random-anchored cumulative-reach hybrids",
        "",
        fixed_eta.to_markdown(index=False, floatfmt=".5g"),
        "",
        "The `eta=0` endpoint is constant Random; `eta=1` is raw target-A",
        "DAPRO. Both are fitted on the same N1=200 Phase-I rows and the same",
        "CRC-adjusted cost limit. Intermediate schedules are convex mixtures",
        "of cumulative reach probabilities, so their Phase-I expected cost is",
        "the same convex mixture and remains within that limit.",
        "The `filled_` columns then mix the schedule toward always-continue by",
        "the unique scalar amount that spends unused Phase-I CRC allowance.",
        "This preserves monotonicity and cannot increase the fixed-target",
        "inverse-propensity objective on the fit population.",
        "",
        "## Offline oracle hybrid (diagnostic only)",
        "",
        oracle.to_markdown(index=False, floatfmt=".5g"),
        "",
        "The oracle eta uses Phase-II labels and is not deployable. It measures",
        "whether an interior Random/DAPRO mixture has useful headroom; a real",
        "method must select eta on an independent tuning fold or with a uniform",
        "finite-sample correction.",
    ]
    output_path.write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=Path("outputs/final_extended_100_all_v3_audit"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/final_extended_100_all_v3_audit/allocation_focus"
        ),
    )
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=100)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stored_raw = pd.read_csv(args.audit_dir / "per_seed_90pct.csv")
    exact = _exact_stored_diagnostics(
        args.audit_dir / "per_seed_90pct.csv"
    )
    exact.to_csv(
        args.output_dir / "exact_stored_allocation_diagnostics.csv",
        index=False,
    )

    policies = (
        "Random hard N1=100",
        "Random CRC N1=100",
        "Unweighted DAPRO N1=100",
        "Raw target DAPRO N1=200",
        "Phase-I target DAPRO N1=100",
        "Phase-I target + global 0.05 N1=100",
    )
    # Finish all Torch/cache work before the NumPy policy solvers. On some
    # Windows Conda installations, alternating independently bundled OpenMP
    # runtimes after either has begun parallel work raises OMP Error #15.
    populations = {
        spec.label: _load_population(spec)
        for spec in DATASETS
    }
    metric_rows = []
    stratum_rows = []
    schedule_store: dict[tuple[str, str], list[np.ndarray]] = {}
    hybrid_rows = []
    for spec in DATASETS:
        population = populations[spec.label]
        stratum_population = {
            "T": _fixed_quantile_strata(
                population["event_time"],
                "T",
            ),
            "q_prior": _fixed_quantile_strata(
                population["q_prior"],
                "q",
            ),
            "active_length": _fixed_quantile_strata(
                population["endpoint"],
                "min(T,q)",
            ),
            "model_score": _fixed_quantile_strata(
                population["score"],
                "score",
            ),
            "target_A": np.where(
                population["raw_a"] > 0,
                "target A=1",
                "target A=0",
            ),
        }
        for seed in range(args.seed_start, args.seed_end):
            cal_indices = _outer_calibration_indices(
                seed,
                len(population["event_time"]),
            )
            for policy in policies:
                fit = _policy_fit(
                    policy,
                    seed,
                    spec,
                    population,
                    cal_indices,
                )
                metrics, pi, expected_queries = _evaluate_policy(
                    spec,
                    population,
                    cal_indices,
                    fit,
                )
                metrics.update({"dataset": spec.label, "seed": seed})
                metric_rows.append(metrics)
                stratum_rows.extend(
                    _stratum_rows(
                        spec,
                        seed,
                        policy,
                        population,
                        cal_indices,
                        fit,
                        pi,
                        expected_queries,
                        stratum_population,
                    )
                )
                schedule_store.setdefault(
                    (spec.label, policy),
                    [],
                ).append(fit["schedule"])
            hybrid_rows.extend(
                _hybrid_rows(
                    spec,
                    seed,
                    population,
                    cal_indices,
                )
            )

    reconstructed = pd.DataFrame(metric_rows)
    reconstructed.to_csv(
        args.output_dir / "current_source_policy_diagnostics.csv",
        index=False,
    )
    stratum_frame = pd.DataFrame(stratum_rows)
    stratum_frame.to_csv(
        args.output_dir / "current_source_stratum_query_shares.csv",
        index=False,
    )
    selected_strata = stratum_frame[
        stratum_frame["policy"].isin(
            [
                "Random CRC N1=100",
                "Raw target DAPRO N1=200",
            ]
        )
        & stratum_frame["stratum_variable"].isin(
            ["active_length", "q_prior", "model_score"]
        )
    ].groupby(
        [
            "dataset",
            "policy",
            "stratum_variable",
            "stratum",
        ],
        as_index=False,
    ).agg(
        sample_share=("sample_share", "mean"),
        target_a_rate=("target_a_rate", "mean"),
        expected_query_share=("expected_query_share", "mean"),
        query_allocation_lift=("query_allocation_lift", "mean"),
        mean_expected_queries=("mean_expected_queries", "mean"),
        mean_terminal_pi=("mean_terminal_pi", "mean"),
        mean_weight=("mean_weight", "mean"),
        target_proxy_contribution_share=(
            "target_proxy_contribution_share",
            "mean",
        ),
    )
    selected_strata.to_csv(
        args.output_dir / "stratum_summary_selected.csv",
        index=False,
    )
    hybrid = pd.DataFrame(hybrid_rows)
    hybrid.to_csv(
        args.output_dir / "random_anchored_hybrid_grid.csv",
        index=False,
    )

    variability_rows = []
    for (dataset, policy), schedules in schedule_store.items():
        values = np.stack(schedules)
        mean_schedule = values.mean(axis=0)
        distance = np.sqrt(
            np.mean(np.square(values - mean_schedule), axis=1)
        )
        row = {
            "dataset": dataset,
            "policy": policy,
            "mean_schedule_rmse_to_cross_seed_mean": float(
                distance.mean()
            ),
            "p90_schedule_rmse_to_cross_seed_mean": float(
                np.quantile(distance, 0.90)
            ),
            "mean_pointwise_reach_sd": float(
                values.std(axis=0).mean()
            ),
            "max_pointwise_reach_sd": float(
                values.std(axis=0).max()
            ),
        }
        for step in (1, 5, 10, 20, 50, 100, 200):
            row[f"mean_r_t{step}"] = float(
                values[:, step - 1].mean()
            )
            row[f"sd_r_t{step}"] = float(
                values[:, step - 1].std()
            )
        variability_rows.append(row)
    variability = pd.DataFrame(variability_rows)
    variability.to_csv(
        args.output_dir / "between_fit_policy_variability.csv",
        index=False,
    )

    check = _reconstruction_check(stored_raw, reconstructed)
    check.to_csv(
        args.output_dir / "reconstruction_check.csv",
        index=False,
    )
    provenance = pd.DataFrame(
        [
            {
                "audit_dir": str(args.audit_dir.resolve()),
                "audit_source_hash": value,
                "current_safety_evaluation_source_hash": (
                    _current_source_hash()
                ),
                "seed_start": args.seed_start,
                "seed_end": args.seed_end,
                "tau_prior": TAU_PRIOR,
                "raw_target_tau": float(TAUS[RAW_INDEX]),
                "terminal_floor": TERMINAL_FLOOR,
            }
            for value in sorted(
                stored_raw["safety_evaluation_source_sha256"]
                .dropna()
                .unique()
            )
        ]
    )
    provenance.to_csv(
        args.output_dir / "provenance.csv",
        index=False,
    )
    _write_markdown_summary(
        args.output_dir / "allocation_focus_report.md",
        exact,
        reconstructed,
        variability,
        hybrid,
        check,
    )


def _current_source_hash() -> str:
    source_root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(source_root.rglob("*.py")):
        digest.update(path.relative_to(source_root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


if __name__ == "__main__":
    main()
