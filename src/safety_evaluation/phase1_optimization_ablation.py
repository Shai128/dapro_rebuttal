"""Optimization-versus-adaptivity ablation and stratified gain analysis.

This module is deliberately analysis-only.  It consumes cached trajectories and
model outputs and never imports the conversation-generation stack.  The phase-I
split, locally adaptive shadow-price rule, DAPRO score, DAPRO inverse-weight
optimizer, projection maps, and LPB calibration rule follow the existing
implementations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.safety_evaluation.budget_allocators.optimization_solver_utils import (
    solve_exact_fast,
)
from src.safety_evaluation.budget_allocators.projected_optimization_utils import (
    project_to_test_beta,
    project_to_test_ir,
    project_to_test_platt,
)
from src.safety_evaluation.calibration.calibration_utils import (
    get_prior,
    solve_optimization,
)
from src.safety_evaluation.survival_utils.compute_mean_time_given_pmf import (
    compute_quantile_survival_time,
)
from src.utils.utils import set_seeds


METHODS = (
    "Static",
    "RandomAdaptive",
    "ScoreAdaptive-Heuristic",
    "LocallyAdaptive",
    "DAPRO",
)
STRATIFIED_METHODS = (
    "Static",
    "ScoreAdaptive-Heuristic",
    "LocallyAdaptive",
    "DAPRO",
)
PAIRINGS = (
    ("DAPRO", "ScoreAdaptive-Heuristic"),
    ("DAPRO", "LocallyAdaptive"),
    ("LocallyAdaptive", "Static"),
    ("ScoreAdaptive-Heuristic", "RandomAdaptive"),
    ("RandomAdaptive", "Static"),
)
PAIR_METRICS = (
    "absolute_coverage_deviation",
    "phase2_mean_latent_terminal_weight",
    "weighted_miscoverage_variance",
    "mean_lpb_size",
    "realized_budget_per_sample",
)
DIFFICULTY_LABELS = ("Q1", "Q2", "Q3", "Q4")
SCORE_LABELS = ("Q1", "Q2", "Q3", "Q4")


@dataclass(frozen=True)
class Config:
    seed_start: int = 0
    seed_end: int = 50
    budget_per_sample: float = 20.0
    tau_prior: float = 0.56
    target_miscoverage: float = 0.10
    n1: int = 100
    p_min: float = 0.005
    slope: float = 5.0
    bisection_tolerance: float = 1e-6
    bootstrap_seed: int = 1729
    bootstrap_resamples: int = 10_000
    projection: str = "platt"
    score: str = "prob"
    max_horizon: int = 200


@dataclass
class CachedData:
    event_times: torch.Tensor
    quantile_est: torch.Tensor
    conditional_grid: torch.Tensor
    probability_est: torch.Tensor
    taus_range: torch.Tensor
    cal_size: int
    test_size: int
    source_files: list[str]


@dataclass
class PolicyResult:
    probabilities: np.ndarray
    phase1_expected_cost: float
    phase1_objective_value: float | None
    tuning_value: float | None
    feasible_boundary: str | None
    fallbacks: list[dict]


def _as_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    neg_exp = np.exp(x[~pos])
    out[~pos] = neg_exp / (1.0 + neg_exp)
    return out.item() if out.ndim == 0 else out


def active_lengths(event_times: np.ndarray, q_prior: np.ndarray, width: int) -> np.ndarray:
    """Return existing DAPRO's number of continuation opportunities.

    Event times and prior horizons in the repository are turn indices.  Existing
    DAPRO spends ``min(T + 1, q + 1)`` phase-I units, hence the +1 here.
    """

    event_times = np.asarray(event_times, dtype=np.int64)
    q_prior = np.asarray(q_prior, dtype=np.int64)
    return np.clip(np.minimum(event_times + 1, q_prior + 1), 0, width)


def expected_cost(probabilities: np.ndarray, lengths: np.ndarray) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.int64)
    width = probabilities.shape[1]
    active = np.arange(width)[None, :] < lengths[:, None]
    cumulative = np.cumprod(np.where(active, probabilities, 1.0), axis=1)
    return float(np.mean(np.sum(cumulative * active, axis=1)))


def terminal_probabilities(probabilities: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    width = probabilities.shape[1]
    active = np.arange(width)[None, :] < np.asarray(lengths)[:, None]
    return np.prod(np.where(active, probabilities, 1.0), axis=1)


def _bisect_largest_feasible(
    cost_fn: Callable[[float], float],
    low: float,
    high: float,
    target: float,
    tolerance: float,
    iterations: int = 100,
) -> tuple[float, float, str | None]:
    low_cost = cost_fn(low)
    high_cost = cost_fn(high)
    if low_cost > target + tolerance:
        return low, low_cost, "lower"
    if high_cost <= target + tolerance:
        return high, high_cost, "upper"
    lo, hi = low, high
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if cost_fn(mid) <= target:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tolerance:
            break
    value = lo
    return value, cost_fn(value), None


def fit_random_policy(
    phase1_lengths: np.ndarray,
    phase2_count: int,
    width: int,
    target: float,
    p_min: float,
    tolerance: float,
) -> PolicyResult:
    phase1_lengths = np.asarray(phase1_lengths, dtype=np.int64)

    def cost(p: float) -> float:
        powers = np.arange(1, width + 1, dtype=np.float64)
        per_time = p**powers
        mask = np.arange(width)[None, :] < phase1_lengths[:, None]
        return float(np.mean(np.sum(per_time[None, :] * mask, axis=1)))

    p, achieved, boundary = _bisect_largest_feasible(
        cost, p_min, 1.0, target, tolerance
    )
    probabilities = np.full((phase2_count, width), p, dtype=np.float64)
    return PolicyResult(probabilities, achieved, None, p, boundary, [])


class MidrankCDF:
    def __init__(self, values: np.ndarray):
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            raise ValueError("MidrankCDF needs at least one value.")
        self.values = np.sort(values, kind="stable")

    def __call__(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        below = np.searchsorted(self.values, values, side="left")
        at_or_below = np.searchsorted(self.values, values, side="right")
        return (below + 0.5 * (at_or_below - below)) / len(self.values)


def fit_score_heuristic(
    phase1_scores: np.ndarray,
    phase1_lengths: np.ndarray,
    phase2_scores: np.ndarray,
    target: float,
    p_min: float,
    slope: float,
    tolerance: float,
) -> PolicyResult:
    phase1_scores = np.asarray(phase1_scores, dtype=np.float64)
    phase2_scores = np.asarray(phase2_scores, dtype=np.float64)
    width = phase1_scores.shape[1]
    cdfs: list[MidrankCDF | None] = []
    ranks1 = np.zeros_like(phase1_scores)
    fallbacks: list[dict] = []
    last_observed: int | None = None
    for t in range(width):
        active = t < phase1_lengths
        if np.any(active):
            cdf = MidrankCDF(phase1_scores[active, t])
            cdfs.append(cdf)
            ranks1[active, t] = cdf(phase1_scores[active, t])
            last_observed = t
        else:
            cdfs.append(None)
            if last_observed is not None:
                fallback = {"time": t + 1, "fallback_time": last_observed + 1}
                fallbacks.append(fallback)
                warnings.warn(
                    "No Phase-I scores at time "
                    f"{fallback['time']}; using time {fallback['fallback_time']}.",
                    RuntimeWarning,
                )

    def probabilities_from_ranks(ranks: np.ndarray, lam: float) -> np.ndarray:
        return p_min + (1.0 - p_min) * _sigmoid(
            slope * (ranks - 0.5) + lam
        )

    def cost(lam: float) -> float:
        return expected_cost(probabilities_from_ranks(ranks1, lam), phase1_lengths)

    lam, achieved, boundary = _bisect_largest_feasible(
        cost, -30.0, 30.0, target, tolerance
    )
    if boundary == "lower":
        warnings.warn(
            "Heuristic cost at lambda=-30 exceeds the target; using -30.",
            RuntimeWarning,
        )
    ranks2 = np.zeros_like(phase2_scores)
    observed_times = [i for i, cdf in enumerate(cdfs) if cdf is not None]
    if not observed_times:
        raise ValueError("No active phase-I scores are available.")
    for t in range(width):
        source_t = t
        if cdfs[source_t] is None:
            earlier = [j for j in observed_times if j < t]
            source_t = earlier[-1] if earlier else observed_times[0]
        ranks2[:, t] = cdfs[source_t](phase2_scores[:, t])
    probabilities = probabilities_from_ranks(ranks2, lam)
    return PolicyResult(probabilities, achieved, None, lam, boundary, fallbacks)


def fit_static_policy(
    phase1_q: np.ndarray,
    phase1_lengths: np.ndarray,
    phase2_q: np.ndarray,
    width: int,
    target: float,
) -> PolicyResult:
    """Apply the existing static optimized closed form, learned on phase I."""

    phase1_q = np.maximum(np.asarray(phase1_q, dtype=np.float64) + 1.0, 1.0)
    phase2_q = np.maximum(np.asarray(phase2_q, dtype=np.float64) + 1.0, 1.0)
    _, lam = solve_optimization(phase1_q, target * len(phase1_q), tol=1e-8)
    if lam is None:
        p1 = np.ones_like(phase1_q)
        p2 = np.ones_like(phase2_q)
        boundary = "upper"
    else:
        p1 = np.minimum(1.0, 1.0 / np.sqrt(lam * phase1_q))
        p2 = np.minimum(1.0, 1.0 / np.sqrt(lam * phase2_q))
        boundary = None
    # Static optimized makes one all-or-nothing decision.
    probabilities = np.ones((len(phase2_q), width), dtype=np.float64)
    probabilities[:, 0] = p2
    phase1_probabilities = np.ones((len(phase1_q), width), dtype=np.float64)
    phase1_probabilities[:, 0] = p1
    objective = float(np.mean(1.0 / p1))
    return PolicyResult(
        probabilities,
        expected_cost(phase1_probabilities, phase1_lengths),
        objective,
        float(lam) if lam is not None else None,
        boundary,
        [],
    )


def _locally_adaptive_probability_paths(
    conditional_grid: np.ndarray,
    event_times: np.ndarray,
    q_prior: np.ndarray,
    lam: float,
) -> tuple[np.ndarray, float]:
    """Evaluate the existing locally adaptive shadow-price policy.

    This mirrors ``AdaptiveOptimizedBudgetAllocator`` without sampling so the
    ablation can reuse paired common random numbers across all dynamic methods.
    """

    # Keep the cached grid's native precision; casting an N×T×T production
    # tensor to float64 would unnecessarily double peak memory.
    conditional_grid = np.asarray(conditional_grid)
    event_times = np.asarray(event_times, dtype=np.int64)
    q_prior = np.asarray(q_prior, dtype=np.int64)
    n, width, _ = conditional_grid.shape
    probabilities = np.ones((n, width), dtype=np.float64)
    cumulative_probability = np.ones(n, dtype=np.float64)
    active = np.ones(n, dtype=bool)
    total_expected_cost = 0.0

    for t_curr in range(width):
        active &= event_times >= t_curr
        active &= q_prior >= t_curr
        if not np.any(active):
            break

        belief = conditional_grid[:, t_curr, t_curr:]
        remaining_steps = np.clip(
            q_prior - t_curr + 1, 0, belief.shape[1]
        ).astype(np.float64)
        future_steps = np.arange(belief.shape[1])
        within_horizon = future_steps[None, :] <= remaining_steps[:, None]
        expected_remaining = np.sum(
            belief
            * within_horizon
            * np.arange(1, belief.shape[1] + 1, dtype=np.float64)[None, :],
            axis=1,
        )
        expected_remaining += remaining_steps * np.sum(
            belief * ~within_horizon, axis=1
        )

        target_terminal_probability = 1.0 / np.sqrt(
            lam * expected_remaining + 1e-12
        )
        target_terminal_probability = np.minimum(target_terminal_probability, 1.0)
        continuation = np.minimum(
            target_terminal_probability
            / np.maximum(cumulative_probability, 1e-9),
            1.0,
        )
        probabilities[active, t_curr] = continuation[active]
        total_expected_cost += float(
            np.sum(cumulative_probability[active] * continuation[active])
        )
        cumulative_probability[active] *= continuation[active]

    return probabilities, total_expected_cost / n


def fit_locally_adaptive_policy(
    phase1_grid: np.ndarray,
    phase1_event_times: np.ndarray,
    phase1_q: np.ndarray,
    phase2_grid: np.ndarray,
    phase2_event_times: np.ndarray,
    phase2_q: np.ndarray,
    target: float,
    tolerance: float,
) -> PolicyResult:
    """Fit the existing locally adaptive method's scalar shadow price."""

    def phase1_cost(lam: float) -> float:
        _, cost = _locally_adaptive_probability_paths(
            phase1_grid, phase1_event_times, phase1_q, lam
        )
        return cost

    low, high = 0.0, 256.0
    low_cost = phase1_cost(low)
    high_cost = phase1_cost(high)
    boundary = None
    if low_cost <= target + tolerance:
        lam = low
        achieved = low_cost
        boundary = "upper"
    elif high_cost > target + tolerance:
        lam = high
        achieved = high_cost
        boundary = "lower"
        warnings.warn(
            "Locally adaptive cost at lambda=256 exceeds the target; using 256.",
            RuntimeWarning,
        )
    else:
        # Match the existing allocator's fixed 25-step shadow-price search.
        for _ in range(25):
            mid = 0.5 * (low + high)
            achieved = phase1_cost(mid)
            if abs(achieved - target) < tolerance:
                low = high = mid
                break
            if achieved > target:
                low = mid
            else:
                high = mid
        lam = 0.5 * (low + high)
        phase1_probabilities, achieved = _locally_adaptive_probability_paths(
            phase1_grid, phase1_event_times, phase1_q, lam
        )

    if boundary is not None:
        phase1_probabilities, achieved = _locally_adaptive_probability_paths(
            phase1_grid, phase1_event_times, phase1_q, lam
        )
    phase2_probabilities, _ = _locally_adaptive_probability_paths(
        phase2_grid, phase2_event_times, phase2_q, lam
    )
    phase1_lengths = active_lengths(
        phase1_event_times, phase1_q, phase1_probabilities.shape[1]
    )
    objective = float(
        np.mean(1.0 / terminal_probabilities(phase1_probabilities, phase1_lengths))
    )
    return PolicyResult(
        phase2_probabilities,
        achieved,
        objective,
        lam,
        boundary,
        [],
    )


def fit_dapro_policy(
    phase1_scores: np.ndarray,
    phase1_event_times: np.ndarray,
    phase1_q: np.ndarray,
    phase2_scores: np.ndarray,
    target: float,
    projection: str,
    device: str,
) -> PolicyResult:
    """Call DAPRO's existing Phase-I solver and score projection unchanged."""

    score1 = torch.as_tensor(phase1_scores, dtype=torch.float32, device=device)
    score2 = torch.as_tensor(phase2_scores, dtype=torch.float32, device=device)
    event1 = torch.as_tensor(phase1_event_times, dtype=torch.long, device=device)
    q1 = torch.as_tensor(phase1_q, dtype=torch.long, device=device)
    max_steps = torch.minimum(event1, q1)
    optimal = solve_exact_fast(score1, max_steps, target, verbose=False)
    optimal[optimal == 0] = 1
    width = score1.shape[1]
    if projection == "ir":
        p2 = project_to_test_ir(
            optimal, score1, score2, q1, event1, width, torch.device(device)
        )
    elif projection == "platt":
        p2 = project_to_test_platt(
            optimal, score1, score2, q1, event1, width, torch.device(device)
        )
    elif projection == "beta":
        p2 = project_to_test_beta(
            optimal, score1, score2, q1, event1, width, torch.device(device)
        )
    else:
        raise ValueError(f"Unknown projection: {projection}")
    mask = np.arange(width)[None, :] < active_lengths(
        phase1_event_times, phase1_q, width
    )[:, None]
    p1_for_cost = np.where(mask, optimal, 1.0)
    objective = float(
        np.mean(1.0 / np.prod(np.where(mask, optimal, 1.0), axis=1))
    )
    return PolicyResult(
        _as_numpy(p2),
        expected_cost(p1_for_cost, active_lengths(phase1_event_times, phase1_q, width)),
        objective,
        None,
        None,
        [],
    )


def simulate_adaptive(
    probabilities: np.ndarray,
    event_times: np.ndarray,
    q_prior: np.ndarray,
    uniforms: np.ndarray,
) -> dict[str, np.ndarray]:
    """Existing DAPRO early-event convention with supplied common uniforms."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    event_times = np.asarray(event_times, dtype=np.int64)
    q_prior = np.asarray(q_prior, dtype=np.int64)
    n, width = probabilities.shape
    reached = np.zeros(n, dtype=np.int64)
    active = np.ones(n, dtype=bool)
    for t in range(width):
        active &= event_times >= t
        active &= q_prior >= t
        if not np.any(active):
            break
        keep = (uniforms[:, t] <= probabilities[:, t]) & active
        reached[keep] += 1
        active &= keep
    succeeded = (reached > q_prior) | (reached > event_times)
    calibration_c = np.where(succeeded, q_prior + 1, 0)
    lengths = active_lengths(event_times, q_prior, width)
    terminal_probability = terminal_probabilities(probabilities, lengths)
    return {
        "reached": reached,
        "succeeded": succeeded.astype(np.int64),
        "calibration_c": calibration_c,
        "realized_cost": reached.copy(),
        "terminal_probability": terminal_probability,
    }


def simulate_locally_adaptive(
    probabilities: np.ndarray,
    event_times: np.ndarray,
    q_prior: np.ndarray,
    uniforms: np.ndarray,
    p_min: float,
) -> dict[str, np.ndarray]:
    """Mirror ``AdaptiveOptimizedBudgetAllocator`` with supplied uniforms."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    event_times = np.asarray(event_times, dtype=np.int64)
    q_prior = np.asarray(q_prior, dtype=np.int64)
    n, width = probabilities.shape
    reached = np.zeros(n, dtype=np.int64)
    active = np.ones(n, dtype=bool)
    cumulative_probability = np.ones(n, dtype=np.float64)
    for t_curr in range(width):
        active &= event_times >= t_curr
        active &= q_prior >= t_curr
        if not np.any(active):
            break
        keep = (uniforms[:, t_curr] <= probabilities[:, t_curr]) & active
        reached[keep] += 1
        cumulative_probability[keep] *= probabilities[keep, t_curr]
        active &= keep

    succeeded = (reached > q_prior) | (reached > event_times)
    calibration_c = np.where(succeeded, q_prior + 1, reached)
    inclusion_probability = np.where(
        succeeded, cumulative_probability, p_min
    )
    return {
        "reached": reached,
        "succeeded": succeeded.astype(np.int64),
        "calibration_c": calibration_c,
        "realized_cost": reached.copy(),
        "terminal_probability": inclusion_probability,
    }


def simulate_static(
    probabilities: np.ndarray,
    event_times: np.ndarray,
    q_prior: np.ndarray,
    uniforms: np.ndarray,
) -> dict[str, np.ndarray]:
    include = uniforms[:, 0] <= probabilities[:, 0]
    lengths = active_lengths(event_times, q_prior, probabilities.shape[1])
    reached = np.where(include, lengths, 0)
    succeeded = include
    return {
        "reached": reached,
        "succeeded": succeeded.astype(np.int64),
        "calibration_c": np.where(include, q_prior + 1, 0),
        "realized_cost": reached.copy(),
        "terminal_probability": probabilities[:, 0].copy(),
    }


def _dapro_scores(conditional_grid: torch.Tensor, score: str) -> torch.Tensor:
    _, width, _ = conditional_grid.shape
    if score == "prob":
        idx = torch.arange(width, device=conditional_grid.device)
        return conditional_grid[:, idx, idx]
    if score == "quantile":
        return 1 / compute_quantile_survival_time(
            conditional_grid, quantile=0.9, tail_distribution="geometric"
        ).squeeze()
    raise ValueError(f"Unknown score: {score}")


def _common_uniforms(seed: int, sample_ids: np.ndarray, width: int) -> np.ndarray:
    """Generate values keyed by (split, global sample id, time)."""

    out = np.empty((len(sample_ids), width), dtype=np.float64)
    for row, sample_id in enumerate(sample_ids):
        digest = hashlib.sha256(
            f"dapro-ablation:{seed}:{int(sample_id)}".encode("utf-8")
        ).digest()
        stream_seed = int.from_bytes(digest[:8], "little", signed=False)
        out[row] = np.random.default_rng(stream_seed).random(width)
    return out


def _split_cached_data(seed: int, data: CachedData):
    """Mirror utils.split_data without importing optional evaluation metrics."""

    set_seeds(seed)
    permutation = np.random.permutation(data.cal_size + data.test_size)
    cal_idx = permutation[: data.cal_size]
    test_idx = permutation[data.cal_size :]
    return (
        data.event_times[cal_idx].detach(),
        data.quantile_est[cal_idx].detach(),
        data.event_times[test_idx].detach(),
        data.quantile_est[test_idx].detach(),
        cal_idx,
        test_idx,
    )


def _select_lpb(
    quantile_cal: np.ndarray,
    event_cal: np.ndarray,
    calibration_c: np.ndarray,
    inclusion_probability: np.ndarray,
    quantile_test: np.ndarray,
    event_test: np.ndarray,
    target_miscoverage: float,
) -> dict[str, float]:
    weights = np.repeat(
        (1.0 / inclusion_probability)[:, None], quantile_cal.shape[1], axis=1
    )
    weights[event_cal[:, None] >= quantile_cal] = 0.0
    weights[quantile_cal > calibration_c[:, None]] = 0.0
    miscoverage = weights.mean(axis=0)
    # This is the exact safe candidate-selection rule used by
    # SurvivalCalibrationWithKnownWeights.get_calibrated_lpb.
    difference = target_miscoverage - miscoverage
    candidate = np.where(difference > 0, 1.0, -np.inf)
    selected_idx = int(np.argmax(np.cumsum(candidate)))
    bound = quantile_test[:, selected_idx]
    coverage = float(np.mean(event_test >= bound))
    selected_values = weights[:, selected_idx]
    variance = (
        float(np.var(selected_values, ddof=1) / len(selected_values))
        if len(selected_values) > 1
        else 0.0
    )
    return {
        "selected_tau_index": selected_idx,
        "selected_tau": float(selected_idx),
        "empirical_coverage": coverage,
        "mean_lpb_size": float(np.mean(bound)),
        "weighted_miscoverage_variance": variance,
    }


def _score_quartiles(initial_scores: np.ndarray) -> np.ndarray:
    values = pd.Series(np.asarray(initial_scores, dtype=np.float64))
    try:
        labels = pd.qcut(values, 4, labels=SCORE_LABELS, duplicates="raise")
        return labels.astype(str).to_numpy()
    except ValueError:
        ranks = values.rank(method="average", pct=True).to_numpy()
        indices = np.minimum((ranks * 4).astype(int), 3)
        return np.asarray(SCORE_LABELS, dtype=object)[indices]


def _difficulty_strata(event_times: np.ndarray, q_prior: np.ndarray) -> np.ndarray:
    ratio = np.minimum(event_times, q_prior) / q_prior
    indices = np.searchsorted([0.25, 0.50, 0.75, 1.0], ratio, side="left")
    indices = np.clip(indices, 0, 3)
    return np.asarray(DIFFICULTY_LABELS, dtype=object)[indices]


def run_split(
    data: CachedData,
    config: Config,
    seed: int,
    device: str,
    dry_limit: int | None = None,
) -> tuple[list[dict], list[dict]]:
    set_seeds(seed)
    (
        event_cal,
        quantile_cal,
        event_test,
        quantile_test,
        cal_idx,
        test_idx,
    ) = _split_cached_data(seed, data)
    grid_cal = data.conditional_grid[cal_idx]
    scores_cal = _dapro_scores(grid_cal, config.score)
    q_cal = get_prior(quantile_cal, data.taus_range, config.tau_prior).long()
    if dry_limit is not None:
        keep_cal = min(dry_limit, len(event_cal))
        keep_test = min(dry_limit, len(event_test))
        event_cal, quantile_cal, q_cal, scores_cal = (
            event_cal[:keep_cal],
            quantile_cal[:keep_cal],
            q_cal[:keep_cal],
            scores_cal[:keep_cal],
        )
        event_test, quantile_test = event_test[:keep_test], quantile_test[:keep_test]
        cal_idx, test_idx = cal_idx[:keep_cal], test_idx[:keep_test]
    n = len(event_cal)
    n1 = min(config.n1, n - 1)
    if n1 <= 0:
        raise ValueError("At least two calibration samples are required.")
    # Existing DAPRO resets the seed immediately before its internal split.
    phase_perm = np.random.RandomState(seed).permutation(n)
    phase1_local, phase2_local = phase_perm[:n1], phase_perm[n1:]
    event1 = _as_numpy(event_cal[phase1_local]).astype(int)
    event2 = _as_numpy(event_cal[phase2_local]).astype(int)
    q1 = _as_numpy(q_cal[phase1_local]).astype(int)
    q2 = _as_numpy(q_cal[phase2_local]).astype(int)
    score1 = _as_numpy(scores_cal[phase1_local])
    score2 = _as_numpy(scores_cal[phase2_local])
    grid1 = _as_numpy(grid_cal[phase1_local])
    grid2 = _as_numpy(grid_cal[phase2_local])
    width = score1.shape[1]
    lengths1 = active_lengths(event1, q1, width)
    phase1_cost = float(np.sum(lengths1))
    target = (
        config.budget_per_sample * n - phase1_cost
    ) / len(phase2_local)
    if target <= 0:
        raise ValueError(
            f"Phase-I cost {phase1_cost} leaves nonpositive Phase-II budget."
        )
    sample_ids = np.asarray(cal_idx)[phase2_local]
    common_u = _common_uniforms(seed, sample_ids, width)
    static_u = _common_uniforms(seed + 10_000_019, sample_ids, width)

    policies: dict[str, PolicyResult] = {}
    policies["Static"] = fit_static_policy(q1, lengths1, q2, width, target)
    policies["RandomAdaptive"] = fit_random_policy(
        lengths1, len(q2), width, target, config.p_min, config.bisection_tolerance
    )
    policies["ScoreAdaptive-Heuristic"] = fit_score_heuristic(
        score1,
        lengths1,
        score2,
        target,
        config.p_min,
        config.slope,
        config.bisection_tolerance,
    )
    policies["LocallyAdaptive"] = fit_locally_adaptive_policy(
        grid1,
        event1,
        q1,
        grid2,
        event2,
        q2,
        target,
        config.bisection_tolerance,
    )
    policies["DAPRO"] = fit_dapro_policy(
        score1,
        event1,
        q1,
        score2,
        target,
        config.projection,
        device,
    )

    split_rows: list[dict] = []
    sample_rows: list[dict] = []
    phase1_quantile = _as_numpy(quantile_cal[phase1_local])
    phase2_quantile = _as_numpy(quantile_cal[phase2_local])
    full_event = np.concatenate([event1, event2])
    full_quantile = np.concatenate([phase1_quantile, phase2_quantile], axis=0)
    test_event_np = _as_numpy(event_test)
    test_quantile_np = _as_numpy(quantile_test)
    score_strata = _score_quartiles(score2[:, 0])
    difficulty_valid = q2 > 0
    difficulty = np.full(len(q2), None, dtype=object)
    difficulty[difficulty_valid] = _difficulty_strata(
        event2[difficulty_valid], q2[difficulty_valid]
    )

    for method in METHODS:
        started = time.perf_counter()
        policy = policies[method]
        probabilities = policy.probabilities
        if not np.all(np.isfinite(probabilities)):
            raise AssertionError(f"{method} produced non-finite probabilities.")
        if not np.all((probabilities > 0) & (probabilities <= 1)):
            raise AssertionError(f"{method} probabilities are outside (0, 1].")
        if method == "Static":
            simulation = simulate_static(probabilities, event2, q2, static_u)
        elif method == "LocallyAdaptive":
            simulation = simulate_locally_adaptive(
                probabilities, event2, q2, common_u, config.p_min
            )
        else:
            simulation = simulate_adaptive(probabilities, event2, q2, common_u)
        terminal = simulation["terminal_probability"]
        latent_weight = 1.0 / terminal
        phase1_c = q1 + 1
        phase1_inclusion = np.ones(len(q1), dtype=np.float64)
        full_c = np.concatenate([phase1_c, simulation["calibration_c"]])
        full_inclusion = np.concatenate([phase1_inclusion, terminal])
        lpb = _select_lpb(
            full_quantile,
            full_event,
            full_c,
            full_inclusion,
            test_quantile_np,
            test_event_np,
            config.target_miscoverage,
        )
        selected_idx = lpb["selected_tau_index"]
        selected_tau = float(data.taus_range[selected_idx].item())
        lpb["selected_tau"] = selected_tau
        realized = float(np.mean(simulation["realized_cost"]))
        target_coverage = 1.0 - config.target_miscoverage
        split_rows.append(
            {
                "split_id": seed,
                "method": method,
                "target_budget_per_sample": target,
                "realized_budget_per_sample": realized,
                "budget_overrun": realized - target,
                "indicator_budget_exceeded": int(realized > target),
                "empirical_coverage": lpb["empirical_coverage"],
                "target_coverage": target_coverage,
                "absolute_coverage_deviation": abs(
                    lpb["empirical_coverage"] - target_coverage
                ),
                "mean_lpb_size": lpb["mean_lpb_size"],
                "selected_tau": selected_tau,
                "phase1_expected_cost": policy.phase1_expected_cost,
                "phase1_objective_value": policy.phase1_objective_value,
                "phase2_mean_latent_terminal_weight": float(np.mean(latent_weight)),
                "phase2_median_latent_terminal_weight": float(np.median(latent_weight)),
                "phase2_p90_latent_terminal_weight": float(
                    np.quantile(latent_weight, 0.90)
                ),
                "phase2_max_latent_terminal_weight": float(np.max(latent_weight)),
                "phase2_resolution_rate": float(np.mean(simulation["succeeded"])),
                "weighted_miscoverage_variance": lpb[
                    "weighted_miscoverage_variance"
                ],
                "runtime": time.perf_counter() - started,
                "tuning_value": policy.tuning_value,
                "feasible_boundary": policy.feasible_boundary,
                "fallback_count": len(policy.fallbacks),
                "fallbacks": json.dumps(policy.fallbacks, separators=(",", ":")),
            }
        )
        for row, sample_id in enumerate(sample_ids):
            probability_path = probabilities[row, : active_lengths(
                event2[row : row + 1], q2[row : row + 1], width
            )[0]]
            sample_rows.append(
                {
                    "split_id": seed,
                    "sample_id": int(sample_id),
                    "method": method,
                    "T_i": int(event2[row]),
                    "q_prior_i": int(q2[row]),
                    "b_i": int(min(event2[row], q2[row])),
                    "initial_score": float(score2[row, 0]),
                    "realized_cost": int(simulation["realized_cost"][row]),
                    "censoring_time": int(simulation["reached"][row]),
                    "resolved_indicator": int(
                        simulation["reached"][row] >= q2[row]
                    ),
                    "event_observed_indicator": int(
                        simulation["reached"][row] > event2[row]
                    ),
                    "latent_terminal_weight": float(latent_weight[row]),
                    "log_latent_terminal_weight": float(np.log(latent_weight[row])),
                    "continuation_probabilities": json.dumps(
                        probability_path.tolist(), separators=(",", ":")
                    ),
                    "difficulty_stratum": difficulty[row],
                    "initial_score_quartile": score_strata[row],
                    "common_uniform_path_hash": hashlib.sha256(
                        common_u[row].tobytes()
                    ).hexdigest(),
                }
            )
    _validate_split(split_rows, sample_rows, policies, target, config)
    return split_rows, sample_rows


def _validate_split(
    split_rows: list[dict],
    sample_rows: list[dict],
    policies: dict[str, PolicyResult],
    target: float,
    config: Config,
) -> None:
    frame = pd.DataFrame(sample_rows)
    ids = [
        tuple(sorted(frame.loc[frame.method == method, "sample_id"].tolist()))
        for method in METHODS
    ]
    assert all(current == ids[0] for current in ids[1:])
    hashes = [
        tuple(
            frame.loc[frame.method == method]
            .sort_values("sample_id")["common_uniform_path_hash"]
        )
        for method in (
            "RandomAdaptive",
            "ScoreAdaptive-Heuristic",
            "LocallyAdaptive",
            "DAPRO",
        )
    ]
    assert all(current == hashes[0] for current in hashes[1:])
    for field in ("difficulty_stratum", "initial_score_quartile"):
        pivot = frame.pivot(index="sample_id", columns="method", values=field)
        assert pivot.nunique(axis=1, dropna=False).eq(1).all()
    random_r = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
    values = config.p_min + (1 - config.p_min) * _sigmoid(
        config.slope * (random_r - 0.5)
    )
    assert np.all(np.diff(values) >= 0)
    for method in (
        "RandomAdaptive",
        "ScoreAdaptive-Heuristic",
        "LocallyAdaptive",
    ):
        result = policies[method]
        if result.feasible_boundary is None:
            assert result.phase1_expected_cost <= target + 5 * config.bisection_tolerance
            assert target - result.phase1_expected_cost <= 10 * config.bisection_tolerance


def _bootstrap_interval(
    values: np.ndarray, seed: int, resamples: int
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    chunk = max(1, min(resamples, 2000))
    means: list[np.ndarray] = []
    remaining = resamples
    while remaining:
        current = min(chunk, remaining)
        indices = rng.integers(0, len(values), size=(current, len(values)))
        means.append(values[indices].mean(axis=1))
        remaining -= current
    boot = np.concatenate(means)
    return tuple(np.quantile(boot, [0.025, 0.975]))


def paired_summary(split_df: pd.DataFrame, config: Config) -> pd.DataFrame:
    rows = []
    for pair_index, (first, second) in enumerate(PAIRINGS):
        for metric_index, metric in enumerate(PAIR_METRICS):
            pivot = split_df.pivot(index="split_id", columns="method", values=metric)
            differences = (pivot[first] - pivot[second]).dropna().to_numpy()
            lo, hi = _bootstrap_interval(
                differences,
                config.bootstrap_seed + pair_index * 100 + metric_index,
                config.bootstrap_resamples,
            )
            lower_is_better = metric != "mean_lpb_size"
            better = differences < 0 if lower_is_better else differences > 0
            rows.append(
                {
                    "first_method": first,
                    "second_method": second,
                    "metric": metric,
                    "mean_paired_difference": float(np.mean(differences)),
                    "median_paired_difference": float(np.median(differences)),
                    "bootstrap_ci_lower": lo,
                    "bootstrap_ci_upper": hi,
                    "fraction_first_better": float(np.mean(better)),
                    "n_splits": len(differences),
                    "better_direction": "lower" if lower_is_better else "higher",
                }
            )
    return pd.DataFrame(rows)


def stratified_outputs(
    sample_df: pd.DataFrame, config: Config
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    focus = sample_df[sample_df.method.isin(STRATIFIED_METHODS)].copy()
    static_weights = (
        focus[focus.method == "Static"]
        .set_index(["split_id", "sample_id"])["latent_terminal_weight"]
    )
    heuristic_weights = (
        focus[focus.method == "ScoreAdaptive-Heuristic"]
        .set_index(["split_id", "sample_id"])["latent_terminal_weight"]
    )
    index = pd.MultiIndex.from_frame(focus[["split_id", "sample_id"]])
    focus["static_weight"] = static_weights.reindex(index).to_numpy()
    focus["heuristic_weight"] = heuristic_weights.reindex(index).to_numpy()
    focus["reduction_vs_static"] = (
        focus["static_weight"] - focus["latent_terminal_weight"]
    )
    focus["reduction_vs_heuristic"] = np.where(
        focus.method == "DAPRO",
        focus["heuristic_weight"] - focus["latent_terminal_weight"],
        np.nan,
    )
    raw_rows = []
    for stratification, column, labels in (
        ("event_time_difficulty", "difficulty_stratum", DIFFICULTY_LABELS),
        ("initial_score_quartile", "initial_score_quartile", SCORE_LABELS),
    ):
        subset = focus.dropna(subset=[column])
        total_mass = subset.groupby(["split_id", "method"])[
            "latent_terminal_weight"
        ].transform("sum")
        subset = subset.assign(weight_mass_share_row=subset.latent_terminal_weight / total_mass)
        for (split_id, method, stratum), group in subset.groupby(
            ["split_id", "method", column], observed=False
        ):
            if stratum not in labels:
                continue
            raw_rows.append(
                {
                    "split_id": split_id,
                    "method": method,
                    "stratification": stratification,
                    "stratum": stratum,
                    "n_samples": len(group),
                    "average_realized_budget": group.realized_cost.mean(),
                    "resolution_rate": group.resolved_indicator.mean(),
                    "event_observation_rate": group.event_observed_indicator.mean(),
                    "mean_latent_terminal_weight": group.latent_terminal_weight.mean(),
                    "median_latent_terminal_weight": group.latent_terminal_weight.median(),
                    "p90_latent_terminal_weight": group.latent_terminal_weight.quantile(0.90),
                    "mean_log_latent_terminal_weight": group.log_latent_terminal_weight.mean(),
                    "weight_mass_share": group.weight_mass_share_row.sum(),
                    "mean_reduction_vs_static": group.reduction_vs_static.mean(),
                    "mean_reduction_vs_heuristic": group.reduction_vs_heuristic.mean(),
                }
            )
    raw = pd.DataFrame(raw_rows)
    for (split_id, method, stratification), group in raw.groupby(
        ["split_id", "method", "stratification"], observed=False
    ):
        expected_count = len(
            focus[
                (focus.split_id == split_id)
                & (focus.method == method)
                & (
                    focus[
                        "difficulty_stratum"
                        if stratification == "event_time_difficulty"
                        else "initial_score_quartile"
                    ].notna()
                )
            ]
        )
        assert int(group.n_samples.sum()) == expected_count
        assert np.isclose(group.weight_mass_share.sum(), 1.0, atol=1e-8)
    metric_columns = [
        col
        for col in raw.columns
        if col not in {"split_id", "method", "stratification", "stratum"}
    ]
    summary_rows = []
    for keys, group in raw.groupby(
        ["method", "stratification", "stratum"], observed=False
    ):
        for metric_index, metric in enumerate(metric_columns):
            values = group[metric].dropna().to_numpy(dtype=float)
            if not len(values):
                continue
            lo, hi = _bootstrap_interval(
                values,
                config.bootstrap_seed + metric_index,
                config.bootstrap_resamples,
            )
            summary_rows.append(
                {
                    "method": keys[0],
                    "stratification": keys[1],
                    "stratum": keys[2],
                    "metric": metric,
                    "mean": float(np.mean(values)),
                    "standard_error": (
                        float(np.std(values, ddof=1) / np.sqrt(len(values)))
                        if len(values) > 1
                        else 0.0
                    ),
                    "bootstrap_ci_lower": lo,
                    "bootstrap_ci_upper": hi,
                    "n_splits": len(values),
                }
            )
    summary = pd.DataFrame(summary_rows)
    tail_rows = []
    for method, group in focus.groupby("method"):
        weights = group.latent_terminal_weight.to_numpy()
        descending = np.sort(weights)[::-1]
        total = descending.sum()
        row = {"method": method}
        for pct in (0.01, 0.05, 0.10):
            count = max(1, int(math.ceil(pct * len(descending))))
            row[f"top_{int(pct * 100)}pct_weight_mass_fraction"] = (
                float(descending[:count].sum() / total)
            )
        tail_rows.append(row)
    tail = pd.DataFrame(tail_rows)
    reduction = focus[focus.method == "DAPRO"].dropna(subset=["difficulty_stratum"])
    reduction_by_bin = reduction.groupby("difficulty_stratum", observed=False)[
        "reduction_vs_static"
    ].sum()
    denominator = reduction_by_bin.sum()
    for label in DIFFICULTY_LABELS:
        tail.loc[
            tail.method == "DAPRO",
            f"dapro_vs_static_reduction_share_{label}",
        ] = (
            float(reduction_by_bin.get(label, 0.0) / denominator)
            if denominator != 0
            else math.nan
        )
    return raw, summary, tail


def _plot_ablation(split_df: pd.DataFrame, output_dir: Path, target_budget: float) -> None:
    colors = {
        "Static": "#4C78A8",
        "RandomAdaptive": "#F58518",
        "ScoreAdaptive-Heuristic": "#54A24B",
        "LocallyAdaptive": "#E45756",
        "DAPRO": "#B279A2",
    }
    panels = (
        ("absolute_coverage_deviation", "A. Absolute coverage deviation"),
        ("phase2_mean_latent_terminal_weight", "B. Mean terminal inverse weight"),
        ("selected_tau", "C. Selected calibration $\\tau$"),
        ("realized_budget_per_sample", "D. Realized budget per sample"),
    )
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), constrained_layout=True)
    rng = np.random.default_rng(7)
    positions = np.arange(len(METHODS))
    for ax, (metric, title) in zip(axes, panels):
        values = [
            split_df.loc[split_df.method == method, metric].dropna().to_numpy()
            for method in METHODS
        ]
        boxes = ax.boxplot(values, positions=positions, widths=0.55, patch_artist=True)
        for patch, method in zip(boxes["boxes"], METHODS):
            patch.set_facecolor(colors[method])
            patch.set_alpha(0.25)
        for x, (method, vals) in enumerate(zip(METHODS, values)):
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(
                x + jitter, vals, s=13, alpha=0.7, color=colors[method], edgecolor="none"
            )
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xticks(positions)
        ax.set_xticklabels(
            ("Static", "Random", "Heuristic", "Local", "DAPRO"), rotation=25
        )
        ax.grid(axis="y", alpha=0.2)
        if metric == "realized_budget_per_sample":
            ax.axhline(target_budget, color="black", linestyle="--", linewidth=1)
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"phase1_optimization_ablation.{suffix}", dpi=300)
    plt.close(fig)


def _plot_stratified(summary: pd.DataFrame, output_dir: Path) -> None:
    colors = {
        "Static": "#4C78A8",
        "ScoreAdaptive-Heuristic": "#54A24B",
        "LocallyAdaptive": "#E45756",
        "DAPRO": "#B279A2",
    }
    panels = (
        ("event_time_difficulty", "average_realized_budget", "A. Realized budget"),
        ("event_time_difficulty", "resolution_rate", "B. Resolution rate"),
        (
            "event_time_difficulty",
            "mean_log_latent_terminal_weight",
            "C. Mean log terminal weight",
        ),
        (
            "initial_score_quartile",
            "mean_log_latent_terminal_weight",
            "D. Mean log terminal weight",
        ),
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.5), constrained_layout=True)
    for ax, (stratification, metric, title) in zip(axes.ravel(), panels):
        part = summary[
            (summary.stratification == stratification) & (summary.metric == metric)
        ]
        labels = DIFFICULTY_LABELS if stratification == "event_time_difficulty" else SCORE_LABELS
        for method in STRATIFIED_METHODS:
            method_part = part[part.method == method].set_index("stratum").reindex(labels)
            mean = method_part["mean"].to_numpy(dtype=float)
            low = mean - method_part["bootstrap_ci_lower"].to_numpy(dtype=float)
            high = method_part["bootstrap_ci_upper"].to_numpy(dtype=float) - mean
            ax.errorbar(
                np.arange(4),
                mean,
                yerr=np.vstack([low, high]),
                marker="o",
                linewidth=1.5,
                capsize=2,
                color=colors[method],
                label=method,
            )
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels(labels)
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=8)
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"stratified_gain_decomposition.{suffix}", dpi=300)
    plt.close(fig)


def _write_latex(frame: pd.DataFrame, path: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        path.write_text(frame.to_latex(index=False, float_format="%.4g"), encoding="utf-8")


def _write_output_readme(
    output_dir: Path,
    args: argparse.Namespace,
    config: Config,
    source_files: list[str],
) -> None:
    source_listing = "\n".join(f"- `{path}`" for path in source_files)
    text = f"""# Phase-I optimization ablation

This directory is generated by
`python -m src.safety_evaluation.phase1_optimization_ablation`.
It contains the split- and sample-level paired ablation, paired bootstrap
comparisons, both stratifications, tail concentration, and publication figures.

## Reproduction

Dry validation (small deterministic numerical trajectories; no model calls):

```bash
python -m src.safety_evaluation.phase1_optimization_ablation --dry-run-fixture
```

Full cached-data experiment:

```bash
python -m src.safety_evaluation.phase1_optimization_ablation \\
  --dataset-name {args.dataset_name or "<dataset>"} \\
  --dataset-setup {args.dataset_setup or "<setup>"} \\
  --budget-per-sample {config.budget_per_sample} --cal-size {args.cal_size} \\
  --tau-prior {config.tau_prior} --seed-start {config.seed_start} \\
  --seed-end {config.seed_end} --n1 {config.n1} \\
  --projection {config.projection} --score {config.score}
```

Parallel seed-sharded run (four workers by default):

```bash
NUM_JOBS=12 DEVICE=cpu \\
  bash src/safety_evaluation/scripts/phase1_optimization_ablation.sh
```

Every worker writes raw outputs to a private temporary shard.  The parent waits
for all workers and validates seed/method completeness before producing the
merged tables and figures.  Each worker loads the cached tensors independently,
so choose `NUM_JOBS` according to available memory as well as CPU count.

The production command requires the repository's cached real-data tensors and
`alg_playground_model/.../probability_est_cal_test.pt`.  Missing caches are a
hard error; this analysis never generates conversations or calls an API.

## Source data

{source_listing}

## Methods and conventions

- **Static** uses the existing optimized all-or-nothing inclusion formula
  `p_i=min(1,1/sqrt(lambda*q_i))`, with lambda fit on Phase I.
- **RandomAdaptive** uses constant per-step continuation `p`, selected by
  bisection against the Phase-II per-sample budget.
- **ScoreAdaptive-Heuristic** uses tie-safe Phase-I midrank percentiles and
  `p_min+(1-p_min)*sigmoid(5*(rank-.5)+lambda)`.  Only lambda is fit.
- **LocallyAdaptive** mirrors the existing `AdaptiveOptimizedBudgetAllocator`:
  it fits a scalar shadow price on Phase I, then recomputes continuation
  probabilities from each trajectory's local conditional grid at every step.
  The probabilities are dynamic but are not jointly optimized by DAPRO's
  inverse-weight objective.
- **DAPRO** directly calls the existing `solve_exact_fast` inverse-weight
  optimizer and existing `{config.projection}` projection code with the existing
  `{config.score}` score.  Those code paths are not modified.

The nominal DAPRO positivity floor is `p_min={config.p_min}`.  Existing DAPRO
projection functions retain their own unchanged numerical epsilon.  Bisection
tolerance is `{config.bisection_tolerance}`; heuristic lambda bounds are
`[-30,30]`.  Missing Phase-I score times fall back to the nearest earlier
observed time and are counted in split metadata.  All adaptive methods use
uniforms deterministically keyed by `(split_id, sample_id, time)`.

Target coverage is `{1-config.target_miscoverage}` and the existing safe LPB
candidate rule is used over the repository tau grid.  Bootstrap intervals use
`{config.bootstrap_resamples}` resamples and seed `{config.bootstrap_seed}`.

## Interpretation

`DAPRO - LocallyAdaptive` directly contrasts globally optimized dynamic
probabilities with the existing local shadow-price rule.
`LocallyAdaptive - Static` isolates the gain from that local dynamic policy,
while `DAPRO - ScoreAdaptive-Heuristic`, `ScoreAdaptive-Heuristic -
RandomAdaptive`, and `RandomAdaptive - Static` retain the original decomposition.
The stratified tables show whether changes concentrate in late/horizon-limited
trajectories or specific initial-score quartiles.

A 50-split CPU run is expected to take tens of minutes to several hours,
depending mainly on calibration size and the selected DAPRO projection.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def _fixture_data(config: Config, device: str) -> CachedData:
    rng = np.random.default_rng(991)
    n, width, tau_count = 72, 8, 80
    base = rng.uniform(0.03, 0.35, size=(n, width))
    grid = np.zeros((n, width, width), dtype=np.float32)
    for current in range(width):
        future = np.maximum(base[:, current:], 1e-3)
        future /= future.sum(axis=1, keepdims=True)
        grid[:, current, current:] = future
    event = rng.integers(1, width + 1, size=n)
    tau_grid = np.logspace(-3, -0.01, tau_count)
    quantile = np.maximum(
        1,
        np.minimum(
            width,
            np.ceil(
                rng.uniform(1, width, size=(n, 1))
                * np.linspace(0.25, 1.0, tau_count)[None, :]
            ),
        ),
    )
    tensor_grid = torch.tensor(grid, device=device)
    return CachedData(
        event_times=torch.tensor(event, device=device),
        quantile_est=torch.tensor(quantile, dtype=torch.float32, device=device),
        conditional_grid=tensor_grid,
        probability_est=tensor_grid,
        taus_range=torch.tensor(tau_grid, device=device),
        cal_size=48,
        test_size=24,
        source_files=["deterministic numerical dry-run fixture"],
    )


def _load_cached_repository_data(
    args: argparse.Namespace, config: Config, device: str
) -> CachedData:
    # Imported only for a production cached-data run.  This utility has optional
    # metric dependencies that the numerical dry-run intentionally does not need.
    from src.safety_evaluation.utils.utils import setup_experiment_data

    prediction_path = (
        Path("alg_playground_model")
        / f"is_real_{args.data_type == 'real'}_dataset_{args.dataset_name}_dataset_{args.dataset_setup}"
        / "probability_est_cal_test.pt"
    )
    if not prediction_path.exists():
        raise FileNotFoundError(
            f"Required cached predictions are missing: {prediction_path.resolve()}. "
            "Refusing to run a model or generate data for this analysis."
        )
    dataset_cache = (
        Path("src")
        / "datasets"
        / "real_data"
        / args.dataset_name
        / args.dataset_setup
    )
    dataset_manifest = dataset_cache / "n_samples_test.npy"
    if not dataset_manifest.exists():
        raise FileNotFoundError(
            f"Required cached trajectory tensors are missing: "
            f"{dataset_manifest.resolve()}. Refusing to reconstruct them from raw "
            "conversation logs for this analysis."
        )
    taus = torch.tensor(np.logspace(-3, -0.01, 1000), device=device)
    (
        _,
        event_times,
        quantile_est,
        probability_est,
        conditional_grid,
        test_size,
    ) = setup_experiment_data(
        args.cal_size,
        args.data_type == "real",
        device,
        args.dataset_name,
        args.dataset_setup,
        taus,
        config.max_horizon,
    )
    return CachedData(
        event_times,
        quantile_est,
        conditional_grid,
        probability_est,
        taus,
        args.cal_size,
        test_size,
        [
            str(prediction_path.resolve()),
            *[str(path.resolve()) for path in sorted(dataset_cache.glob("*.npy"))],
        ],
    )


def _config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        seed_start=args.seed_start,
        seed_end=args.seed_end,
        budget_per_sample=args.budget_per_sample,
        tau_prior=args.tau_prior,
        target_miscoverage=args.target_miscoverage,
        n1=args.n1,
        p_min=args.p_min,
        bisection_tolerance=args.tolerance,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_resamples=args.bootstrap_resamples,
        projection=args.projection,
        score=args.score,
        max_horizon=args.max_horizon,
    )


def _write_raw_outputs(
    split_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    split_df.sort_values(["split_id", "method"]).to_csv(
        output_dir / "split_level_results.csv", index=False
    )
    sample_df.sort_values(["split_id", "sample_id", "method"]).to_csv(
        output_dir / "sample_level_results.csv", index=False
    )


def _finalize_outputs(
    split_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    output_dir: Path,
    args: argparse.Namespace,
    config: Config,
    source_files: list[str],
    devices: list[str],
    parallel_workers: int,
) -> None:
    split_df = split_df.sort_values(["split_id", "method"]).reset_index(drop=True)
    sample_df = sample_df.sort_values(
        ["split_id", "sample_id", "method"]
    ).reset_index(drop=True)
    _write_raw_outputs(split_df, sample_df, output_dir)
    paired = paired_summary(split_df, config)
    paired.to_csv(output_dir / "paired_comparisons.csv", index=False)
    _write_latex(paired, output_dir / "paired_comparisons.tex")
    strat_raw, strat_summary, tail = stratified_outputs(sample_df, config)
    strat_raw.to_csv(output_dir / "stratified_results_raw.csv", index=False)
    strat_summary.to_csv(output_dir / "stratified_results_summary.csv", index=False)
    tail.to_csv(output_dir / "tail_concentration.csv", index=False)
    _plot_ablation(split_df, output_dir, split_df.target_budget_per_sample.mean())
    _plot_stratified(strat_summary, output_dir)
    metadata = {
        "config": asdict(config),
        "device": devices[0] if len(devices) == 1 else devices,
        "dry_run": bool(args.dry_run or args.dry_run_fixture),
        "fixture": bool(args.dry_run_fixture),
        "source_files": source_files,
        "methods": list(METHODS),
        "parallel_workers": parallel_workers,
        "common_random_numbers": "sha256 keyed by split_id/sample_id, one path shared by adaptive methods",
        "clipping": {
            "heuristic": "none beyond the explicit p_min parameterization",
            "random": "bounded bisection domain [p_min,1]",
            "locally_adaptive": (
                "existing early-stop inclusion-probability floor p_min; "
                "continuation probabilities otherwise follow the shadow-price rule"
            ),
            "dapro": "unchanged projection-specific epsilon in repository utility",
        },
        "indexing_convention": (
            "Repository event times are stored as 1-based stopping times while "
            "predicted prior horizons are grid indices. Acquisition follows existing "
            "DAPRO exactly: active opportunities are min(T+1,q_prior+1), capped by "
            "the saved trajectory width; b_i preserves the requested raw min(T,q_prior)."
        ),
        "new_llm_or_api_calls": False,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    _write_output_readme(output_dir, args, config, source_files)


def run(args: argparse.Namespace) -> Path:
    device = args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu"
    config = _config_from_args(args)
    data = _fixture_data(config, device) if args.dry_run_fixture else _load_cached_repository_data(args, config, device)
    output_dir = Path(args.output_dir)
    if args.dry_run_fixture:
        output_dir /= "dry_run"
    split_rows, sample_rows = [], []
    for seed in range(config.seed_start, config.seed_end):
        current_split, current_samples = run_split(
            data,
            config,
            seed,
            device,
            dry_limit=args.dry_limit if args.dry_run else None,
        )
        split_rows.extend(current_split)
        sample_rows.extend(current_samples)
    split_df = pd.DataFrame(split_rows)
    sample_df = pd.DataFrame(sample_rows)
    if args.raw_only:
        _write_raw_outputs(split_df, sample_df, output_dir)
        worker_metadata = {
            "config": asdict(config),
            "device": device,
            "source_files": data.source_files,
            "methods": list(METHODS),
        }
        (output_dir / "worker_metadata.json").write_text(
            json.dumps(worker_metadata, indent=2), encoding="utf-8"
        )
    else:
        _finalize_outputs(
            split_df,
            sample_df,
            output_dir,
            args,
            config,
            data.source_files,
            [device],
            parallel_workers=1,
        )
    return output_dir


def merge_shards(args: argparse.Namespace) -> Path:
    """Merge isolated seed shards and generate the final analysis outputs."""

    config = _config_from_args(args)
    shard_dirs = [Path(path) for path in args.merge_shards]
    split_frames = []
    sample_frames = []
    worker_metadata = []
    for shard_dir in shard_dirs:
        split_path = shard_dir / "split_level_results.csv"
        sample_path = shard_dir / "sample_level_results.csv"
        metadata_path = shard_dir / "worker_metadata.json"
        missing = [
            str(path)
            for path in (split_path, sample_path, metadata_path)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Shard {shard_dir} is incomplete; missing: {missing}"
            )
        split_frames.append(pd.read_csv(split_path))
        sample_frames.append(pd.read_csv(sample_path))
        worker_metadata.append(json.loads(metadata_path.read_text(encoding="utf-8")))

    split_df = pd.concat(split_frames, ignore_index=True)
    sample_df = pd.concat(sample_frames, ignore_index=True)
    expected_seeds = set(range(config.seed_start, config.seed_end))
    actual_seeds = set(split_df["split_id"].astype(int).unique())
    if actual_seeds != expected_seeds:
        missing = sorted(expected_seeds - actual_seeds)
        extra = sorted(actual_seeds - expected_seeds)
        raise ValueError(f"Shard seed mismatch; missing={missing}, extra={extra}")
    split_counts = split_df.groupby("split_id")["method"].agg(["size", "nunique"])
    if not (
        split_counts["size"].eq(len(METHODS))
        & split_counts["nunique"].eq(len(METHODS))
    ).all():
        raise ValueError("Each merged seed must contain every method exactly once.")
    if split_df.duplicated(["split_id", "method"]).any():
        raise ValueError("Duplicate split/method rows found across shards.")
    if sample_df.duplicated(["split_id", "sample_id", "method"]).any():
        raise ValueError("Duplicate split/sample/method rows found across shards.")
    sample_counts = sample_df.groupby(["split_id", "sample_id"])["method"].agg(
        ["size", "nunique"]
    )
    if not (
        sample_counts["size"].eq(len(METHODS))
        & sample_counts["nunique"].eq(len(METHODS))
    ).all():
        raise ValueError(
            "Each merged split/sample pair must contain every method exactly once."
        )

    reference_config = {
        key: value
        for key, value in worker_metadata[0]["config"].items()
        if key not in {"seed_start", "seed_end"}
    }
    requested_config = {
        key: value
        for key, value in asdict(config).items()
        if key not in {"seed_start", "seed_end"}
    }
    if reference_config != requested_config:
        raise ValueError("Merged command configuration does not match the shards.")
    for metadata in worker_metadata:
        current_config = {
            key: value
            for key, value in metadata["config"].items()
            if key not in {"seed_start", "seed_end"}
        }
        if current_config != reference_config:
            raise ValueError("Shard configurations do not match.")
        if metadata["methods"] != list(METHODS):
            raise ValueError("Shard method lists do not match the current analysis.")
    source_files = worker_metadata[0]["source_files"]
    if any(metadata["source_files"] != source_files for metadata in worker_metadata[1:]):
        raise ValueError("Shard source-file manifests do not match.")
    devices = sorted({metadata["device"] for metadata in worker_metadata})

    output_dir = Path(args.output_dir)
    if args.dry_run_fixture:
        output_dir /= "dry_run"
    output_dir.mkdir(parents=True, exist_ok=True)
    _finalize_outputs(
        split_df,
        sample_df,
        output_dir,
        args,
        config,
        source_files,
        devices,
        parallel_workers=len(shard_dirs),
    )
    return output_dir


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name")
    parser.add_argument("--dataset-setup")
    parser.add_argument("--data-type", choices=("real", "synthetic"), default="real")
    parser.add_argument("--cal-size", type=int, default=3000)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=50)
    parser.add_argument("--budget-per-sample", type=float, default=20.0)
    parser.add_argument("--tau-prior", type=float, default=0.56)
    parser.add_argument("--target-miscoverage", type=float, default=0.10)
    parser.add_argument("--n1", type=int, default=100)
    parser.add_argument("--p-min", type=float, default=0.005)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--bootstrap-seed", type=int, default=1729)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--projection", choices=("ir", "platt", "beta"), default="platt")
    parser.add_argument("--score", choices=("prob", "quantile"), default="prob")
    parser.add_argument("--max-horizon", type=int, default=200)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="results/phase1_optimization_ablation")
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="Write only raw split/sample outputs for one parallel worker.",
    )
    parser.add_argument(
        "--merge-shards",
        nargs="+",
        metavar="SHARD_DIR",
        help="Merge raw worker directories and generate final tables and figures.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Run one split on cached-data subsets.")
    parser.add_argument("--dry-limit", type=int, default=200)
    parser.add_argument(
        "--dry-run-fixture",
        action="store_true",
        help="Run one split on deterministic numeric trajectories when repository caches are unavailable.",
    )
    args = parser.parse_args(argv)
    if args.dry_run or args.dry_run_fixture:
        args.bootstrap_resamples = min(args.bootstrap_resamples, 500)
        args.n1 = min(args.n1, 16)
    if (args.dry_run or args.dry_run_fixture) and not args.merge_shards:
        args.seed_end = args.seed_start + 1
    if args.raw_only and args.merge_shards:
        parser.error("--raw-only and --merge-shards are mutually exclusive.")
    if not args.dry_run_fixture and (not args.dataset_name or not args.dataset_setup):
        parser.error("--dataset-name and --dataset-setup are required for cached-data runs.")
    if args.seed_end <= args.seed_start:
        parser.error("--seed-end must be greater than --seed-start.")
    if not 0 < args.p_min <= 1:
        parser.error("--p-min must be in (0,1].")
    if not 0 < args.target_miscoverage < 1:
        parser.error("--target-miscoverage must be in (0,1).")
    return args


def main() -> None:
    args = parse_args()
    output = merge_shards(args) if args.merge_shards else run(args)
    print(f"Wrote phase-I optimization ablation to {output.resolve()}")


if __name__ == "__main__":
    main()
