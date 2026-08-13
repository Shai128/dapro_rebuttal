"""Real-data pilot for LPB selector-aware DAPRO objectives.

This file is deliberately an offline diagnostic, not production allocation
code.  It uses the production LPB candidate grid, N1/Phase-II split, direct
binned solver, budget correction, Horvitz--Thompson calibration, and strict
prefix selector.  Alternative objectives are confined to this file.

Information audit
-----------------
* Policy fitting uses only the N1 fully observed Phase-I calibration rows,
  their model predictions, and causal prefix predictions for Phase II.
* ``selection_value`` is computed from the conditional distribution available
  at the current prefix; it never reads a Phase-II event time.
* Test outcomes enter only after a policy has been frozen, to evaluate the
  selected LPB.  They never enter a score, objective, bin, or probability.
* The pivotal variant uses artificial common-random Bernoulli draws on the
  Phase-I rows and a Phase-I plug-in output curve.  It is a pilot sensitivity
  estimate, not an oracle/test-assisted policy.

The stabilized analytic objective is

    w_i = sum_j Delta_j A_ij / ((alpha - m_j)^2 + tau^2),

where ``m_j`` and ``Delta_j = m_j - m_{j-1}`` are computed on Phase I,
``A_ij = 1{T_i < q_ij}``, and
``tau = tau_multiplier * sqrt(alpha*(1-alpha)/N1)``.  Its soft causal version
uses at prefix t the event-time mass

    h_i(t | H_it) * sum_j c_j 1{t < q_ij},

with ``c_j = Delta_j / ((alpha-m_j)^2+tau^2)``.  Thus it is the conditional
expectation of the hard row weight under the frozen conditional PMF.
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

from analysis.diagnostics.dapro_binning_audit import (
    SETUPS,
    bin_stats,
    load_setup,
)
from analysis.diagnostics.lpb_dapro_binning_audit import (
    lpb_quantiles,
    strict_select,
    target_value_scores,
)
from src.predictive_bounds.budget_allocators.optimization_solver_utils import (
    solve_binned_deployable_policy,
)
from src.predictive_bounds.budget_allocators.projected_optimization_utils import (
    correct_projected_cumulative_probabilities_to_budget,
)
from src.predictive_bounds.calibration.calibration_utils import get_prior
from src.predictive_bounds.utils.utils import make_lpb_tau_grid


ALPHA = 0.10
WIDTH = 200


@dataclass(frozen=True)
class Variant:
    objective: str
    score: str
    bins: int


VARIANTS = {
    "reference_soft_anchor_hazard_k2": Variant("reference_soft", "hazard", 2),
    "reference_soft_anchor_value_k4": Variant("reference_soft", "anchor_value", 4),
    "margin_hard_value_k2": Variant("margin_hard", "anchor_value", 2),
    "margin_hard_value_k4": Variant("margin_hard", "anchor_value", 4),
    "margin_soft_value_k2": Variant("margin_soft", "selection_value", 2),
    "margin_soft_value_k4": Variant("margin_soft", "selection_value", 4),
    "pivotal_hard_value_k4": Variant("pivotal_hard", "anchor_value", 4),
    "smooth_influence_hard_value_k4": Variant(
        "smooth_influence_hard", "anchor_value", 4
    ),
    "smooth_acquisition_influence_hard_value_k4": Variant(
        "smooth_acquisition_influence_hard", "anchor_value", 4
    ),
}


def _thresholds(candidate_events: np.ndarray) -> np.ndarray:
    """First true candidate per row for a row-wise nested event matrix."""
    events = np.asarray(candidate_events, dtype=bool)
    any_event = events.any(axis=1)
    threshold = np.full(len(events), events.shape[1], dtype=np.int64)
    threshold[any_event] = events[any_event].argmax(axis=1)
    return threshold


def _curve_from_thresholds(
    thresholds: np.ndarray,
    weights: np.ndarray,
    candidate_count: int,
) -> np.ndarray:
    increments = np.bincount(
        thresholds,
        weights=weights,
        minlength=candidate_count + 1,
    )[:candidate_count]
    return np.cumsum(increments)


def _strict_select_monotone(values: np.ndarray, alpha: float = ALPHA) -> int:
    """Fast equivalent of the production selector for monotone curves."""
    crossing = int(np.searchsorted(values, alpha, side="left"))
    return min(max(crossing - 1, 0), len(values) - 1)


def analytic_margin_components(
    fit_events: np.ndarray,
    *,
    alpha: float,
    tau_multiplier: float,
    regularization: float,
    candidate_window_multiplier: float,
    weight_clip_quantile: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Return Phase-I row weights and frozen candidate coefficients."""
    n1 = len(fit_events)
    rates = fit_events.mean(axis=0, dtype=np.float64)
    deltas = np.zeros_like(rates)
    deltas[1:] = np.maximum(np.diff(rates), 0.0)
    tau = tau_multiplier * np.sqrt(alpha * (1 - alpha) / n1)
    coefficients = deltas / ((alpha - rates) ** 2 + tau**2)
    if np.isfinite(candidate_window_multiplier):
        window = candidate_window_multiplier * tau
        coefficients = coefficients * (np.abs(alpha - rates) <= window)
    row_weights = fit_events.astype(np.float64) @ coefficients

    positive = row_weights > 0
    clip_value = np.inf
    if np.any(positive) and weight_clip_quantile < 1.0:
        clip_value = float(np.quantile(
            row_weights[positive],
            weight_clip_quantile,
        ))
        row_weights = np.minimum(row_weights, clip_value)

    # A tiny non-target event term preserves exploration for Phase-I outcomes
    # assigned zero selector weight.  Normalize first so the regularizer has a
    # fixed interpretation and cannot depend on the arbitrary coefficient
    # scale.  This is the same 0.001 convention used by Definitive DAPRO.
    positive = row_weights > 0
    scale = float(row_weights[positive].mean()) if np.any(positive) else 1.0
    row_weights = row_weights / max(scale, np.finfo(np.float64).tiny)
    any_event = fit_events[:, -1].astype(np.float64)
    row_weights = (row_weights + regularization * any_event) / (
        1.0 + regularization
    )
    return row_weights, coefficients, {
        "margin_tau": float(tau),
        "margin_tau_multiplier": float(tau_multiplier),
        "margin_regularization": float(regularization),
        "margin_candidate_window_multiplier": float(candidate_window_multiplier),
        "margin_weight_clip_quantile": float(weight_clip_quantile),
        "margin_weight_clip_value": float(clip_value),
        "margin_nonzero_deltas": int(np.count_nonzero(deltas)),
        "margin_coefficient_sum": float(coefficients.sum()),
        "margin_weight_mean": float(row_weights.mean()),
        "margin_weight_max": float(row_weights.max()),
        "phase1_selected_index": int(strict_select(rates)),
        "phase1_selected_rate": float(rates[strict_select(rates)]),
        "phase1_min_abs_margin": float(np.abs(alpha - rates).min()),
    }


def event_weight_table(
    quantiles: np.ndarray,
    coefficients: np.ndarray,
    width: int,
) -> np.ndarray:
    """``sum_j c_j 1{event_time < q_ij}`` for every row/event time."""
    quantiles = np.asarray(quantiles, dtype=np.int64)
    output = np.empty((len(quantiles), width), dtype=np.float64)
    for row, horizons in enumerate(quantiles):
        grouped = np.bincount(
            horizons,
            weights=coefficients,
            minlength=width + 2,
        )
        # At event time t in {1,...,M}, A_ij=1 iff t < q_ij.
        output[row] = np.cumsum(grouped[::-1])[::-1][2 : width + 2]
    return output


def selection_value_score(
    grid: torch.Tensor,
    event_weights: np.ndarray,
    rows: np.ndarray,
    width: int,
) -> np.ndarray:
    """Causal remaining selector value per unit expected remaining cost.

    The numerator is the conditional expectation of the frozen analytic row
    weight over future event times.  The denominator is the conditional
    expected remaining acquisition cost.  Their square-root ratio is the
    one-block Neyman solution and supplies an ordering only; the DAPRO solver
    still optimizes all continuation probabilities and the exact budget.
    """
    output = np.zeros((len(rows), width), dtype=np.float64)
    weights = torch.as_tensor(event_weights[rows], dtype=grid.dtype)
    selected_grid = grid[rows]
    tiny = torch.finfo(selected_grid.dtype).tiny
    for step in range(width):
        pmf = selected_grid[:, step, step:width]
        valid = selected_grid[:, step, step:].sum(dim=1).clamp_min(tiny)
        future_weight = weights[:, step:width]
        numerator = (pmf * future_weight).sum(dim=1) / valid
        remaining = torch.arange(
            1,
            width - step + 1,
            dtype=selected_grid.dtype,
        )
        denominator = (pmf * remaining).sum(dim=1)
        if selected_grid.shape[2] > width:
            denominator = denominator + (
                selected_grid[:, step, width:].sum(dim=1) * (width - step)
            )
        denominator = (denominator / valid).clamp_min(tiny)
        output[:, step] = torch.sqrt(
            numerator.clamp_min(0) / denominator
        ).to(torch.float64).numpy()
    return output


def pivotal_phase1_weights(
    fit_events: np.ndarray,
    reference_pi: np.ndarray,
    *,
    draws: int,
    seed: int,
    regularization: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Common-random finite-difference selector sensitivity on Phase I.

    ``g`` is the Phase-I plug-in coverage curve ``1-m_j``.  For every common
    draw of all other Bernoulli acquisitions, row i is forced to zero and one;
    the mean squared change in selected ``g`` is its pivotal weight.
    """
    n1, candidates = fit_events.shape
    pi = np.clip(np.asarray(reference_pi, dtype=np.float64), 1e-8, 1.0)
    threshold = _thresholds(fit_events)
    rates = fit_events.mean(axis=0, dtype=np.float64)
    output_curve = 1.0 - rates
    rng = np.random.default_rng(seed)
    square_change = np.zeros(n1, dtype=np.float64)
    change = np.zeros(n1, dtype=np.float64)
    switch = np.zeros(n1, dtype=np.float64)

    for _ in range(draws):
        z = rng.random(n1) < pi
        base_curve = _curve_from_thresholds(
            threshold,
            z.astype(np.float64) / pi,
            candidates,
        ) / n1
        for row in range(n1):
            without = base_curve.copy()
            if z[row] and threshold[row] < candidates:
                without[threshold[row] :] -= 1.0 / (pi[row] * n1)
            selected_zero = _strict_select_monotone(without)
            with_row = without.copy()
            if threshold[row] < candidates:
                with_row[threshold[row] :] += 1.0 / (pi[row] * n1)
            selected_one = _strict_select_monotone(with_row)
            difference = output_curve[selected_one] - output_curve[selected_zero]
            change[row] += abs(difference)
            square_change[row] += difference**2
            switch[row] += selected_one != selected_zero

    weights = square_change / draws
    positive = weights > 0
    scale = float(weights[positive].mean()) if np.any(positive) else 1.0
    weights = weights / max(scale, np.finfo(np.float64).tiny)
    weights = (weights + regularization * fit_events[:, -1]) / (
        1.0 + regularization
    )
    return weights, {
        "pivotal_draws": int(draws),
        "pivotal_nonzero_rows": int(np.count_nonzero(positive)),
        "pivotal_mean_abs_output_change": float(change.mean() / draws),
        "pivotal_mean_force_switch_probability": float(switch.mean() / draws),
        "pivotal_weight_mean": float(weights.mean()),
        "pivotal_weight_max": float(weights.max()),
    }


def smooth_selection_influence_weights(
    fit_events: np.ndarray,
    fit_quantiles: np.ndarray,
    *,
    alpha: float,
    bandwidth_multiplier: float,
    winsor_quantile: float,
    anchor_shrink: float,
    regularization: float,
    centered: bool,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return a stabilized first-order smooth-selector influence target.

    Consecutive candidates with identical Phase-I event columns are compressed
    before differentiation.  On the compressed grid, freeze the same-sample
    plug-in output curve ``g_j = 1 - m_j`` and define

        G_h(m) = g_0 + sum_{j>=1} Delta g_j Phi((alpha-m_j)/h).

    Its frozen-curve gradient is

        gamma_j = -Delta g_j phi_normal((alpha-m_j)/h) / h.

    With ``centered=True``, the row influence
    ``sum_j gamma_j(A_ij-m_j)`` is formed *before* squaring, retaining signed
    cancellation among nested candidates.  This is the superpopulation split
    influence.  With ``centered=False``, the function instead forms the
    fixed-population HT acquisition influence ``sum_j gamma_j A_ij``.  The
    resulting squared influence is winsorized, normalized, and shrunk toward
    the Phase-I selected hard event.  Finally a constant 0.001-style term is
    added to every row, preserving exploration even for a zero plug-in
    influence.  All ingredients use Phase I only.
    """
    events = np.asarray(fit_events, dtype=bool)
    quantiles = np.asarray(fit_quantiles, dtype=np.int64)
    n1, raw_candidates = events.shape
    if quantiles.shape != events.shape:
        raise ValueError("Phase-I events and quantiles must have equal shape.")

    event_change = np.concatenate([
        np.ones(1, dtype=bool),
        np.any(events[:, 1:] != events[:, :-1], axis=0),
    ])
    compressed_indices = np.flatnonzero(event_change)
    compressed_events = events[:, compressed_indices].astype(np.float64)
    rates = compressed_events.mean(axis=0)

    quantile_change = np.concatenate([
        np.ones(1, dtype=bool),
        np.any(quantiles[:, 1:] != quantiles[:, :-1], axis=0),
    ])
    integer_pattern_count = int(quantile_change.sum())

    bandwidth_base = np.sqrt(alpha * (1 - alpha) / n1)
    bandwidth = bandwidth_multiplier * bandwidth_base
    output_curve = 1.0 - rates
    delta_g = np.diff(output_curve)
    transition_rates = rates[1:]
    z = (alpha - transition_rates) / bandwidth
    kernel_density = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    gamma = -delta_g * kernel_density / bandwidth
    influence_features = compressed_events[:, 1:]
    if centered:
        influence_features = influence_features - transition_rates[None, :]
    contributions = influence_features * gamma[None, :]
    influence = contributions.sum(axis=1)
    raw_weights = np.square(influence)

    positive = raw_weights > np.finfo(np.float64).tiny
    winsor_cap = 0.0
    clipped = raw_weights.copy()
    if np.any(positive):
        winsor_cap = float(np.quantile(raw_weights[positive], winsor_quantile))
        clipped = np.minimum(clipped, winsor_cap)
    clipped_mean = max(float(clipped.mean()), np.finfo(np.float64).tiny)
    normalized_influence = clipped / clipped_mean

    raw_rates = events.mean(axis=0, dtype=np.float64)
    selected_index = strict_select(raw_rates, target=alpha)
    anchor = events[:, selected_index].astype(np.float64)
    anchor_mean = max(float(anchor.mean()), np.finfo(np.float64).tiny)
    normalized_anchor = anchor / anchor_mean
    weights = (
        (1.0 - anchor_shrink) * normalized_influence
        + anchor_shrink * normalized_anchor
    )
    weights = (weights + regularization) / (1.0 + regularization)

    absolute_contribution = np.abs(contributions).sum(axis=1)
    nonzero_contribution = absolute_contribution > np.finfo(np.float64).tiny
    cancellation = np.zeros(n1, dtype=np.float64)
    cancellation[nonzero_contribution] = 1.0 - (
        np.abs(influence[nonzero_contribution])
        / absolute_contribution[nonzero_contribution]
    )

    def effective_size(values: np.ndarray) -> float:
        total = float(values.sum())
        square = float(np.square(values).sum())
        return total**2 / square if square > 0 else 0.0

    gamma_abs = np.abs(gamma)
    gamma_total = float(gamma_abs.sum())
    gamma_probability = (
        gamma_abs / gamma_total
        if gamma_total > 0
        else np.zeros_like(gamma_abs)
    )
    gamma_effective_candidates = (
        1.0 / float(np.square(gamma_probability).sum())
        if np.any(gamma_probability)
        else 0.0
    )
    return weights, {
        "influence_raw_candidate_count": int(raw_candidates),
        "influence_centered": int(centered),
        "influence_integer_pattern_count": integer_pattern_count,
        "influence_event_pattern_count": int(len(compressed_indices)),
        "influence_duplicate_event_share": float(
            1.0 - len(compressed_indices) / raw_candidates
        ),
        "influence_bandwidth_base": float(bandwidth_base),
        "influence_bandwidth_multiplier": float(bandwidth_multiplier),
        "influence_bandwidth": float(bandwidth),
        "influence_winsor_quantile": float(winsor_quantile),
        "influence_winsor_cap": float(winsor_cap),
        "influence_anchor_shrink": float(anchor_shrink),
        "influence_selected_index": int(selected_index),
        "influence_selected_rate": float(raw_rates[selected_index]),
        "influence_gamma_nonzero_count": int(np.count_nonzero(gamma_abs > 1e-15)),
        "influence_gamma_positive_count": int(np.count_nonzero(gamma > 1e-15)),
        "influence_gamma_negative_count": int(np.count_nonzero(gamma < -1e-15)),
        "influence_gamma_abs_sum": gamma_total,
        "influence_gamma_effective_candidates": float(gamma_effective_candidates),
        "influence_phi_positive_share": float(np.mean(influence > 1e-12)),
        "influence_phi_negative_share": float(np.mean(influence < -1e-12)),
        "influence_phi_zero_share": float(np.mean(np.abs(influence) <= 1e-12)),
        "influence_phi_abs_mean": float(np.abs(influence).mean()),
        "influence_phi_abs_max": float(np.abs(influence).max()),
        "influence_mean_cancellation": float(
            cancellation[nonzero_contribution].mean()
            if np.any(nonzero_contribution)
            else 0.0
        ),
        "influence_median_cancellation": float(
            np.median(cancellation[nonzero_contribution])
            if np.any(nonzero_contribution)
            else 0.0
        ),
        "influence_raw_weight_effective_rows": float(effective_size(raw_weights)),
        "influence_clipped_weight_effective_rows": float(effective_size(clipped)),
        "influence_final_weight_effective_rows": float(effective_size(weights)),
        "influence_final_weight_max": float(weights.max()),
        "influence_exploration_regularization": float(regularization),
    }


@dataclass
class FittedPolicy:
    variant: str
    calibration_idx: np.ndarray
    test_idx: np.ndarray
    fit_local: np.ndarray
    deploy_local: np.ndarray
    fit_lengths: np.ndarray
    deploy_lengths: np.ndarray
    fit_pi: np.ndarray
    deploy_q: np.ndarray
    deploy_pi: np.ndarray
    deploy_active: np.ndarray
    phase1_cost: float
    expected_total_cost: float
    diagnostics: dict


def fit_variant(
    *,
    variant_name: str,
    data_seed: int,
    policy_seed: int,
    times: np.ndarray,
    quantiles: np.ndarray,
    prior: np.ndarray,
    grid: torch.Tensor,
    hazard: np.ndarray,
    anchor_value: np.ndarray,
    anchor_horizons: np.ndarray,
    budget: float,
    n1: int,
    projection_margin: float,
    tau_multiplier: float,
    regularization: float,
    pivotal_draws: int,
    candidate_window_multiplier: float,
    weight_clip_quantile: float,
    anchor_shrink: float,
    influence_bandwidth_multiplier: float,
    influence_winsor_quantile: float,
    influence_anchor_shrink: float,
    selection_score_cache: dict[tuple[int, int, float, float], np.ndarray] | None = None,
) -> FittedPolicy:
    variant = VARIANTS[variant_name]
    outer_idx = np.random.RandomState(data_seed).permutation(len(times))
    calibration_idx = outer_idx[:3000]
    test_idx = outer_idx[3000:]
    inner = np.random.RandomState(policy_seed).permutation(len(calibration_idx))
    fit_local = inner[:n1]
    deploy_local = inner[n1:]
    fit_idx = calibration_idx[fit_local]
    deploy_idx = calibration_idx[deploy_local]
    fit_lengths = np.minimum(times[fit_idx], prior[fit_idx]).astype(np.int64)
    deploy_lengths = np.minimum(times[deploy_idx], prior[deploy_idx]).astype(np.int64)
    phase1_cost = float(fit_lengths.sum())
    target_budget = (
        (budget * len(calibration_idx) - phase1_cost) / len(deploy_idx)
        - projection_margin
    )
    if target_budget <= 0:
        raise ValueError("Phase-I rows exhaust the LPB budget.")

    fit_events = times[fit_idx, None] < quantiles[fit_idx]
    margin_weights, coefficients, margin_diagnostics = analytic_margin_components(
        fit_events,
        alpha=ALPHA,
        tau_multiplier=tau_multiplier,
        regularization=regularization,
        candidate_window_multiplier=candidate_window_multiplier,
        weight_clip_quantile=weight_clip_quantile,
    )
    active_fit = np.arange(WIDTH)[None, :] < fit_lengths[:, None]
    objective_weights = None
    objective_masses = None
    extra_diagnostics: dict[str, float] = dict(margin_diagnostics)

    if variant.objective == "reference_soft":
        target_mask = (
            np.arange(1, WIDTH + 1)[None, :]
            < anchor_horizons[fit_idx, None]
        )
        objective_masses = (
            hazard[fit_idx] * active_fit * target_mask
            + regularization * hazard[fit_idx] * active_fit
        ) / (1.0 + regularization)
    elif variant.objective == "margin_hard":
        selected_index = int(margin_diagnostics["phase1_selected_index"])
        anchor_weights = fit_events[:, selected_index].astype(np.float64)
        margin_mean = max(margin_weights.mean(), np.finfo(np.float64).tiny)
        anchor_mean = max(anchor_weights.mean(), np.finfo(np.float64).tiny)
        objective_weights = (
            (1.0 - anchor_shrink) * margin_weights / margin_mean
            + anchor_shrink * anchor_weights / anchor_mean
        )
    elif variant.objective == "margin_soft":
        all_event_weights = event_weight_table(quantiles, coefficients, WIDTH)
        margin_mass = hazard[fit_idx] * active_fit * all_event_weights[fit_idx]
        selected_index = int(margin_diagnostics["phase1_selected_index"])
        selected_horizons = quantiles[fit_idx, selected_index]
        selected_mask = (
            np.arange(1, WIDTH + 1)[None, :] < selected_horizons[:, None]
        )
        anchor_mass = hazard[fit_idx] * active_fit * selected_mask
        margin_mean = max(
            float(margin_mass.sum() / max(active_fit.sum(), 1)),
            np.finfo(np.float64).tiny,
        )
        anchor_mean = max(
            float(anchor_mass.sum() / max(active_fit.sum(), 1)),
            np.finfo(np.float64).tiny,
        )
        objective_masses = (
            (1.0 - anchor_shrink) * margin_mass / margin_mean
            + anchor_shrink * anchor_mass / anchor_mean
            + regularization * hazard[fit_idx] * active_fit
        ) / (1.0 + regularization)
    elif variant.objective == "pivotal_hard":
        # Fit a reference K2 policy recursively, then estimate black-box
        # selector sensitivity on the same Phase-I rows.  Recursion terminates
        # because the requested reference objective is different.
        reference = fit_variant(
            variant_name="reference_soft_anchor_hazard_k2",
            data_seed=data_seed,
            policy_seed=policy_seed,
            times=times,
            quantiles=quantiles,
            prior=prior,
            grid=grid,
            hazard=hazard,
            anchor_value=anchor_value,
            anchor_horizons=anchor_horizons,
            budget=budget,
            n1=n1,
            projection_margin=projection_margin,
            tau_multiplier=tau_multiplier,
            regularization=regularization,
            pivotal_draws=0,
            candidate_window_multiplier=candidate_window_multiplier,
            weight_clip_quantile=weight_clip_quantile,
            anchor_shrink=anchor_shrink,
            influence_bandwidth_multiplier=influence_bandwidth_multiplier,
            influence_winsor_quantile=influence_winsor_quantile,
            influence_anchor_shrink=influence_anchor_shrink,
            selection_score_cache=selection_score_cache,
        )
        objective_weights, pivotal_diagnostics = pivotal_phase1_weights(
            fit_events,
            reference.fit_pi,
            draws=pivotal_draws,
            seed=policy_seed + 70_001,
            regularization=regularization,
        )
        extra_diagnostics.update(pivotal_diagnostics)
    elif variant.objective in {
        "smooth_influence_hard",
        "smooth_acquisition_influence_hard",
    }:
        objective_weights, influence_diagnostics = (
            smooth_selection_influence_weights(
                fit_events,
                quantiles[fit_idx],
                alpha=ALPHA,
                bandwidth_multiplier=influence_bandwidth_multiplier,
                winsor_quantile=influence_winsor_quantile,
                anchor_shrink=influence_anchor_shrink,
                regularization=regularization,
                centered=(variant.objective == "smooth_influence_hard"),
            )
        )
        extra_diagnostics.update(influence_diagnostics)
    else:  # pragma: no cover - guarded by the fixed registry
        raise ValueError(variant.objective)

    if variant.score == "hazard":
        fit_scores = hazard[fit_idx]
        deploy_scores = hazard[deploy_idx]
    elif variant.score == "anchor_value":
        fit_scores = anchor_value[fit_idx]
        deploy_scores = anchor_value[deploy_idx]
    elif variant.score == "selection_value":
        key = (
            data_seed,
            policy_seed,
            float(tau_multiplier),
            float(candidate_window_multiplier),
        )
        values = None if selection_score_cache is None else selection_score_cache.get(key)
        if values is None:
            all_event_weights = event_weight_table(quantiles, coefficients, WIDTH)
            selected_rows = np.concatenate([fit_idx, deploy_idx])
            values = selection_value_score(
                grid,
                all_event_weights,
                selected_rows,
                WIDTH,
            )
            if selection_score_cache is not None:
                selection_score_cache[key] = values
        fit_scores = values[: len(fit_idx)]
        deploy_scores = values[len(fit_idx) :]
    else:  # pragma: no cover
        raise ValueError(variant.score)

    started = time.perf_counter()
    _, fit_raw_q, deploy_raw_q, solver_diagnostics = solve_binned_deployable_policy(
        torch.as_tensor(fit_scores, dtype=torch.float64),
        torch.as_tensor(deploy_scores, dtype=torch.float64),
        torch.as_tensor(fit_lengths),
        target_budget,
        None if objective_weights is None else torch.as_tensor(objective_weights),
        variant.bins,
        objective_masses=(
            None if objective_masses is None else torch.as_tensor(objective_masses)
        ),
    )
    fit_q, deploy_q_tensor, correction = (
        correct_projected_cumulative_probabilities_to_budget(
            fit_raw_q.cumprod(dim=1),
            deploy_raw_q.cumprod(dim=1),
            torch.as_tensor(fit_lengths),
            torch.as_tensor(prior[fit_idx]),
            torch.as_tensor(prior[deploy_idx]),
            target_budget,
            terminal_pi_min=0.005,
        )
    )
    fit_rho = fit_q.cumprod(dim=1).numpy()
    deploy_q = deploy_q_tensor.numpy()
    deploy_rho = np.cumprod(deploy_q, axis=1)
    deploy_active = np.arange(WIDTH)[None, :] < deploy_lengths[:, None]
    fit_pi = fit_rho[np.arange(n1), fit_lengths - 1]
    deploy_pi = deploy_rho[np.arange(len(deploy_idx)), deploy_lengths - 1]
    expected_deploy_cost = float((deploy_rho * deploy_active).sum())
    expected_total_cost = phase1_cost + expected_deploy_cost
    diagnostics = {
        **extra_diagnostics,
        **solver_diagnostics,
        **correction,
        **bin_stats(fit_scores, fit_lengths, fit_raw_q, variant.bins),
        "fit_runtime_seconds": float(time.perf_counter() - started),
        "target_phase2_budget": float(target_budget),
        "anchor_shrink": float(anchor_shrink),
    }
    return FittedPolicy(
        variant=variant_name,
        calibration_idx=calibration_idx,
        test_idx=test_idx,
        fit_local=fit_local,
        deploy_local=deploy_local,
        fit_lengths=fit_lengths,
        deploy_lengths=deploy_lengths,
        fit_pi=fit_pi,
        deploy_q=deploy_q,
        deploy_pi=deploy_pi,
        deploy_active=deploy_active,
        phase1_cost=phase1_cost,
        expected_total_cost=expected_total_cost,
        diagnostics=diagnostics,
    )


def acquire_and_select(
    *,
    fitted: FittedPolicy,
    acquisition_seed: int,
    times: np.ndarray,
    quantiles: np.ndarray,
    prior: np.ndarray,
    taus: np.ndarray,
) -> dict[str, float | int]:
    cal_idx = fitted.calibration_idx
    fit_idx = cal_idx[fitted.fit_local]
    deploy_idx = cal_idx[fitted.deploy_local]
    uniforms = np.random.default_rng(acquisition_seed).random(
        (len(times), WIDTH)
    )[deploy_idx]
    sequential_keep = np.logical_and.accumulate(
        uniforms < fitted.deploy_q,
        axis=1,
    )
    acquired = sequential_keep & fitted.deploy_active
    endpoint = sequential_keep[
        np.arange(len(deploy_idx)), fitted.deploy_lengths - 1
    ]

    fit_events = times[fit_idx, None] < quantiles[fit_idx]
    deploy_events = times[deploy_idx, None] < quantiles[deploy_idx]
    weighted = endpoint.astype(np.float64) / fitted.deploy_pi
    rates = (
        fit_events.sum(axis=0) + weighted @ deploy_events.astype(np.float64)
    ) / len(cal_idx)
    selected = strict_select(rates)

    all_events = times[cal_idx, None] < quantiles[cal_idx]
    oracle_rates = all_events.mean(axis=0)
    oracle_selected = strict_select(oracle_rates)
    test_events = times[fitted.test_idx, None] < quantiles[fitted.test_idx]
    test_coverage = 1.0 - test_events.mean(axis=0)
    test_size = quantiles[fitted.test_idx].mean(axis=0)

    return {
        "acquisition_seed": int(acquisition_seed),
        "selected_index": int(selected),
        "selected_tau": float(taus[selected]),
        "estimated_miscoverage": float(rates[selected]),
        "coverage_pct": float(100 * test_coverage[selected]),
        "selected_size": float(test_size[selected]),
        "oracle_selected_index": int(oracle_selected),
        "oracle_selected_tau": float(taus[oracle_selected]),
        "oracle_coverage_pct": float(100 * test_coverage[oracle_selected]),
        "oracle_selected_size": float(test_size[oracle_selected]),
        "switched_from_oracle": int(selected != oracle_selected),
        "selected_index_minus_oracle": int(selected - oracle_selected),
        "coverage_minus_oracle_pp": float(
            100 * (test_coverage[selected] - test_coverage[oracle_selected])
        ),
        "realized_cost_per_sample": float(
            (fitted.phase1_cost + acquired.sum()) / len(cal_idx)
        ),
    }


def nonlinear_selector_moments(
    *,
    fitted: FittedPolicy,
    times: np.ndarray,
    quantiles: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, float | int]:
    """Monte Carlo acquisition variance of the complete nonlinear selector.

    Endpoint observation is exactly Bernoulli with ``deploy_pi``.  Sampling
    that endpoint coin directly is distributionally identical to replaying all
    sequential continuation coins and avoids a needless 2950x200 array per
    draw.  Nested candidate events are represented by their first true index,
    reducing each draw from a dense 2950x1257 matrix product to one histogram.
    """
    cal_idx = fitted.calibration_idx
    fit_idx = cal_idx[fitted.fit_local]
    deploy_idx = cal_idx[fitted.deploy_local]
    fit_events = times[fit_idx, None] < quantiles[fit_idx]
    deploy_events = times[deploy_idx, None] < quantiles[deploy_idx]
    fit_threshold = _thresholds(fit_events)
    deploy_threshold = _thresholds(deploy_events)
    candidates = quantiles.shape[1]
    fit_curve = _curve_from_thresholds(
        fit_threshold,
        np.ones(len(fit_idx)),
        candidates,
    )
    test_events = times[fitted.test_idx, None] < quantiles[fitted.test_idx]
    coverage_curve = 100 * (1.0 - test_events.mean(axis=0))
    rng = np.random.default_rng(seed)
    outputs = np.empty(draws, dtype=np.float64)
    selected_indices = np.empty(draws, dtype=np.int64)
    inverse_pi = 1.0 / fitted.deploy_pi
    for draw in range(draws):
        included = rng.random(len(deploy_idx)) < fitted.deploy_pi
        deploy_curve = _curve_from_thresholds(
            deploy_threshold,
            included.astype(np.float64) * inverse_pi,
            candidates,
        )
        rates = (fit_curve + deploy_curve) / len(cal_idx)
        selected = _strict_select_monotone(rates)
        selected_indices[draw] = selected
        outputs[draw] = coverage_curve[selected]
    oracle_rates = (
        fit_events.sum(axis=0) + deploy_events.sum(axis=0)
    ) / len(cal_idx)
    oracle_selected = _strict_select_monotone(oracle_rates)
    return {
        "nonlinear_draws": int(draws),
        "nonlinear_mean_coverage_pct": float(outputs.mean()),
        "nonlinear_acquisition_variance_pp2": float(outputs.var(ddof=1)),
        "nonlinear_switch_probability": float(
            np.mean(selected_indices != oracle_selected)
        ),
        "nonlinear_selected_index_mean": float(selected_indices.mean()),
        "nonlinear_selected_index_sd": float(selected_indices.std(ddof=1)),
        "nonlinear_oracle_index": int(oracle_selected),
        "nonlinear_oracle_coverage_pct": float(coverage_curve[oracle_selected]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", choices=SETUPS, required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--n1", type=int, default=50)
    parser.add_argument("--budget", type=float, default=20.0)
    parser.add_argument("--projection-margin", type=float, default=1.0)
    parser.add_argument("--tau-multiplier", type=float, default=1.0)
    parser.add_argument("--regularization", type=float, default=0.001)
    parser.add_argument("--candidate-window-multiplier", type=float, default=float("inf"))
    parser.add_argument("--weight-clip-quantile", type=float, default=1.0)
    parser.add_argument("--anchor-shrink", type=float, default=0.0)
    parser.add_argument("--influence-bandwidth-multiplier", type=float, default=1.0)
    parser.add_argument("--influence-winsor-quantile", type=float, default=0.95)
    parser.add_argument("--influence-anchor-shrink", type=float, default=0.25)
    parser.add_argument("--pivotal-draws", type=int, default=1000)
    parser.add_argument("--nonlinear-draws", type=int, default=0)
    parser.add_argument("--nonlinear-seeds", default="0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.n1 <= 0 or args.n1 >= 3000:
        raise ValueError("N1 must lie strictly between 0 and 3000.")
    if args.tau_multiplier <= 0:
        raise ValueError("Tau multiplier must be positive.")
    if args.candidate_window_multiplier <= 0:
        raise ValueError("Candidate-window multiplier must be positive.")
    if not 0 < args.weight_clip_quantile <= 1:
        raise ValueError("Weight-clip quantile must lie in (0,1].")
    if not 0 <= args.anchor_shrink <= 1:
        raise ValueError("Anchor shrink must lie in [0,1].")
    if args.influence_bandwidth_multiplier <= 0:
        raise ValueError("Influence bandwidth multiplier must be positive.")
    if not 0 < args.influence_winsor_quantile <= 1:
        raise ValueError("Influence winsor quantile must lie in (0,1].")
    if not 0 <= args.influence_anchor_shrink <= 1:
        raise ValueError("Influence anchor shrink must lie in [0,1].")
    variants = args.variants.split(",")
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")

    grid, times_tensor, dataset, setup = load_setup(args.setup)
    times = times_tensor.numpy().astype(np.int64)
    taus_tensor = make_lpb_tau_grid(device="cpu")
    taus = taus_tensor.numpy()
    quantiles_tensor = lpb_quantiles(grid, taus_tensor, WIDTH)
    quantiles = quantiles_tensor.numpy().astype(np.int64)
    prior = get_prior(quantiles_tensor, taus_tensor, 0.56).numpy().astype(np.int64)
    anchor_index = strict_select(taus, target=ALPHA)
    anchor_horizons = quantiles[:, anchor_index]
    step = torch.arange(WIDTH)
    hazard = grid[:, step, step].to(torch.float64).numpy()
    anchor_value = target_value_scores(
        grid,
        quantiles_tensor[:, anchor_index],
        WIDTH,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    seeds = [int(value) for value in args.seeds.split(",")]
    nonlinear_seeds = {
        int(value) for value in args.nonlinear_seeds.split(",") if value
    }
    rows: list[dict] = []
    nonlinear_rows: list[dict] = []
    selection_score_cache: dict[tuple[int, int, float, float], np.ndarray] = {}
    for seed in seeds:
        for variant_name in variants:
            print(json.dumps({
                "setup": args.setup,
                "seed": seed,
                "variant": variant_name,
            }), flush=True)
            fitted = fit_variant(
                variant_name=variant_name,
                data_seed=seed,
                policy_seed=seed,
                times=times,
                quantiles=quantiles,
                prior=prior,
                grid=grid,
                hazard=hazard,
                anchor_value=anchor_value,
                anchor_horizons=anchor_horizons,
                budget=args.budget,
                n1=args.n1,
                projection_margin=args.projection_margin,
                tau_multiplier=args.tau_multiplier,
                regularization=args.regularization,
                pivotal_draws=args.pivotal_draws,
                candidate_window_multiplier=args.candidate_window_multiplier,
                weight_clip_quantile=args.weight_clip_quantile,
                anchor_shrink=args.anchor_shrink,
                influence_bandwidth_multiplier=(
                    args.influence_bandwidth_multiplier
                ),
                influence_winsor_quantile=args.influence_winsor_quantile,
                influence_anchor_shrink=args.influence_anchor_shrink,
                selection_score_cache=selection_score_cache,
            )
            evaluation = acquire_and_select(
                fitted=fitted,
                acquisition_seed=seed,
                times=times,
                quantiles=quantiles,
                prior=prior,
                taus=taus,
            )
            row = {
                "setup_key": args.setup,
                "dataset": dataset,
                "data_setup": setup,
                "seed": seed,
                "variant": variant_name,
                "objective": VARIANTS[variant_name].objective,
                "score": VARIANTS[variant_name].score,
                "bins": VARIANTS[variant_name].bins,
                "n1": args.n1,
                "budget": args.budget,
                "projection_margin": args.projection_margin,
                "expected_cost_per_sample": (
                    fitted.expected_total_cost / len(fitted.calibration_idx)
                ),
                **evaluation,
                **fitted.diagnostics,
            }
            rows.append(row)
            pd.DataFrame(rows).to_csv(args.output, index=False)

            if args.nonlinear_draws > 0 and seed in nonlinear_seeds:
                moments = nonlinear_selector_moments(
                    fitted=fitted,
                    times=times,
                    quantiles=quantiles,
                    draws=args.nonlinear_draws,
                    seed=seed + 900_001,
                )
                nonlinear_rows.append({
                    "setup_key": args.setup,
                    "seed": seed,
                    "variant": variant_name,
                    "expected_cost_per_sample": (
                        fitted.expected_total_cost / len(fitted.calibration_idx)
                    ),
                    **moments,
                })
                pd.DataFrame(nonlinear_rows).to_csv(
                    args.output.with_name(args.output.stem + "_nonlinear.csv"),
                    index=False,
                )


if __name__ == "__main__":
    main()
