"""Convexity and deployability tests for the isolated Basis-DAPRO prototype."""

import numpy as np
import torch

from src.predictive_bounds.budget_allocators.DAPRO import (
    _apply_shared_cumulative_row_cost_envelope,
)
from src.predictive_bounds.budget_allocators.optimization_solver_utils import (
    solve_binned_deployable_policy,
)
from src.predictive_bounds.experiments.basis_dapro_prototype import (
    FrozenScoreBasis,
    apply_shared_envelope_crc,
    basis_problem_complexity,
    causal_target_value_score,
    fit_basis_dapro,
)


def _variance_proxy(masses, conditionals):
    reach = np.cumprod(conditionals, axis=1)
    return float(np.mean(np.sum(masses * (1.0 / reach - 1.0), axis=1)))


def test_step_k2_one_hot_time_is_existing_binned_policy_class():
    scores = np.array([
        [0.1, 0.2],
        [0.2, 0.5],
        [0.3, 0.1],
        [0.4, 0.7],
        [0.6, 0.4],
        [0.7, 0.8],
        [0.8, 0.3],
        [0.9, 0.9],
    ])
    lengths = np.full(len(scores), 2)
    masses = np.array([
        [0.2, 0.3],
        [0.5, 0.1],
        [0.4, 0.7],
        [0.3, 0.2],
        [0.8, 0.4],
        [0.6, 0.5],
        [0.1, 0.9],
        [0.7, 0.6],
    ])
    budget = 1.1
    model = fit_basis_dapro(
        scores,
        lengths,
        masses,
        np.ones_like(masses),
        budget,
        score_basis_kind="step_bins",
        score_basis_size=2,
        time_basis_size=2,
        terminal_reach_floor=None,
    )
    _, existing_fit, _, _ = solve_binned_deployable_policy(
        torch.as_tensor(scores),
        torch.as_tensor(scores),
        torch.as_tensor(lengths),
        budget,
        objective_weights=None,
        n_bins=2,
        objective_masses=masses,
    )

    prototype_fit = model.conditionals(scores)
    np.testing.assert_allclose(
        prototype_fit,
        existing_fit.numpy(),
        atol=2e-7,
        rtol=2e-7,
    )
    np.testing.assert_allclose(
        model.objective_value,
        _variance_proxy(masses, existing_fit.numpy()),
        atol=2e-7,
        rtol=2e-7,
    )
    assert model.parameter_count == 4
    assert model.expected_cost <= budget + 1e-8


def test_one_turn_convex_solution_beats_exhaustive_probability_grid():
    scores = np.array([[0.1], [0.2], [0.8], [0.9]])
    lengths = np.ones(4, dtype=np.int64)
    masses = np.array([[0.2], [0.6], [0.9], [1.3]])
    costs = np.ones_like(masses)
    budget = 0.62
    model = fit_basis_dapro(
        scores,
        lengths,
        masses,
        costs,
        budget,
        score_basis_kind="step_bins",
        score_basis_size=2,
        time_basis_size=1,
        terminal_reach_floor=None,
    )

    # Brute force the same convex K=2 class directly in probability space.
    grid = np.linspace(0.01, 1.0, 991)
    best = np.inf
    for low in grid:
        feasible_high = grid[(grid >= low) & ((low + grid) / 2 <= budget)]
        if len(feasible_high) == 0:
            continue
        objectives = 0.5 * (
            masses[:2].mean() * (1.0 / low - 1.0)
            + masses[2:].mean() * (1.0 / feasible_high - 1.0)
        )
        best = min(best, float(objectives.min()))

    assert model.expected_cost <= budget + 1e-8
    assert model.objective_value <= best + 2e-3


def test_continuous_rank_basis_is_causal_monotone_positive_and_crc_ready():
    rng = np.random.default_rng(12)
    n, width = 48, 5
    scores = rng.normal(size=(n, width))
    lengths = np.full(n, width)
    masses = 0.05 + rng.random((n, width))
    model = fit_basis_dapro(
        scores,
        lengths,
        masses,
        np.ones_like(masses),
        budget_per_sample=2.5,
        score_basis_kind="linear_rank",
        score_basis_size=4,
        time_basis_size=3,
        terminal_reach_floor=0.02,
    )

    ordered = np.repeat(
        np.linspace(scores.min() - 1, scores.max() + 1, 21)[:, None],
        width,
        axis=1,
    )
    ordered_probabilities = model.conditionals(ordered)
    assert np.all(np.diff(ordered_probabilities, axis=0) >= -2e-8)
    assert np.all(model.cumulative_reach(ordered) >= 0.02 - 2e-7)

    # Future-score changes cannot alter earlier continuation decisions.
    paired = np.zeros((2, width))
    paired[1, 3:] = 100.0
    paired_probabilities = model.conditionals(paired)
    np.testing.assert_allclose(
        paired_probabilities[0, :3],
        paired_probabilities[1, :3],
        atol=0,
        rtol=0,
    )

    base = model.conditionals(ordered[:4])
    envelope = np.array([0.80, 0.60, 0.45, 0.35, 0.30])
    selected = []
    for alpha in [1.0, 0.5, 0.0]:
        conditionals = apply_shared_envelope_crc(
            base,
            np.full(4, width),
            envelope,
            alpha=alpha,
            terminal_reach_floor=0.02,
        )
        cumulative = np.cumprod(conditionals, axis=1)
        assert np.all(cumulative >= 0.02 - 1e-12)
        assert np.all(cumulative.sum(axis=1) <= envelope.sum() + 1e-12)
        selected.append(cumulative)
    assert np.all(selected[1] <= selected[0] + 1e-12)
    assert np.all(selected[2] <= selected[1] + 1e-12)

    production_cap, _ = _apply_shared_cumulative_row_cost_envelope(
        torch.as_tensor(base),
        torch.full((4,), width),
        torch.as_tensor(envelope),
    )
    np.testing.assert_allclose(
        np.cumprod(production_cap.numpy(), axis=1),
        selected[0],
        atol=1e-12,
        rtol=1e-12,
    )


def test_value_score_uses_only_current_prefix_and_supports_both_targets():
    n, width = 2, 4
    grid = np.zeros((n, width, width + 1), dtype=np.float64)
    for row in range(n):
        for step in range(width):
            grid[row, step, step] = 0.2 + 0.05 * row
            grid[row, step, -1] = 0.8 - 0.05 * row
    metric = causal_target_value_score(
        grid,
        acquisition_horizons=width,
        target_horizons=width,
        strict=False,
    )
    lpb = causal_target_value_score(
        grid,
        acquisition_horizons=np.array([4, 4]),
        target_horizons=np.array([3, 2]),
        strict=True,
    )
    assert metric.shape == lpb.shape == (n, width)
    assert np.all(metric >= 0) and np.all(lpb >= 0)

    changed = grid.copy()
    changed[:, 2:, :] = np.roll(changed[:, 2:, :], 1, axis=2)
    changed_metric = causal_target_value_score(
        changed,
        acquisition_horizons=width,
        target_horizons=width,
        strict=False,
    )
    np.testing.assert_allclose(changed_metric[:, :2], metric[:, :2])


def test_frozen_lookup_is_deterministic_and_complexity_is_low_dimensional():
    scores = np.array([
        [0.0, 0.2, 0.4],
        [0.0, 0.5, 0.7],
        [1.0, 0.8, 0.9],
        [1.0, 0.9, 1.0],
    ])
    lookup = FrozenScoreBasis.fit(
        scores,
        np.full(4, 3),
        kind="linear_rank",
        size=3,
    )
    first = lookup.transform(scores)
    second = lookup.transform(scores.copy())
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first.sum(axis=2), 1.0)

    complexity = basis_problem_complexity(200, 200, 6, 4)
    assert complexity["parameters"] == 24
    assert complexity["parameters"] < 200 * 4

