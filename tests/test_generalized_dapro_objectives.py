import numpy as np
import torch

from src.predictive_bounds.budget_allocators.DAPRO import (
    SoftTargetCRCDAPRO,
    SoftTargetDAPRO,
)
from src.predictive_bounds.budget_allocators.dapro_objectives import (
    history_soft_objective_coefficients,
    initial_pmf_objective_coefficients,
    realized_target_weights,
)
from src.predictive_bounds.budget_allocators.optimization_solver_utils import (
    solve_exact_fast,
)


def test_terminal_weights_and_equivalent_prefix_masses_solve_identically():
    scores = torch.tensor([
        [0.1, 0.2, 0.3],
        [0.3, 0.2, 0.1],
        [0.2, 0.4, 0.5],
    ], dtype=torch.float64)
    lengths = torch.tensor([1, 2, 3])
    weights = np.array([1.0, 0.5, 0.2])
    masses = np.zeros((3, 3), dtype=np.float64)
    masses[np.arange(3), lengths.numpy() - 1] = weights

    hard = solve_exact_fast(
        scores,
        lengths,
        1.2,
        objective_weights=weights,
        verbose=False,
    )
    generalized = solve_exact_fast(
        scores,
        lengths,
        1.2,
        objective_masses=masses,
        verbose=False,
    )
    np.testing.assert_allclose(generalized, hard, atol=1e-8, rtol=1e-8)


def test_prefix_mass_solver_matches_two_turn_cumulative_optimum():
    conditionals = solve_exact_fast(
        torch.zeros((1, 2), dtype=torch.float64),
        torch.tensor([2]),
        1.0,
        objective_masses=np.array([[0.2, 0.8]]),
        verbose=False,
    )
    cumulative = np.cumprod(conditionals, axis=1)
    np.testing.assert_allclose(cumulative, [[0.5, 0.5]], atol=1e-6)


def test_general_target_coefficients_cover_hard_initial_and_history_forms():
    times = torch.tensor([2, 3, 4])
    np.testing.assert_array_equal(
        realized_target_weights(times, 3, strict=False).numpy(),
        [1.0, 1.0, 0.0],
    )
    np.testing.assert_array_equal(
        realized_target_weights(times, 3, strict=True).numpy(),
        [1.0, 0.0, 0.0],
    )

    initial = initial_pmf_objective_coefficients(
        np.array([[0.2, 0.3, 0.1]]),
        np.array([[1.0, 0.8, 0.5]]),
        2,
        strict=False,
        target_kind="fixed_horizon",
    )
    np.testing.assert_array_equal(initial.event_mass, [[0.2, 0.3, 0.0]])
    np.testing.assert_array_equal(initial.cost_mass, [[1.0, 0.8, 0.5]])

    grid = torch.zeros((2, 3, 4), dtype=torch.float64)
    grid[0, torch.arange(3), torch.arange(3)] = torch.tensor(
        [0.1, 0.2, 0.3], dtype=torch.float64
    )
    grid[1, torch.arange(3), torch.arange(3)] = torch.tensor(
        [0.4, 0.5, 0.6], dtype=torch.float64
    )
    soft = history_soft_objective_coefficients(
        grid,
        torch.tensor([2, 3]),
        torch.tensor([2, 3]),
        strict=False,
        target_kind="fixed_horizon",
    )
    np.testing.assert_allclose(
        soft.event_mass,
        [[0.1, 0.2, 0.0], [0.4, 0.5, 0.6]],
    )
    np.testing.assert_array_equal(
        soft.cost_mass,
        [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]],
    )


def test_soft_target_dapro_runs_as_a_dynamic_metric_backend():
    n, width = 80, 4
    grid = torch.zeros((n, width, width + 1), dtype=torch.float64)
    row = torch.arange(n)
    for step in range(width):
        hazard = 0.05 + 0.2 * ((row % 5) / 4)
        grid[:, step, step] = hazard
        grid[:, step, -1] = 1 - hazard
    event_times = (row % (width + 1)) + 1
    quantiles = torch.full((n, 5), width, dtype=torch.float32)
    allocator = SoftTargetDAPRO(
        grid,
        2.0,
        torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]),
        0.5,
        width,
        n1=20,
        metric_estimation_horizon=width,
        projection_budget_margin=0.2,
    )
    allocator.set_acquisition_randomness(
        seed=7,
        uniforms=np.random.default_rng(7).random((n, width)),
    )

    result = allocator.allocate_budget(
        grid,
        None,
        event_times,
        quantiles,
    )

    assert result.C.shape == (n,)
    assert torch.all(result.C_probs > 0)
    assert result.additional_metrics["generalized_dapro"] == 1
    assert (
        result.additional_metrics["generalized_dapro_policy_class"]
        == "time_score_bin_dynamic"
    )
    assert result.additional_metrics[
        "generalized_dapro_uses_current_prefix_x_it"
    ] == 1


def test_soft_target_dapro_supports_a_four_bin_policy_class():
    allocator = SoftTargetDAPRO(
        torch.full((20, 4, 5), 0.2, dtype=torch.float64),
        2.0,
        torch.tensor([0.1, 0.56]),
        0.56,
        4,
        n1=10,
        score_bin_count=4,
        metric_estimation_horizon=4,
        projection_budget_margin=0.2,
    )

    assert allocator.score_bin_count == 4
    assert "soft_prefix_bins_4_metric_horizon_4" in allocator.name


def _synthetic_history_grid(n: int, width: int) -> torch.Tensor:
    grid = torch.zeros((n, width, width + 1), dtype=torch.float64)
    row = torch.arange(n)
    for step in range(width):
        hazard = 0.03 + 0.12 * ((row % 7) / 6)
        grid[:, step, step] = hazard
        grid[:, step, -1] = 1 - hazard
    return grid


def test_soft_target_crc_runs_for_metric_and_lpb_targets():
    n, width = 1000, 20
    grid = _synthetic_history_grid(n, width)
    rows = torch.arange(n)
    event_times = (rows % (width + 1)) + 1
    quantiles = torch.full((n, 10), width, dtype=torch.float32)
    taus = torch.arange(0.1, 1.1, 0.1)

    for metric_horizon in (width, None):
        allocator = SoftTargetCRCDAPRO(
            grid,
            8.0,
            taus,
            0.5,
            width,
            n1=100,
            budget_control_size=50,
            metric_estimation_horizon=metric_horizon,
            row_cost_cap_multiplier=2.0,
        )
        allocator.set_acquisition_randomness(
            seed=7,
            uniforms=np.random.default_rng(7).random((n, width)),
        )

        result = allocator.allocate_budget(
            grid,
            None,
            event_times,
            quantiles,
        )

        assert torch.all(result.C_probs > 0)
        assert result.additional_metrics["generalized_dapro"] == 1
        assert result.additional_metrics[
            "generalized_dapro_budget_control_mode"
        ] == "crc"
        assert result.additional_metrics["risk_budget_selector_valid"] == 1
        assert result.additional_metrics[
            "expected_budget_guarantee_kind"
        ] == "crc_marginal_expected_total_budget"
        assert result.additional_metrics[
            "expected_budget_guarantee_is_marginal_finite_sample"
        ] == 1
        assert result.additional_metrics[
            "risk_budget_maximum_candidate_cost_per_sample"
        ] == 16.0
        if metric_horizon is None:
            assert "lpb_alpha" in allocator.name
        else:
            assert "metric_horizon" in allocator.name
