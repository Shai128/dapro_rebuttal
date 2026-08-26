import numpy as np

from analysis.diagnostics.realized_residual_crc_audit import _grid_floor


def test_candidate_grid_floor_matches_descending_linear_crc_grid():
    grid = np.linspace(1.0, 0.0, 401)
    for value in (0.0, 0.001, 0.137, 0.8626, 0.9999, 1.0):
        feasible = grid[grid <= value + 1e-12]
        expected = feasible[0] if len(feasible) else 0.0
        assert abs(_grid_floor(value, len(grid)) - expected) <= 1e-12


def test_realized_residual_selector_has_exact_finite_budget_counterexample():
    probabilities = np.array([0.6819741, 0.1070096, 0.2110163])
    pilot = np.array([0.4, 0.4, 0.8])
    # Rows are population types; columns are ordered aggressive->conservative.
    costs = np.array(
        [
            [1.0, 1.0, 0.4],
            [0.8, 0.2, 0.0],
            [1.0, 0.2, 0.0],
        ]
    )
    candidate_bound = 1.0
    remaining_budget = 1.4
    population_cost = probabilities @ costs

    expected_total = 0.0
    for control_type, probability in enumerate(probabilities):
        target = remaining_budget - pilot[control_type]
        selector_values = (costs[control_type] + candidate_bound) / 2.0
        feasible = np.flatnonzero(selector_values <= target + 1e-12)
        assert len(feasible) > 0
        selected = int(feasible[0])
        expected_total += probability * (
            pilot[control_type] + population_cost[selected]
        )

    assert abs(expected_total - 1.4138338) < 1e-6
    assert expected_total > remaining_budget
