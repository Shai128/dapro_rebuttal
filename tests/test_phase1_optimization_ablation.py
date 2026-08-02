import numpy as np
from src.predictive_bounds.ablations.phase1_optimization import (
    MidrankCDF,
    active_lengths,
    expected_cost,
    fit_locally_adaptive_policy,
    fit_random_policy,
    fit_score_heuristic,
    fit_static_policy,
    simulate_adaptive,
    simulate_locally_adaptive,
    quantiles_to_interaction_counts,
)


def test_midrank_cdf_ties_are_deterministic():
    cdf = MidrankCDF(np.array([1.0, 1.0, 2.0, 4.0]))
    np.testing.assert_allclose(cdf(np.array([1.0, 2.0, 3.0])), [0.25, 0.625, 0.75])


def test_random_policy_hits_feasible_budget():
    lengths = np.array([1, 2, 3, 4, 4])
    result = fit_random_policy(lengths, 3, 4, 1.6, 0.005, 1e-7)
    assert result.feasible_boundary is None
    assert result.phase1_expected_cost <= 1.6 + 1e-7
    assert 1.6 - result.phase1_expected_cost <= 1e-5
    assert np.all((result.probabilities > 0) & (result.probabilities <= 1))


def test_static_policy_matches_budget_using_realized_active_lengths():
    prior = np.array([4, 4, 4, 4])
    lengths = np.array([1, 2, 3, 4])
    result = fit_static_policy(
        prior, lengths, prior, width=4, target=1.5
    )
    assert result.feasible_boundary is None
    assert abs(result.phase1_expected_cost - 1.5) <= 1e-6
    assert np.all((result.probabilities > 0) & (result.probabilities <= 1))


def test_heuristic_is_monotone_and_budgeted():
    scores = np.array(
        [
            [0.1, 0.2, 0.1],
            [0.3, 0.4, 0.2],
            [0.5, 0.6, 0.3],
            [0.7, 0.8, 0.4],
        ]
    )
    lengths = np.array([3, 3, 3, 3])
    result = fit_score_heuristic(
        scores, lengths, scores, 1.4, p_min=0.005, slope=5.0, tolerance=1e-7
    )
    assert np.all(np.diff(result.probabilities[:, 0]) >= 0)
    assert expected_cost(
        result.probabilities, lengths
    ) <= 1.4 + 1e-5


def test_locally_adaptive_policy_is_finite_and_budgeted():
    grid = np.array(
        [
            [[0.2, 0.3, 0.5], [0.0, 0.4, 0.6], [0.0, 0.0, 1.0]],
            [[0.6, 0.3, 0.1], [0.0, 0.7, 0.3], [0.0, 0.0, 1.0]],
            [[0.1, 0.2, 0.7], [0.0, 0.2, 0.8], [0.0, 0.0, 1.0]],
            [[0.4, 0.4, 0.2], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]],
        ]
    )
    event = np.array([2, 2, 1, 2])
    prior = np.array([2, 2, 2, 1])
    result = fit_locally_adaptive_policy(
        grid, event, prior, grid, event, prior, target=1.4, tolerance=1e-6
    )
    assert result.phase1_expected_cost <= 1.4 + 1e-5
    assert np.all(np.isfinite(result.probabilities))
    assert np.all((result.probabilities > 0) & (result.probabilities <= 1))


def test_locally_adaptive_simulation_keeps_partial_censoring_and_latent_weight():
    probabilities = np.full((2, 3), 0.5)
    event = np.array([3, 1])
    prior = np.array([2, 2])
    uniforms = np.array([[0.1, 0.9, 0.1], [0.1, 0.1, 0.1]])
    result = simulate_locally_adaptive(
        probabilities, event, prior, uniforms, p_min=0.005
    )
    np.testing.assert_array_equal(result["calibration_c"], [1, 2])
    np.testing.assert_allclose(result["terminal_probability"], [0.25, 0.5])


def test_common_uniforms_make_simulation_deterministic():
    p = np.full((3, 4), 0.7)
    event = np.array([2, 3, 4])
    prior = np.array([3, 3, 3])
    uniforms = np.array(
        [[0.1, 0.2, 0.9, 0.1], [0.8, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2]]
    )
    first = simulate_adaptive(p, event, prior, uniforms)
    second = simulate_adaptive(p, event, prior, uniforms)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    np.testing.assert_array_equal(active_lengths(event, prior, 4), [2, 3, 3])


def test_simulation_uses_paper_interaction_count_convention():
    probabilities = np.array(
        [[0.8, 0.7, 0.6, 0.5], [0.9, 0.4, 0.8, 0.3], [0.6, 0.6, 0.6, 0.6]]
    )
    event = np.array([1, 3, 4])
    prior = np.array([3, 2, 3])
    uniforms = np.array(
        [[0.1, 0.99, 0.99, 0.99], [0.1, 0.2, 0.1, 0.1], [0.5, 0.5, 0.5, 0.5]]
    )
    paired = simulate_adaptive(
        probabilities, event, prior, uniforms
    )
    # A first-turn event resolves after one successful acquisition. The second
    # sample resolves at its two-interaction prior; the third resolves at three.
    np.testing.assert_array_equal(paired["realized_cost"], [1, 2, 3])
    np.testing.assert_array_equal(paired["calibration_c"], [3, 2, 3])
    np.testing.assert_allclose(
        paired["terminal_probability"],
        [0.8, 0.9 * 0.4, 0.6**3],
        rtol=1e-6,
    )


def test_saved_quantile_grid_indices_are_converted_to_event_time_units():
    raw = np.array([0, 1, 199, 200])
    np.testing.assert_array_equal(
        quantiles_to_interaction_counts(raw, width=200),
        [1, 2, 200, 200],
    )
    np.testing.assert_array_equal(
        quantiles_to_interaction_counts(raw, width=200, upper_bound=20),
        [1, 2, 20, 20],
    )
