"""Tests for the exact nonlinear LPB selector-variance diagnostic."""

import numpy as np
import pytest

from analysis.diagnostics.lpb_selection_variance import (
    analytic_margin_bounds,
    exact_selector_moments,
    monte_carlo_selector_moments,
    pivotal_variance_bounds,
    toy_problem,
)


def test_nested_crossing_formula_matches_direct_selected_output_variance():
    events, pi, alpha, coverages = toy_problem()
    exact = exact_selector_moments(events, pi, alpha, coverages)
    simulated = monte_carlo_selector_moments(
        events,
        pi,
        alpha,
        coverages,
        draws=50_000,
        seed=17,
        batch_size=10_000,
    )

    assert exact.crossing_formula_variance == pytest.approx(
        exact.direct_variance,
        abs=2e-14,
    )
    # The empirical crossing identity uses the same draws and is algebraic,
    # not an asymptotic comparison.
    assert simulated.crossing_formula_variance == pytest.approx(
        simulated.direct_variance,
        abs=2e-14,
    )
    assert simulated.direct_variance == pytest.approx(
        exact.direct_variance,
        abs=2.5e-4,
    )


def test_pivotal_and_margin_bounds_cover_exact_selector_variance():
    events, pi, alpha, coverages = toy_problem()
    exact = exact_selector_moments(events, pi, alpha, coverages)
    pivotal = pivotal_variance_bounds(events, pi, alpha, coverages)
    margin = analytic_margin_bounds(events, pi, alpha, coverages)

    assert pivotal.lower_bound <= exact.direct_variance + 2e-14
    assert exact.direct_variance <= pivotal.upper_bound + 2e-14
    assert exact.direct_variance <= margin.cantelli_bound + 2e-14
    assert margin.cantelli_bound <= margin.linear_bound + 2e-14
    assert np.all(margin.row_weights >= 0)


def test_non_nested_candidates_are_rejected():
    events, pi, alpha, coverages = toy_problem()
    invalid = events.copy()
    invalid[0] = [True, False, True]
    with pytest.raises(ValueError, match="nested"):
        exact_selector_moments(invalid, pi, alpha, coverages)


def test_zero_complete_data_margin_routes_to_pivotal_method():
    events, pi, _, coverages = toy_problem()
    boundary_alpha = float(events.mean(axis=0)[1])
    with pytest.raises(ValueError, match="nonregular"):
        analytic_margin_bounds(events, pi, boundary_alpha, coverages)
