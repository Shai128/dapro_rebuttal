import numpy as np
import pytest

from src.predictive_bounds.budget_allocators.risk_controlled_budget import (
    affine_cumulative_policy_family,
    cumulative_policy_costs,
    row_local_horizon_budget_cap,
    select_crc_budget_candidate,
    select_hoeffding_budget_candidate,
    solve_constant_continuation_policy,
)


def test_affine_family_is_nested_and_preserves_base_shape():
    base = np.array([0.9, 0.6, 0.3])
    parameters = np.array([1.0, 0.5, 0.0, -0.5, -1.0])

    family = affine_cumulative_policy_family(
        base,
        parameters,
        terminal_pi_min=0.05,
    )

    np.testing.assert_allclose(family[0], 1.0)
    np.testing.assert_allclose(family[2], base)
    np.testing.assert_allclose(family[-1], 0.05)
    assert np.all(np.diff(family, axis=0) <= 1e-12)
    assert np.all(np.diff(family, axis=1) <= 1e-12)


def test_cumulative_policy_costs_use_strict_count_lengths():
    candidates = np.array(
        [
            [1.0, 0.8, 0.4],
            [0.5, 0.3, 0.1],
        ]
    )
    lengths = np.array([0, 1, 3])

    costs = cumulative_policy_costs(candidates, lengths)

    np.testing.assert_allclose(
        costs,
        [
            [0.0, 0.0],
            [1.0, 0.5],
            [2.2, 0.9],
        ],
    )


def test_constant_continuation_policy_meets_cost_with_hard_floor():
    lengths = np.array([1, 2, 3, 4])
    cumulative, probability, achieved_cost = (
        solve_constant_continuation_policy(
            lengths,
            budget_per_sample=1.5,
            time_width=4,
            terminal_pi_min=0.05,
        )
    )

    assert 0 < probability < 1
    assert np.all(np.diff(cumulative) <= 1e-12)
    assert np.all(cumulative >= 0.05)
    expected = cumulative_policy_costs(
        cumulative[None, :],
        lengths,
    )[:, 0].mean()
    np.testing.assert_allclose(achieved_cost, expected)
    np.testing.assert_allclose(achieved_cost, 1.5, atol=1e-8)


def test_constant_continuation_policy_checks_floor_feasibility():
    with pytest.raises(ValueError, match="terminal floor"):
        solve_constant_continuation_policy(
            np.array([4, 4]),
            budget_per_sample=1.5,
            time_width=4,
            terminal_pi_min=0.5,
        )


def test_crc_selects_first_feasible_nested_candidate():
    costs = np.array(
        [
            [3.0, 2.0, 0.2],
            [3.0, 1.8, 0.2],
            [2.5, 1.9, 0.2],
        ]
    )
    pilot = np.array([2.0, 1.0, 3.0])
    # CRC controls c + (n/m)b against B_remaining/m = 4.5.
    result = select_crc_budget_candidate(
        costs,
        pilot,
        total_budget_after_policy_fit=22.5,
        deployment_sample_count=5,
        maximum_cost_per_sample=4.0,
    )

    # rho=3/5 and envelope=(1 + rho)*4=6.4. Candidate 0 has
    # selector 4.625; candidate 1 has selector 3.925 and is first feasible.
    assert result.selected_index == 1
    np.testing.assert_allclose(result.selector_left_side_per_sample, 3.925)
    np.testing.assert_allclose(result.deployment_budget_per_sample, 4.5)
    assert result.guarantee_kind == "crc_marginal_expected_total_budget"


def test_crc_uses_separate_candidate_and_pilot_envelopes():
    costs = np.tile([2.0, 1.0], (10, 1))
    pilot = np.full(10, 20.0)
    shared = select_crc_budget_candidate(
        costs,
        pilot,
        total_budget_after_policy_fit=320.0,
        deployment_sample_count=10,
        maximum_cost_per_sample=20.0,
    )
    separate = select_crc_budget_candidate(
        costs,
        pilot,
        total_budget_after_policy_fit=320.0,
        deployment_sample_count=10,
        maximum_cost_per_sample=20.0,
        maximum_candidate_cost_per_sample=2.0,
        maximum_pilot_cost_per_sample=20.0,
    )

    assert separate.correction_per_sample < shared.correction_per_sample
    assert separate.selected_index <= shared.selected_index


def test_crc_exact_enumeration_controls_expected_total_cost():
    # Two equiprobable row types, with correlated pilot and candidate costs.
    # Enumerate every ordered control sample so the expectation is exact.
    candidate_cost_by_type = np.array([
        [1.0, 0.2],
        [2.0, 0.4],
    ])
    pilot_cost_by_type = np.array([0.5, 1.5])
    control_count = 2
    deployment_count = 4
    remaining_budget = 9.2
    population_candidate_means = candidate_cost_by_type.mean(axis=0)

    expected_total = 0.0
    for first_type in range(2):
        for second_type in range(2):
            types = np.array([first_type, second_type])
            costs = candidate_cost_by_type[types]
            pilot = pilot_cost_by_type[types]
            selected = select_crc_budget_candidate(
                costs,
                pilot,
                total_budget_after_policy_fit=remaining_budget,
                deployment_sample_count=deployment_count,
                maximum_cost_per_sample=2.0,
                maximum_candidate_cost_per_sample=2.0,
                maximum_pilot_cost_per_sample=1.5,
            )
            total = (
                pilot.sum()
                + deployment_count
                * population_candidate_means[selected.selected_index]
            )
            expected_total += 0.25 * total

    assert expected_total <= remaining_budget + 1e-12


def test_crc_requires_policy_family_fixed_and_nested_in_declared_order():
    costs = np.array(
        [
            [0.2, 0.3],
            [0.4, 0.1],
        ]
    )
    with pytest.raises(ValueError, match="most aggressive"):
        select_crc_budget_candidate(
            costs,
            np.zeros(2),
            total_budget_after_policy_fit=10,
            deployment_sample_count=5,
            maximum_cost_per_sample=1,
        )


def test_crc_fails_loudly_when_even_safest_candidate_is_infeasible():
    with pytest.raises(ValueError, match="No candidate policy"):
        select_crc_budget_candidate(
            np.ones((2, 1)),
            np.full(2, 2.0),
            total_budget_after_policy_fit=1.0,
            deployment_sample_count=5,
            maximum_cost_per_sample=2.0,
        )


def test_hoeffding_selector_controls_all_fixed_candidates_simultaneously():
    n = 200
    costs = np.tile([0.8, 0.5, 0.1], (n, 1))
    pilot = np.zeros(n)
    result = select_hoeffding_budget_candidate(
        costs,
        pilot,
        total_budget_after_policy_fit=65.0,
        deployment_sample_count=100,
        maximum_cost_per_sample=1.0,
        delta=0.1,
    )

    # radius=sqrt(log(30)/(400)) ~= .0922. Candidate 0 is above .65,
    # while candidate 1 is below it.
    assert result.selected_index == 1
    assert result.selector_left_side_per_sample <= 0.65
    assert result.guarantee_kind == (
        "hoeffding_high_probability_deployment_budget"
    )


def test_row_local_cap_is_label_free_and_bounds_every_event_length():
    cumulative = np.tile([0.9, 0.7, 0.4, 0.2], (2, 1))
    horizons = np.array([4, 3])

    result = row_local_horizon_budget_cap(
        cumulative,
        horizons,
        budget_per_sample=1.5,
        terminal_pi_min=0.05,
    )

    np.testing.assert_allclose(result.capped_horizon_costs, 1.5)
    assert np.all(result.mixture_coefficients < 1)
    assert np.all(
        result.cumulative_probabilities[:, :2]
        >= result.cumulative_probabilities[:, 1:3] - 1e-12
    )

    # These event-derived lengths are used only to verify the guarantee.  They
    # are not inputs to the cap, so changing them cannot change the policy.
    for event_lengths in (np.array([1, 2]), np.array([4, 3])):
        actual_costs = np.array([
            result.cumulative_probabilities[row, :event_length].sum()
            for row, event_length in enumerate(event_lengths)
        ])
        assert np.all(actual_costs <= 1.5 + 1e-10)


def test_row_local_cap_leaves_under_budget_rows_unchanged():
    cumulative = np.array([[0.5, 0.2, 0.1]])
    result = row_local_horizon_budget_cap(
        cumulative,
        np.array([3]),
        budget_per_sample=1.0,
        terminal_pi_min=0.05,
    )

    np.testing.assert_allclose(
        result.cumulative_probabilities,
        cumulative,
    )
    np.testing.assert_allclose(result.mixture_coefficients, 1.0)


def test_row_local_cap_rejects_floor_budget_incompatibility():
    with pytest.raises(ValueError, match="terminal floor"):
        row_local_horizon_budget_cap(
            np.array([[0.8, 0.7, 0.6, 0.5]]),
            np.array([4]),
            budget_per_sample=1.5,
            terminal_pi_min=0.5,
        )
