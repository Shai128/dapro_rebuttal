"""Tests for the isolated real-data Basis-DAPRO diagnostic."""

import itertools

import numpy as np

from src.predictive_bounds.experiments.basis_dapro_metric_diagnostic import (
    exact_sequential_aht_variance,
    prefix_metric_predictions_and_scores,
    sequential_aht_components,
)


def test_sequential_components_force_the_terminal_metric_update():
    predictions = np.array([
        [0.2, 0.7],
        [0.3, 0.4],
    ])
    event_times = np.array([2, 3])

    increment_mass, residual_mass, increments, active = (
        sequential_aht_components(predictions, event_times, horizon=2)
    )

    np.testing.assert_array_equal(active, np.ones((2, 2), dtype=bool))
    # Row 1 terminates inside the horizon (A=1); row 2 is known to have
    # survived the horizon (A=0).  In both rows the last update is exact.
    np.testing.assert_allclose(increments, [[0.5, 0.3], [0.1, -0.4]])
    np.testing.assert_allclose(increment_mass, np.square(increments))
    np.testing.assert_allclose(
        residual_mass,
        [[0.8**2, 0.3**2], [0.3**2, 0.4**2]],
    )


def test_exact_sequential_variance_matches_enumeration_of_nested_reach():
    predictions = np.array([[0.2, 0.7]])
    event_times = np.array([2])
    _, residual_mass, increments, active = sequential_aht_components(
        predictions,
        event_times,
        horizon=2,
    )
    conditionals = np.array([[0.6, 0.5]])
    reach = np.cumprod(conditionals, axis=1)
    closed_form = exact_sequential_aht_variance(
        residual_mass,
        active,
        reach,
        total_sample_count=1,
    )

    enumerated = 0.0
    for first, second in itertools.product([0, 1], repeat=2):
        if second and not first:
            continue
        probability = (
            (0.4 if not first else 1.0)
            if not first
            else (0.6 * (0.5 if second else 0.5))
        )
        estimate = predictions[0, 0]
        if first:
            estimate += increments[0, 0] / reach[0, 0]
        if second:
            estimate += increments[0, 1] / reach[0, 1]
        enumerated += probability * (estimate - 1.0) ** 2

    np.testing.assert_allclose(closed_form, enumerated, atol=1e-14)
    np.testing.assert_allclose(
        closed_form,
        0.8**2 * (1 / 0.6 - 1) + 0.3**2 * (1 / 0.3 - 1 / 0.6),
    )


def test_prefix_scores_are_causal_and_match_manual_two_turn_values():
    # outcome index 0/1 denotes termination at the next/current indexed turn;
    # index 2 is the survival tail beyond the metric horizon.
    grid = np.array([[
        [0.25, 0.25, 0.50],
        [0.00, 0.40, 0.60],
    ]], dtype=np.float32)
    hazard, prediction, value, information = (
        prefix_metric_predictions_and_scores(
            __import__("torch").as_tensor(grid),
            horizon=2,
            chunk_size=1,
        )
    )

    np.testing.assert_allclose(hazard, [[0.25, 0.40]])
    np.testing.assert_allclose(prediction, [[0.50, 0.40]])
    # At t=0 the expected remaining acquisition cost is
    # .25*1 + .25*2 + .50*2 = 1.75; at t=1 it is one.
    np.testing.assert_allclose(value, [[np.sqrt(0.5 / 1.75), np.sqrt(0.4)]])
    np.testing.assert_allclose(
        information,
        [[np.sqrt(0.5 * 0.5 / 1.75), np.sqrt(0.4 * 0.6)]],
    )

    changed = grid.copy()
    changed[:, 1, :] = [0.0, 0.9, 0.1]
    changed_result = prefix_metric_predictions_and_scores(
        __import__("torch").as_tensor(changed),
        horizon=2,
        chunk_size=1,
    )
    for original, modified in zip(
            (hazard, prediction, value, information), changed_result):
        np.testing.assert_array_equal(original[:, 0], modified[:, 0])
