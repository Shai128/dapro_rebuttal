import itertools

import numpy as np

from analysis.diagnostics.metric_augmented_ht_audit import (
    _conditionals_from_cumulative,
    _sequential_augmented_estimate_and_variance,
)


def test_sequential_augmented_ht_exact_enumeration():
    cumulative = np.array([[0.8, 0.4, 0.2]], dtype=np.float64)
    prediction_before = np.array([[0.25, 0.50, 0.70]], dtype=np.float64)
    target = np.array([1.0], dtype=np.float64)
    lengths = np.array([3], dtype=np.int64)
    conditionals = _conditionals_from_cumulative(cumulative)[0]

    outcomes = []
    for decisions in itertools.product([0, 1], repeat=3):
        reached = np.cumprod(np.asarray(decisions, dtype=np.int8))
        # Infeasible histories after the first stop have zero probability.
        if any(decisions[index] for index in range(1, 3) if not reached[index - 1]):
            continue
        probability = 1.0
        active = True
        for decision, conditional in zip(decisions, conditionals):
            if not active:
                break
            probability *= conditional if decision else (1.0 - conditional)
            active = bool(decision)
        estimate, exact_variance = _sequential_augmented_estimate_and_variance(
            target=target,
            lengths=lengths,
            prediction_before=prediction_before,
            cumulative=cumulative,
            acquired=reached.reshape(1, -1),
        )
        outcomes.append((probability, estimate, exact_variance))

    probabilities = np.asarray([row[0] for row in outcomes])
    estimates = np.asarray([row[1] for row in outcomes])
    assert np.isclose(probabilities.sum(), 1.0)
    assert np.isclose(np.sum(probabilities * estimates), target[0])
    enumerated_variance = np.sum(
        probabilities * (estimates - target[0]) ** 2
    )
    assert np.isclose(enumerated_variance, outcomes[0][2])


def test_sequential_augmented_ht_reduces_to_terminal_aht_without_updates():
    cumulative = np.array([[0.9, 0.6, 0.3]], dtype=np.float64)
    prediction_before = np.array([[0.4, 0.4, 0.4]], dtype=np.float64)
    target = np.array([1.0], dtype=np.float64)
    lengths = np.array([3], dtype=np.int64)
    acquired = np.ones((1, 3), dtype=np.int8)

    estimate, variance = _sequential_augmented_estimate_and_variance(
        target=target,
        lengths=lengths,
        prediction_before=prediction_before,
        cumulative=cumulative,
        acquired=acquired,
    )
    expected_estimate = 0.4 + (1.0 - 0.4) / 0.3
    expected_variance = (1.0 - 0.4) ** 2 * (1.0 / 0.3 - 1.0)
    assert np.isclose(estimate, expected_estimate)
    assert np.isclose(variance, expected_variance)
