import numpy as np
import pytest

from src.predictive_bounds.budget_allocators.optimization_solver_utils import (
    _generalized_pav,
    _generalized_pav_compiled,
)


def _literal_dictionary_generalized_pav(
        alpha,
        beta,
        lower,
        ordered_scores,
):
    """Unmodified reference formerly nested in ``solve_exact_fast``."""

    def value(block):
        if block["alpha"] <= 0:
            raw = -np.inf
        else:
            raw = 0.5 * (
                np.log(block["alpha"]) - np.log(block["beta"])
            )
        return min(0.0, max(block["lower"], raw))

    blocks = []
    start = 0
    while start < len(alpha):
        stop = start + 1
        while (
                stop < len(alpha)
                and ordered_scores[stop] == ordered_scores[start]
        ):
            stop += 1
        block = {
            "start": start,
            "stop": stop,
            "alpha": float(alpha[start:stop].sum()),
            "beta": float(beta[start:stop].sum()),
            "lower": float(lower[start:stop].max()),
        }
        block["value"] = value(block)
        blocks.append(block)
        while (
                len(blocks) > 1
                and blocks[-2]["value"] > blocks[-1]["value"] + 1e-12
        ):
            right = blocks.pop()
            left = blocks.pop()
            merged = {
                "start": left["start"],
                "stop": right["stop"],
                "alpha": left["alpha"] + right["alpha"],
                "beta": left["beta"] + right["beta"],
                "lower": max(left["lower"], right["lower"]),
            }
            merged["value"] = value(merged)
            blocks.append(merged)
        start = stop
    result = np.empty(len(alpha), dtype=np.float64)
    for block in blocks:
        result[block["start"]:block["stop"]] = block["value"]
    return result


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("size", [0, 1, 2, 7, 31, 100, 129, 257, 400])
@pytest.mark.parametrize("score_kind", ["unique", "ties"])
def test_numeric_stack_generalized_pav_is_bitwise_reference_equivalent(
        seed,
        size,
        score_kind,
):
    rng = np.random.default_rng(seed)
    alpha = np.exp(rng.uniform(-20, 20, size=size))
    if size:
        alpha[rng.random(size) < 0.25] = 0.0
    beta = np.exp(rng.uniform(-20, 20, size=size))
    lower = rng.uniform(-700, -0.01, size=size)
    if score_kind == "unique":
        ordered_scores = np.sort(rng.normal(size=size))
    else:
        ordered_scores = np.sort(rng.integers(0, 5, size=size))

    expected = _literal_dictionary_generalized_pav(
        alpha,
        beta,
        lower,
        ordered_scores,
    )
    actual = _generalized_pav(
        alpha,
        beta,
        lower,
        ordered_scores,
    )

    np.testing.assert_array_equal(actual, expected)


def test_numeric_stack_generalized_pav_rejects_mismatched_inputs():
    with pytest.raises(ValueError, match="equal length"):
        _generalized_pav(
            np.ones(2),
            np.ones(1),
            np.ones(2),
            np.ones(2),
        )


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("size", [0, 1, 2, 7, 31, 100, 129, 257, 400])
@pytest.mark.parametrize("score_kind", ["unique", "ties"])
def test_compiled_pav_is_bitwise_reference_equivalent(
        seed,
        size,
        score_kind,
):
    rng = np.random.default_rng(seed + 500)
    alpha = np.exp(rng.uniform(-20, 20, size=size))
    if size:
        alpha[rng.random(size) < 0.25] = 0.0
    beta = np.exp(rng.uniform(-20, 20, size=size))
    lower = rng.uniform(-700, -0.01, size=size)
    if score_kind == "unique":
        ordered_scores = np.arange(size, dtype=np.float64)
    else:
        ordered_scores = np.sort(rng.integers(0, 5, size=size))

    expected = _literal_dictionary_generalized_pav(
        alpha,
        beta,
        lower,
        ordered_scores,
    )
    actual = _generalized_pav_compiled(
        alpha,
        beta,
        lower,
        ordered_scores,
    )

    np.testing.assert_array_equal(actual, expected)
