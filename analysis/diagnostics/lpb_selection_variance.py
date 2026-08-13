"""Executable diagnostics for nonlinear LPB selector acquisition variance.

The LPB candidates are nested.  Column ``j`` of ``candidate_events`` is the
complete-data event indicator for candidate ``j + 1``; the first entry of
``candidate_coverages`` is the fallback candidate zero.  The strict selector
therefore returns the number of HT candidate rates below ``alpha``.

Run directly with::

    python -m analysis.diagnostics.lpb_selection_variance

Only NumPy is required.  Exact enumeration is intentionally restricted to
small diagnostic examples; the Monte Carlo routine is batched.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import json

import numpy as np


@dataclass(frozen=True)
class SelectorMoments:
    """Crossing probabilities and selected-output moments."""

    crossing_probabilities: np.ndarray
    mean: float
    direct_variance: float
    crossing_formula_variance: float


@dataclass(frozen=True)
class PivotalBounds:
    """First-order Hoeffding lower and Efron--Stein upper bounds."""

    lower_bound: float
    upper_bound: float
    mean_finite_differences: np.ndarray
    mean_squared_finite_differences: np.ndarray


@dataclass(frozen=True)
class MarginBounds:
    """Analytic candidate-margin bounds and DAPRO-compatible row weights."""

    complete_rates: np.ndarray
    candidate_margins: np.ndarray
    candidate_variances: np.ndarray
    row_weights: np.ndarray
    cantelli_bound: float
    linear_bound: float


def _validated_inputs(
        candidate_events: np.ndarray,
        propensities: np.ndarray,
        alpha: float,
        candidate_coverages: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    events = np.asarray(candidate_events, dtype=bool)
    pi = np.asarray(propensities, dtype=np.float64).reshape(-1)
    coverages = np.asarray(candidate_coverages, dtype=np.float64).reshape(-1)
    if events.ndim != 2 or events.shape[0] == 0 or events.shape[1] == 0:
        raise ValueError("`candidate_events` must be a nonempty matrix.")
    if len(pi) != len(events):
        raise ValueError("There must be one propensity per calibration row.")
    if np.any(~np.isfinite(pi)) or np.any((pi <= 0) | (pi > 1)):
        raise ValueError("Propensities must be finite and lie in (0, 1].")
    if len(coverages) != events.shape[1] + 1:
        raise ValueError(
            "`candidate_coverages` must contain candidate zero followed by "
            "one value per event column."
        )
    if np.any(~np.isfinite(coverages)):
        raise ValueError("Candidate coverages must be finite.")
    if not np.isfinite(alpha):
        raise ValueError("`alpha` must be finite.")
    if np.any(np.diff(events.astype(np.int8), axis=1) < 0):
        raise ValueError("Candidate-event columns must be row-wise nested.")
    return events, pi, float(alpha), coverages


def _selected_output(
        acquisitions: np.ndarray,
        events: np.ndarray,
        pi: np.ndarray,
        alpha: float,
        coverages: np.ndarray,
) -> float:
    rates = (acquisitions / pi) @ events.astype(np.float64) / len(events)
    selected_index = int(np.count_nonzero(rates < alpha))
    return float(coverages[selected_index])


def selector_variance_from_crossings(
        crossing_probabilities: np.ndarray,
        candidate_coverages: np.ndarray,
) -> tuple[float, float]:
    """Return exact mean and variance from marginal nested crossings.

    If ``F[j] = P(M_{j+1} < alpha)``, nestedness gives
    ``Cov(I_j, I_k) = F[max(j, k)] - F[j] F[k]``.  No Gaussian assumption is
    used here.
    """
    crossing = np.asarray(crossing_probabilities, dtype=np.float64).reshape(-1)
    coverages = np.asarray(candidate_coverages, dtype=np.float64).reshape(-1)
    if len(coverages) != len(crossing) + 1:
        raise ValueError("Coverage and crossing dimensions do not agree.")
    if np.any(~np.isfinite(crossing)) or np.any((crossing < 0) | (crossing > 1)):
        raise ValueError("Crossing probabilities must lie in [0, 1].")
    differences = np.diff(coverages)
    index = np.arange(len(crossing))
    nested_second_moment = crossing[np.maximum.outer(index, index)]
    covariance = nested_second_moment - np.outer(crossing, crossing)
    mean = float(coverages[0] + differences @ crossing)
    variance = float(differences @ covariance @ differences)
    return mean, variance


def monte_carlo_selector_moments(
        candidate_events: np.ndarray,
        propensities: np.ndarray,
        alpha: float,
        candidate_coverages: np.ndarray,
        *,
        draws: int = 100_000,
        seed: int = 0,
        batch_size: int = 20_000,
) -> SelectorMoments:
    """Estimate crossings and directly simulate the selected output."""
    events, pi, alpha, coverages = _validated_inputs(
        candidate_events,
        propensities,
        alpha,
        candidate_coverages,
    )
    if draws <= 0 or batch_size <= 0:
        raise ValueError("`draws` and `batch_size` must be positive.")
    rng = np.random.default_rng(seed)
    event_float = events.astype(np.float64)
    crossing_counts = np.zeros(events.shape[1], dtype=np.int64)
    output_sum = 0.0
    output_square_sum = 0.0
    completed = 0
    while completed < draws:
        size = min(batch_size, draws - completed)
        acquisitions = rng.random((size, len(events))) < pi[None, :]
        rates = (
            (acquisitions.astype(np.float64) / pi[None, :]) @ event_float
            / len(events)
        )
        feasible = rates < alpha
        # Nested event columns and common positive weights make every row of
        # `feasible` an initial prefix.
        selected_indices = feasible.sum(axis=1)
        outputs = coverages[selected_indices]
        crossing_counts += feasible.sum(axis=0)
        output_sum += float(outputs.sum())
        output_square_sum += float(np.square(outputs).sum())
        completed += size

    crossing = crossing_counts / draws
    direct_mean = output_sum / draws
    direct_variance = output_square_sum / draws - direct_mean**2
    formula_mean, formula_variance = selector_variance_from_crossings(
        crossing,
        coverages,
    )
    if abs(formula_mean - direct_mean) > 5e-13:
        raise AssertionError("The nested-crossing mean identity failed.")
    return SelectorMoments(
        crossing_probabilities=crossing,
        mean=direct_mean,
        direct_variance=float(direct_variance),
        crossing_formula_variance=formula_variance,
    )


def exact_selector_moments(
        candidate_events: np.ndarray,
        propensities: np.ndarray,
        alpha: float,
        candidate_coverages: np.ndarray,
        *,
        maximum_rows: int = 20,
) -> SelectorMoments:
    """Enumerate every acquisition vector for a small diagnostic problem."""
    events, pi, alpha, coverages = _validated_inputs(
        candidate_events,
        propensities,
        alpha,
        candidate_coverages,
    )
    if len(events) > maximum_rows:
        raise ValueError("Exact enumeration is restricted to small examples.")
    crossing = np.zeros(events.shape[1], dtype=np.float64)
    output_mean = 0.0
    output_second = 0.0
    for values in product((0.0, 1.0), repeat=len(events)):
        acquisition = np.asarray(values, dtype=np.float64)
        probability = float(np.prod(np.where(acquisition > 0, pi, 1 - pi)))
        rates = acquisition / pi @ events.astype(np.float64) / len(events)
        feasible = rates < alpha
        output = float(coverages[int(feasible.sum())])
        crossing += probability * feasible
        output_mean += probability * output
        output_second += probability * output**2
    formula_mean, formula_variance = selector_variance_from_crossings(
        crossing,
        coverages,
    )
    if abs(formula_mean - output_mean) > 2e-13:
        raise AssertionError("The exact nested-crossing mean identity failed.")
    return SelectorMoments(
        crossing_probabilities=crossing,
        mean=output_mean,
        direct_variance=float(output_second - output_mean**2),
        crossing_formula_variance=formula_variance,
    )


def pivotal_variance_bounds(
        candidate_events: np.ndarray,
        propensities: np.ndarray,
        alpha: float,
        candidate_coverages: np.ndarray,
        *,
        maximum_rows: int = 20,
) -> PivotalBounds:
    """Exactly enumerate Hoeffding and Efron--Stein pivotal bounds."""
    events, pi, alpha, coverages = _validated_inputs(
        candidate_events,
        propensities,
        alpha,
        candidate_coverages,
    )
    n = len(events)
    if n > maximum_rows:
        raise ValueError("Pivotal enumeration is restricted to small examples.")
    mean_delta = np.zeros(n, dtype=np.float64)
    mean_delta_square = np.zeros(n, dtype=np.float64)
    for row in range(n):
        other_rows = np.delete(np.arange(n), row)
        for values in product((0.0, 1.0), repeat=n - 1):
            other = np.asarray(values, dtype=np.float64)
            probability = float(np.prod(np.where(
                other > 0,
                pi[other_rows],
                1 - pi[other_rows],
            )))
            z_zero = np.zeros(n, dtype=np.float64)
            z_zero[other_rows] = other
            z_one = z_zero.copy()
            z_one[row] = 1.0
            difference = (
                _selected_output(z_one, events, pi, alpha, coverages)
                - _selected_output(z_zero, events, pi, alpha, coverages)
            )
            mean_delta[row] += probability * difference
            mean_delta_square[row] += probability * difference**2
    bernoulli_variance = pi * (1 - pi)
    return PivotalBounds(
        lower_bound=float(np.sum(bernoulli_variance * mean_delta**2)),
        upper_bound=float(np.sum(bernoulli_variance * mean_delta_square)),
        mean_finite_differences=mean_delta,
        mean_squared_finite_differences=mean_delta_square,
    )


def analytic_margin_bounds(
        candidate_events: np.ndarray,
        propensities: np.ndarray,
        alpha: float,
        candidate_coverages: np.ndarray,
        *,
        margin_tolerance: float = 1e-12,
) -> MarginBounds:
    """Compute Cantelli and linear margin bounds for the hard selector."""
    events, pi, alpha, coverages = _validated_inputs(
        candidate_events,
        propensities,
        alpha,
        candidate_coverages,
    )
    n = len(events)
    complete_rates = events.mean(axis=0, dtype=np.float64)
    margins = np.abs(alpha - complete_rates)
    if np.any(margins <= margin_tolerance):
        raise ValueError(
            "A candidate is on the nonregular boundary; use pivotal crossing "
            "weights instead of the linear margin bound."
        )
    inverse_excess = 1 / pi - 1
    candidate_variances = (
        (events.astype(np.float64) * inverse_excess[:, None]).sum(axis=0)
        / n**2
    )
    coverage_gaps = np.abs(np.diff(coverages))
    total_variation = float(coverage_gaps.sum())
    coefficients = coverage_gaps / margins**2
    row_weights = events.astype(np.float64) @ coefficients
    cantelli = total_variation * float(np.sum(
        coverage_gaps
        * candidate_variances
        / (candidate_variances + margins**2)
    ))
    linear = total_variation * float(np.sum(
        coverage_gaps * candidate_variances / margins**2
    ))
    # This identity verifies that the final bound has the ordinary weighted
    # DAPRO form D_c / n^2 sum_i w_i (1 / pi_i - 1).
    dapro_form = (
        total_variation / n**2 * float(row_weights @ inverse_excess)
    )
    if abs(linear - dapro_form) > 5e-13:
        raise AssertionError("The analytic margin-weight identity failed.")
    return MarginBounds(
        complete_rates=complete_rates,
        candidate_margins=margins,
        candidate_variances=candidate_variances,
        row_weights=row_weights,
        cantelli_bound=cantelli,
        linear_bound=linear,
    )


def toy_problem() -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Return a heterogeneous nested problem with a near-boundary candidate."""
    first_event_band = np.asarray([1, 1, 2, 2, 3, 3, 4, 4])
    candidate_index = np.arange(1, 4)[None, :]
    events = candidate_index >= first_event_band[:, None]
    pi = np.asarray([0.65, 0.80, 0.55, 0.90, 0.72, 0.60, 0.85, 1.00])
    alpha = 0.58
    coverages = np.asarray([0.99, 0.94, 0.86, 0.72])
    return events, pi, alpha, coverages


def main() -> None:
    events, pi, alpha, coverages = toy_problem()
    exact = exact_selector_moments(events, pi, alpha, coverages)
    monte_carlo = monte_carlo_selector_moments(
        events,
        pi,
        alpha,
        coverages,
        draws=250_000,
        seed=20260812,
    )
    pivotal = pivotal_variance_bounds(events, pi, alpha, coverages)
    margin = analytic_margin_bounds(events, pi, alpha, coverages)
    summary = {
        "exact_variance": exact.direct_variance,
        "exact_crossing_formula_variance": exact.crossing_formula_variance,
        "monte_carlo_variance": monte_carlo.direct_variance,
        "monte_carlo_crossing_formula_variance": (
            monte_carlo.crossing_formula_variance
        ),
        "pivotal_lower_bound": pivotal.lower_bound,
        "pivotal_upper_bound": pivotal.upper_bound,
        "margin_cantelli_bound": margin.cantelli_bound,
        "margin_linear_bound": margin.linear_bound,
        "margin_row_weights": margin.row_weights.tolist(),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
