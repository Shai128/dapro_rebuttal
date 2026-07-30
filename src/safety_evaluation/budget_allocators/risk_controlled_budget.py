"""Non-leaky budget control for a policy learned on an independent fold.

This module separates three different guarantees that are easy to conflate:

* :func:`select_crc_budget_candidate` gives a distribution-free *marginal
  expected-total-budget* guarantee.  The nested candidate family must be fixed
  before observing the independent budget-control fold.
* :func:`select_hoeffding_budget_candidate` gives a finite-grid,
  high-probability guarantee that the deployment population's expected cost
  fits the budget remaining after the realized pilot costs.
* :func:`row_local_horizon_budget_cap` gives a deterministic, per-row upper
  bound using only that row's label-free prior horizon.  It is conservative
  because an event may terminate the trajectory before that horizon.

None of these functions may be fit using Phase-II event times.  In particular,
``min(T_i, q_i)`` is a valid active length on an independent, fully observed
budget-control fold, but it is not a label-free Phase-II input.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BudgetCandidateSelection:
    """Diagnostics for a selected member of a nested policy family."""

    selected_index: int
    empirical_expected_cost_per_sample: float
    deployment_budget_per_sample: float
    selector_left_side_per_sample: float
    correction_per_sample: float
    guarantee_kind: str


@dataclass(frozen=True)
class RowLocalHorizonCap:
    """A row-local cumulative policy and its label-free budget diagnostics."""

    cumulative_probabilities: np.ndarray
    raw_horizon_costs: np.ndarray
    capped_horizon_costs: np.ndarray
    mixture_coefficients: np.ndarray
    budget_per_sample: float
    terminal_pi_min: float


def _validate_cost_inputs(
        expected_costs: np.ndarray,
        pilot_costs: np.ndarray,
        deployment_sample_count: int,
        maximum_candidate_cost_per_sample: float,
        maximum_pilot_cost_per_sample: float,
        *,
        require_nested: bool,
        tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    costs = np.asarray(expected_costs, dtype=np.float64)
    pilot = np.asarray(pilot_costs, dtype=np.float64).reshape(-1)
    if costs.ndim != 2 or costs.shape[0] == 0 or costs.shape[1] == 0:
        raise ValueError(
            "`expected_costs` must have shape "
            "(budget-control rows, candidate policies)."
        )
    if len(pilot) != costs.shape[0]:
        raise ValueError(
            "`pilot_costs` must have one value per budget-control row."
        )
    if deployment_sample_count <= 0:
        raise ValueError("`deployment_sample_count` must be positive.")
    for name, value in [
        (
            "maximum_candidate_cost_per_sample",
            maximum_candidate_cost_per_sample,
        ),
        ("maximum_pilot_cost_per_sample", maximum_pilot_cost_per_sample),
    ]:
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"`{name}` must be finite and positive.")
    if not np.all(np.isfinite(costs)) or not np.all(np.isfinite(pilot)):
        raise ValueError("Budget-control costs must be finite.")
    if np.any(costs < -tolerance) or np.any(pilot < -tolerance):
        raise ValueError("Budget-control costs must be nonnegative.")
    if (
            np.any(
                costs
                > maximum_candidate_cost_per_sample + tolerance
            )
            or np.any(
                pilot
                > maximum_pilot_cost_per_sample + tolerance
            )
    ):
        raise ValueError(
            "Budget-control costs exceed their declared per-sample bounds."
        )
    if require_nested and np.any(np.diff(costs, axis=1) > tolerance):
        raise ValueError(
            "Candidate policies must be ordered from most aggressive to most "
            "conservative, with row-wise non-increasing expected costs."
        )
    return costs, pilot


def select_crc_budget_candidate(
        expected_costs: np.ndarray,
        pilot_costs: np.ndarray,
        *,
        total_budget_after_policy_fit: float,
        deployment_sample_count: int,
        maximum_cost_per_sample: float,
        maximum_candidate_cost_per_sample: float | None = None,
        maximum_pilot_cost_per_sample: float | None = None,
        tolerance: float = 1e-12,
) -> BudgetCandidateSelection:
    """Select the strongest nested policy with a CRC expected-budget guarantee.

    The rows must form an i.i.d. budget-control fold that is independent of the
    data used to learn the candidate policy shapes.  Columns are a fixed,
    nested family ordered from most aggressive to most conservative.
    ``expected_costs[i, k]`` is the exact expected acquisition cost of policy
    ``k`` conditional on fully observed control trajectory ``i``.

    Let ``n`` be the control-fold size, ``m`` the deployment size, ``b_i`` the
    fully observed pilot costs, and ``c_i(k)`` the candidate costs.  Apply CRC
    to the nested loss

        ell_i(k) = c_i(k) + (n / m) b_i.

    The rule is

        (sum_i ell_i(k) + L) / (n + 1) <= B_remaining / m,

    where ``L = M_candidate + (n / m) * M_pilot`` bounds ``ell``.  Therefore
    CRC gives

        m E[c_new(k_hat)] + n E[b_new] <= B_remaining,

    which is exactly the expected control-plus-deployment cost that remains
    after conditioning on the policy-fit fold.  The two component bounds may
    differ: a fixed conservative policy family can have much smaller maximum
    expected cost than the fully observed pilot horizon.  If the optional
    separate bounds are omitted, both default to
    ``maximum_cost_per_sample`` for backward compatibility.  Under
    exchangeability, monotonicity, and feasibility, the selected policy
    satisfies the configured *total* budget in marginal expectation,
    conditional on the policy-fitting fold.

    This guarantee does not hold if the policy shapes were learned on these
    same rows; use a separate policy-fit fold for optimized DAPRO.
    """
    candidate_bound = (
        float(maximum_cost_per_sample)
        if maximum_candidate_cost_per_sample is None
        else float(maximum_candidate_cost_per_sample)
    )
    pilot_bound = (
        float(maximum_cost_per_sample)
        if maximum_pilot_cost_per_sample is None
        else float(maximum_pilot_cost_per_sample)
    )
    costs, pilot = _validate_cost_inputs(
        expected_costs,
        pilot_costs,
        deployment_sample_count,
        candidate_bound,
        pilot_bound,
        require_nested=True,
        tolerance=tolerance,
    )
    if (
            not np.isfinite(total_budget_after_policy_fit)
            or total_budget_after_policy_fit < 0
    ):
        raise ValueError(
            "`total_budget_after_policy_fit` must be finite and nonnegative."
        )

    n_control = len(pilot)
    target = float(total_budget_after_policy_fit) / deployment_sample_count
    rho = n_control / deployment_sample_count
    envelope = candidate_bound + rho * pilot_bound
    selector_values = (
        costs.sum(axis=0)
        + rho * pilot.sum()
        + envelope
    ) / (n_control + 1)
    feasible = np.flatnonzero(selector_values <= target + tolerance)
    if len(feasible) == 0:
        raise ValueError(
            "No candidate policy satisfies the CRC budget selector. Add a more "
            "conservative candidate, reduce the terminal floor, or increase "
            "the budget/control-fold size."
        )
    selected = int(feasible[0])
    return BudgetCandidateSelection(
        selected_index=selected,
        empirical_expected_cost_per_sample=float(costs[:, selected].mean()),
        deployment_budget_per_sample=float(max(target, 0.0)),
        selector_left_side_per_sample=float(selector_values[selected]),
        correction_per_sample=float(envelope / (n_control + 1)),
        guarantee_kind="crc_marginal_expected_total_budget",
    )


def select_hoeffding_budget_candidate(
        expected_costs: np.ndarray,
        pilot_costs: np.ndarray,
        *,
        total_budget_after_policy_fit: float,
        deployment_sample_count: int,
        maximum_cost_per_sample: float,
        maximum_candidate_cost_per_sample: float | None = None,
        maximum_pilot_cost_per_sample: float | None = None,
        delta: float,
        tolerance: float = 1e-12,
) -> BudgetCandidateSelection:
    """Select a fixed finite candidate with a high-probability budget UCB.

    Candidate policies must be fixed before observing the independent
    budget-control fold.  With probability at least ``1-delta`` over that
    fold, all candidate population costs are below their simultaneous
    one-sided Hoeffding bounds.  Selecting any candidate whose bound fits the
    *realized* remaining deployment budget therefore guarantees

        policy-fit cost + realized control cost
            + E[deployment cost | selected policy] <= configured budget.

    Unlike CRC, this is a conditional high-probability certificate rather than
    a marginal-expectation result.  It is often conservative for long horizons.
    """
    candidate_bound = (
        float(maximum_cost_per_sample)
        if maximum_candidate_cost_per_sample is None
        else float(maximum_candidate_cost_per_sample)
    )
    pilot_bound = (
        float(maximum_cost_per_sample)
        if maximum_pilot_cost_per_sample is None
        else float(maximum_pilot_cost_per_sample)
    )
    costs, pilot = _validate_cost_inputs(
        expected_costs,
        pilot_costs,
        deployment_sample_count,
        candidate_bound,
        pilot_bound,
        require_nested=True,
        tolerance=tolerance,
    )
    if not 0 < delta < 1:
        raise ValueError("`delta` must lie in (0, 1).")
    if (
            not np.isfinite(total_budget_after_policy_fit)
            or total_budget_after_policy_fit < 0
    ):
        raise ValueError(
            "`total_budget_after_policy_fit` must be finite and nonnegative."
        )

    remaining = float(total_budget_after_policy_fit - pilot.sum())
    target = remaining / deployment_sample_count
    if target < -tolerance:
        raise ValueError(
            "The fully observed budget-control fold already exceeds the budget "
            "remaining after policy fitting."
        )

    n_control, candidate_count = costs.shape
    radius = candidate_bound * np.sqrt(
        np.log(candidate_count / delta) / (2 * n_control)
    )
    upper_bounds = np.minimum(
        candidate_bound,
        costs.mean(axis=0) + radius,
    )
    feasible = np.flatnonzero(upper_bounds <= target + tolerance)
    if len(feasible) == 0:
        raise ValueError(
            "No candidate policy has a Hoeffding upper confidence bound below "
            "the remaining deployment budget."
        )
    selected = int(feasible[0])
    return BudgetCandidateSelection(
        selected_index=selected,
        empirical_expected_cost_per_sample=float(costs[:, selected].mean()),
        deployment_budget_per_sample=float(max(target, 0.0)),
        selector_left_side_per_sample=float(upper_bounds[selected]),
        correction_per_sample=float(radius),
        guarantee_kind="hoeffding_high_probability_deployment_budget",
    )


def affine_cumulative_policy_family(
        base_cumulative_probabilities: np.ndarray,
        mixture_parameters: np.ndarray,
        *,
        terminal_pi_min: float,
        tolerance: float = 1e-12,
) -> np.ndarray:
    """Construct a nested family without changing cumulative-policy ordering.

    ``mixture_parameters`` must be non-increasing and lie in ``[-1, 1]``.
    Parameter ``0`` returns the base policy, ``1`` returns always-continue, and
    ``-1`` returns the minimum-cost trajectory mixture with cumulative
    propensity ``terminal_pi_min``.  Intermediate values are affine mixtures.

    The output has shape ``(number of candidates, time width)`` and is ordered
    from most aggressive to most conservative.
    """
    base = np.asarray(base_cumulative_probabilities, dtype=np.float64).reshape(-1)
    parameters = np.asarray(mixture_parameters, dtype=np.float64).reshape(-1)
    if len(base) == 0 or len(parameters) == 0:
        raise ValueError("The base policy and mixture grid must be non-empty.")
    if not 0 < terminal_pi_min <= 1:
        raise ValueError("`terminal_pi_min` must lie in (0, 1].")
    if not np.all(np.isfinite(base)) or not np.all(np.isfinite(parameters)):
        raise ValueError("Policy probabilities and mixture values must be finite.")
    if (
            np.any(base < terminal_pi_min - tolerance)
            or np.any(base > 1 + tolerance)
    ):
        raise ValueError(
            "The base cumulative policy must respect the terminal floor and "
            "lie in [terminal_pi_min, 1]."
        )
    if np.any(np.diff(base) > tolerance):
        raise ValueError("The base cumulative policy must be non-increasing.")
    if (
            np.any(parameters < -1 - tolerance)
            or np.any(parameters > 1 + tolerance)
            or np.any(np.diff(parameters) > tolerance)
    ):
        raise ValueError(
            "Mixture parameters must be non-increasing and lie in [-1, 1]."
        )

    family = np.empty((len(parameters), len(base)), dtype=np.float64)
    for index, parameter in enumerate(parameters):
        if parameter >= 0:
            family[index] = base + parameter * (1 - base)
        else:
            family[index] = (
                terminal_pi_min
                + (1 + parameter) * (base - terminal_pi_min)
            )
    return np.clip(family, terminal_pi_min, 1.0)


def cumulative_policy_costs(
        cumulative_policy_candidates: np.ndarray,
        active_lengths: np.ndarray,
) -> np.ndarray:
    """Evaluate fixed cumulative policies on fully observed control lengths.

    ``active_lengths`` may be ``min(T_i, q_i)`` only when these rows belong to
    an independent, fully observed budget-control fold.  The returned matrix
    has shape ``(rows, candidates)``.
    """
    candidates = np.asarray(
        cumulative_policy_candidates,
        dtype=np.float64,
    )
    lengths = np.asarray(active_lengths).reshape(-1)
    if candidates.ndim != 2 or candidates.shape[0] == 0:
        raise ValueError(
            "`cumulative_policy_candidates` must have shape "
            "(candidate policies, time width)."
        )
    if not np.all(np.isfinite(candidates)):
        raise ValueError("Cumulative policy candidates must be finite.")
    if np.any(candidates < 0) or np.any(candidates > 1):
        raise ValueError("Cumulative policy candidates must lie in [0, 1].")
    if np.any(np.diff(candidates, axis=1) > 1e-12):
        raise ValueError("Every cumulative policy must be non-increasing.")
    if not np.issubdtype(lengths.dtype, np.integer):
        if not np.all(np.isfinite(lengths)) or not np.all(lengths == np.floor(lengths)):
            raise ValueError("`active_lengths` must contain finite integers.")
        lengths = lengths.astype(np.int64)
    else:
        lengths = lengths.astype(np.int64, copy=False)
    width = candidates.shape[1]
    if np.any(lengths < 0) or np.any(lengths > width):
        raise ValueError(f"`active_lengths` must lie in [0, {width}].")
    active = np.arange(width)[None, :] < lengths[:, None]
    return active.astype(np.float64) @ candidates.T


def solve_constant_continuation_policy(
        active_lengths: np.ndarray,
        *,
        budget_per_sample: float,
        time_width: int,
        terminal_pi_min: float,
        tolerance: float = 1e-10,
) -> tuple[np.ndarray, float, float]:
    """Fit the exact hard-floor constant-continuation reference schedule.

    A constant conditional continuation probability ``p`` has cumulative
    reach ``p ** (t + 1)``.  The hard terminal floor changes this to
    ``max(p ** (t + 1), terminal_pi_min)``.  Bisection chooses the largest
    feasible ``p`` on the supplied fully observed lengths.

    The returned tuple is ``(cumulative_schedule, p, expected_cost)``.  This
    helper is useful for Random-anchored optimized policies: both the simple
    reference schedule and a learned DAPRO schedule can be fit to the same
    Phase-I budget before taking a convex cumulative-probability mixture.
    """
    lengths = np.asarray(active_lengths).reshape(-1)
    if time_width <= 0:
        raise ValueError("`time_width` must be positive.")
    if not 0 < terminal_pi_min <= 1:
        raise ValueError("`terminal_pi_min` must lie in (0, 1].")
    if not np.isfinite(budget_per_sample) or budget_per_sample < 0:
        raise ValueError("`budget_per_sample` must be finite and nonnegative.")
    if not np.issubdtype(lengths.dtype, np.integer):
        if (
                not np.all(np.isfinite(lengths))
                or not np.all(lengths == np.floor(lengths))
        ):
            raise ValueError("`active_lengths` must contain finite integers.")
        lengths = lengths.astype(np.int64)
    else:
        lengths = lengths.astype(np.int64, copy=False)
    if len(lengths) == 0:
        raise ValueError("At least one active length is required.")
    if np.any(lengths < 0) or np.any(lengths > time_width):
        raise ValueError(
            f"`active_lengths` must lie in [0, {time_width}]."
        )

    survival = (
        np.arange(time_width)[None, :] < lengths[:, None]
    ).mean(axis=0)
    floor_cost = float(terminal_pi_min * survival.sum())
    maximum_cost = float(survival.sum())
    if budget_per_sample < floor_cost - tolerance:
        raise ValueError(
            "The constant policy budget is infeasible under the terminal "
            f"floor: target={budget_per_sample}, minimum={floor_cost}."
        )

    powers = np.arange(1, time_width + 1, dtype=np.float64)

    def schedule(probability: float) -> np.ndarray:
        return np.maximum(
            probability ** powers,
            terminal_pi_min,
        )

    if budget_per_sample >= maximum_cost - tolerance:
        probability = 1.0
        cumulative = np.ones(time_width, dtype=np.float64)
        achieved_cost = maximum_cost
    else:
        low, high = 0.0, 1.0
        cumulative = schedule(low)
        achieved_cost = float(np.dot(survival, cumulative))
        for _ in range(80):
            midpoint = (low + high) / 2
            midpoint_schedule = schedule(midpoint)
            midpoint_cost = float(np.dot(survival, midpoint_schedule))
            if midpoint_cost <= budget_per_sample:
                low = midpoint
                cumulative = midpoint_schedule
                achieved_cost = midpoint_cost
            else:
                high = midpoint
            if (
                    high - low <= tolerance
                    or abs(achieved_cost - budget_per_sample)
                    <= tolerance * max(1.0, budget_per_sample)
            ):
                break
        probability = low
    return cumulative, float(probability), achieved_cost


def row_local_horizon_budget_cap(
        cumulative_probabilities: np.ndarray,
        prior_horizons: np.ndarray,
        *,
        budget_per_sample: float,
        terminal_pi_min: float,
        tolerance: float = 1e-10,
) -> RowLocalHorizonCap:
    """Cap each row using only its own label-free prior horizon.

    For an over-budget row, cumulative propensities ``R`` are mixed toward the
    floor by

        R' = epsilon + lambda * (R - epsilon).

    This preserves the row's temporal ordering and relative deviations from
    the floor.  It guarantees ``sum_{t<q_i} R'_i(t) <= budget_per_sample`` and
    hence also bounds the true expected cost, because
    ``min(T_i, q_i) <= q_i``.  Rows do not share information, so the operation
    is compatible with a per-row censoring mechanism.

    The cap deliberately does not redistribute slack from short-horizon rows.
    Such redistribution would require a Phase-II pool statistic and would no
    longer be row-local under the current coverage proof.
    """
    cumulative = np.asarray(cumulative_probabilities, dtype=np.float64)
    horizons = np.asarray(prior_horizons).reshape(-1)
    if cumulative.ndim != 2 or cumulative.shape[0] == 0:
        raise ValueError(
            "`cumulative_probabilities` must have shape (rows, time width)."
        )
    if len(horizons) != len(cumulative):
        raise ValueError("`prior_horizons` must have one value per row.")
    if not np.isfinite(budget_per_sample) or budget_per_sample < 0:
        raise ValueError("`budget_per_sample` must be finite and nonnegative.")
    if not 0 < terminal_pi_min <= 1:
        raise ValueError("`terminal_pi_min` must lie in (0, 1].")
    if not np.all(np.isfinite(cumulative)):
        raise ValueError("Cumulative probabilities must be finite.")
    if (
            np.any(cumulative < terminal_pi_min - tolerance)
            or np.any(cumulative > 1 + tolerance)
    ):
        raise ValueError(
            "Cumulative probabilities must lie in [terminal_pi_min, 1]."
        )
    if not np.issubdtype(horizons.dtype, np.integer):
        if (
                not np.all(np.isfinite(horizons))
                or not np.all(horizons == np.floor(horizons))
        ):
            raise ValueError("`prior_horizons` must contain finite integers.")
        horizons = horizons.astype(np.int64)
    else:
        horizons = horizons.astype(np.int64, copy=False)
    width = cumulative.shape[1]
    if np.any(horizons < 0) or np.any(horizons > width):
        raise ValueError(f"`prior_horizons` must lie in [0, {width}].")

    for row, horizon in enumerate(horizons):
        if horizon > 1 and np.any(
                np.diff(cumulative[row, :horizon]) > tolerance
        ):
            raise ValueError(
                "Every active cumulative-probability path must be "
                "non-increasing."
            )

    active = np.arange(width)[None, :] < horizons[:, None]
    raw_costs = (cumulative * active).sum(axis=1)
    floor_costs = terminal_pi_min * horizons.astype(np.float64)
    infeasible = floor_costs > budget_per_sample + tolerance
    if np.any(infeasible):
        first = int(np.flatnonzero(infeasible)[0])
        raise ValueError(
            "The terminal floor is incompatible with the row-local budget: "
            f"row {first} has minimum horizon cost {floor_costs[first]:.6g} "
            f"> {budget_per_sample:.6g}."
        )

    corrected = cumulative.copy()
    coefficients = np.ones(len(cumulative), dtype=np.float64)
    over_budget = raw_costs > budget_per_sample + tolerance
    for row in np.flatnonzero(over_budget):
        denominator = raw_costs[row] - floor_costs[row]
        if denominator <= tolerance:
            coefficients[row] = 0.0
        else:
            coefficients[row] = np.clip(
                (budget_per_sample - floor_costs[row]) / denominator,
                0.0,
                1.0,
            )
        horizon = horizons[row]
        corrected[row, :horizon] = (
            terminal_pi_min
            + coefficients[row]
            * (cumulative[row, :horizon] - terminal_pi_min)
        )

    # Values after q_i are structurally irrelevant.  Setting them to one makes
    # later cumulative-to-conditional conversion unambiguous.
    corrected = np.where(active, corrected, 1.0)
    capped_costs = (corrected * active).sum(axis=1)
    if np.any(capped_costs > budget_per_sample + 10 * tolerance):
        raise RuntimeError("Row-local horizon correction failed numerically.")
    return RowLocalHorizonCap(
        cumulative_probabilities=corrected,
        raw_horizon_costs=raw_costs,
        capped_horizon_costs=capped_costs,
        mixture_coefficients=coefficients,
        budget_per_sample=float(budget_per_sample),
        terminal_pi_min=float(terminal_pi_min),
    )


__all__ = [
    "BudgetCandidateSelection",
    "RowLocalHorizonCap",
    "affine_cumulative_policy_family",
    "cumulative_policy_costs",
    "row_local_horizon_budget_cap",
    "solve_constant_continuation_policy",
    "select_crc_budget_candidate",
    "select_hoeffding_budget_candidate",
]
