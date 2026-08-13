"""Isolated prototype of low-dimensional, continuous Basis-DAPRO.

This module is intentionally not registered as a production allocator.  It
tests a deployable policy class that is shared by LPB and metric estimation:
the target changes only the nonnegative prefix event masses supplied to
``fit_basis_dapro``.

Let ``r_it`` be a frozen empirical-rank transform of a causal scalar score and
let ``phi(t)`` and ``psi(r_it)`` be time and score bases.  Conditional log
continuation is

    y_it = phi(t)^T Theta psi(r_it),       p_it = exp(y_it),

and cumulative log reach is ``z_it = sum_{s<=t} y_is``.  For prefix event
mass ``a_it`` and acquisition-cost mass ``d_it``, the fitted coefficients
minimize

    mean_i sum_t a_it [exp(-z_it) - 1]

subject to ``mean_i sum_t d_it exp(z_it) <= B``.  Both exponentials are
convex functions of the basis coefficients, and score monotonicity,
``p_it <= 1``, and a worst-score cumulative-reach floor are linear
constraints.  The resulting finite-dimensional problem is therefore convex.

Two score bases are provided:

* ``linear_rank`` is continuous piecewise-linear interpolation over empirical
  score rank.  With a small time basis this is the proposed Basis-DAPRO.
* ``step_bins`` is a deterministic quantile-bin lookup.  With one-hot time
  basis it exactly contains the existing K-bin deployable DAPRO policy class;
  K=2 is consequently a special case of this prototype.

The score lookup and fitted coefficients use policy-fit data only.  At
deployment, turn ``t`` depends solely on the current score and frozen lookup,
so the policy is causal.  ``apply_shared_envelope_crc`` shows that its output
can enter the same shared cumulative PAV cap and nested affine CRC family used
by production DAPRO.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import LinearConstraint, minimize


ScoreBasisKind = Literal["linear_rank", "step_bins"]


def _as_float_matrix(value, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or not np.all(np.isfinite(result)):
        raise ValueError(f"`{name}` must be a finite matrix.")
    return result


def _active_mask(lengths: np.ndarray, width: int) -> np.ndarray:
    lengths = np.asarray(lengths, dtype=np.int64).reshape(-1)
    if np.any(lengths < 0) or np.any(lengths > width):
        raise ValueError(f"Active lengths must lie in [0, {width}].")
    return np.arange(width)[None, :] < lengths[:, None]


def causal_target_value_score(
        conditional_grid,
        acquisition_horizons,
        target_horizons,
        *,
        strict: bool,
) -> np.ndarray:
    """Return a causal square-root target-value-per-cost score.

    At prefix ``t``, only ``conditional_grid[i,t,:]`` is used.  The numerator
    is remaining probability of the named target event and the denominator is
    expected remaining acquisition cost, truncated at the known acquisition
    horizon.  Metric estimation uses ``target_horizons=T_max`` with a
    non-strict event; LPB uses its frozen candidate quantile with a strict
    lower-tail event.  Thus this score definition, like the optimizer, is
    shared across tasks.
    """
    grid = np.asarray(conditional_grid, dtype=np.float64)
    if grid.ndim != 3 or not np.all(np.isfinite(grid)):
        raise ValueError("`conditional_grid` must be a finite rank-three array.")
    if np.any(grid < -1e-10):
        raise ValueError("Conditional probabilities must be nonnegative.")
    grid = np.maximum(grid, 0.0)
    n, width, outcomes = grid.shape

    def broadcast_horizon(value, name: str) -> np.ndarray:
        result = np.asarray(value, dtype=np.float64).reshape(-1)
        if len(result) == 1:
            result = np.repeat(result, n)
        if len(result) != n or not np.all(np.isfinite(result)):
            raise ValueError(f"`{name}` must be finite and scalar or rowwise.")
        return result

    acquisition = broadcast_horizon(
        acquisition_horizons,
        "acquisition_horizons",
    )
    target = broadcast_horizon(target_horizons, "target_horizons")
    result = np.zeros((n, width), dtype=np.float64)
    outcome_index = np.arange(outcomes, dtype=np.float64)
    one_based_outcome = outcome_index + 1.0
    comparison = np.less if strict else np.less_equal
    for step in range(width):
        remaining_capacity = np.maximum(acquisition - step, 0.0)
        future = outcome_index >= step
        future_mass = np.sum(grid[:, step, future], axis=1)
        active = (remaining_capacity > 0) & (future_mass > 0)
        if not np.any(active):
            continue
        target_event = future[None, :] & comparison(
            one_based_outcome[None, :],
            target[:, None],
        )
        target_mass = np.sum(
            grid[:, step, :] * target_event,
            axis=1,
        )
        raw_steps = np.maximum(outcome_index - step + 1.0, 1.0)
        remaining_cost = np.sum(
            grid[:, step, :]
            * future[None, :]
            * np.minimum(
                raw_steps[None, :],
                remaining_capacity[:, None],
            ),
            axis=1,
        )
        probability = np.divide(
            target_mass,
            future_mass,
            out=np.zeros(n, dtype=np.float64),
            where=future_mass > 0,
        )
        expected_cost = np.divide(
            remaining_cost,
            future_mass,
            out=np.ones(n, dtype=np.float64),
            where=future_mass > 0,
        )
        result[active, step] = np.sqrt(
            probability[active]
            / np.maximum(expected_cost[active], np.finfo(np.float64).tiny)
        )
    return result


def _linear_interpolation_basis(
        points: np.ndarray,
        knots: np.ndarray,
) -> np.ndarray:
    """Return nonnegative linear-hat weights that sum to one."""
    values = np.asarray(points, dtype=np.float64)
    knots = np.asarray(knots, dtype=np.float64).reshape(-1)
    if len(knots) == 0 or not np.all(np.isfinite(knots)):
        raise ValueError("Interpolation knots must be nonempty and finite.")
    if len(knots) > 1 and np.any(knots[1:] <= knots[:-1]):
        raise ValueError("Interpolation knots must be strictly increasing.")
    if len(knots) == 1:
        return np.ones(values.shape + (1,), dtype=np.float64)

    clipped = np.clip(values, knots[0], knots[-1])
    right = np.searchsorted(knots, clipped, side="right")
    right = np.clip(right, 1, len(knots) - 1)
    left = right - 1
    span = knots[right] - knots[left]
    right_weight = (clipped - knots[left]) / span
    result = np.zeros(values.shape + (len(knots),), dtype=np.float64)
    flat = result.reshape(-1, len(knots))
    row = np.arange(flat.shape[0])
    flat[row, left.reshape(-1)] = 1.0 - right_weight.reshape(-1)
    flat[row, right.reshape(-1)] += right_weight.reshape(-1)
    return result


def linear_time_basis(width: int, size: int) -> np.ndarray:
    """Piecewise-linear time basis evaluated at the deployable integer turns."""
    if width <= 0 or not 1 <= size <= width:
        raise ValueError("Time-basis size must lie in [1, width].")
    if size == width:
        return np.eye(width, dtype=np.float64)
    knots = np.linspace(0.0, width - 1.0, size)
    return _linear_interpolation_basis(np.arange(width), knots)


@dataclass(frozen=True)
class FrozenScoreBasis:
    """Frozen, causal map from a scalar current-prefix score to basis weights."""

    kind: ScoreBasisKind
    size: int
    width: int
    sorted_fit_values: tuple[np.ndarray, ...]
    step_edges: tuple[np.ndarray, ...]
    source_times: tuple[int, ...]

    @classmethod
    def fit(
            cls,
            scores,
            active_lengths,
            *,
            kind: ScoreBasisKind,
            size: int,
    ) -> "FrozenScoreBasis":
        score_matrix = _as_float_matrix(scores, "scores")
        n, width = score_matrix.shape
        if size <= 0:
            raise ValueError("Score-basis size must be positive.")
        if kind not in {"linear_rank", "step_bins"}:
            raise ValueError("Unknown score-basis kind.")
        lengths = np.asarray(active_lengths, dtype=np.int64).reshape(-1)
        if len(lengths) != n:
            raise ValueError("Scores and active lengths must have equal rows.")
        active = _active_mask(lengths, width)
        observed = [step for step in range(width) if np.any(active[:, step])]
        if not observed:
            raise ValueError("At least one active policy-fit prefix is required.")

        sorted_values: list[np.ndarray] = []
        edges: list[np.ndarray] = []
        sources: list[int] = []
        for step in range(width):
            if np.any(active[:, step]):
                source = step
            else:
                earlier = [candidate for candidate in observed if candidate < step]
                source = earlier[-1] if earlier else observed[0]
            values = np.sort(
                score_matrix[active[:, source], source],
                kind="stable",
            )
            sorted_values.append(values)
            if kind == "step_bins" and size > 1:
                cutpoints = np.unique(np.quantile(
                    values,
                    np.arange(1, size, dtype=np.float64) / size,
                ))
            else:
                cutpoints = np.empty(0, dtype=np.float64)
            edges.append(cutpoints)
            sources.append(source)
        return cls(
            kind=kind,
            size=int(size),
            width=width,
            sorted_fit_values=tuple(sorted_values),
            step_edges=tuple(edges),
            source_times=tuple(sources),
        )

    def transform(self, scores) -> np.ndarray:
        """Map rows to ``(row, time, score_basis)`` weights."""
        score_matrix = _as_float_matrix(scores, "scores")
        if score_matrix.shape[1] != self.width:
            raise ValueError("Deployment score width differs from fitted width.")
        n = len(score_matrix)
        result = np.zeros((n, self.width, self.size), dtype=np.float64)
        if self.kind == "step_bins":
            for step, edges in enumerate(self.step_edges):
                bins = np.searchsorted(
                    edges,
                    score_matrix[:, step],
                    side="right",
                )
                result[np.arange(n), step, bins] = 1.0
            return result

        knots = np.linspace(0.0, 1.0, self.size)
        ranks = np.empty((n, self.width), dtype=np.float64)
        for step, reference in enumerate(self.sorted_fit_values):
            below = np.searchsorted(
                reference,
                score_matrix[:, step],
                side="left",
            )
            at_or_below = np.searchsorted(
                reference,
                score_matrix[:, step],
                side="right",
            )
            ranks[:, step] = (
                below + 0.5 * (at_or_below - below)
            ) / len(reference)
        return _linear_interpolation_basis(ranks, knots)


def _tensor_design(
        time_basis: np.ndarray,
        score_basis: np.ndarray,
) -> np.ndarray:
    """Return conditional-log design with flattened time-by-score parameters."""
    return np.einsum(
        "ta,ntk->ntak",
        time_basis,
        score_basis,
    ).reshape(
        score_basis.shape[0],
        score_basis.shape[1],
        time_basis.shape[1] * score_basis.shape[2],
    )


def _policy_vertex_design(time_basis: np.ndarray, score_size: int) -> np.ndarray:
    """Evaluate the tensor design at every score-basis vertex."""
    width, time_size = time_basis.shape
    vertices = np.zeros(
        (width, score_size, time_size * score_size),
        dtype=np.float64,
    )
    for step in range(width):
        for score_index in range(score_size):
            vertices[step, score_index] = np.outer(
                time_basis[step],
                np.eye(score_size)[score_index],
            ).reshape(-1)
    return vertices


def _constant_score_initialization(
        time_basis: np.ndarray,
        score_size: int,
        terminal_reach_floor: float | None,
) -> np.ndarray:
    """Construct a low-cost feasible initialization equal across score rank."""
    width, time_size = time_basis.shape
    if terminal_reach_floor is None:
        log_terminal = -30.0
    else:
        log_terminal = float(np.log(terminal_reach_floor))
    coefficients = np.zeros(time_size, dtype=np.float64)
    first_mass = float(time_basis[:, 0].sum())
    if first_mass <= 0:
        raise ValueError("The first time basis must have positive support.")
    coefficients[0] = log_terminal / first_mass
    return np.repeat(coefficients[:, None], score_size, axis=1).reshape(-1)


@dataclass(frozen=True)
class BasisDAPROModel:
    """Fitted deployable Basis-DAPRO policy."""

    coefficients: np.ndarray
    time_basis: np.ndarray
    score_basis: FrozenScoreBasis
    terminal_reach_floor: float | None
    objective_value: float
    expected_cost: float
    optimizer_iterations: int
    optimizer_message: str

    @property
    def parameter_count(self) -> int:
        return int(self.coefficients.size)

    def conditional_log_probabilities(self, scores) -> np.ndarray:
        score_weights = self.score_basis.transform(scores)
        design = _tensor_design(self.time_basis, score_weights)
        values = np.einsum("ntd,d->nt", design, self.coefficients)
        if np.any(values > 2e-7):
            raise RuntimeError("Fitted log continuation exceeds zero.")
        return np.minimum(values, 0.0)

    def conditionals(self, scores) -> np.ndarray:
        return np.exp(self.conditional_log_probabilities(scores))

    def cumulative_reach(self, scores) -> np.ndarray:
        return np.cumprod(self.conditionals(scores), axis=1)


def fit_basis_dapro(
        fit_scores,
        active_lengths,
        event_masses,
        cost_masses,
        budget_per_sample: float,
        *,
        score_basis_kind: ScoreBasisKind = "linear_rank",
        score_basis_size: int = 4,
        time_basis_size: int = 6,
        terminal_reach_floor: float | None = 0.005,
        max_iterations: int = 1000,
        tolerance: float = 1e-10,
) -> BasisDAPROModel:
    """Fit the convex Basis-DAPRO prototype on a policy-fit fold.

    ``event_masses`` chooses the task.  Metric estimation supplies unsafe-event
    prefix mass through its fixed horizon; LPB supplies the corresponding
    candidate-specific influence/event mass.  No other part of the fitter
    changes.
    """
    scores = _as_float_matrix(fit_scores, "fit_scores")
    event = _as_float_matrix(event_masses, "event_masses")
    cost = _as_float_matrix(cost_masses, "cost_masses")
    if event.shape != scores.shape or cost.shape != scores.shape:
        raise ValueError("Scores, event masses, and cost masses must agree.")
    if np.any(event < 0) or np.any(cost < 0):
        raise ValueError("Objective and cost masses must be nonnegative.")
    if not np.isfinite(budget_per_sample) or budget_per_sample <= 0:
        raise ValueError("Budget per sample must be finite and positive.")
    if terminal_reach_floor is not None and not (
            0 < terminal_reach_floor <= 1
    ):
        raise ValueError("Terminal reach floor must lie in (0, 1].")

    n, width = scores.shape
    lengths = np.asarray(active_lengths, dtype=np.int64).reshape(-1)
    if len(lengths) != n:
        raise ValueError("Active lengths must have one entry per score row.")
    active = _active_mask(lengths, width).astype(np.float64)
    event = event * active
    cost = cost * active
    maximum_cost = float(cost.sum(axis=1).mean())
    if budget_per_sample >= maximum_cost - tolerance:
        budget_per_sample = maximum_cost

    score_lookup = FrozenScoreBasis.fit(
        scores,
        lengths,
        kind=score_basis_kind,
        size=score_basis_size,
    )
    time_weights = linear_time_basis(width, time_basis_size)
    score_weights = score_lookup.transform(scores)
    conditional_design = _tensor_design(time_weights, score_weights)
    cumulative_design = np.cumsum(conditional_design, axis=1)
    parameter_count = time_basis_size * score_basis_size
    vertices = _policy_vertex_design(time_weights, score_basis_size)

    # Linear constraints are sufficient globally: both score bases interpolate
    # between these vertices.  Vertex log-probabilities are nonpositive and
    # nondecreasing in score.  Their worst-score cumulative sums stay above the
    # requested floor, hence every possible causal score path is positive.
    constraint_rows = []
    lower_bounds = []
    upper_bounds = []
    for step in range(width):
        for score_index in range(score_basis_size):
            constraint_rows.append(vertices[step, score_index])
            lower_bounds.append(-np.inf)
            upper_bounds.append(0.0)
        for score_index in range(score_basis_size - 1):
            constraint_rows.append(
                vertices[step, score_index + 1]
                - vertices[step, score_index]
            )
            lower_bounds.append(0.0)
            upper_bounds.append(np.inf)
    if terminal_reach_floor is not None:
        cumulative_worst = np.cumsum(vertices[:, 0, :], axis=0)
        for step in range(width):
            constraint_rows.append(cumulative_worst[step])
            lower_bounds.append(np.log(terminal_reach_floor))
            upper_bounds.append(np.inf)
    linear_constraint = LinearConstraint(
        np.asarray(constraint_rows, dtype=np.float64),
        np.asarray(lower_bounds, dtype=np.float64),
        np.asarray(upper_bounds, dtype=np.float64),
    )

    def reach_and_inverse(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        log_reach = np.einsum("ntd,d->nt", cumulative_design, theta)
        return np.exp(np.clip(log_reach, -700, 0)), np.exp(
            np.clip(-log_reach, 0, 700)
        )

    def objective(theta: np.ndarray) -> float:
        _, inverse = reach_and_inverse(theta)
        return float(np.mean(np.sum(event * (inverse - 1.0), axis=1)))

    def objective_gradient(theta: np.ndarray) -> np.ndarray:
        _, inverse = reach_and_inverse(theta)
        return -np.einsum(
            "nt,ntd->d",
            event * inverse,
            cumulative_design,
        ) / n

    def expected_cost(theta: np.ndarray) -> float:
        reach, _ = reach_and_inverse(theta)
        return float(np.mean(np.sum(cost * reach, axis=1)))

    def cost_gradient(theta: np.ndarray) -> np.ndarray:
        reach, _ = reach_and_inverse(theta)
        return np.einsum(
            "nt,ntd->d",
            cost * reach,
            cumulative_design,
        ) / n

    low_cost_initial = _constant_score_initialization(
        time_weights,
        score_basis_size,
        terminal_reach_floor,
    )
    minimum_start_cost = expected_cost(low_cost_initial)
    if minimum_start_cost > budget_per_sample + 1e-8:
        raise ValueError(
            "The requested budget is infeasible in this temporal basis under "
            "the cumulative-reach floor. Increase the time-basis size, budget, "
            "or lower the floor."
        )
    if maximum_cost <= budget_per_sample + tolerance:
        solution = np.zeros(parameter_count, dtype=np.float64)
        return BasisDAPROModel(
            coefficients=solution,
            time_basis=time_weights,
            score_basis=score_lookup,
            terminal_reach_floor=terminal_reach_floor,
            objective_value=objective(solution),
            expected_cost=expected_cost(solution),
            optimizer_iterations=0,
            optimizer_message="full_acquisition_budget",
        )

    # Start strictly inside the nonlinear budget constraint without exposing
    # SLSQP to the enormous inverse reaches of the deliberately low-cost
    # construction above.  Scaling a feasible log-policy toward zero preserves
    # every linear constraint, including the cumulative floor.
    interior_target = minimum_start_cost + 0.9 * (
        budget_per_sample - minimum_start_cost
    )
    scale_low, scale_high = 0.0, 1.0
    for _ in range(80):
        scale_mid = 0.5 * (scale_low + scale_high)
        if expected_cost(scale_mid * low_cost_initial) > interior_target:
            scale_low = scale_mid
        else:
            scale_high = scale_mid
    initial = scale_high * low_cost_initial

    # Normalize only the optimizer-facing objective.  Sparse hazard and
    # squared-increment masses can make its natural scale 1e-3 or smaller,
    # which triggers spurious SLSQP line-search failures although the problem
    # is convex.  Positive rescaling leaves the minimizer unchanged.
    objective_scale = max(
        float(np.mean(np.sum(event, axis=1))),
        1e-8,
    )
    constraints = [
        linear_constraint,
        {
            "type": "ineq",
            "fun": lambda theta: budget_per_sample - expected_cost(theta),
            "jac": lambda theta: -cost_gradient(theta),
        },
    ]

    def solve_slsqp(start: np.ndarray, ftol: float):
        return minimize(
            lambda theta: objective(theta) / objective_scale,
            start,
            method="SLSQP",
            jac=lambda theta: objective_gradient(theta) / objective_scale,
            constraints=constraints,
            options={
                "maxiter": int(max_iterations),
                "ftol": float(ftol),
                "disp": False,
            },
        )

    result = solve_slsqp(initial, max(float(tolerance), 1e-10))
    first_message = str(result.message)
    if not result.success:
        retry_start = (
            np.asarray(result.x, dtype=np.float64)
            if (
                np.all(np.isfinite(result.x))
                and expected_cost(result.x) <= budget_per_sample + 1e-7
            )
            else initial
        )
        result = solve_slsqp(retry_start, max(float(tolerance), 1e-8))
        if result.success:
            result.message = (
                f"scaled SLSQP retry after: {first_message}; {result.message}"
            )
        else:
            candidate = np.asarray(result.x, dtype=np.float64)
            constraint_values = np.asarray(constraint_rows) @ candidate
            lower = np.asarray(lower_bounds)
            upper = np.asarray(upper_bounds)
            linear_violation = max(
                float(np.max(np.where(
                    np.isfinite(lower),
                    lower - constraint_values,
                    -np.inf,
                ))),
                float(np.max(np.where(
                    np.isfinite(upper),
                    constraint_values - upper,
                    -np.inf,
                ))),
                0.0,
            )
            cost_violation = expected_cost(candidate) - budget_per_sample
            # SLSQP status 8 is frequently a roundoff-only termination exactly
            # on the active exponential-budget boundary.  Accept it only after
            # independently checking every convex constraint and improvement
            # over the strictly feasible start.  The downstream cumulative
            # projection then removes the allowed 5e-4 cost roundoff (at most
            # 3e-5 relative to the real-data policy budget).
            numerically_valid = (
                np.all(np.isfinite(candidate))
                and linear_violation <= 2e-7
                and cost_violation <= 5e-4
                and objective(candidate) <= objective(initial) + 1e-10
            )
            if numerically_valid:
                if cost_violation > 0:
                    low, high = 0.0, 1.0
                    for _ in range(80):
                        middle = 0.5 * (low + high)
                        interpolated = initial + middle * (
                            candidate - initial
                        )
                        if expected_cost(interpolated) <= budget_per_sample:
                            low = middle
                        else:
                            high = middle
                    result.x = initial + low * (candidate - initial)
                result.success = True
                result.message = (
                    "accepted feasible SLSQP boundary iterate after: "
                    f"{first_message}; retry={result.message}; "
                    f"linear_violation={linear_violation:.3g}; "
                    f"cost_violation={cost_violation:.3g}"
                )
            else:
                raise RuntimeError(
                    "Basis-DAPRO convex solve failed: "
                    f"first={first_message}; retry={result.message}; "
                    f"linear_violation={linear_violation:.3g}; "
                    f"cost_violation={cost_violation:.3g}"
                )
    achieved_cost = expected_cost(result.x)
    if achieved_cost > budget_per_sample + 2e-7:
        raise RuntimeError("Basis-DAPRO optimizer returned an over-budget policy.")
    vertex_values = np.einsum("tkd,d->tk", vertices, result.x)
    if np.any(vertex_values > 2e-7):
        raise RuntimeError("Basis-DAPRO optimizer violated p <= 1.")
    if np.any(np.diff(vertex_values, axis=1) < -2e-7):
        raise RuntimeError("Basis-DAPRO optimizer violated score monotonicity.")
    if terminal_reach_floor is not None:
        worst_reach = np.exp(np.cumsum(vertex_values[:, 0]))
        if np.any(worst_reach < terminal_reach_floor - 2e-7):
            raise RuntimeError("Basis-DAPRO optimizer violated positivity.")
    return BasisDAPROModel(
        coefficients=np.asarray(result.x, dtype=np.float64),
        time_basis=time_weights,
        score_basis=score_lookup,
        terminal_reach_floor=terminal_reach_floor,
        objective_value=objective(result.x),
        expected_cost=achieved_cost,
        optimizer_iterations=int(result.nit),
        optimizer_message=str(result.message),
    )


def apply_shared_envelope_crc(
        base_conditionals,
        active_lengths,
        shared_cumulative_envelope,
        *,
        alpha: float,
        terminal_reach_floor: float,
) -> np.ndarray:
    """Apply the production shared-envelope then nested affine CRC family.

    The envelope must be learned and frozen from the policy-fit fold before
    control/deployment scores are inspected.  Pointwise intersection preserves
    causality.  The affine family is nested in ``alpha`` and retains the same
    pathwise horizon-cost cap.
    """
    conditionals = _as_float_matrix(base_conditionals, "base_conditionals")
    n, width = conditionals.shape
    if np.any(conditionals <= 0) or np.any(conditionals > 1):
        raise ValueError("Base conditionals must lie in (0, 1].")
    lengths = np.asarray(active_lengths, dtype=np.int64).reshape(-1)
    if len(lengths) != n:
        raise ValueError("Active lengths must have one entry per row.")
    active = _active_mask(lengths, width)
    envelope = np.asarray(shared_cumulative_envelope, dtype=np.float64).reshape(-1)
    if len(envelope) != width or np.any(~np.isfinite(envelope)):
        raise ValueError("Shared envelope has the wrong shape or is nonfinite.")
    if (
            np.any(envelope < terminal_reach_floor)
            or np.any(envelope > 1)
            or np.any(np.diff(envelope) > 1e-12)
    ):
        raise ValueError("Shared envelope must be decreasing and respect floor.")
    if not 0 <= alpha <= 1 or not 0 < terminal_reach_floor <= 1:
        raise ValueError("CRC alpha/floor are outside their valid range.")

    cumulative = np.cumprod(conditionals, axis=1)
    if np.any(cumulative[active] < terminal_reach_floor - 1e-10):
        raise ValueError("Base policy does not respect the terminal floor.")
    capped = np.minimum(cumulative, envelope[None, :])
    selected = terminal_reach_floor + alpha * (
        capped - terminal_reach_floor
    )
    selected = np.where(active, selected, 1.0)
    previous = np.concatenate(
        [np.ones((n, 1), dtype=np.float64), selected[:, :-1]],
        axis=1,
    )
    result = np.divide(selected, previous, out=np.ones_like(selected))
    return np.where(active, np.clip(result, 0.0, 1.0), 1.0)


def basis_problem_complexity(
        sample_count: int,
        width: int,
        time_basis_size: int,
        score_basis_size: int,
) -> dict[str, int | str]:
    """Return explicit fit/deployment complexity for experiment reports."""
    parameters = time_basis_size * score_basis_size
    return {
        "parameters": parameters,
        "score_lookup_fit": "O(N*T*log(N))",
        "objective_gradient_per_iteration": f"O({sample_count * width * parameters})",
        "linear_constraints": (
            width * score_basis_size
            + width * max(score_basis_size - 1, 0)
            + width
        ),
        "deployment_per_prefix": f"O(log(N)+{parameters})",
    }


__all__ = [
    "BasisDAPROModel",
    "FrozenScoreBasis",
    "apply_shared_envelope_crc",
    "basis_problem_complexity",
    "causal_target_value_score",
    "fit_basis_dapro",
    "linear_time_basis",
]
