"""
solve_exact_fast.py
────────────────────────────────────────────────────────────────────────────────
Algorithm: Dual decomposition (bisection on λ) + Gauss-Seidel BCD inner loop.

Key optimisations over previous versions
─────────────────────────────────────────
1. EXACT PER-COORDINATE MINIMISER  (derivation below)
   For fixed λ and all y[i,j] j≠t, the Lagrangian in u = y[i,t] is:
     L_i(u) = A_i·exp(-u) + B_i·exp(u)   (A_i, B_i > 0, independent of u)
   Minimiser: u* = 0.5·log(A_i / B_i)
   After substitution of current cumY values:
     u*[i] = y[i,t] − 0.5·(log λ + log N + log suffix_t[i] + cumY[i, C_i−1])
   No step size, no iteration — one formula per (i,t).

2. INCREMENTAL cumY UPDATE  (avoids full O(NL) recompute per column)
   After updating y[:,t], cumY[:,s] shifts by Δ for all s≥t.
   Done in one numpy broadcast: cumY[:,t:] += delta[:,None]

3. SKIP PAV WHEN NOT NEEDED
   PAV violations drop 88 → 52 → 14 columns per pass.
   Check for violations before calling isotonic_regression.
   Later passes and warm-started outer iterations spend almost no time on PAV.

4. TWO-SIDED WARM START
   Maintain separate Y_lo / Y_hi for the two bisection sides.
   Warm-starting from the closest-budget solution halves the required
   inner passes after the first few outer iterations.

Derivation of per-coordinate minimiser
───────────────────────────────────────
Define (for fixed t and all other variables):
  z_t⁻  = cumY[i, t−1]   (prefix sum, doesn't include y[i,t])
  R_i   = cumY[i, C_i−1] − cumY[i, t]   (suffix sum beyond t)
  W_i   = Σ_{s≥t} exp(cumY[i,s] − cumY[i,t]) · mask[i,s]
        = N · suffix_t[i] · exp(−cumY[i,t])

Then: A_i = (1/N)·exp(−z_t⁻ − R_i)
      B_i = (λ/N)·W_i·exp(z_t⁻)

u* = 0.5·log(A_i/B_i)
   = 0.5·(−z_t⁻ − R_i − log λ − log W_i − z_t⁻)
   = y[i,t] − 0.5·(log λ + log N + log suffix_t[i] + cumY[i, C_i−1])
"""

from __future__ import annotations
import numpy as np
import torch

try:
    from numba import njit
except ImportError:  # pragma: no cover - exercised in minimal installations
    njit = None

if njit is None:  # pragma: no cover - exercised in minimal installations
    def _compile_if_available(**_kwargs):
        return lambda function: function
else:
    _compile_if_available = njit


def _generalized_pav(
        alpha: np.ndarray,
        beta: np.ndarray,
        lower: np.ndarray,
        ordered_scores: np.ndarray,
) -> np.ndarray:
    """Exact ordered minimizer of ``sum(alpha*e^-u + beta*e^u)``.

    This is the numeric-stack equivalent of the former dictionary-based PAV
    loop in :func:`solve_exact_fast`.  Keeping one preallocated array per
    block field avoids allocating, hashing, popping, and rebuilding Python
    dictionaries in every coordinate update.  The grouping, block-value
    formula, merge tolerance, and left-to-right reduction order are unchanged.
    """
    size = len(alpha)
    if not (
            len(beta) == size
            and len(lower) == size
            and len(ordered_scores) == size
    ):
        raise ValueError("All generalized-PAV inputs must have equal length.")
    if size == 0:
        return np.empty(0, dtype=np.float64)

    starts = np.empty(size, dtype=np.intp)
    stops = np.empty(size, dtype=np.intp)
    alpha_sums = np.empty(size, dtype=np.float64)
    beta_sums = np.empty(size, dtype=np.float64)
    lower_maxima = np.empty(size, dtype=np.float64)
    values = np.empty(size, dtype=np.float64)
    block_count = 0
    start = 0

    while start < size:
        stop = start + 1
        while (
                stop < size
                and ordered_scores[stop] == ordered_scores[start]
        ):
            stop += 1

        block_alpha = float(alpha[start:stop].sum())
        block_beta = float(beta[start:stop].sum())
        block_lower = float(lower[start:stop].max())
        if block_alpha <= 0:
            raw_value = -np.inf
        else:
            raw_value = 0.5 * (
                np.log(block_alpha) - np.log(block_beta)
            )
        block_value = min(0.0, max(block_lower, raw_value))

        starts[block_count] = start
        stops[block_count] = stop
        alpha_sums[block_count] = block_alpha
        beta_sums[block_count] = block_beta
        lower_maxima[block_count] = block_lower
        values[block_count] = block_value
        block_count += 1

        while (
                block_count > 1
                and values[block_count - 2]
                > values[block_count - 1] + 1e-12
        ):
            left = block_count - 2
            right = block_count - 1
            stops[left] = stops[right]
            alpha_sums[left] += alpha_sums[right]
            beta_sums[left] += beta_sums[right]
            lower_maxima[left] = max(
                lower_maxima[left],
                lower_maxima[right],
            )
            if alpha_sums[left] <= 0:
                raw_value = -np.inf
            else:
                raw_value = 0.5 * (
                    np.log(alpha_sums[left])
                    - np.log(beta_sums[left])
                )
            values[left] = min(
                0.0,
                max(lower_maxima[left], raw_value),
            )
            block_count -= 1
        start = stop

    result = np.empty(size, dtype=np.float64)
    for block in range(block_count):
        result[starts[block]:stops[block]] = values[block]
    return result


@_compile_if_available(cache=True, nogil=True)
def _numpy_pairwise_sum(
        values: np.ndarray,
        start: int,
        stop: int,
) -> float:
    """Reproduce NumPy's contiguous float64 reduction order.

    NumPy uses eight accumulators for arrays up to 128 elements and recursively
    bisects larger arrays on an eight-element boundary.  Mirroring that order
    lets the compiled tied-score path remain bitwise identical to
    ``values[start:stop].sum()`` instead of merely numerically close.
    """
    size = stop - start
    if size < 8:
        result = -0.0
        for index in range(start, stop):
            result += values[index]
        return result
    if size <= 128:
        r0 = values[start]
        r1 = values[start + 1]
        r2 = values[start + 2]
        r3 = values[start + 3]
        r4 = values[start + 4]
        r5 = values[start + 5]
        r6 = values[start + 6]
        r7 = values[start + 7]
        offset = 8
        while offset <= size - 8:
            r0 += values[start + offset]
            r1 += values[start + offset + 1]
            r2 += values[start + offset + 2]
            r3 += values[start + offset + 3]
            r4 += values[start + offset + 4]
            r5 += values[start + offset + 5]
            r6 += values[start + offset + 6]
            r7 += values[start + offset + 7]
            offset += 8
        result = (
            ((r0 + r1) + (r2 + r3))
            + ((r4 + r5) + (r6 + r7))
        )
        while offset < size:
            result += values[start + offset]
            offset += 1
        return result

    left_size = size // 2
    left_size -= left_size % 8
    return (
        _numpy_pairwise_sum(values, start, start + left_size)
        + _numpy_pairwise_sum(values, start + left_size, stop)
    )


@_compile_if_available(cache=True, nogil=True)
def _generalized_pav_compiled(
        alpha: np.ndarray,
        beta: np.ndarray,
        lower: np.ndarray,
        ordered_scores: np.ndarray,
) -> np.ndarray:
    """Compiled numeric-stack PAV with exact tied-score reductions.

    When Numba is unavailable the optional decorator leaves this as ordinary
    Python, so solver correctness does not depend on the acceleration package.
    """
    size = len(alpha)
    starts = np.empty(size, dtype=np.int64)
    stops = np.empty(size, dtype=np.int64)
    alpha_sums = np.empty(size, dtype=np.float64)
    beta_sums = np.empty(size, dtype=np.float64)
    lower_maxima = np.empty(size, dtype=np.float64)
    values = np.empty(size, dtype=np.float64)
    block_count = 0

    start = 0
    while start < size:
        stop = start + 1
        while (
                stop < size
                and ordered_scores[stop] == ordered_scores[start]
        ):
            stop += 1
        block_alpha = _numpy_pairwise_sum(alpha, start, stop)
        block_beta = _numpy_pairwise_sum(beta, start, stop)
        block_lower = lower[start]
        for row in range(start + 1, stop):
            block_lower = max(block_lower, lower[row])
        if block_alpha <= 0:
            raw_value = -np.inf
        else:
            raw_value = 0.5 * (
                np.log(block_alpha) - np.log(block_beta)
            )
        block_value = min(0.0, max(block_lower, raw_value))

        starts[block_count] = start
        stops[block_count] = stop
        alpha_sums[block_count] = block_alpha
        beta_sums[block_count] = block_beta
        lower_maxima[block_count] = block_lower
        values[block_count] = block_value
        block_count += 1

        while (
                block_count > 1
                and values[block_count - 2]
                > values[block_count - 1] + 1e-12
        ):
            left = block_count - 2
            right = block_count - 1
            stops[left] = stops[right]
            alpha_sums[left] += alpha_sums[right]
            beta_sums[left] += beta_sums[right]
            lower_maxima[left] = max(
                lower_maxima[left],
                lower_maxima[right],
            )
            if alpha_sums[left] <= 0:
                raw_value = -np.inf
            else:
                raw_value = 0.5 * (
                    np.log(alpha_sums[left])
                    - np.log(beta_sums[left])
                )
            values[left] = min(
                0.0,
                max(lower_maxima[left], raw_value),
            )
            block_count -= 1
        start = stop

    result = np.empty(size, dtype=np.float64)
    for block in range(block_count):
        for row in range(starts[block], stops[block]):
            result[row] = values[block]
    return result


def solve_time_only_cumulative_policy(
        lengths: torch.Tensor | np.ndarray,
        budget_per_sample: float,
        objective_weights: torch.Tensor | np.ndarray,
        width: int,
        terminal_pi_min: float,
        tolerance: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Solve the exact deployable time-only target-variance policy.

    Let ``r_t`` be the cumulative probability of reaching interaction ``t``.
    With endpoint lengths ``L_i`` and fixed nonnegative weights ``W_i``, the
    problem is

        min_r mean_i W_i / r[L_i - 1]
        s.t. mean_i sum_{t < L_i} r_t <= budget,
             1 >= r_0 >= ... >= r_{width-1} >= terminal_pi_min.

    Grouping endpoints gives ``sum_t h_t / r_t`` with linear cost
    ``sum_t s_t r_t``.  Pool-adjacent-violators on ``sqrt(h_t / s_t)``
    enforces temporal monotonicity, and a scalar dual bisection meets the
    budget.  The resulting class contains every constant-per-step Random
    policy but deploys by direct time lookup, without a projection model. If
    the objective has a zero-weight tail, the optimum can be flat before the
    budget is exhausted; the returned ``direct_time_budget_boundary`` then
    records ``"objective_plateau"`` rather than claiming budget equality.
    """
    if width <= 0:
        raise ValueError("`width` must be positive.")
    if not 0 < terminal_pi_min <= 1:
        raise ValueError("`terminal_pi_min` must lie in (0, 1].")
    lengths_np = np.asarray(
        lengths.detach().cpu() if torch.is_tensor(lengths) else lengths,
        dtype=np.int64,
    ).reshape(-1)
    weights_np = np.asarray(
        (
            objective_weights.detach().cpu()
            if torch.is_tensor(objective_weights)
            else objective_weights
        ),
        dtype=np.float64,
    ).reshape(-1)
    if len(lengths_np) != len(weights_np):
        raise ValueError("`lengths` and `objective_weights` must agree.")
    if np.any(lengths_np < 0) or np.any(lengths_np > width):
        raise ValueError(f"`lengths` must lie in [0, {width}].")
    if not np.all(np.isfinite(weights_np)) or np.any(weights_np < 0):
        raise ValueError("`objective_weights` must be finite and nonnegative.")
    if not np.isfinite(budget_per_sample) or budget_per_sample < 0:
        raise ValueError("`budget_per_sample` must be finite and nonnegative.")

    n = len(lengths_np)
    if n == 0:
        raise ValueError("At least one Phase-I row is required.")
    time = np.arange(width)
    active = time[None, :] < lengths_np[:, None]
    survival_coefficients = active.mean(axis=0).astype(np.float64)
    endpoint_coefficients = np.zeros(width, dtype=np.float64)
    positive_length = lengths_np > 0
    np.add.at(
        endpoint_coefficients,
        lengths_np[positive_length] - 1,
        weights_np[positive_length] / n,
    )
    last_active = int(np.count_nonzero(survival_coefficients > 0))
    maximum_cost = float(survival_coefficients.sum())
    minimum_cost = float(
        terminal_pi_min * survival_coefficients.sum()
    )
    if budget_per_sample < minimum_cost - tolerance:
        raise ValueError(
            "Time-only policy budget is infeasible under the terminal floor: "
            f"target={budget_per_sample}, minimum={minimum_cost}."
        )

    # PAV blocks for a non-increasing sequence.  The unconstrained scale of a
    # block is sqrt(sum(h) / sum(s)); the common dual multiplier is applied
    # only after pooling and therefore cannot change the block ordering.
    blocks = []
    for step in range(last_active):
        block = {
            "start": step,
            "stop": step + 1,
            "h": float(endpoint_coefficients[step]),
            "s": float(survival_coefficients[step]),
        }
        block["base"] = np.sqrt(block["h"] / block["s"])
        blocks.append(block)
        while (
                len(blocks) > 1
                and blocks[-2]["base"] < blocks[-1]["base"] - 1e-15
        ):
            right = blocks.pop()
            left = blocks.pop()
            merged = {
                "start": left["start"],
                "stop": right["stop"],
                "h": left["h"] + right["h"],
                "s": left["s"] + right["s"],
            }
            merged["base"] = np.sqrt(merged["h"] / merged["s"])
            blocks.append(merged)

    def schedule(dual):
        cumulative = np.ones(width, dtype=np.float64)
        scale = np.sqrt(max(dual, np.finfo(np.float64).tiny))
        for block in blocks:
            value = np.clip(
                block["base"] / scale,
                terminal_pi_min,
                1.0,
            )
            cumulative[block["start"]:block["stop"]] = value
        if last_active:
            cumulative[last_active:] = cumulative[last_active - 1]
        return cumulative

    if budget_per_sample >= maximum_cost - tolerance:
        cumulative = np.ones(width, dtype=np.float64)
        dual = 0.0
        achieved_cost = maximum_cost
        boundary = "maximum"
    else:
        low, high = 1e-16, 1e16
        low_schedule = schedule(low)
        high_schedule = schedule(high)
        low_cost = float(np.dot(survival_coefficients, low_schedule))
        high_cost = float(np.dot(survival_coefficients, high_schedule))
        if high_cost > budget_per_sample + tolerance:
            raise RuntimeError(
                "Dual bracket did not reach the minimum-cost policy."
            )
        for _ in range(100):
            mid = np.sqrt(low * high)
            mid_schedule = schedule(mid)
            mid_cost = float(
                np.dot(survival_coefficients, mid_schedule)
            )
            if mid_cost > budget_per_sample:
                low = mid
            else:
                high = mid
                high_schedule = mid_schedule
                high_cost = mid_cost
            if abs(mid_cost - budget_per_sample) <= tolerance * max(
                    budget_per_sample,
                    1.0,
            ):
                high_schedule = mid_schedule
                high_cost = mid_cost
                high = mid
                break
        cumulative = high_schedule
        achieved_cost = high_cost
        dual = high
        boundary = None
        if achieved_cost < budget_per_sample - tolerance * max(
                budget_per_sample,
                1.0,
        ):
            boundary = "objective_plateau"

    previous = np.concatenate([[1.0], cumulative[:-1]])
    conditionals = np.clip(
        cumulative / np.maximum(previous, np.finfo(np.float64).tiny),
        0.0,
        1.0,
    )
    terminal = np.ones(n, dtype=np.float64)
    terminal[positive_length] = cumulative[lengths_np[positive_length] - 1]
    objective = float(np.mean(weights_np / terminal))
    return conditionals, cumulative, {
        "direct_time_dual": dual,
        "direct_time_expected_cost": achieved_cost,
        "direct_time_budget_slack": (
            budget_per_sample - achieved_cost
        ),
        "direct_time_minimum_cost": minimum_cost,
        "direct_time_maximum_cost": maximum_cost,
        "direct_time_objective": objective,
        "direct_time_pav_blocks": len(blocks),
        "direct_time_budget_boundary": boundary,
    }


def solve_exact_fast(
        S: torch.Tensor,
        C: torch.Tensor,
        B_bar: float,
        objective_weights: torch.Tensor | np.ndarray | None = None,
        objective_masses: torch.Tensor | np.ndarray | None = None,
        terminal_pi_min: float | None = None,
        max_outer: int = 60,
        max_inner: int = 30,
        tol: float = 1e-9,
        verbose: bool = True,
) -> np.ndarray:
    """Solve the score-monotone acquisition problem in log probabilities.

    ``C[i]`` is the strict active length ``min(T_i, q_i)``. With cumulative
    reach ``rho[i,t] = prod_{s<=t} P[i,s]``, ``objective_masses`` solves

        mean_i sum_t objective_masses[i,t] / rho[i,t].

    The omitted ``-objective_masses`` term is constant. ``objective_weights``
    is the backward-compatible terminal-event shorthand: weight ``i`` is
    placed at time ``C[i]-1``. Passing binary target indicators therefore
    minimizes the usual terminal-event conditional-variance proxy, while soft
    prefix masses implement its Rao--Blackwellized/model-integrated version.

    A terminal-propensity floor is deliberately not solved here: it couples
    coordinates and the previous coordinate-descent implementation could stop
    at a path-dependent, severely budget-slack solution. Positivity for the
    deployed policy is instead provided by the explicit exploration mixture
    applied after projection.
    """
    if not torch.is_tensor(S) or not torch.is_tensor(C):
        raise TypeError("`S` and `C` must be torch tensors.")
    S_np = np.asarray(S.detach().cpu(), dtype=np.float64)
    C_np = np.asarray(C.detach().cpu(), dtype=np.int64)
    if S_np.ndim != 2:
        raise ValueError(f"`S` must have shape (N,L); got {S_np.shape}.")
    n, width = S_np.shape
    if C_np.shape != (n,):
        raise ValueError(f"`C` must have shape {(n,)}; got {C_np.shape}.")
    if np.any(C_np < 0) or np.any(C_np > width):
        raise ValueError(f"`C` must lie in [0, {width}].")
    if not np.isfinite(B_bar) or B_bar < 0:
        raise ValueError(f"`B_bar` must be finite and nonnegative; got {B_bar}.")
    time = np.arange(width)
    mask = (time[None, :] < C_np[:, None]).astype(np.float64)

    if objective_weights is not None and objective_masses is not None:
        raise ValueError(
            "Pass either `objective_weights` or `objective_masses`, not both."
        )
    if objective_masses is not None:
        if torch.is_tensor(objective_masses):
            objective_masses = objective_masses.detach().cpu().numpy()
        masses = np.asarray(objective_masses, dtype=np.float64)
        if masses.shape != (n, width):
            raise ValueError(
                "`objective_masses` must have shape "
                f"{(n, width)}; got {masses.shape}."
            )
        if not np.all(np.isfinite(masses)) or np.any(masses < 0):
            raise ValueError(
                "`objective_masses` must be finite and nonnegative."
            )
        masses = masses * mask
    else:
        if objective_weights is None:
            weights = np.ones(n, dtype=np.float64)
        else:
            if torch.is_tensor(objective_weights):
                objective_weights = objective_weights.detach().cpu().numpy()
            weights = np.asarray(objective_weights, dtype=np.float64)
            if weights.shape != (n,):
                raise ValueError(
                    f"`objective_weights` must have shape {(n,)}; got {weights.shape}."
                )
            if not np.all(np.isfinite(weights)) or np.any(weights < 0):
                raise ValueError("`objective_weights` must be finite and nonnegative.")
        masses = np.zeros((n, width), dtype=np.float64)
        positive_length = C_np > 0
        masses[
            np.flatnonzero(positive_length),
            C_np[positive_length] - 1,
        ] = weights[positive_length]

    if terminal_pi_min is not None:
        raise NotImplementedError(
            "A terminal floor couples coordinates and is not solved reliably "
            "by coordinate descent. Enforce positivity in the deployed policy "
            "with an explicit exploration mixture instead."
        )
    maximum_budget = float(np.mean(C_np))
    if B_bar >= maximum_budget - tol:
        return mask.copy()

    ordered_rows_by_time = []
    for t in range(width):
        valid = np.flatnonzero(mask[:, t])
        if len(valid):
            order = np.argsort(S_np[valid, t], kind="stable")
            rows = valid[order]
            ordered_rows_by_time.append(rows)
        else:
            ordered_rows_by_time.append(None)

    if np.all(masses == 0):
        y_minimum = np.zeros((n, width), dtype=np.float64)
        y_minimum[C_np > 0, 0] = -700.0
        y_minimum[mask == 0] = -1e9
        return np.exp(y_minimum) * mask
    p_initial = float(np.clip(B_bar / (1.0 + B_bar), 1e-4, 1 - 1e-4))
    y_initial = np.full((n, width), np.log(p_initial), dtype=np.float64)
    y_initial[mask == 0] = -1e9

    def compute_budget(y):
        cumulative = np.exp(np.clip(np.cumsum(y, axis=1), -700, 0))
        return float(np.mean(np.sum(cumulative * mask, axis=1)))

    def compute_weighted_objective(y):
        cumulative_y = np.cumsum(y, axis=1)
        inverse_reach = np.exp(np.clip(-cumulative_y, 0, 700))
        return float(np.mean(np.sum(
            masses * (inverse_reach - 1.0),
            axis=1,
        )))

    def inner_solve(lam, y_start):
        y = y_start.copy()
        log_lam = np.log(max(lam, 1e-300))
        log_n = np.log(n)
        for _ in range(max_inner):
            previous = y.copy()
            cumulative_y = np.cumsum(y, axis=1)
            for t, rows in enumerate(ordered_rows_by_time):
                if rows is None:
                    continue
                old_column = y[:, t].copy()
                suffix = np.sum(
                    np.exp(np.clip(cumulative_y[rows, t:], -700, 0))
                    * mask[rows, t:],
                    axis=1,
                )
                inverse_objective_suffix = np.sum(
                    masses[rows, t:]
                    * np.exp(np.clip(-cumulative_y[rows, t:], 0, 700)),
                    axis=1,
                )
                alpha = (
                    inverse_objective_suffix
                    * np.exp(old_column[rows])
                    / n
                )
                beta = np.exp(np.clip(
                    log_lam
                    - log_n
                    + np.log(np.maximum(suffix, 1e-300))
                    - old_column[rows],
                    -700,
                    700,
                ))
                lower = np.full(len(rows), -700.0, dtype=np.float64)
                y[rows, t] = _generalized_pav_compiled(
                    alpha,
                    beta,
                    lower,
                    S_np[rows, t],
                )
                y[mask[:, t] == 0, t] = -1e9
                delta = y[:, t] - old_column
                cumulative_y[:, t:] += delta[:, None]
            if np.max(np.abs((y - previous)[mask > 0])) < 5e-4:
                break
        return y

    lambda_low, lambda_high = 1e-12, 1e14
    y_low = y_initial.copy()
    y_high = y_initial.copy()
    budget_low = compute_budget(y_low)
    budget_high = budget_low
    y_best = None
    best_budget = np.nan
    best_objective = np.inf
    if verbose:
        print(
            f"  p_init={p_initial:.4f} initial_budget={budget_low:.4f} "
            f"target={B_bar}"
        )

    for outer in range(max_outer):
        lambda_mid = float(np.sqrt(lambda_low * lambda_high))
        warm = (
            y_low
            if abs(budget_low - B_bar) < abs(budget_high - B_bar)
            else y_high
        )
        y_mid = inner_solve(lambda_mid, warm)
        budget_mid = compute_budget(y_mid)
        if budget_mid > B_bar:
            lambda_low, y_low, budget_low = lambda_mid, y_mid.copy(), budget_mid
        else:
            lambda_high, y_high, budget_high = lambda_mid, y_mid.copy(), budget_mid
            objective_mid = compute_weighted_objective(y_mid)
            objective_tolerance = tol * max(
                abs(best_objective) if np.isfinite(best_objective) else 1.0,
                1.0,
            )
            if (
                    objective_mid < best_objective - objective_tolerance
                    or (
                        abs(objective_mid - best_objective)
                        <= objective_tolerance
                        and (
                            not np.isfinite(best_budget)
                            or abs(budget_mid - B_bar)
                            < abs(best_budget - B_bar)
                        )
                    )
            ):
                y_best = y_mid.copy()
                best_budget = budget_mid
                best_objective = objective_mid
        if verbose:
            print(
                f"  outer {outer + 1:3d} lambda={lambda_mid:.4e} "
                f"budget={budget_mid:.6f}/{B_bar}"
            )
        if abs(budget_mid - B_bar) <= tol * max(B_bar, 1.0):
            break

    if y_best is None:
        raise RuntimeError(
            "The dual search did not find a numerically budget-feasible "
            "candidate."
        )
    final_probabilities = np.exp(y_best) * mask
    final_budget = compute_budget(y_best)
    monotone = all(
        not np.any(
            final_probabilities[rows, t][:-1]
            > final_probabilities[rows, t][1:] + 1e-9
        )
        for t, rows in enumerate(ordered_rows_by_time)
        if rows is not None
    )
    terminal_log = np.sum(np.where(mask > 0, y_best, 0.0), axis=1)
    terminal_probability = np.exp(np.clip(terminal_log, -700, 0))
    weighted_objective = compute_weighted_objective(y_best)
    if verbose:
        print(
            f"\nBudget: {final_budget:.10f} <= {B_bar} "
            f"{'OK' if final_budget <= B_bar + 1e-8 else 'VIOLATED'}"
        )
        print(f"Monotonicity: {'OK' if monotone else 'VIOLATED'}")
        print(f"Weighted objective: {weighted_objective:.6f}")
    assert final_budget <= B_bar + 1e-8
    assert monotone
    return final_probabilities


def solve_binned_deployable_policy(
        validation_scores: torch.Tensor,
        deployment_scores: torch.Tensor,
        validation_lengths: torch.Tensor,
        budget_per_sample: float,
        objective_weights: torch.Tensor | np.ndarray | None,
        n_bins: int,
        objective_masses: torch.Tensor | np.ndarray | None = None,
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, dict]:
    """Optimize a score-bin policy that deploys without regression.

    Phase-I scores are quantized independently at each time.  Equal bin IDs are
    tied by ``solve_exact_fast``'s equal-score PAV blocks, producing one
    continuation probability per time/bin.  Phase II then uses direct bin
    lookup.  This removes the high-variance oracle-to-Platt regression while
    retaining a small, score-adaptive policy class.
    """
    if n_bins < 1:
        raise ValueError("`n_bins` must be positive.")
    if validation_scores.ndim != 2 or deployment_scores.ndim != 2:
        raise ValueError("Score matrices must be two-dimensional.")
    if validation_scores.shape[1] != deployment_scores.shape[1]:
        raise ValueError("Validation and deployment score widths must agree.")
    n_validation, width = validation_scores.shape
    lengths_np = np.asarray(
        validation_lengths.detach().cpu(),
        dtype=np.int64,
    ).reshape(-1)
    if len(lengths_np) != n_validation:
        raise ValueError(
            "`validation_lengths` must have one value per validation row."
        )
    validation_np = np.asarray(
        validation_scores.detach().cpu(),
        dtype=np.float64,
    )
    deployment_np = np.asarray(
        deployment_scores.detach().cpu(),
        dtype=np.float64,
    )
    validation_bins = np.zeros((n_validation, width), dtype=np.int64)
    deployment_bins = np.zeros(
        (len(deployment_np), width),
        dtype=np.int64,
    )
    cutpoints = []
    empty_training_steps = 0
    for step in range(width):
        active = lengths_np > step
        if not np.any(active):
            cutpoints.append(np.empty(0, dtype=np.float64))
            empty_training_steps += 1
            continue
        edges = np.unique(np.quantile(
            validation_np[active, step],
            np.arange(1, n_bins, dtype=np.float64) / n_bins,
        ))
        cutpoints.append(edges)
        validation_bins[:, step] = np.searchsorted(
            edges,
            validation_np[:, step],
            side="right",
        )
        deployment_bins[:, step] = np.searchsorted(
            edges,
            deployment_np[:, step],
            side="right",
        )

    binned_scores = torch.as_tensor(
        validation_bins,
        dtype=torch.float64,
        device=validation_scores.device,
    )
    optimal = solve_exact_fast(
        binned_scores,
        validation_lengths,
        budget_per_sample,
        objective_weights=objective_weights,
        objective_masses=objective_masses,
        terminal_pi_min=None,
        verbose=False,
    )
    active_mask = (
        np.arange(width)[None, :] < lengths_np[:, None]
    )
    optimal[~active_mask] = 1.0

    table = np.ones((n_bins, width), dtype=np.float64)
    empty_cells = 0
    maximum_within_bin_spread = 0.0
    for step in range(width):
        active = active_mask[:, step]
        if not np.any(active):
            continue
        observed_bins = []
        for bin_index in range(n_bins):
            rows = active & (validation_bins[:, step] == bin_index)
            if not np.any(rows):
                empty_cells += 1
                continue
            values = optimal[rows, step]
            table[bin_index, step] = float(np.mean(values))
            maximum_within_bin_spread = max(
                maximum_within_bin_spread,
                float(np.ptp(values)),
            )
            observed_bins.append(bin_index)
        if observed_bins:
            for bin_index in range(n_bins):
                if bin_index in observed_bins:
                    continue
                nearest = min(
                    observed_bins,
                    key=lambda observed: abs(observed - bin_index),
                )
                table[bin_index, step] = table[nearest, step]
            table[:, step] = np.maximum.accumulate(table[:, step])

    validation_policy = table[
        validation_bins,
        np.arange(width)[None, :],
    ]
    deployment_policy = table[
        deployment_bins,
        np.arange(width)[None, :],
    ]
    return (
        optimal,
        torch.as_tensor(
            validation_policy,
            dtype=torch.float64,
            device=validation_scores.device,
        ),
        torch.as_tensor(
            deployment_policy,
            dtype=torch.float64,
            device=deployment_scores.device,
        ),
        {
            "direct_score_bin_count": n_bins,
            "direct_score_bin_empty_cells": empty_cells,
            "direct_score_bin_empty_training_steps": empty_training_steps,
            "direct_score_bin_max_within_bin_probability_spread": (
                maximum_within_bin_spread
            ),
            "direct_score_bin_unique_cutpoints_mean": float(np.mean([
                len(edges) for edges in cutpoints
            ])),
        },
    )


def fit_isotonic_maps(S_cal, P_opt, C_cal):
    """Fit one isotonic map per time step in log-space for generalisation."""
    from sklearn.isotonic import IsotonicRegression as _IR
    S_cal = np.asarray(S_cal, dtype=np.float64)
    Y_opt = np.log(np.clip(np.asarray(P_opt, dtype=np.float64), 1e-15, 1.0))
    C_cal = np.asarray(C_cal, dtype=np.int64)
    N, L = S_cal.shape
    mask_np = np.arange(L)[None, :] < C_cal[:, None]
    models = []
    for t in range(L):
        valid = np.where(mask_np[:, t])[0]
        if len(valid) >= 2:
            ir = _IR(increasing=True, out_of_bounds="clip")
            ir.fit(S_cal[valid, t], Y_opt[valid, t])
            models.append(ir)
        else:
            models.append(None)
    return models


def predict_isotonic_maps(S_test, C_test, models):
    """Apply fitted M_t maps to new scores. Returns P_test (N_test, L)."""
    S_test = np.asarray(S_test, dtype=np.float64)
    C_test = np.asarray(C_test, dtype=np.int64)
    N, L = S_test.shape
    mask_np = np.arange(L)[None, :] < C_test[:, None]
    Y_out = np.full((N, L), -1e9, dtype=np.float64)
    for t, ir in enumerate(models):
        if ir is None:
            continue
        valid = np.where(mask_np[:, t])[0]
        if len(valid) > 0:
            Y_out[valid, t] = np.minimum(ir.predict(S_test[valid, t]), 0.0)
    return np.exp(Y_out) * mask_np


if __name__ == "__main__":
    import time

    rng = np.random.default_rng(42)
    N, L = 100, 200
    S = rng.standard_normal((N, L))
    C = rng.integers(L // 2, L + 1, size=N)
    B_bar = 8.0
    print(f"Problem: N={N}, L={L}, B_bar={B_bar}\n")
    t0 = time.perf_counter()
    P = solve_exact_fast(S, C, B_bar, verbose=True)
    print(f"\nWall-clock: {time.perf_counter() - t0:.3f}s")
