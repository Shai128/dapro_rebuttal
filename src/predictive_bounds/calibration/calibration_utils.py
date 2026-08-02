import numpy as np
import torch
from scipy.optimize import bisect


def indexed_tensor_metrics(
        tensors_by_prefix: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Convert equal-length metric vectors to indexed Python scalars at once.

    Calling ``.item()`` once per target forces one host synchronization per
    scalar on CUDA.  Stacking these small vectors first preserves the values
    while requiring a single device-to-host transfer for the whole metric
    block.
    """
    if not tensors_by_prefix:
        return {}
    items = list(tensors_by_prefix.items())
    vectors = []
    expected_length = None
    expected_device = None
    for prefix, values in items:
        if not isinstance(prefix, str) or not prefix:
            raise ValueError("Metric prefixes must be non-empty strings.")
        if not torch.is_tensor(values) or values.ndim != 1:
            raise ValueError(
                f"Metric vector {prefix!r} must be a one-dimensional tensor."
            )
        if not values.is_floating_point():
            raise ValueError(
                f"Metric vector {prefix!r} must use a floating dtype."
            )
        if expected_length is None:
            expected_length = len(values)
            expected_device = values.device
        elif len(values) != expected_length:
            raise ValueError("All indexed metric vectors must have equal length.")
        elif values.device != expected_device:
            raise ValueError("All indexed metric vectors must share one device.")
        vectors.append(values.detach())

    rows = torch.stack(vectors, dim=0).cpu().tolist()
    return {
        f"{prefix}_{index}": float(value)
        for (prefix, _), row in zip(items, rows)
        for index, value in enumerate(row)
    }


def select_calibration_positions(
        miscoverage: torch.Tensor,
        target_taus: torch.Tensor,
) -> torch.Tensor:
    """Select the largest initial candidate prefix strictly below each target.

    This is the common LPB selector used throughout the calibration code.  The
    strict comparison intentionally matches the historical implementation:
    for each target alpha, return the last candidate before the first
    miscoverage value that is not smaller than alpha.  Restricting the choice
    to an initial prefix is important when finite-sample estimates are not
    perfectly monotone.
    """
    if not torch.is_tensor(miscoverage) or miscoverage.ndim != 1:
        raise ValueError("`miscoverage` must be a one-dimensional torch tensor.")
    if not torch.is_tensor(target_taus) or target_taus.ndim != 1:
        raise ValueError("`target_taus` must be a one-dimensional torch tensor.")
    if len(miscoverage) == 0:
        raise ValueError("`miscoverage` must contain at least one candidate.")
    if len(target_taus) == 0:
        raise ValueError("`target_taus` must contain at least one target.")
    target_taus = target_taus.to(
        device=miscoverage.device,
        dtype=miscoverage.dtype,
    )
    feasible = target_taus.unsqueeze(0) - miscoverage.unsqueeze(1) > 0
    prefix_scores = torch.where(
        feasible,
        torch.ones_like(feasible, dtype=miscoverage.dtype),
        torch.full(
            feasible.shape,
            -torch.inf,
            dtype=miscoverage.dtype,
            device=miscoverage.device,
        ),
    ).cumsum(dim=0)
    return prefix_scores.argmax(dim=0)


def quantiles_to_interaction_counts(
        quantiles: torch.Tensor | np.ndarray,
        width: int,
        upper_bound: float | None = None,
) -> torch.Tensor | np.ndarray:
    """Convert saved zero-based quantile indices to interaction counts.

    Event times in the repository are one-based counts in ``[1, width]``,
    whereas the survival quantile routines return zero-based grid indices.
    The maximum-horizon sentinel is already ``width``, so clipping after the
    increment maps both the last grid point and the sentinel to ``width``.
    """
    if width <= 0:
        raise ValueError(f"`width` must be positive; got {width}.")
    cap = float(width)
    if upper_bound is not None:
        if not np.isfinite(upper_bound) or upper_bound <= 0:
            raise ValueError(
                f"`upper_bound` must be finite and positive; got {upper_bound}."
            )
        cap = min(cap, float(upper_bound))
    if torch.is_tensor(quantiles):
        return torch.clamp(quantiles + 1, max=cap)
    return np.minimum(np.asarray(quantiles) + 1, cap)


def get_prior(q: torch.Tensor, taus_range: torch.Tensor, tau_prior: float):
    idx = abs(taus_range - tau_prior).argmin()
    return q[..., idx]

def sample_calibration_set(
        prior_quantile_est,
        C_probs,
        num_attempts
):
    # Determine the number of samples per instance.
    # Note: C is computed as a (n, 1) array. The generic runner accepts per-prompt max attempts.
    C = torch.where(torch.rand(len(prior_quantile_est), device=C_probs.device) < C_probs, prior_quantile_est,
                    0).int().reshape(
        -1, 1)
    num_attempts = num_attempts.reshape(-1, 1)
    T_tilde = torch.minimum(num_attempts, C)
    return T_tilde, C


def sample_calibration_set_new(
        prior_quantile_est,
        C_probs,
        num_attempts
):
    # Determine the number of samples per instance.
    # Note: C is computed as a (n, 1) array. The generic runner accepts per-prompt max attempts.
    C = torch.where(torch.rand(len(prior_quantile_est), device=C_probs.device) < C_probs, prior_quantile_est,
                    0).int().reshape(
        -1, 1)
    num_attempts = num_attempts.reshape(-1, 1)
    T_tilde = torch.minimum(num_attempts, C)
    return T_tilde, C


def constraint_violation(lambda_val, w, b):
    """
    Computes the difference between the left-hand side of the constraint
    sum(w_i * p_i) and b, where p_i = min{1, 1/sqrt(lambda*w_i)}.
    """
    p = np.minimum(1, 1 / np.sqrt(lambda_val * w))
    return np.sum(w * p) - b


def solve_optimization(w, b, tol=1e-8):
    """
    Solves the optimization problem:
         min   sum(1/p_i)
         s.t.  sum(w_i * p_i) = b,   0 < p_i <= 1,
    by finding the Lagrange multiplier lambda such that the constraint holds.

    Parameters:
        w   : array-like, weights (all positive)
        b   : positive scalar, right-hand side of the constraint
        tol : tolerance for the bisection algorithm

    Returns:
        p   : optimal vector p*, where each p_i = min{1, 1/sqrt(lambda*w_i)}
        lambda_star : the Lagrange multiplier found
    """
    w = np.array(w, dtype=float)

    # Check feasibility: b must be no more than sum(w) since p_i <= 1.
    if b > np.sum(w):
        return np.ones_like(w), None

    # # Set a lower bound for lambda.
    # lambda_low = 1e-12
    # # Set an initial upper bound: choose lambda_high so that for every i, 1/sqrt(lambda_high*w_i) < 1.
    # # A sufficient condition is lambda_high > max(1/w).
    # lambda_high = max(1 / w) * 10.0
    #
    # # Ensure that our bounds bracket a zero of the function.
    # # When lambda is very small, p_i = 1 for each i, so the constraint is sum(w) - b (which is >= 0).
    # # We need constraint_violation(lambda_low, w, b) >= 0 and constraint_violation(lambda_high, w, b) <= 0.
    # f_low = constraint_violation(lambda_low, w, b)
    # f_high = constraint_violation(lambda_high, w, b)
    #
    # # Increase lambda_high until f_high is negative.
    # while f_high > 0:
    #     lambda_high *= 2
    #     f_high = constraint_violation(lambda_high, w, b)
    #
    # # Use bisection to find lambda such that the constraint is met.
    # lambda_star = bisect(constraint_violation, lambda_low, lambda_high, args=(w, b), xtol=tol)

    # With lambda_star in hand, compute the optimal p.
    lambda_star = (( (1/(b/len(w))) * np.sqrt(w).mean())) ** 2
    assert abs(lambda_star - (( (1 / b) * np.sqrt(w).sum() ) ** 2)) < 1e-5
    p_opt = np.minimum(1,1/ (( (1/(b/len(w))) * np.sqrt(w).mean()) * np.sqrt(w)))
    # p_opt = np.minimum(1, 1 / np.sqrt(lambda_star * w))
    return p_opt, lambda_star
