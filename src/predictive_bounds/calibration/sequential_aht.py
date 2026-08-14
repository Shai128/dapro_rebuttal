"""Sequential augmented Horvitz--Thompson estimators for event targets.

The acquisition policy and the estimator are deliberately separated.  An
allocator opts into one of these estimators by returning its executed
conditional continuation probabilities.  The prediction path may be
misspecified: telescoping to the resolved binary target is what gives design
unbiasedness.
"""

from __future__ import annotations

import torch


def _normalized_pmf(grid: torch.Tensor, current: int) -> torch.Tensor:
    pmf = grid[:, current, :].to(torch.float64).clamp_min(0.0)
    return pmf / pmf.sum(dim=1, keepdim=True).clamp_min(1e-15)


def lower_event_probability(
        pmf: torch.Tensor,
        horizons: torch.Tensor,
        *,
        strict: bool,
) -> torch.Tensor:
    """Return ``P(T<h)`` or ``P(T<=h)`` for row-specific integer ``h``."""
    values = pmf.to(torch.float64)
    h = horizons.to(device=values.device, dtype=torch.long).reshape(-1)
    if len(values) != len(h):
        raise ValueError("The PMF and horizon vectors must have equal length.")
    outcome_times = torch.arange(
        1, values.shape[1] + 1, device=values.device
    )[None, :]
    mask = outcome_times < h[:, None] if strict else outcome_times <= h[:, None]
    return (values * mask.to(values.dtype)).sum(dim=1)


def terminal_residual_lower_curve(
        event_times: torch.Tensor,
        candidates: torch.Tensor,
        acquisition_horizons: torch.Tensor,
        terminal_propensities: torch.Tensor,
        candidate_propensities: torch.Tensor,
        conditional_grid: torch.Tensor,
        *,
        strict: bool = True,
) -> torch.Tensor:
    """Terminal residual-AHT estimates of a lower-event candidate curve.

    For LPBs the target is ``1{T<f}`` (``strict=True``).  For a fixed metric
    horizon the same routine can estimate ``1{T<=f}``.  A positive event is
    resolved at its event time; a negative target is resolved by reaching the
    candidate horizon.
    """
    grid = conditional_grid.to(torch.float64)
    if grid.ndim != 3 or candidates.ndim != 2:
        raise ValueError("Expected an N-by-time PMF grid and N-by-J candidates.")
    n = len(grid)
    if len(candidates) != n:
        raise ValueError("Candidate and PMF rows must agree.")
    candidate_pi = candidate_propensities.to(
        device=grid.device, dtype=torch.float64
    )
    if candidate_pi.shape != candidates.shape:
        raise ValueError("Candidate propensities must match candidates.")
    terminal_pi = terminal_propensities.reshape(-1, 1).to(
        device=grid.device, dtype=torch.float64
    )
    if torch.any(candidate_pi <= 0) or torch.any(terminal_pi <= 0):
        raise ValueError("AHT propensities must be strictly positive.")

    times = event_times.reshape(-1, 1).to(grid.device)
    acquired = acquisition_horizons.reshape(-1, 1).to(grid.device)
    f = candidates.to(device=grid.device, dtype=torch.long)
    target = (times < f) if strict else (times <= f)
    positive_observed = target & (acquired >= times)
    negative_observed = (~target) & (acquired >= f)
    observed = positive_observed | negative_observed
    observation_pi = torch.where(target, terminal_pi, candidate_pi)

    pmf0 = _normalized_pmf(grid, 0)
    expanded = pmf0[:, None, :].expand(-1, f.shape[1], -1)
    outcome_times = torch.arange(
        1, pmf0.shape[1] + 1, device=grid.device
    )[None, None, :]
    mask = outcome_times < f[:, :, None] if strict else outcome_times <= f[:, :, None]
    model = (expanded * mask.to(expanded.dtype)).sum(dim=2)
    return model + observed.to(torch.float64) * (
        target.to(torch.float64) - model
    ) / observation_pi


def sequential_lower_curve(
        event_times: torch.Tensor,
        candidates: torch.Tensor,
        acquisition_horizons: torch.Tensor,
        continuation_probabilities: torch.Tensor,
        conditional_grid: torch.Tensor,
        *,
        strict: bool = True,
) -> torch.Tensor:
    """All-prefix sequential-AHT estimates for lower-event candidates."""
    grid = conditional_grid.to(torch.float64)
    conditionals = continuation_probabilities.to(
        device=grid.device, dtype=torch.float64
    )
    if grid.ndim != 3:
        raise ValueError("Sequential AHT requires a conditional-PMF grid.")
    n, width, _ = grid.shape
    if conditionals.shape != (n, width):
        raise ValueError("Continuation probabilities must match the PMF grid.")
    if torch.any(conditionals <= 0) or torch.any(conditionals > 1):
        raise ValueError("Continuation probabilities must lie in (0, 1].")
    if candidates.ndim != 2 or len(candidates) != n:
        raise ValueError("Candidates must have shape N-by-J.")

    times = event_times.reshape(-1).to(device=grid.device, dtype=torch.long)
    acquired = acquisition_horizons.reshape(-1).to(
        device=grid.device, dtype=torch.long
    )
    cumulative = conditionals.cumprod(dim=1)
    integer_horizons = torch.arange(1, width + 1, device=grid.device)

    def curve_at(current: int) -> torch.Tensor:
        pmf = _normalized_pmf(grid, current)
        cdf = pmf[:, :width].cumsum(dim=1)
        if strict:
            return torch.cat(
                [torch.zeros((n, 1), dtype=cdf.dtype, device=cdf.device),
                 cdf[:, :-1]],
                dim=1,
            )
        return cdf

    previous = curve_at(0)
    estimate = previous.clone()
    for turn in range(1, width + 1):
        active = times >= turn
        if not torch.any(active):
            break
        post = previous.clone()
        event_now = times == turn
        if torch.any(event_now):
            target_now = (
                turn < integer_horizons
                if strict
                else turn <= integer_horizons
            ).to(torch.float64)
            post[event_now] = target_now
        survived = times > turn
        if torch.any(survived):
            if turn < width:
                post[survived] = curve_at(turn)[survived]
            else:
                post[survived] = 0.0
        delta = post - previous
        reached = (acquired >= turn).to(torch.float64)
        estimate += (
            reached / cumulative[:, turn - 1]
        )[:, None] * delta
        previous[active] = post[active]

    f = candidates.to(device=grid.device, dtype=torch.long)
    indices = f.clamp(min=1, max=width) - 1
    return estimate.gather(1, indices)


def metric_aht_contributions(
        event_times: torch.Tensor,
        acquisition_horizons: torch.Tensor,
        terminal_propensities: torch.Tensor,
        conditional_grid: torch.Tensor,
        horizon: int,
        *,
        continuation_probabilities: torch.Tensor | None,
) -> torch.Tensor:
    """Return terminal- or sequential-AHT contributions for ``1{T<=M}``."""
    n = len(event_times)
    candidate = torch.full(
        (n, 1), int(horizon), dtype=torch.long,
        device=conditional_grid.device,
    )
    if continuation_probabilities is not None:
        return sequential_lower_curve(
            event_times,
            candidate,
            acquisition_horizons,
            continuation_probabilities,
            conditional_grid,
            strict=False,
        ).reshape(-1)
    candidate_pi = terminal_propensities.reshape(-1, 1)
    return terminal_residual_lower_curve(
        event_times,
        candidate,
        acquisition_horizons,
        terminal_propensities,
        candidate_pi,
        conditional_grid,
        strict=False,
    ).reshape(-1)


def metric_aht_path_variance(
        event_times: torch.Tensor,
        terminal_propensities: torch.Tensor,
        conditional_grid: torch.Tensor,
        horizon: int,
        *,
        continuation_probabilities: torch.Tensor | None,
) -> torch.Tensor:
    """Exact row-wise conditional acquisition variance of metric AHT."""
    grid = conditional_grid.to(torch.float64)
    times = event_times.reshape(-1).to(device=grid.device, dtype=torch.long)
    h = torch.full_like(times, int(horizon))
    target = (times <= h).to(torch.float64)
    previous = lower_event_probability(
        _normalized_pmf(grid, 0), h, strict=False
    )
    if continuation_probabilities is None:
        pi = terminal_propensities.reshape(-1).to(
            device=grid.device, dtype=torch.float64
        )
        return (target - previous).square() * (pi.reciprocal() - 1.0)

    conditionals = continuation_probabilities.to(
        device=grid.device, dtype=torch.float64
    )
    cumulative = conditionals.cumprod(dim=1)
    variance = torch.zeros_like(previous)
    for turn in range(1, int(horizon) + 1):
        active = times >= turn
        if not torch.any(active):
            break
        inverse_now = cumulative[:, turn - 1].reciprocal()
        inverse_before = (
            torch.ones_like(inverse_now)
            if turn == 1
            else cumulative[:, turn - 2].reciprocal()
        )
        variance += active.to(torch.float64) * (
            inverse_now - inverse_before
        ) * (target - previous).square()
        post = previous.clone()
        event_now = times == turn
        post[event_now] = 1.0
        survived = times > turn
        if torch.any(survived):
            if turn < int(horizon):
                next_h = torch.full_like(times, int(horizon))
                next_probability = lower_event_probability(
                    _normalized_pmf(grid, turn), next_h, strict=False
                )
                post[survived] = next_probability[survived]
            else:
                post[survived] = 0.0
        previous[active] = post[active]
    return variance


__all__ = [
    "lower_event_probability",
    "metric_aht_contributions",
    "metric_aht_path_variance",
    "sequential_lower_curve",
    "terminal_residual_lower_curve",
]
