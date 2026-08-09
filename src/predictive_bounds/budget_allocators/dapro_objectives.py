"""Shared target coefficients for generalized DAPRO allocation objectives.

Every target is represented by nonnegative event/influence mass ``a`` and
at-risk cost mass ``d``.  Given cumulative reach ``rho``, the common design
problem is

    minimize sum a * (1 / rho - 1)
    subject to sum d * rho <= budget.

Hard-label DAPRO, soft history-adaptive DAPRO, and the pre-run initial-PMF
allocator differ only in how they estimate these coefficients and in the
policy class over which ``rho`` is optimized.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class DAPROObjectiveCoefficients:
    """Event/influence and at-risk cost masses on a common prefix grid."""

    event_mass: np.ndarray
    cost_mass: np.ndarray
    target_kind: str
    coefficient_estimator: str

    def __post_init__(self) -> None:
        event = np.asarray(self.event_mass, dtype=np.float64)
        cost = np.asarray(self.cost_mass, dtype=np.float64)
        if event.ndim != 2 or cost.shape != event.shape:
            raise ValueError("DAPRO coefficient arrays must be equal matrices.")
        if not np.all(np.isfinite(event)) or not np.all(np.isfinite(cost)):
            raise ValueError("DAPRO coefficients must be finite.")
        if np.any(event < 0) or np.any(cost < 0):
            raise ValueError("DAPRO coefficients must be nonnegative.")


def target_time_mask(
        width: int,
        horizons: int | float | np.ndarray | torch.Tensor,
        *,
        strict: bool,
) -> np.ndarray:
    """Return the one-based target-event mask ``t < h`` or ``t <= h``."""
    if width <= 0:
        raise ValueError("`width` must be positive.")
    values = (
        horizons.detach().cpu().numpy()
        if torch.is_tensor(horizons)
        else np.asarray(horizons)
    )
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 0:
        values = values.reshape(1)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("Target horizons must be a finite scalar or vector.")
    one_based_time = np.arange(1, width + 1, dtype=np.float64)[None, :]
    comparison = np.less if strict else np.less_equal
    return comparison(one_based_time, values[:, None])


def realized_target_weights(
        event_times: torch.Tensor,
        horizons: int | float | torch.Tensor,
        *,
        strict: bool,
) -> torch.Tensor:
    """Return hard terminal target weights under common one-based semantics."""
    times = event_times.reshape(-1).to(torch.float64)
    horizon = torch.as_tensor(
        horizons,
        dtype=torch.float64,
        device=times.device,
    ).reshape(-1)
    if len(horizon) == 1:
        horizon = horizon.expand_as(times)
    if len(horizon) != len(times):
        raise ValueError("Target horizons must be scalar or one per row.")
    return (times < horizon if strict else times <= horizon).to(torch.float64)


def initial_pmf_objective_coefficients(
        event_mass: np.ndarray,
        at_risk_mass: np.ndarray,
        horizons: int | float | np.ndarray | torch.Tensor,
        *,
        strict: bool,
        target_kind: str,
) -> DAPROObjectiveCoefficients:
    """Build pre-run model-integrated coefficients from an initial PMF."""
    event = np.asarray(event_mass, dtype=np.float64)
    risk = np.asarray(at_risk_mass, dtype=np.float64)
    if event.ndim != 2 or risk.shape != event.shape:
        raise ValueError("Initial event and at-risk masses must agree.")
    mask = target_time_mask(
        event.shape[1],
        horizons,
        strict=strict,
    )
    if len(mask) == 1 and len(event) != 1:
        mask = np.broadcast_to(mask, event.shape)
    if mask.shape != event.shape:
        raise ValueError("Target horizons must be scalar or one per PMF row.")
    return DAPROObjectiveCoefficients(
        event_mass=event * mask,
        cost_mass=risk,
        target_kind=target_kind,
        coefficient_estimator="initial_pmf_model_integrated",
    )


def history_soft_objective_coefficients(
        conditional_grid: torch.Tensor,
        active_lengths: torch.Tensor,
        horizons: int | float | torch.Tensor,
        *,
        strict: bool,
        target_kind: str,
        global_regularization: float = 0.0,
) -> DAPROObjectiveCoefficients:
    """Rao--Blackwellized coefficients from causal prefix hazards.

    ``conditional_grid[i,t,t]`` is the event hazard available at prefix
    ``X_it`` before acquiring interaction ``t+1``. Only prefixes that exist in
    the fully observed policy-fit fold contribute. A small regularization may
    mix the named target with all event mass inside the acquisition envelope.
    """
    if not torch.is_tensor(conditional_grid) or conditional_grid.ndim != 3:
        raise ValueError(
            "`conditional_grid` must have shape (N, current_time, outcome)."
        )
    n, width, outcomes = conditional_grid.shape
    if outcomes < width:
        raise ValueError("The conditional grid does not cover its time width.")
    lengths = np.asarray(
        active_lengths.reshape(-1).detach().cpu(),
        dtype=np.int64,
    )
    if lengths.shape != (n,) or np.any(lengths < 0) or np.any(lengths > width):
        raise ValueError("Active lengths must lie in the conditional-grid range.")
    if not np.isfinite(global_regularization) or global_regularization < 0:
        raise ValueError("`global_regularization` must be nonnegative.")

    step = torch.arange(width, device=conditional_grid.device)
    hazard = np.asarray(
        conditional_grid[:, step, step].detach().cpu(),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(hazard)):
        raise ValueError("Prefix hazards must be finite.")
    if np.any(hazard < -1e-8) or np.any(hazard > 1 + 1e-8):
        raise ValueError("Prefix hazards must lie in [0, 1].")
    hazard = np.clip(hazard, 0.0, 1.0)
    active = np.arange(width)[None, :] < lengths[:, None]
    target = target_time_mask(width, horizons, strict=strict)
    if len(target) == 1 and n != 1:
        target = np.broadcast_to(target, (n, width))
    if target.shape != (n, width):
        raise ValueError("Target horizons must be scalar or one per prefix row.")

    named_mass = hazard * active * target
    if global_regularization > 0:
        all_event_mass = hazard * active
        named_mass = (
            named_mass + global_regularization * all_event_mass
        ) / (1.0 + global_regularization)
    return DAPROObjectiveCoefficients(
        event_mass=named_mass,
        cost_mass=active.astype(np.float64),
        target_kind=target_kind,
        coefficient_estimator="history_prefix_hazard_model_integrated",
    )


__all__ = [
    "DAPROObjectiveCoefficients",
    "history_soft_objective_coefficients",
    "initial_pmf_objective_coefficients",
    "realized_target_weights",
    "target_time_mask",
]
