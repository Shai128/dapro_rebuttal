"""Metric-specific allocation for Horvitz--Thompson event-rate estimation.

The unsafe-event-rate contribution of row ``i`` has conditional variance

    A_i * (1 / rho_i(T_i) - 1),

where ``A_i = 1{T_i <= M}`` and ``rho_i(t)`` is the cumulative probability of
reaching interaction ``t``.  This module therefore optimizes cumulative reach
directly, rather than a mean inverse weight at an unrelated LPB horizon.

Only the model distribution available before the first interaction is used.
For event mass ``a_it`` and at-risk probability ``d_it``, the plug-in problem

    min sum_it a_it / rho_it
    s.t. mean_i sum_t d_it rho_it <= B,
         1 >= rho_i1 >= ... >= rho_iM >= epsilon

is convex.  Antitonic pool-adjacent-violators gives a per-row base path, and a
single scalar dual variable meets the predicted budget.  The CRC subclass
uses a fixed nested, row-capped family and an independent fully observed
control fold to obtain the repository's marginal expected-total-budget
guarantee.
"""

from __future__ import annotations

import numpy as np
import torch

from src.predictive_bounds.budget_allocators.budget_allocator import (
    BudgetAllocationResult,
    BudgetAllocator,
    summarize_expected_budget,
)
from src.predictive_bounds.budget_allocators.dapro_objectives import (
    initial_pmf_objective_coefficients,
)
from src.predictive_bounds.budget_allocators.risk_controlled_budget import (
    select_crc_budget_candidate,
)
from src.predictive_bounds.budget_allocators.vectorized_adaptive_allocator_patch import (
    simulate_process_vectorized,
)


def initial_event_and_at_risk_probabilities(
        probability_est: torch.Tensor,
        width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return causal event masses and at-risk probabilities in float64.

    Real caches have ``width + 1`` outcome classes: one for each event time
    and a final survival class.  Synthetic/legacy experiments instead pass a
    two-dimensional matrix of discrete hazards.  Any PMF classes beyond the
    event window are treated as survival through the metric horizon.
    """
    if not torch.is_tensor(probability_est) or probability_est.ndim not in (2, 3):
        raise ValueError(
            "`probability_est` must have shape (N, time) or "
            "(N, current_time, outcome)."
        )
    if width <= 0:
        raise ValueError("`width` must be positive.")
    outcome_count = probability_est.shape[-1]
    if (probability_est.ndim == 3 and probability_est.shape[1] < 1) or (
            outcome_count < width
    ):
        raise ValueError(
            "The probability tensor does not cover the requested horizon."
        )

    if probability_est.ndim == 2:
        hazard = np.asarray(
            probability_est[:, :width].detach().cpu(),
            dtype=np.float64,
        )
        if not np.all(np.isfinite(hazard)):
            raise ValueError("Initial model hazards must be finite.")
        if np.any(hazard < -1e-8) or np.any(hazard > 1 + 1e-8):
            raise ValueError("Initial model hazards must lie in [0, 1].")
        hazard = np.clip(hazard, 0.0, 1.0)
        at_risk = np.concatenate(
            [
                np.ones((len(hazard), 1), dtype=np.float64),
                np.cumprod(1.0 - hazard[:, :-1], axis=1),
            ],
            axis=1,
        )
        return hazard * at_risk, np.clip(
            at_risk,
            np.finfo(np.float64).tiny,
            1.0,
        )

    # Deliberately read current_time zero only.  Future rows depend on future
    # interaction histories and would leak information into a pre-run policy.
    initial_tensor = probability_est[:, 0, :]
    initial = np.asarray(initial_tensor.detach().cpu(), dtype=np.float64)
    if not np.all(np.isfinite(initial)):
        raise ValueError("Initial model probabilities must be finite.")
    if np.any(initial < -1e-8):
        raise ValueError("Initial model probabilities must be nonnegative.")
    initial = np.maximum(initial, 0.0)

    event = initial[:, :width].copy()
    if initial.shape[1] > width:
        tail = initial[:, width:].sum(axis=1)
    else:
        tail = np.maximum(1.0 - event.sum(axis=1), 0.0)
    total = event.sum(axis=1) + tail
    if np.any(total <= 0):
        raise ValueError("Every initial PMF row must have positive mass.")
    event /= total[:, None]

    # d_it = P(T >= t | X_i0).  At t=1 every row is at risk.  Computing this
    # from preceding event masses automatically includes the normalized tail.
    at_risk = 1.0 - np.concatenate(
        [
            np.zeros((len(event), 1), dtype=np.float64),
            np.cumsum(event[:, :-1], axis=1),
        ],
        axis=1,
    )
    at_risk = np.clip(
        at_risk,
        np.finfo(np.float64).tiny,
        1.0,
    )
    return event, at_risk


def prefix_remaining_event_cost_index(
        conditional_grid: torch.Tensor,
        width: int,
) -> np.ndarray:
    """Return a causal Neyman index at every observed prefix.

    At prefix ``X_it`` the index is

        sqrt(P(T_i <= M | T_i > t, X_it)
             / E[min(T_i, M) - t | T_i > t, X_it]).

    It is the square-root value-per-cost rule obtained by treating acquisition
    of the remaining suffix as a local block decision.  Each column uses only
    the conditional PMF available at that prefix.  Outcome classes before the
    current prefix are ignored explicitly, so malformed stale mass cannot
    influence a later decision.
    """
    if not torch.is_tensor(conditional_grid) or conditional_grid.ndim != 3:
        raise ValueError(
            "`conditional_grid` must have shape "
            "(N, current_time, outcome)."
        )
    n, current_count, outcome_count = conditional_grid.shape
    if width <= 0 or current_count < width or outcome_count < width:
        raise ValueError(
            "The conditional grid does not cover the requested horizon."
        )
    grid = conditional_grid[:, :width, :]
    if not bool(torch.isfinite(grid).all()):
        raise ValueError("Prefix probabilities must be finite.")
    if bool((grid < -1e-8).any()):
        raise ValueError("Prefix probabilities must be nonnegative.")
    grid = grid.clamp_min(0).to(torch.float64)

    current = torch.arange(width, device=grid.device).view(width, 1)
    outcome = torch.arange(outcome_count, device=grid.device).view(1, -1)
    event = (outcome >= current) & (outcome < width)
    tail = outcome >= width
    valid = event | tail
    event_weight = event.to(torch.float64)
    valid_weight = valid.to(torch.float64)
    remaining_cost_weight = torch.where(
        event,
        (outcome - current + 1).to(torch.float64),
        torch.where(
            tail,
            (width - current).to(torch.float64),
            torch.zeros((), dtype=torch.float64, device=grid.device),
        ),
    )

    valid_mass = torch.einsum("nto,to->nt", grid, valid_weight)
    if bool((valid_mass <= 0).any()):
        raise ValueError(
            "Every causal prefix must assign positive mass to its future."
        )
    remaining_event = (
        torch.einsum("nto,to->nt", grid, event_weight) / valid_mass
    )
    remaining_cost = (
        torch.einsum("nto,to->nt", grid, remaining_cost_weight)
        / valid_mass
    )
    if bool((remaining_cost <= 0).any()):
        raise ValueError("Expected remaining acquisition cost must be positive.")
    index = torch.sqrt(remaining_event / remaining_cost)
    if index.shape != (n, width) or not bool(torch.isfinite(index).all()):
        raise ValueError("The prefix Neyman index must be finite.")
    return np.asarray(index.detach().cpu(), dtype=np.float64)


def antitonic_pav_bases(
        numerator: np.ndarray,
        denominator: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pool ``sqrt(sum numerator / sum denominator)`` into decreasing paths.

    The common scalar multiplier and monotone clipping are applied after PAV,
    so the block partition does not depend on the budget.  The second return
    value is the number of blocks in each row and is useful diagnostically.
    """
    a = np.asarray(numerator, dtype=np.float64)
    d = np.asarray(denominator, dtype=np.float64)
    if a.ndim != 2 or d.shape != a.shape:
        raise ValueError("PAV coefficients must be equally shaped matrices.")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(d)):
        raise ValueError("PAV coefficients must be finite.")
    if np.any(a < 0) or np.any(d <= 0):
        raise ValueError(
            "PAV numerators must be nonnegative and denominators positive."
        )

    n, width = a.shape
    bases = np.empty_like(a)
    block_counts = np.empty(n, dtype=np.int64)
    tolerance = 1e-15

    # Numeric-stack PAV.  Keeping arrays preallocated makes this O(NM) while
    # retaining a transparent implementation that can be checked on tiny
    # convex programs in tests.
    starts = np.empty(width, dtype=np.int64)
    stops = np.empty(width, dtype=np.int64)
    numerator_sums = np.empty(width, dtype=np.float64)
    denominator_sums = np.empty(width, dtype=np.float64)
    values = np.empty(width, dtype=np.float64)
    for row in range(n):
        blocks = 0
        for step in range(width):
            starts[blocks] = step
            stops[blocks] = step + 1
            numerator_sums[blocks] = a[row, step]
            denominator_sums[blocks] = d[row, step]
            values[blocks] = np.sqrt(
                numerator_sums[blocks] / denominator_sums[blocks]
            )
            blocks += 1

            # Cumulative reach must be non-increasing.  Merge whenever the
            # left unconstrained block would have smaller reach than the right.
            while (
                    blocks > 1
                    and values[blocks - 2] < values[blocks - 1] - tolerance
            ):
                left = blocks - 2
                right = blocks - 1
                stops[left] = stops[right]
                numerator_sums[left] += numerator_sums[right]
                denominator_sums[left] += denominator_sums[right]
                values[left] = np.sqrt(
                    numerator_sums[left] / denominator_sums[left]
                )
                blocks -= 1

        for block in range(blocks):
            bases[row, starts[block]:stops[block]] = values[block]
        block_counts[row] = blocks
    return bases, block_counts


def cumulative_paths_for_scale(
        bases: np.ndarray,
        scale: float | np.ndarray,
        floor: float,
) -> np.ndarray:
    """Apply a scalar or row-specific scale to antitonic PAV bases."""
    values = np.asarray(bases, dtype=np.float64)
    row_scale = np.asarray(scale, dtype=np.float64)
    if row_scale.ndim == 1:
        row_scale = row_scale[:, None]
    # The mathematical extension at an infinite scale is one for every
    # positive base and the floor for a zero base.  ``np.inf * 0`` is NaN, so
    # spell out that boundary instead of relying on raw multiplication.
    scaled = np.zeros_like(values)
    np.multiply(
        values,
        row_scale,
        out=scaled,
        where=values > 0,
    )
    cumulative = np.clip(scaled, floor, 1.0)
    return cumulative


def solve_common_scale(
        bases: np.ndarray,
        cost_coefficients: np.ndarray,
        budget_per_sample: float,
        floor: float,
        *,
        tolerance: float = 1e-10,
) -> tuple[np.ndarray, float, str]:
    """Find the largest common scale satisfying a linear expected budget."""
    b = np.asarray(bases, dtype=np.float64)
    costs = np.asarray(cost_coefficients, dtype=np.float64)
    if b.ndim != 2 or costs.shape != b.shape:
        raise ValueError("Bases and cost coefficients must be equal matrices.")
    if not np.isfinite(budget_per_sample) or budget_per_sample < 0:
        raise ValueError("`budget_per_sample` must be finite and nonnegative.")
    if not 0 < floor <= 1:
        raise ValueError("`floor` must lie in (0, 1].")

    def mean_cost(scale: float) -> float:
        return float(
            np.mean(
                np.sum(
                    costs * cumulative_paths_for_scale(b, scale, floor),
                    axis=1,
                )
            )
        )

    minimum_cost = mean_cost(0.0)
    if budget_per_sample < minimum_cost - tolerance:
        raise ValueError(
            "The requested budget is infeasible under the propensity floor: "
            f"target={budget_per_sample:.6g}, minimum={minimum_cost:.6g}."
        )
    maximum_cost = float(np.mean(costs.sum(axis=1)))
    if budget_per_sample >= maximum_cost - tolerance:
        return np.ones_like(b), np.inf, "maximum"

    low, high = 0.0, 1.0
    while mean_cost(high) < budget_per_sample and high < 1e16:
        high *= 2.0
    for _ in range(80):
        middle = (low + high) / 2.0
        if mean_cost(middle) <= budget_per_sample:
            low = middle
        else:
            high = middle
    cumulative = cumulative_paths_for_scale(b, low, floor)
    boundary = (
        "objective_plateau"
        if float(np.mean(np.sum(costs * cumulative, axis=1)))
        < budget_per_sample - tolerance
        else "budget"
    )
    return cumulative, float(low), boundary


def row_horizon_cap_scales(
        bases: np.ndarray,
        row_cost_cap: float,
        floor: float,
) -> np.ndarray:
    """Largest row-specific scales whose full-horizon reach sums meet a cap."""
    b = np.asarray(bases, dtype=np.float64)
    if b.ndim != 2:
        raise ValueError("`bases` must be a matrix.")
    _, width = b.shape
    if row_cost_cap < width * floor - 1e-10:
        raise ValueError("The row cap is infeasible under the propensity floor.")
    if row_cost_cap >= width - 1e-10:
        return np.full(len(b), np.inf, dtype=np.float64)

    low = np.zeros(len(b), dtype=np.float64)
    high = np.ones(len(b), dtype=np.float64)
    positive = b.max(axis=1) > 0
    for _ in range(60):
        cost = cumulative_paths_for_scale(b, high, floor).sum(axis=1)
        expand = positive & (cost < row_cost_cap - 1e-10)
        if not np.any(expand):
            break
        high[expand] *= 2.0
    for _ in range(70):
        middle = (low + high) / 2.0
        cost = cumulative_paths_for_scale(b, middle, floor).sum(axis=1)
        feasible = cost <= row_cost_cap
        low[feasible] = middle[feasible]
        high[~feasible] = middle[~feasible]
    # All-zero objective rows remain at the floor for every useful policy.
    low[~positive] = 0.0
    return low


def cumulative_to_conditionals(cumulative: np.ndarray) -> np.ndarray:
    """Convert cumulative reach to exact conditional continuation chances."""
    rho = np.asarray(cumulative, dtype=np.float64)
    if rho.ndim != 2:
        raise ValueError("`cumulative` must be a matrix.")
    previous = np.concatenate(
        [np.ones((len(rho), 1), dtype=np.float64), rho[:, :-1]],
        axis=1,
    )
    conditionals = rho / np.maximum(previous, np.finfo(np.float64).tiny)
    return np.clip(conditionals, 0.0, 1.0)


class _MetricOptimalPMFBase(BudgetAllocator):
    """Shared causal PMF policy construction and trajectory simulation."""

    def __init__(
            self,
            budget_per_sample: float,
            taus_range: torch.Tensor,
            tau_prior: float,
            m_upper_bound: int,
            *,
            terminal_pi_min: float | None = None,
    ):
        super().__init__(budget_per_sample, taus_range, tau_prior)
        if not isinstance(m_upper_bound, (int, np.integer)) or m_upper_bound < 1:
            raise ValueError("`m_upper_bound` must be a positive integer.")
        self.m_upper_bound = int(m_upper_bound)
        self.terminal_pi_min = (
            1.0 / self.m_upper_bound
            if terminal_pi_min is None
            else float(terminal_pi_min)
        )
        if not 0 < self.terminal_pi_min <= 1:
            raise ValueError("`terminal_pi_min` must lie in (0, 1].")
        self.last_cumulative_probabilities = None

    def _model_bases(
            self,
            probability_est: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        event, at_risk = initial_event_and_at_risk_probabilities(
            probability_est,
            self.m_upper_bound,
        )
        coefficients = initial_pmf_objective_coefficients(
            event,
            at_risk,
            self.m_upper_bound,
            strict=False,
            target_kind="unsafe_event_rate",
        )
        bases, block_counts = antitonic_pav_bases(
            coefficients.event_mass,
            coefficients.cost_mass,
        )
        return (
            coefficients.event_mass,
            coefficients.cost_mass,
            bases,
            block_counts,
        )

    def _simulate_rows(
            self,
            cumulative: np.ndarray,
            row_indices: np.ndarray,
            t: torch.Tensor,
            *,
            all_uniforms: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        device = t.device
        width = self.m_upper_bound
        conditionals = torch.as_tensor(
            cumulative_to_conditionals(cumulative),
            dtype=torch.float64,
            device=device,
        )
        indices = torch.as_tensor(row_indices, dtype=torch.long, device=device)
        row_t = t.reshape(-1)[indices]
        prior = torch.full(
            (len(row_indices),),
            width,
            dtype=torch.long,
            device=device,
        )
        row_uniforms = (
            None
            if all_uniforms is None
            else all_uniforms[indices]
        )
        if all_uniforms is None:
            self.reset_acquisition_rng()
        return simulate_process_vectorized(
            torch.ones_like(conditionals),
            prior,
            row_t,
            0.0,
            stochastic=True,
            uniforms=row_uniforms,
            pi_func=lambda _: conditionals,
            terminal_pi_min=None,
            terminal_floor_mode="none",
        )

    def _common_metrics(
            self,
            *,
            cumulative: np.ndarray,
            block_counts: np.ndarray,
            t: torch.Tensor,
            total_expected_budget: float,
            cost_semantics: str,
    ) -> dict:
        n = len(cumulative)
        active_lengths = np.minimum(
            np.asarray(t.reshape(-1).detach().cpu(), dtype=np.int64),
            self.m_upper_bound,
        ).clip(min=1)
        active = (
            np.arange(self.m_upper_bound)[None, :]
            < active_lengths[:, None]
        )
        endpoint = cumulative[np.arange(n), active_lengths - 1]
        return {
            "generalized_dapro": 1,
            "generalized_dapro_target": "unsafe_event_rate",
            "generalized_dapro_coefficient_estimator": (
                "initial_pmf_model_integrated"
            ),
            "generalized_dapro_policy_class": (
                "precommitted_row_time_cumulative_reach"
            ),
            "objective_kind": (
                "model_expected_metric_event_weighted_inverse_probability"
            ),
            "target_metric": "unsafe_event_rate",
            "target_metric_horizon": self.m_upper_bound,
            "metric_policy_uses_initial_pmf_only": 1,
            "metric_policy_uses_future_history": 0,
            "metric_policy_uses_latent_event_times": 0,
            "metric_policy_cumulative_reach_optimized": 1,
            "metric_policy_pav_mean_block_count": float(block_counts.mean()),
            "metric_policy_pav_max_block_count": int(block_counts.max()),
            "metric_policy_terminal_pi_min": self.terminal_pi_min,
            "metric_policy_mean_endpoint_probability": float(endpoint.mean()),
            "metric_policy_min_endpoint_probability": float(endpoint.min()),
            "metric_policy_max_endpoint_probability": float(endpoint.max()),
            "metric_policy_true_expected_cost_per_sample": float(
                np.sum(cumulative * active) / n
            ),
            **summarize_expected_budget(
                total_expected_budget,
                n,
                self.budget_per_sample,
                cost_semantics=cost_semantics,
            ),
        }


class MetricOptimalPMFAllocator(_MetricOptimalPMFBase):
    """Exact plug-in PMF optimum under the model-predicted budget."""

    @property
    def name(self) -> str:
        return "metric_optimal_pmf_model_budget"

    def allocate_budget(
            self,
            probability_est: torch.Tensor,
            x: torch.Tensor,
            t: torch.Tensor,
            quantile_est: torch.Tensor,
    ) -> BudgetAllocationResult:
        del x
        n = len(t)
        if len(probability_est) != n or len(quantile_est) != n:
            raise ValueError("Allocator inputs must have the same row count.")
        event, at_risk, bases, block_counts = self._model_bases(
            probability_est
        )
        cumulative, scale, boundary = solve_common_scale(
            bases,
            at_risk,
            self.budget_per_sample,
            self.terminal_pi_min,
        )
        self.last_cumulative_probabilities = cumulative.copy()
        all_uniforms = self.get_acquisition_uniforms(
            n,
            self.m_upper_bound,
            device=t.device,
            dtype=torch.float64,
        )
        rows = np.arange(n, dtype=np.int64)
        censoring, propensities, realized_cost = self._simulate_rows(
            cumulative,
            rows,
            t,
            all_uniforms=all_uniforms,
        )
        predicted_cost = float(np.mean(np.sum(at_risk * cumulative, axis=1)))
        active_lengths = np.minimum(
            np.asarray(t.reshape(-1).detach().cpu(), dtype=np.int64),
            self.m_upper_bound,
        ).clip(min=1)
        active = np.arange(self.m_upper_bound)[None, :] < active_lengths[:, None]
        true_expected_total = float(np.sum(cumulative * active))
        inverse = 1 / propensities
        metrics = {
            **self._common_metrics(
                cumulative=cumulative,
                block_counts=block_counts,
                t=t,
                total_expected_budget=true_expected_total,
                cost_semantics=(
                    "initial_pmf_optimal_expected_interactions_with_event_stopping"
                ),
            ),
            "metric_policy_budget_control_mode": "model_predicted",
            "metric_policy_model_expected_cost_per_sample": predicted_cost,
            "metric_policy_model_budget_gap_per_sample": (
                predicted_cost - self.budget_per_sample
            ),
            "metric_policy_common_scale": scale,
            "metric_policy_budget_boundary": boundary,
            "metric_policy_predicted_event_rate": float(event.sum(axis=1).mean()),
        }
        return BudgetAllocationResult(
            quantile_est,
            censoring,
            propensities,
            int(realized_cost),
            mean_weight=float(inverse.mean().item()),
            max_weight=float(inverse.max().item()),
            additional_metrics=metrics,
        )


class MetricOptimalPMFCRCAllocator(_MetricOptimalPMFBase):
    """Row-capped PMF-optimal family with independent CRC budget control."""

    def __init__(
            self,
            budget_per_sample: float,
            taus_range: torch.Tensor,
            tau_prior: float,
            m_upper_bound: int,
            *,
            control_size: int = 100,
            row_cost_cap_multiplier: float = 2.0,
            candidate_count: int = 1001,
            terminal_pi_min: float | None = None,
    ):
        super().__init__(
            budget_per_sample,
            taus_range,
            tau_prior,
            m_upper_bound,
            terminal_pi_min=terminal_pi_min,
        )
        if not isinstance(control_size, (int, np.integer)) or control_size <= 0:
            raise ValueError("`control_size` must be a positive integer.")
        if not np.isfinite(row_cost_cap_multiplier) or row_cost_cap_multiplier <= 0:
            raise ValueError("`row_cost_cap_multiplier` must be positive.")
        if not isinstance(candidate_count, (int, np.integer)) or candidate_count < 2:
            raise ValueError("`candidate_count` must be at least two.")
        self.control_size = int(control_size)
        self.row_cost_cap_multiplier = float(row_cost_cap_multiplier)
        self.candidate_count = int(candidate_count)
        self.last_control_indices = None
        self.last_deployment_indices = None
        self.last_selected_candidate_index = None

    @property
    def name(self) -> str:
        cap = (
            f"{self.row_cost_cap_multiplier:.2f}"
            .replace(".", "p")
        )
        return (
            "metric_optimal_pmf_crc_control_"
            f"{self.control_size}_row_cap_{cap}x_budget"
        )

    def allocate_budget(
            self,
            probability_est: torch.Tensor,
            x: torch.Tensor,
            t: torch.Tensor,
            quantile_est: torch.Tensor,
    ) -> BudgetAllocationResult:
        del x
        n = len(t)
        if len(probability_est) != n or len(quantile_est) != n:
            raise ValueError("Allocator inputs must have the same row count.")
        if self.control_size >= n:
            raise ValueError("`control_size` must be smaller than the row count.")

        event, at_risk, bases, block_counts = self._model_bases(
            probability_est
        )
        row_cap = min(
            float(self.m_upper_bound),
            self.row_cost_cap_multiplier * self.budget_per_sample,
        )
        cap_scales = row_horizon_cap_scales(
            bases,
            row_cap,
            self.terminal_pi_min,
        )

        permutation = (
            np.random.permutation(n)
            if self.acquisition_seed is None
            else np.random.RandomState(
                int(self.acquisition_seed)
            ).permutation(n)
        )
        control_indices = permutation[:self.control_size]
        deployment_indices = permutation[self.control_size:]
        self.last_control_indices = control_indices.copy()
        self.last_deployment_indices = deployment_indices.copy()
        active_lengths = np.minimum(
            np.asarray(t.reshape(-1).detach().cpu(), dtype=np.int64),
            self.m_upper_bound,
        ).clip(min=1)
        control_lengths = active_lengths[control_indices]
        control_active = (
            np.arange(self.m_upper_bound)[None, :]
            < control_lengths[:, None]
        )

        # A fixed, data-independent parameter grid covers scales [infinity, 0].
        # Candidate shapes may use pretrained, label-free PMFs, but no event
        # time enters until the independent control costs are evaluated below.
        alphas = np.linspace(1.0, 0.0, self.candidate_count)
        scales = np.empty_like(alphas)
        scales[0] = np.inf
        scales[1:] = alphas[1:] / np.maximum(1.0 - alphas[1:], 1e-300)
        candidate_costs = np.empty(
            (self.control_size, self.candidate_count),
            dtype=np.float64,
        )
        control_bases = bases[control_indices]
        control_caps = cap_scales[control_indices]
        for column, scale in enumerate(scales):
            effective = np.minimum(scale, control_caps)
            cumulative = cumulative_paths_for_scale(
                control_bases,
                effective,
                self.terminal_pi_min,
            )
            candidate_costs[:, column] = np.sum(
                cumulative * control_active,
                axis=1,
            )

        selection = select_crc_budget_candidate(
            candidate_costs,
            control_lengths.astype(np.float64),
            total_budget_after_policy_fit=(self.budget_per_sample * n),
            deployment_sample_count=len(deployment_indices),
            maximum_cost_per_sample=self.m_upper_bound,
            maximum_candidate_cost_per_sample=row_cap,
            maximum_pilot_cost_per_sample=self.m_upper_bound,
        )
        selected_scale = scales[selection.selected_index]
        self.last_selected_candidate_index = int(selection.selected_index)
        deployment_effective = np.minimum(
            selected_scale,
            cap_scales[deployment_indices],
        )
        deployment_cumulative = cumulative_paths_for_scale(
            bases[deployment_indices],
            deployment_effective,
            self.terminal_pi_min,
        )

        all_uniforms = self.get_acquisition_uniforms(
            n,
            self.m_upper_bound,
            device=t.device,
            dtype=torch.float64,
        )
        deploy_c, deploy_propensity, deploy_realized_cost = self._simulate_rows(
            deployment_cumulative,
            deployment_indices,
            t,
            all_uniforms=all_uniforms,
        )
        device = t.device
        final_c = torch.full(
            (n,),
            self.m_upper_bound,
            dtype=torch.long,
            device=device,
        )
        final_propensity = torch.ones(n, dtype=torch.float64, device=device)
        deployment_tensor = torch.as_tensor(
            deployment_indices,
            dtype=torch.long,
            device=device,
        )
        final_c[deployment_tensor] = deploy_c.to(torch.long)
        final_propensity[deployment_tensor] = deploy_propensity

        control_cost = float(control_lengths.sum())
        deployment_active = (
            np.arange(self.m_upper_bound)[None, :]
            < active_lengths[deployment_indices, None]
        )
        deployment_expected_total = float(
            np.sum(deployment_cumulative * deployment_active)
        )
        total_expected = control_cost + deployment_expected_total

        # Build the full selected cumulative table for common diagnostics only.
        full_cumulative = np.ones((n, self.m_upper_bound), dtype=np.float64)
        full_cumulative[deployment_indices] = deployment_cumulative
        self.last_cumulative_probabilities = full_cumulative.copy()
        inverse = 1 / final_propensity
        metrics = {
            **self._common_metrics(
                cumulative=full_cumulative,
                block_counts=block_counts,
                t=t,
                total_expected_budget=total_expected,
                cost_semantics=(
                    "fully_observed_crc_control_plus_row_capped_initial_pmf_"
                    "expected_interactions"
                ),
            ),
            "metric_policy_budget_control_mode": "crc",
            "metric_policy_crc_control_size": self.control_size,
            "metric_policy_crc_deployment_size": len(deployment_indices),
            "metric_policy_crc_candidate_count": self.candidate_count,
            "metric_policy_crc_selected_index": selection.selected_index,
            "metric_policy_crc_selected_alpha": float(
                alphas[selection.selected_index]
            ),
            "metric_policy_crc_selected_scale": float(selected_scale),
            "metric_policy_crc_row_cost_cap": row_cap,
            "metric_policy_crc_row_cost_cap_multiplier": (
                self.row_cost_cap_multiplier
            ),
            "metric_policy_crc_empirical_control_policy_cost_per_sample": (
                selection.empirical_expected_cost_per_sample
            ),
            "metric_policy_crc_control_pilot_cost_per_sample": float(
                control_lengths.mean()
            ),
            "metric_policy_crc_selector_left_side_per_sample": (
                selection.selector_left_side_per_sample
            ),
            "metric_policy_crc_deployment_target_per_sample": (
                selection.deployment_budget_per_sample
            ),
            "metric_policy_crc_correction_per_sample": (
                selection.correction_per_sample
            ),
            "metric_policy_crc_guarantee_kind": selection.guarantee_kind,
            "metric_policy_crc_selector_valid": int(
                selection.selector_left_side_per_sample
                <= selection.deployment_budget_per_sample + 1e-12
            ),
            "metric_policy_predicted_event_rate": float(event.sum(axis=1).mean()),
            "metric_policy_control_indices_use_labels_only_for_crc_cost": 1,
        }
        return BudgetAllocationResult(
            quantile_est,
            final_c,
            final_propensity,
            int(control_cost + deploy_realized_cost),
            mean_weight=float(inverse.mean().item()),
            max_weight=float(inverse.max().item()),
            additional_metrics=metrics,
        )


class MetricOptimalPooledTimeAllocator(_MetricOptimalPMFBase):
    """Simple shared Neyman schedule derived from the pooled initial PMF.

    Unlike :class:`MetricOptimalPMFAllocator`, this policy does not personalize
    reach by row.  It averages the initial event and at-risk masses over the
    evaluation covariates, solves one antitonic PAV problem of length ``M``,
    and deploys the same time-varying schedule to every trajectory.  This is
    the exact plug-in optimum within the shared precommitted time-schedule
    class and has no policy-fitting split or DAPRO score-map optimization.
    """

    @property
    def name(self) -> str:
        return "metric_optimal_pooled_time_model_budget"

    def _pooled_model_shape(
            self,
            probability_est: torch.Tensor,
    ) -> tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
    ]:
        event, at_risk = initial_event_and_at_risk_probabilities(
            probability_est,
            self.m_upper_bound,
        )
        pooled_event = event.mean(axis=0, keepdims=True)
        pooled_at_risk = at_risk.mean(axis=0, keepdims=True)
        bases, block_counts = antitonic_pav_bases(
            pooled_event,
            pooled_at_risk,
        )
        return event, at_risk, pooled_event, pooled_at_risk, bases, block_counts

    def allocate_budget(
            self,
            probability_est: torch.Tensor,
            x: torch.Tensor,
            t: torch.Tensor,
            quantile_est: torch.Tensor,
    ) -> BudgetAllocationResult:
        del x
        n = len(t)
        if len(probability_est) != n or len(quantile_est) != n:
            raise ValueError("Allocator inputs must have the same row count.")

        (
            _event,
            _at_risk,
            pooled_event,
            pooled_at_risk,
            bases,
            block_counts,
        ) = self._pooled_model_shape(probability_est)
        pooled_cumulative, scale, boundary = solve_common_scale(
            bases,
            pooled_at_risk,
            self.budget_per_sample,
            self.terminal_pi_min,
        )
        cumulative = np.broadcast_to(
            pooled_cumulative,
            (n, self.m_upper_bound),
        ).copy()
        self.last_cumulative_probabilities = cumulative.copy()

        all_uniforms = self.get_acquisition_uniforms(
            n,
            self.m_upper_bound,
            device=t.device,
            dtype=torch.float64,
        )
        rows = np.arange(n, dtype=np.int64)
        censoring, propensities, realized_cost = self._simulate_rows(
            cumulative,
            rows,
            t,
            all_uniforms=all_uniforms,
        )

        predicted_cost = float(
            np.sum(pooled_at_risk * pooled_cumulative)
        )
        active_lengths = np.minimum(
            np.asarray(t.reshape(-1).detach().cpu(), dtype=np.int64),
            self.m_upper_bound,
        ).clip(min=1)
        prefix_cost = np.cumsum(pooled_cumulative[0])
        true_expected_total = float(prefix_cost[active_lengths - 1].sum())
        inverse = 1 / propensities
        metrics = {
            **self._common_metrics(
                cumulative=cumulative,
                block_counts=block_counts,
                t=t,
                total_expected_budget=true_expected_total,
                cost_semantics=(
                    "shared_pooled_pmf_time_schedule_with_event_stopping"
                ),
            ),
            "generalized_dapro": 0,
            "simple_metric_policy": 1,
            "objective_kind": (
                "model_expected_pooled_time_event_weighted_inverse_probability"
            ),
            "generalized_dapro_coefficient_estimator": "not_applicable",
            "generalized_dapro_policy_class": "not_applicable",
            "metric_policy_budget_control_mode": "model_predicted",
            "metric_policy_pooled_across_rows": 1,
            "metric_policy_row_personalized": 0,
            "metric_policy_shared_time_schedule": 1,
            "metric_policy_model_expected_cost_per_sample": predicted_cost,
            "metric_policy_model_budget_gap_per_sample": (
                predicted_cost - self.budget_per_sample
            ),
            "metric_policy_common_scale": scale,
            "metric_policy_budget_boundary": boundary,
            "metric_policy_predicted_event_rate": float(pooled_event.sum()),
            "metric_policy_pooled_pav_block_count": int(block_counts[0]),
        }
        return BudgetAllocationResult(
            quantile_est,
            censoring,
            propensities,
            int(realized_cost),
            mean_weight=float(inverse.mean().item()),
            max_weight=float(inverse.max().item()),
            additional_metrics=metrics,
        )


class MetricOptimalPooledTimeCRCAllocator(MetricOptimalPooledTimeAllocator):
    """CRC-controlled nested family of shared pooled Neyman schedules."""

    def __init__(
            self,
            budget_per_sample: float,
            taus_range: torch.Tensor,
            tau_prior: float,
            m_upper_bound: int,
            *,
            control_size: int = 50,
            row_cost_cap_multiplier: float = 2.0,
            candidate_count: int = 1001,
            terminal_pi_min: float | None = None,
    ):
        super().__init__(
            budget_per_sample,
            taus_range,
            tau_prior,
            m_upper_bound,
            terminal_pi_min=terminal_pi_min,
        )
        if not isinstance(control_size, (int, np.integer)) or control_size <= 0:
            raise ValueError("`control_size` must be a positive integer.")
        if not np.isfinite(row_cost_cap_multiplier) or row_cost_cap_multiplier <= 0:
            raise ValueError("`row_cost_cap_multiplier` must be positive.")
        if not isinstance(candidate_count, (int, np.integer)) or candidate_count < 2:
            raise ValueError("`candidate_count` must be at least two.")
        self.control_size = int(control_size)
        self.row_cost_cap_multiplier = float(row_cost_cap_multiplier)
        self.candidate_count = int(candidate_count)
        self.last_control_indices = None
        self.last_deployment_indices = None
        self.last_selected_candidate_index = None

    @property
    def name(self) -> str:
        cap = f"{self.row_cost_cap_multiplier:.2f}".replace(".", "p")
        return (
            "metric_optimal_pooled_time_crc_control_"
            f"{self.control_size}_row_cap_{cap}x_budget"
        )

    def allocate_budget(
            self,
            probability_est: torch.Tensor,
            x: torch.Tensor,
            t: torch.Tensor,
            quantile_est: torch.Tensor,
    ) -> BudgetAllocationResult:
        del x
        n = len(t)
        if len(probability_est) != n or len(quantile_est) != n:
            raise ValueError("Allocator inputs must have the same row count.")
        if self.control_size >= n:
            raise ValueError("`control_size` must be smaller than the row count.")

        (
            _event,
            _at_risk,
            pooled_event,
            _pooled_at_risk,
            bases,
            block_counts,
        ) = self._pooled_model_shape(probability_est)
        row_cap = min(
            float(self.m_upper_bound),
            self.row_cost_cap_multiplier * self.budget_per_sample,
        )
        cap_scale = row_horizon_cap_scales(
            bases,
            row_cap,
            self.terminal_pi_min,
        )[0]

        permutation = (
            np.random.permutation(n)
            if self.acquisition_seed is None
            else np.random.RandomState(int(self.acquisition_seed)).permutation(n)
        )
        control_indices = permutation[:self.control_size]
        deployment_indices = permutation[self.control_size:]
        self.last_control_indices = control_indices.copy()
        self.last_deployment_indices = deployment_indices.copy()

        active_lengths = np.minimum(
            np.asarray(t.reshape(-1).detach().cpu(), dtype=np.int64),
            self.m_upper_bound,
        ).clip(min=1)
        control_lengths = active_lengths[control_indices]
        alphas = np.linspace(1.0, 0.0, self.candidate_count)
        scales = np.empty_like(alphas)
        scales[0] = np.inf
        scales[1:] = alphas[1:] / np.maximum(
            1.0 - alphas[1:],
            1e-300,
        )

        candidate_cumulative = np.empty(
            (self.candidate_count, self.m_upper_bound),
            dtype=np.float64,
        )
        for column, scale in enumerate(scales):
            candidate_cumulative[column] = cumulative_paths_for_scale(
                bases,
                min(scale, cap_scale),
                self.terminal_pi_min,
            )[0]
        candidate_prefix_cost = np.cumsum(candidate_cumulative, axis=1)
        candidate_costs = candidate_prefix_cost[
            :,
            control_lengths - 1,
        ].T

        selection = select_crc_budget_candidate(
            candidate_costs,
            control_lengths.astype(np.float64),
            total_budget_after_policy_fit=self.budget_per_sample * n,
            deployment_sample_count=len(deployment_indices),
            maximum_cost_per_sample=self.m_upper_bound,
            maximum_candidate_cost_per_sample=row_cap,
            maximum_pilot_cost_per_sample=self.m_upper_bound,
        )
        selected_cumulative = candidate_cumulative[selection.selected_index]
        self.last_selected_candidate_index = int(selection.selected_index)
        deployment_cumulative = np.broadcast_to(
            selected_cumulative,
            (len(deployment_indices), self.m_upper_bound),
        ).copy()

        all_uniforms = self.get_acquisition_uniforms(
            n,
            self.m_upper_bound,
            device=t.device,
            dtype=torch.float64,
        )
        deploy_c, deploy_propensity, deploy_realized_cost = self._simulate_rows(
            deployment_cumulative,
            deployment_indices,
            t,
            all_uniforms=all_uniforms,
        )
        device = t.device
        final_c = torch.full(
            (n,),
            self.m_upper_bound,
            dtype=torch.long,
            device=device,
        )
        final_propensity = torch.ones(n, dtype=torch.float64, device=device)
        deployment_tensor = torch.as_tensor(
            deployment_indices,
            dtype=torch.long,
            device=device,
        )
        final_c[deployment_tensor] = deploy_c.to(torch.long)
        final_propensity[deployment_tensor] = deploy_propensity

        control_cost = float(control_lengths.sum())
        prefix_cost = np.cumsum(selected_cumulative)
        deployment_expected_total = float(
            prefix_cost[active_lengths[deployment_indices] - 1].sum()
        )
        total_expected = control_cost + deployment_expected_total
        full_cumulative = np.ones((n, self.m_upper_bound), dtype=np.float64)
        full_cumulative[deployment_indices] = deployment_cumulative
        self.last_cumulative_probabilities = full_cumulative.copy()
        inverse = 1 / final_propensity
        metrics = {
            **self._common_metrics(
                cumulative=full_cumulative,
                block_counts=block_counts,
                t=t,
                total_expected_budget=total_expected,
                cost_semantics=(
                    "fully_observed_crc_control_plus_shared_row_capped_"
                    "pooled_pmf_time_schedule"
                ),
            ),
            "generalized_dapro": 0,
            "simple_metric_policy": 1,
            "objective_kind": (
                "model_expected_pooled_time_event_weighted_inverse_probability"
            ),
            "generalized_dapro_coefficient_estimator": "not_applicable",
            "generalized_dapro_policy_class": "not_applicable",
            "metric_policy_budget_control_mode": "crc",
            "metric_policy_pooled_across_rows": 1,
            "metric_policy_row_personalized": 0,
            "metric_policy_shared_time_schedule": 1,
            "metric_policy_crc_control_size": self.control_size,
            "metric_policy_crc_deployment_size": len(deployment_indices),
            "metric_policy_crc_candidate_count": self.candidate_count,
            "metric_policy_crc_selected_index": selection.selected_index,
            "metric_policy_crc_selected_alpha": float(
                alphas[selection.selected_index]
            ),
            "metric_policy_crc_selected_scale": float(
                scales[selection.selected_index]
            ),
            "metric_policy_crc_row_cost_cap": row_cap,
            "metric_policy_crc_row_cost_cap_multiplier": (
                self.row_cost_cap_multiplier
            ),
            "metric_policy_crc_empirical_control_policy_cost_per_sample": (
                selection.empirical_expected_cost_per_sample
            ),
            "metric_policy_crc_control_pilot_cost_per_sample": float(
                control_lengths.mean()
            ),
            "metric_policy_crc_selector_left_side_per_sample": (
                selection.selector_left_side_per_sample
            ),
            "metric_policy_crc_deployment_target_per_sample": (
                selection.deployment_budget_per_sample
            ),
            "metric_policy_crc_correction_per_sample": (
                selection.correction_per_sample
            ),
            "metric_policy_crc_guarantee_kind": selection.guarantee_kind,
            "metric_policy_crc_selector_valid": int(
                selection.selector_left_side_per_sample
                <= selection.deployment_budget_per_sample + 1e-12
            ),
            "metric_policy_predicted_event_rate": float(pooled_event.sum()),
            "metric_policy_pooled_pav_block_count": int(block_counts[0]),
            "metric_policy_control_indices_use_labels_only_for_crc_cost": 1,
        }
        return BudgetAllocationResult(
            quantile_est,
            final_c,
            final_propensity,
            int(control_cost + deploy_realized_cost),
            mean_weight=float(inverse.mean().item()),
            max_weight=float(inverse.max().item()),
            additional_metrics=metrics,
        )


class MetricPrefixNeymanCRCAllocator(MetricOptimalPooledTimeCRCAllocator):
    """Closed-form, history-adaptive metric allocation with CRC control.

    This is the lightweight history-adaptive counterpart of the pooled
    schedule.  It learns no time/score bins and performs no DAPRO coordinate
    optimization.  At each acquired prefix it evaluates the square-root ratio
    of the model's remaining unsafe-event probability to expected remaining
    acquisition cost.  Cumulative reach is the running minimum of this index,
    scaled by one scalar and bounded by a shared front-loaded envelope.

    The scalar is selected on fully observed control rows by the same nested-
    family CRC rule used elsewhere in the repository.  The envelope has
    horizon sum at most ``row_cost_cap_multiplier * B``; this supplies the
    finite candidate-cost bound required for a useful CRC correction.
    """

    def __init__(
            self,
            conditional_grid: torch.Tensor,
            budget_per_sample: float,
            taus_range: torch.Tensor,
            tau_prior: float,
            m_upper_bound: int,
            *,
            control_size: int = 50,
            row_cost_cap_multiplier: float = 2.0,
            candidate_count: int = 401,
            terminal_pi_min: float | None = None,
    ):
        super().__init__(
            budget_per_sample,
            taus_range,
            tau_prior,
            m_upper_bound,
            control_size=control_size,
            row_cost_cap_multiplier=row_cost_cap_multiplier,
            candidate_count=candidate_count,
            terminal_pi_min=terminal_pi_min,
        )
        self.conditional_grid = conditional_grid
        self.last_prefix_index = None

    @property
    def name(self) -> str:
        cap = f"{self.row_cost_cap_multiplier:.2f}".replace(".", "p")
        return (
            "metric_prefix_neyman_crc_control_"
            f"{self.control_size}_row_cap_{cap}x_budget"
        )

    def allocate_budget(
            self,
            probability_est: torch.Tensor,
            x: torch.Tensor,
            t: torch.Tensor,
            quantile_est: torch.Tensor,
    ) -> BudgetAllocationResult:
        del x
        n = len(t)
        if len(probability_est) != n or len(quantile_est) != n:
            raise ValueError("Allocator inputs must have the same row count.")
        if not torch.is_tensor(self.conditional_grid):
            raise ValueError("Prefix-Neyman allocation requires a conditional grid.")
        if len(self.conditional_grid) != n:
            raise ValueError(
                "The conditional grid and allocator inputs must be row-aligned."
            )
        if self.control_size >= n:
            raise ValueError("`control_size` must be smaller than the row count.")

        (
            _event,
            _at_risk,
            pooled_event,
            _pooled_at_risk,
            bases,
            block_counts,
        ) = self._pooled_model_shape(probability_est)
        row_cap = min(
            float(self.m_upper_bound),
            self.row_cost_cap_multiplier * self.budget_per_sample,
        )
        minimum_horizon_cost = self.m_upper_bound * self.terminal_pi_min
        if row_cap < minimum_horizon_cost - 1e-12:
            raise ValueError(
                "The row cost cap is smaller than the terminal-floor horizon "
                "cost. Increase the budget/cap or reduce the propensity floor."
            )
        cap_scale = row_horizon_cap_scales(
            bases,
            row_cap,
            self.terminal_pi_min,
        )[0]
        envelope = cumulative_paths_for_scale(
            bases,
            cap_scale,
            self.terminal_pi_min,
        )[0]

        raw_index = prefix_remaining_event_cost_index(
            self.conditional_grid,
            self.m_upper_bound,
        )
        running_index = np.minimum.accumulate(raw_index, axis=1)
        self.last_prefix_index = raw_index.copy()

        permutation = (
            np.random.permutation(n)
            if self.acquisition_seed is None
            else np.random.RandomState(int(self.acquisition_seed)).permutation(n)
        )
        control_indices = permutation[:self.control_size]
        deployment_indices = permutation[self.control_size:]
        self.last_control_indices = control_indices.copy()
        self.last_deployment_indices = deployment_indices.copy()

        active_lengths = np.minimum(
            np.asarray(t.reshape(-1).detach().cpu(), dtype=np.int64),
            self.m_upper_bound,
        ).clip(min=1)
        control_lengths = active_lengths[control_indices]
        alphas = np.linspace(1.0, 0.0, self.candidate_count)
        scales = np.empty_like(alphas)
        scales[0] = np.inf
        scales[1:] = alphas[1:] / np.maximum(
            1.0 - alphas[1:],
            1e-300,
        )

        control_index = running_index[control_indices]
        scaled = np.zeros(
            (
                self.control_size,
                self.candidate_count,
                self.m_upper_bound,
            ),
            dtype=np.float64,
        )
        np.multiply(
            control_index[:, None, :],
            scales[None, :, None],
            out=scaled,
            where=control_index[:, None, :] > 0,
        )
        candidate_cumulative = np.minimum(
            envelope[None, None, :],
            np.clip(scaled, self.terminal_pi_min, 1.0),
        )
        candidate_prefix_cost = np.cumsum(candidate_cumulative, axis=2)
        candidate_costs = np.take_along_axis(
            candidate_prefix_cost,
            (control_lengths - 1)[:, None, None],
            axis=2,
        )[:, :, 0]
        del scaled, candidate_cumulative, candidate_prefix_cost

        selection = select_crc_budget_candidate(
            candidate_costs,
            control_lengths.astype(np.float64),
            total_budget_after_policy_fit=self.budget_per_sample * n,
            deployment_sample_count=len(deployment_indices),
            maximum_cost_per_sample=self.m_upper_bound,
            maximum_candidate_cost_per_sample=row_cap,
            maximum_pilot_cost_per_sample=self.m_upper_bound,
        )
        selected_scale = scales[selection.selected_index]
        deployment_cumulative = np.minimum(
            envelope[None, :],
            cumulative_paths_for_scale(
                running_index[deployment_indices],
                selected_scale,
                self.terminal_pi_min,
            ),
        )
        self.last_selected_candidate_index = int(selection.selected_index)

        all_uniforms = self.get_acquisition_uniforms(
            n,
            self.m_upper_bound,
            device=t.device,
            dtype=torch.float64,
        )
        deploy_c, deploy_propensity, deploy_realized_cost = self._simulate_rows(
            deployment_cumulative,
            deployment_indices,
            t,
            all_uniforms=all_uniforms,
        )
        device = t.device
        final_c = torch.full(
            (n,),
            self.m_upper_bound,
            dtype=torch.long,
            device=device,
        )
        final_propensity = torch.ones(n, dtype=torch.float64, device=device)
        deployment_tensor = torch.as_tensor(
            deployment_indices,
            dtype=torch.long,
            device=device,
        )
        final_c[deployment_tensor] = deploy_c.to(torch.long)
        final_propensity[deployment_tensor] = deploy_propensity

        control_cost = float(control_lengths.sum())
        deployment_prefix_cost = np.cumsum(deployment_cumulative, axis=1)
        deployment_expected_total = float(
            deployment_prefix_cost[
                np.arange(len(deployment_indices)),
                active_lengths[deployment_indices] - 1,
            ].sum()
        )
        total_expected = control_cost + deployment_expected_total
        full_cumulative = np.ones((n, self.m_upper_bound), dtype=np.float64)
        full_cumulative[deployment_indices] = deployment_cumulative
        self.last_cumulative_probabilities = full_cumulative.copy()
        inverse = 1 / final_propensity
        metrics = {
            **self._common_metrics(
                cumulative=full_cumulative,
                block_counts=block_counts,
                t=t,
                total_expected_budget=total_expected,
                cost_semantics=(
                    "fully_observed_crc_control_plus_causal_prefix_neyman_"
                    "schedule_with_event_stopping"
                ),
            ),
            "generalized_dapro": 0,
            "simple_metric_policy": 1,
            "objective_kind": "prefix_remaining_event_per_cost_neyman_index",
            "generalized_dapro_coefficient_estimator": "not_applicable",
            "generalized_dapro_policy_class": "not_applicable",
            "metric_policy_budget_control_mode": "crc",
            "metric_policy_uses_initial_pmf_only": 0,
            "metric_policy_uses_future_history": 0,
            "metric_policy_uses_current_prefix": 1,
            "metric_policy_cumulative_reach_optimized": 0,
            "metric_policy_pooled_across_rows": 0,
            "metric_policy_row_personalized": 1,
            "metric_policy_shared_time_schedule": 0,
            "metric_policy_closed_form_no_dapro_optimization": 1,
            "metric_policy_prefix_index": (
                "sqrt_remaining_event_probability_per_expected_remaining_cost"
            ),
            "metric_policy_prefix_ancestry_operator": "running_minimum",
            "metric_policy_crc_control_size": self.control_size,
            "metric_policy_crc_deployment_size": len(deployment_indices),
            "metric_policy_crc_candidate_count": self.candidate_count,
            "metric_policy_crc_selected_index": selection.selected_index,
            "metric_policy_crc_selected_alpha": float(
                alphas[selection.selected_index]
            ),
            "metric_policy_crc_selected_scale": float(selected_scale),
            "metric_policy_crc_row_cost_cap": row_cap,
            "metric_policy_crc_row_cost_cap_multiplier": (
                self.row_cost_cap_multiplier
            ),
            "metric_policy_crc_empirical_control_policy_cost_per_sample": (
                selection.empirical_expected_cost_per_sample
            ),
            "metric_policy_crc_control_pilot_cost_per_sample": float(
                control_lengths.mean()
            ),
            "metric_policy_crc_selector_left_side_per_sample": (
                selection.selector_left_side_per_sample
            ),
            "metric_policy_crc_deployment_target_per_sample": (
                selection.deployment_budget_per_sample
            ),
            "metric_policy_crc_correction_per_sample": (
                selection.correction_per_sample
            ),
            "metric_policy_crc_guarantee_kind": selection.guarantee_kind,
            "metric_policy_crc_selector_valid": int(
                selection.selector_left_side_per_sample
                <= selection.deployment_budget_per_sample + 1e-12
            ),
            "metric_policy_predicted_event_rate": float(pooled_event.sum()),
            "metric_policy_pooled_pav_block_count": int(block_counts[0]),
            "metric_policy_control_indices_use_labels_only_for_crc_cost": 1,
            "metric_policy_envelope_horizon_cost": float(envelope.sum()),
        }
        return BudgetAllocationResult(
            quantile_est,
            final_c,
            final_propensity,
            int(control_cost + deploy_realized_cost),
            mean_weight=float(inverse.mean().item()),
            max_weight=float(inverse.max().item()),
            additional_metrics=metrics,
        )


__all__ = [
    "MetricOptimalPMFAllocator",
    "MetricOptimalPMFCRCAllocator",
    "MetricOptimalPooledTimeAllocator",
    "MetricOptimalPooledTimeCRCAllocator",
    "MetricPrefixNeymanCRCAllocator",
    "antitonic_pav_bases",
    "cumulative_paths_for_scale",
    "cumulative_to_conditionals",
    "initial_event_and_at_risk_probabilities",
    "prefix_remaining_event_cost_index",
    "row_horizon_cap_scales",
    "solve_common_scale",
]
