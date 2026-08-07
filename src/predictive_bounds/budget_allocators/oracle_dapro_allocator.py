"""Oracle Target-A DAPRO benchmarks.

These allocators deliberately use latent event times in the allocation rule.
They are not deployable methods.  Their purpose is to isolate the loss caused
by learning and projecting DAPRO probabilities from the loss inherent in the
budget itself.

For active length ``L_i=min(T_i,q_i)`` and target indicator ``A_i``, let
``R_i(t)`` be the cumulative probability of reaching interaction ``t``.  The
oracle solves

    minimize  mean_i A_i * (1 / R_i(L_i) - 1)
    subject to mean_i sum_{t <= L_i} R_i(t) <= B.

For a fixed terminal reach ``r_i``, the cheapest feasible path has
``R_i(t)=r_i`` at every active time.  Consequently the matrix problem reduces
to a one-dimensional water-filling solution

    r_i = clip(s * sqrt(A_i / L_i), epsilon, 1),

where ``s`` is chosen to meet the expected budget.  One equivalent conditional
policy draws the row-level inclusion coin at its first interaction and then
continues with probability one.  This is the exact unrestricted optimum; it
does not retain DAPRO's deployability or score-monotonicity constraints.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.predictive_bounds.budget_allocators.budget_allocator import (
    BudgetAllocationResult,
    BudgetAllocator,
    summarize_expected_budget,
)
from src.predictive_bounds.budget_allocators.risk_controlled_budget import (
    select_crc_budget_candidate,
)
from src.predictive_bounds.budget_allocators.vectorized_adaptive_allocator_patch import (
    simulate_process_vectorized,
)
from src.predictive_bounds.calibration.calibration_utils import (
    get_prior,
    select_calibration_positions,
)


@dataclass(frozen=True)
class OraclePolicySolution:
    """Exact unrestricted conditional policy and its diagnostics."""

    continuation_probabilities: torch.Tensor
    terminal_probabilities: torch.Tensor
    expected_cost_per_sample: float
    objective_per_sample: float
    scale: float
    budget_boundary: str


def _terminal_probabilities_for_scale(
        lengths: np.ndarray,
        objective_weights: np.ndarray,
        scale: float,
        minimum_terminal_probability: float,
) -> np.ndarray:
    terminal = np.full(
        len(lengths),
        minimum_terminal_probability,
        dtype=np.float64,
    )
    positive = (lengths > 0) & (objective_weights > 0)
    terminal[positive] = np.clip(
        scale * np.sqrt(
            objective_weights[positive] / lengths[positive]
        ),
        minimum_terminal_probability,
        1.0,
    )
    terminal[lengths == 0] = 1.0
    return terminal


def _solution_at_fixed_scale(
        active_lengths: torch.Tensor,
        objective_weights: torch.Tensor,
        scale: float,
        width: int,
        minimum_terminal_probability: float,
        budget_boundary: str,
) -> OraclePolicySolution:
    lengths = np.asarray(active_lengths.detach().cpu(), dtype=np.int64)
    weights = np.asarray(objective_weights.detach().cpu(), dtype=np.float64)
    terminal = _terminal_probabilities_for_scale(
        lengths,
        weights,
        scale,
        minimum_terminal_probability,
    )
    conditionals = np.ones((len(lengths), width), dtype=np.float64)
    conditionals[lengths > 0, 0] = terminal[lengths > 0]
    positive = (lengths > 0) & (weights > 0)
    contributions = np.zeros(len(lengths), dtype=np.float64)
    contributions[positive] = weights[positive] * (
        1 / terminal[positive] - 1
    )
    return OraclePolicySolution(
        continuation_probabilities=torch.as_tensor(
            conditionals,
            dtype=torch.float64,
            device=active_lengths.device,
        ),
        terminal_probabilities=torch.as_tensor(
            terminal,
            dtype=torch.float64,
            device=active_lengths.device,
        ),
        expected_cost_per_sample=float(np.mean(lengths * terminal)),
        objective_per_sample=float(np.mean(contributions)),
        scale=float(scale),
        budget_boundary=budget_boundary,
    )


def solve_unrestricted_oracle_policy(
        active_lengths: torch.Tensor | np.ndarray,
        objective_weights: torch.Tensor | np.ndarray,
        budget_per_sample: float,
        width: int,
        *,
        minimum_terminal_probability: float = 1e-12,
        tolerance: float = 1e-10,
) -> OraclePolicySolution:
    """Solve the exact full-information Target-A allocation problem."""
    lengths = np.asarray(
        (
            active_lengths.detach().cpu()
            if torch.is_tensor(active_lengths)
            else active_lengths
        ),
        dtype=np.int64,
    ).reshape(-1)
    weights = np.asarray(
        (
            objective_weights.detach().cpu()
            if torch.is_tensor(objective_weights)
            else objective_weights
        ),
        dtype=np.float64,
    ).reshape(-1)
    if width <= 0:
        raise ValueError("`width` must be positive.")
    if len(lengths) == 0 or len(lengths) != len(weights):
        raise ValueError("Lengths and objective weights must be nonempty and agree.")
    if np.any(lengths < 0) or np.any(lengths > width):
        raise ValueError(f"Active lengths must lie in [0, {width}].")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("Objective weights must be finite and nonnegative.")
    if not np.isfinite(budget_per_sample) or budget_per_sample < 0:
        raise ValueError("`budget_per_sample` must be finite and nonnegative.")
    if not 0 < minimum_terminal_probability <= 1:
        raise ValueError(
            "`minimum_terminal_probability` must lie in (0, 1]."
        )
    if np.any((weights > 0) & (lengths == 0)):
        raise ValueError(
            "A positive Target-A indicator cannot have zero active length."
        )

    minimum_terminal = _terminal_probabilities_for_scale(
        lengths,
        weights,
        0.0,
        minimum_terminal_probability,
    )
    minimum_cost = float(np.mean(lengths * minimum_terminal))
    if budget_per_sample < minimum_cost - tolerance:
        raise ValueError(
            "The oracle budget is infeasible under the numerical positivity "
            f"floor: target={budget_per_sample:.6g}, minimum={minimum_cost:.6g}."
        )

    positive = (lengths > 0) & (weights > 0)
    saturation_scale = (
        float(np.max(np.sqrt(lengths[positive] / weights[positive])))
        if np.any(positive)
        else 0.0
    )
    saturated_terminal = _terminal_probabilities_for_scale(
        lengths,
        weights,
        saturation_scale,
        minimum_terminal_probability,
    )
    saturation_cost = float(np.mean(lengths * saturated_terminal))

    if budget_per_sample >= saturation_cost - tolerance:
        scale = saturation_scale
        terminal = saturated_terminal
        boundary = "objective_plateau"
    else:
        low, high = 0.0, saturation_scale
        for _ in range(100):
            mid = (low + high) / 2
            candidate = _terminal_probabilities_for_scale(
                lengths,
                weights,
                mid,
                minimum_terminal_probability,
            )
            cost = float(np.mean(lengths * candidate))
            if cost <= budget_per_sample:
                low = mid
            else:
                high = mid
        scale = low
        terminal = _terminal_probabilities_for_scale(
            lengths,
            weights,
            scale,
            minimum_terminal_probability,
        )
        boundary = "budget"

    tensor_lengths = torch.as_tensor(
        lengths,
        dtype=torch.long,
        device=(
            active_lengths.device
            if torch.is_tensor(active_lengths)
            else torch.device("cpu")
        ),
    )
    tensor_weights = torch.as_tensor(
        weights,
        dtype=torch.float64,
        device=tensor_lengths.device,
    )
    return _solution_at_fixed_scale(
        tensor_lengths,
        tensor_weights,
        scale,
        width,
        minimum_terminal_probability,
        boundary,
    )


class OracleTargetADAPRO(BudgetAllocator):
    """Full-trajectory Target-A oracle with optional DAPRO/CRC splitting."""

    _VALID_SPLIT_MODES = {"phase1", "phase1_crc", "none"}

    def __init__(
            self,
            conditional_grid: torch.Tensor,
            budget_per_sample: float,
            taus_range: torch.Tensor,
            tau_prior: float,
            m_upper_bound: float,
            *,
            split_mode: str,
            n1: int = 200,
            budget_control_size: int = 100,
            target_alpha: float = 0.10,
            metric_estimation_horizon: int | None = None,
            minimum_terminal_probability: float = 1e-12,
            budget_candidate_count: int = 2001,
            reach_t_max_is_success: bool = False,
    ):
        super().__init__(budget_per_sample, taus_range, tau_prior)
        if split_mode not in self._VALID_SPLIT_MODES:
            raise ValueError(
                f"`split_mode` must be one of {sorted(self._VALID_SPLIT_MODES)}."
            )
        if not 0 < target_alpha < tau_prior:
            raise ValueError("`target_alpha` must lie in (0, tau_prior).")
        if n1 <= 0:
            raise ValueError("`n1` must be positive.")
        if split_mode == "phase1_crc" and not 0 < budget_control_size < n1:
            raise ValueError("CRC requires 0 < budget_control_size < n1.")
        if budget_candidate_count < 2:
            raise ValueError("`budget_candidate_count` must be at least two.")
        if not np.isfinite(m_upper_bound) or m_upper_bound < 1:
            raise ValueError("`m_upper_bound` must be finite and at least one.")
        self.conditional_grid = conditional_grid
        self.split_mode = split_mode
        self.n1 = int(n1)
        self.budget_control_size = int(budget_control_size)
        self.target_alpha = float(target_alpha)
        if metric_estimation_horizon is not None and metric_estimation_horizon <= 0:
            raise ValueError("`metric_estimation_horizon` must be positive.")
        self.metric_estimation_horizon = (
            None
            if metric_estimation_horizon is None
            else int(metric_estimation_horizon)
        )
        self.minimum_terminal_probability = float(
            minimum_terminal_probability
        )
        self.budget_candidate_count = int(budget_candidate_count)
        self.reach_t_max_is_success = bool(reach_t_max_is_success)
        self.m_upper_bound = float(m_upper_bound)
        self.last_continuation_probabilities = None
        self.last_phase1_indices = None
        self.last_control_indices = None
        self.last_phase2_indices = None

    @property
    def name(self) -> str:
        alpha = f"{self.target_alpha:.2f}".replace(".", "p")
        if self.split_mode == "none":
            return f"oracle_target_a_dapro_no_split_alpha_{alpha}"
        base = f"oracle_target_a_dapro_alpha_{alpha}"
        if self.split_mode == "phase1_crc":
            base += f"_crc_control_{self.budget_control_size}"
        return f"{base}_n1_{self.n1}"

    def _target_indicator(
            self,
            event_times: torch.Tensor,
            quantile_est: torch.Tensor,
            prior_q: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        if self.metric_estimation_horizon is not None:
            return (
                (
                    event_times.reshape(-1)
                    <= self.metric_estimation_horizon
                ).to(torch.float64),
                -1,
            )
        prior_index = int(
            torch.abs(self.taus_range - self.tau_prior).argmin().item()
        )
        candidate_taus = self.taus_range[:prior_index + 1]
        anchor_index = int(select_calibration_positions(
            candidate_taus,
            torch.tensor(
                [self.target_alpha],
                dtype=candidate_taus.dtype,
                device=candidate_taus.device,
            ),
        ).item())
        anchor_q = quantile_est[:, anchor_index].reshape(-1)
        if torch.any(anchor_q > prior_q.reshape(-1) + 1e-6):
            raise ValueError(
                "The oracle Target-A anchor must lie inside q_prior."
            )
        return (
            (event_times.reshape(-1) < anchor_q).to(torch.float64),
            anchor_index,
        )

    def _crc_scale(
            self,
            control_lengths: torch.Tensor,
            control_weights: torch.Tensor,
            policy_fit_cost: float,
            deployment_count: int,
            total_sample_count: int,
            width: int,
    ) -> tuple[float, dict]:
        lengths = np.asarray(control_lengths.detach().cpu(), dtype=np.int64)
        weights = np.asarray(control_weights.detach().cpu(), dtype=np.float64)
        # Target-A weights are binary, so sqrt(width) saturates every possible
        # positive row.  Crucially this endpoint, and hence the entire nested
        # candidate family, is fixed before the CRC fold is observed.
        maximum_scale = float(np.sqrt(width))
        scales = np.linspace(
            maximum_scale,
            0.0,
            self.budget_candidate_count,
        )
        base = np.zeros(len(lengths), dtype=np.float64)
        positive = (lengths > 0) & (weights > 0)
        base[positive] = np.sqrt(weights[positive] / lengths[positive])
        terminal = np.clip(
            base[:, None] * scales[None, :],
            self.minimum_terminal_probability,
            1.0,
        )
        terminal[lengths == 0] = 1.0
        candidate_costs = lengths[:, None] * terminal
        selection = select_crc_budget_candidate(
            candidate_costs,
            lengths.astype(np.float64),
            total_budget_after_policy_fit=(
                self.budget_per_sample * total_sample_count
                - policy_fit_cost
            ),
            deployment_sample_count=deployment_count,
            maximum_cost_per_sample=width,
            maximum_candidate_cost_per_sample=width,
            maximum_pilot_cost_per_sample=width,
        )
        return float(scales[selection.selected_index]), {
            "risk_budget_control_enabled": 1,
            "risk_budget_control_mode": "crc",
            "risk_budget_control_size": len(lengths),
            "risk_budget_candidate_count": self.budget_candidate_count,
            "risk_budget_selected_index": selection.selected_index,
            "risk_budget_selector_left_side_per_sample": (
                selection.selector_left_side_per_sample
            ),
            "risk_budget_deployment_target_per_sample": (
                selection.deployment_budget_per_sample
            ),
            "risk_budget_correction_per_sample": selection.correction_per_sample,
            "risk_budget_guarantee_kind": selection.guarantee_kind,
            "risk_budget_selector_valid": int(
                selection.selector_left_side_per_sample
                <= selection.deployment_budget_per_sample + 1e-12
            ),
        }

    def allocate_budget(
            self,
            probability_est: torch.Tensor,
            x: torch.Tensor,
            t: torch.Tensor,
            quantile_est: torch.Tensor,
    ) -> BudgetAllocationResult:
        del probability_est, x
        device = self.conditional_grid.device
        n, width, _ = self.conditional_grid.shape
        if len(t) != n or len(quantile_est) != n:
            raise ValueError("Allocator inputs must match the grid row count.")
        if self.split_mode != "none" and self.n1 >= n:
            raise ValueError("`n1` must be smaller than the sample count.")

        prior_q = get_prior(quantile_est, self.taus_range, self.tau_prior)
        lengths = torch.minimum(t.reshape(-1), prior_q.reshape(-1)).to(
            device=device,
            dtype=torch.long,
        ).clamp(min=0, max=width)
        target_a, anchor_index = self._target_indicator(
            t,
            quantile_est,
            prior_q,
        )
        target_a = target_a.to(device=device)

        if self.split_mode == "none":
            phase1_indices = np.empty(0, dtype=np.int64)
            control_indices = np.empty(0, dtype=np.int64)
            phase2_indices = np.arange(n, dtype=np.int64)
        else:
            permutation = np.random.permutation(n)
            phase1_indices = permutation[:self.n1]
            phase2_indices = permutation[self.n1:]
            if self.split_mode == "phase1_crc":
                fit_count = self.n1 - self.budget_control_size
                control_indices = phase1_indices[fit_count:]
            else:
                control_indices = np.empty(0, dtype=np.int64)

        self.last_phase1_indices = phase1_indices.copy()
        self.last_control_indices = control_indices.copy()
        self.last_phase2_indices = phase2_indices.copy()
        phase1_cost = float(lengths[phase1_indices].sum().item())
        remaining_total_budget = self.budget_per_sample * n - phase1_cost
        if remaining_total_budget < 0:
            raise ValueError(
                "Fully observing the oracle Phase-I rows exceeds the total budget."
            )

        crc_metrics = {
            "risk_budget_control_enabled": 0,
            "risk_budget_control_mode": "none",
        }
        if self.split_mode == "phase1_crc":
            fit_count = self.n1 - self.budget_control_size
            fit_indices = phase1_indices[:fit_count]
            policy_fit_cost = float(lengths[fit_indices].sum().item())
            crc_scale, crc_metrics = self._crc_scale(
                lengths[control_indices],
                target_a[control_indices],
                policy_fit_cost,
                len(phase2_indices),
                n,
                width,
            )
            exact_budget_solution = solve_unrestricted_oracle_policy(
                lengths[phase2_indices],
                target_a[phase2_indices],
                remaining_total_budget / len(phase2_indices),
                width,
                minimum_terminal_probability=(
                    self.minimum_terminal_probability
                ),
            )
            deployed_scale = min(crc_scale, exact_budget_solution.scale)
            deployment_solution = _solution_at_fixed_scale(
                lengths[phase2_indices],
                target_a[phase2_indices],
                deployed_scale,
                width,
                self.minimum_terminal_probability,
                (
                    "crc"
                    if crc_scale <= exact_budget_solution.scale
                    else "exact_phase2_budget_cap"
                ),
            )
            crc_metrics.update({
                "oracle_crc_selected_scale": crc_scale,
                "oracle_crc_deployed_scale": deployed_scale,
                "oracle_crc_exact_phase2_budget_cap_active": int(
                    crc_scale > exact_budget_solution.scale
                ),
            })
        else:
            deployment_solution = solve_unrestricted_oracle_policy(
                lengths[phase2_indices],
                target_a[phase2_indices],
                remaining_total_budget / len(phase2_indices),
                width,
                minimum_terminal_probability=(
                    self.minimum_terminal_probability
                ),
            )

        full_probabilities = torch.ones(
            (n, width),
            dtype=torch.float64,
            device=device,
        )
        full_probabilities[phase2_indices] = (
            deployment_solution.continuation_probabilities
        )
        self.last_continuation_probabilities = full_probabilities.detach().clone()

        acquisition_uniforms = self.get_acquisition_uniforms(
            n,
            width,
            device=device,
            dtype=torch.float64,
        )
        phase2_uniforms = (
            None
            if acquisition_uniforms is None
            else acquisition_uniforms[phase2_indices]
        )
        if acquisition_uniforms is None:
            self.reset_acquisition_rng()
        phase2_c, phase2_propensity, phase2_realized_cost = (
            simulate_process_vectorized(
                torch.ones(
                    (len(phase2_indices), width),
                    dtype=torch.float64,
                    device=device,
                ),
                prior_q[phase2_indices],
                t[phase2_indices],
                0.0,
                stochastic=True,
                reach_t_max_is_success=self.reach_t_max_is_success,
                uniforms=phase2_uniforms,
                pi_func=lambda _: deployment_solution.continuation_probabilities,
                terminal_pi_min=None,
                terminal_floor_mode="none",
            )
        )

        final_c = torch.empty(n, dtype=torch.long, device=device)
        final_propensity = torch.ones(n, dtype=torch.float64, device=device)
        final_c[phase1_indices] = prior_q[phase1_indices].to(torch.long)
        final_c[phase2_indices] = phase2_c.to(torch.long)
        final_propensity[phase2_indices] = phase2_propensity

        phase2_expected_total = (
            deployment_solution.expected_cost_per_sample * len(phase2_indices)
        )
        total_expected = phase1_cost + phase2_expected_total
        expected_metrics = summarize_expected_budget(
            total_expected,
            n,
            self.budget_per_sample,
            cost_semantics=(
                "fully_observed_phase1_plus_oracle_phase2_expected_interactions"
                if self.split_mode != "none"
                else "oracle_expected_interactions_all_rows"
            ),
        )
        inverse = 1 / final_propensity
        objective_contributions = target_a * (inverse - 1)
        additional_metrics = {
            "objective_kind": (
                "mean_metric_event_weighted_inverse_probability_minus_one"
                if self.metric_estimation_horizon is not None
                else "mean_target_a_weighted_inverse_probability_minus_one"
            ),
            "oracle_dapro": 1,
            "oracle_uses_full_trajectories": 1,
            "oracle_split_mode": self.split_mode,
            "oracle_policy_is_deployable": 0,
            "oracle_score_monotonicity_constraint": 0,
            "target_anchor_alpha": self.target_alpha,
            "target_anchor_index": anchor_index,
            "target_anchor_tau": (
                np.nan
                if anchor_index < 0
                else float(self.taus_range[anchor_index].item())
            ),
            "target_event_kind": (
                "metric_event_by_fixed_horizon"
                if self.metric_estimation_horizon is not None
                else "lpb_anchor_event"
            ),
            "target_metric": (
                "unsafe_event_rate"
                if self.metric_estimation_horizon is not None
                else "lpb_miscoverage"
            ),
            "target_metric_horizon": (
                self.metric_estimation_horizon
                if self.metric_estimation_horizon is not None
                else np.nan
            ),
            "target_a_rate": float(target_a.mean().item()),
            "oracle_mean_target_a_weighted_inverse_probability_minus_one": (
                float(objective_contributions.mean().item())
            ),
            "oracle_phase2_optimized_objective": (
                deployment_solution.objective_per_sample
            ),
            "oracle_water_filling_scale": deployment_solution.scale,
            "oracle_budget_boundary": deployment_solution.budget_boundary,
            "minimum_terminal_probability": self.minimum_terminal_probability,
            "phase1_sample_count": len(phase1_indices),
            "phase1_expected_cost_total": phase1_cost,
            "phase1_expected_cost_per_sample": (
                phase1_cost / len(phase1_indices)
                if len(phase1_indices)
                else 0.0
            ),
            "phase1_all_probabilities_one": 1,
            "phase2_sample_count": len(phase2_indices),
            "phase2_expected_cost_total": phase2_expected_total,
            "phase2_expected_cost_per_sample": (
                deployment_solution.expected_cost_per_sample
            ),
            "phase2_realized_cost_per_sample": (
                phase2_realized_cost / len(phase2_indices)
            ),
            "crc_split_sample_count": len(control_indices),
            "crc_split_all_probabilities_one": 1,
            **crc_metrics,
            **expected_metrics,
        }
        return BudgetAllocationResult(
            quantile_est,
            final_c,
            final_propensity,
            int(phase1_cost + phase2_realized_cost),
            mean_weight=float(inverse.mean().item()),
            max_weight=float(inverse.max().item()),
            additional_metrics=additional_metrics,
        )


class SplitOracleTargetADAPRO(OracleTargetADAPRO):
    """Oracle with the ordinary DAPRO Phase-I/Phase-II split."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, split_mode="phase1", **kwargs)


class CRCOracleTargetADAPRO(OracleTargetADAPRO):
    """Oracle with policy-fit, independent CRC, and Phase-II splits."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, split_mode="phase1_crc", **kwargs)


class GlobalOracleTargetADAPRO(OracleTargetADAPRO):
    """No-split oracle optimizing all calibration rows under nominal budget."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, split_mode="none", **kwargs)
