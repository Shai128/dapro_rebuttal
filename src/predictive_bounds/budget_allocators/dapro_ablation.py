"""Controlled LPB and metric ablations for history-adaptive DAPRO.

The production allocator defaults are intentionally unchanged. These wrappers
expose orthogonal experimental axes: coefficient estimator, representation,
score definition, and budget controller.
"""

from __future__ import annotations

import math
import re

import numpy as np
import torch

from src.predictive_bounds.budget_allocators.DAPRO import (
    DefinitiveCRCDAPRO,
    DefinitiveDAPRO,
    SoftTargetCRCDAPRO,
    SoftTargetDAPRO,
)
from src.predictive_bounds.budget_allocators.budget_allocator import (
    BudgetAllocationResult,
    BudgetAllocator,
    candidate_reach_probabilities,
    summarize_expected_budget,
)
from src.predictive_bounds.budget_allocators.risk_controlled_budget import (
    cumulative_policy_costs,
    select_crc_budget_candidate,
)
from src.predictive_bounds.budget_allocators.vectorized_adaptive_allocator_patch import (
    simulate_process_vectorized,
)
from src.predictive_bounds.calibration.calibration_utils import (
    get_prior,
)


_VALID_ABLATION_KINDS = {
    "n1", "score_noise", "budget", "hard_soft", "representation",
    "score", "cmax", "attacker_shift", "optimization",
}
_VALID_SCORE_KINDS = {
    "hazard", "remaining_quantile", "target_value", "random",
    "oracle_remaining_time",
}


def _float_token(value: float) -> str:
    return np.format_float_positional(float(value), trim="-").replace(".", "p")


def _name_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


class _DAPROAblationMixin:
    """Identity, diagnostics, and score alternatives for an ablation."""

    def _initialize_ablation(
            self, *, ablation_kind: str, ablation_value: float,
            ablation_label: str = "", score_kind: str = "hazard",
            score_noise_lambda: float = 0.0,
            score_noise_seed: int = 314159) -> None:
        kind = str(ablation_kind).lower()
        if kind not in _VALID_ABLATION_KINDS:
            raise ValueError(
                f"Unknown DAPRO ablation kind {ablation_kind!r}; expected "
                f"one of {sorted(_VALID_ABLATION_KINDS)}."
            )
        value = float(ablation_value)
        if not math.isfinite(value):
            raise ValueError("`ablation_value` must be finite.")
        score_kind = str(score_kind).lower()
        if score_kind not in _VALID_SCORE_KINDS:
            raise ValueError(
                f"Unknown score kind {score_kind!r}; expected one of "
                f"{sorted(_VALID_SCORE_KINDS)}."
            )
        noise = float(score_noise_lambda)
        if not 0.0 <= noise <= 1.0:
            raise ValueError("`score_noise_lambda` must lie in [0, 1].")
        self.ablation_kind = kind
        self.ablation_value = value
        self.ablation_label = str(ablation_label or _float_token(value))
        self.ablation_score_kind = score_kind
        self.score_noise_lambda = noise
        self.score_noise_seed = int(score_noise_seed)
        self._ablation_score_mean_timewise_correlation = np.nan
        self._ablation_score_k2_bin_agreement = np.nan

    @property
    def name(self) -> str:
        return (
            f"{super().name}_ablation_{self.ablation_kind}_"
            f"{_float_token(self.ablation_value)}_"
            f"{_name_token(self.ablation_label)}"
        )

    def _fixed_alpha_horizons(
            self, quantile_est: torch.Tensor) -> torch.Tensor:
        taus = torch.as_tensor(
            self.taus_range, dtype=torch.float64, device=quantile_est.device
        )
        eligible = torch.nonzero(
            taus < self.target_alpha, as_tuple=False
        ).reshape(-1)
        if len(eligible) == 0:
            raise ValueError(
                "The LPB tau grid has no candidate strictly below target alpha."
            )
        index = int(eligible[-1].item())
        return quantile_est[:, index].reshape(-1).clamp(
            min=1, max=self.conditional_grid.shape[1]
        )

    def _task_target_horizons(
            self, quantile_est: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """Return row horizons and whether the target endpoint is strict.

        LPB uses ``1{T < q_alpha(X)}``.  Metric estimation instead targets
        the unsafe-event rate ``1{T <= M}``, where ``M`` is the fixed metric
        horizon.  Keeping this choice in one helper makes every score-quality
        anchor use exactly the same target as the fitted DAPRO objective.
        """
        metric_horizon = getattr(self, "metric_estimation_horizon", None)
        if metric_horizon is None:
            return self._fixed_alpha_horizons(quantile_est), True
        horizons = torch.full(
            (len(quantile_est),),
            int(metric_horizon),
            dtype=torch.long,
            device=quantile_est.device,
        )
        return horizons, False

    def _target_value_scores(
            self, quantile_est: torch.Tensor) -> torch.Tensor:
        """Causal remaining probability of the task-specific target event.

        For LPB this is ``P(t < T < q_alpha(X) | T > t, X_it)``.  For the
        event-rate metric it is ``P(t < T <= M | T > t, X_it)``.  PMF outcome
        column zero corresponds to event time one, hence the strict LPB and
        inclusive metric targets differ by one column at the upper endpoint.
        """
        grid = self.conditional_grid
        width = grid.shape[1]
        horizons, strict = self._task_target_horizons(quantile_est)
        horizons = horizons.to(torch.long)
        result = torch.zeros(
            (len(grid), width), dtype=torch.float64, device=grid.device
        )
        for step in range(width):
            pmf = grid[:, step, :]
            cumulative = pmf.cumsum(dim=1)
            upper_offset = 2 if strict else 1
            upper = (horizons - upper_offset).clamp(
                min=0, max=pmf.shape[1] - 1
            )
            mass = cumulative.gather(1, upper[:, None]).squeeze(1)
            if step:
                mass = mass - cumulative[:, step - 1]
            future_mass = pmf[:, step:].sum(dim=1).clamp_min(
                torch.finfo(pmf.dtype).tiny
            )
            target_is_still_possible = (
                horizons > step + 1 if strict else horizons >= step + 1
            )
            result[:, step] = torch.where(
                target_is_still_possible,
                mass.clamp_min(0).to(torch.float64)
                / future_mass.to(torch.float64),
                0.0,
            )
        return result

    def _remaining_quantile_scores(self) -> torch.Tensor:
        """Inverse model median remaining event time at every prefix."""
        grid = self.conditional_grid
        n, width, outcomes = grid.shape
        result = torch.zeros((n, width), dtype=torch.float64, device=grid.device)
        outcome = torch.arange(outcomes, device=grid.device)
        for step in range(width):
            pmf = grid[:, step, :].clamp_min(0)
            future_pmf = pmf * (outcome >= step)[None, :]
            total = future_pmf.sum(dim=1).clamp_min(
                torch.finfo(pmf.dtype).tiny
            )
            cdf = future_pmf.cumsum(dim=1)
            median = (cdf >= 0.5 * total[:, None]).to(torch.int64).argmax(dim=1)
            remaining = (median - step + 1).clamp_min(1).to(torch.float64)
            result[:, step] = remaining.reciprocal()
        return result

    def _permute_timewise(self, scores: torch.Tensor) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.score_noise_seed)
        corrupted = torch.empty_like(scores)
        for time_idx in range(scores.shape[1]):
            permutation = torch.randperm(
                scores.shape[0], generator=generator, device="cpu"
            ).to(scores.device)
            corrupted[:, time_idx] = scores[permutation, time_idx]
        return corrupted

    def _apply_score_noise(self, scores: torch.Tensor) -> torch.Tensor:
        lam = self.score_noise_lambda
        if lam == 0.0:
            self._ablation_score_mean_timewise_correlation = 1.0
            self._ablation_score_k2_bin_agreement = 1.0
            return scores
        corrupted = self._permute_timewise(scores)
        mixed = (1.0 - lam) * scores + lam * corrupted
        centered_original = scores - scores.mean(dim=0, keepdim=True)
        centered_mixed = mixed - mixed.mean(dim=0, keepdim=True)
        denominator = torch.sqrt(
            centered_original.square().sum(dim=0)
            * centered_mixed.square().sum(dim=0)
        )
        valid = denominator > 1e-15
        if bool(valid.any()):
            correlations = (
                (centered_original * centered_mixed).sum(dim=0)[valid]
                / denominator[valid]
            )
            self._ablation_score_mean_timewise_correlation = float(
                correlations.mean().detach().cpu()
            )
        original_bin = scores >= scores.median(dim=0).values[None, :]
        mixed_bin = mixed >= mixed.median(dim=0).values[None, :]
        self._ablation_score_k2_bin_agreement = float(
            (original_bin == mixed_bin).to(torch.float64).mean().cpu()
        )
        return mixed

    def policy_scores(self, quantile_est: torch.Tensor) -> torch.Tensor:
        kind = self.ablation_score_kind
        if kind in {"hazard", "random"}:
            scores = super().policy_scores(quantile_est)
            if kind == "random":
                scores = self._permute_timewise(scores)
        elif kind == "remaining_quantile":
            scores = self._remaining_quantile_scores()
        elif kind == "target_value":
            scores = self._target_value_scores(quantile_est)
        elif kind == "oracle_remaining_time":
            raise RuntimeError("Oracle scores require allocation event times.")
        else:  # pragma: no cover
            raise AssertionError(kind)
        return self._apply_score_noise(scores)

    def policy_scores_for_allocation(
            self, quantile_est: torch.Tensor,
            event_times: torch.Tensor) -> torch.Tensor:
        if self.ablation_score_kind != "oracle_remaining_time":
            return self.policy_scores(quantile_est)
        width = self.conditional_grid.shape[1]
        horizons, strict = self._task_target_horizons(quantile_est)
        horizons = horizons.to(event_times.device)
        times = event_times.reshape(-1).to(torch.float64)
        target = (
            times < horizons if strict else times <= horizons
        ).to(torch.float64)
        step = torch.arange(width, device=times.device, dtype=torch.float64)
        remaining = (times[:, None] - step[None, :]).clamp_min(1.0)
        # Intentionally noncausal: a full-information score-quality anchor.
        return target[:, None] / remaining

    def objective_metadata(self) -> dict:
        metadata = super().objective_metadata()
        is_metric = getattr(self, "metric_estimation_horizon", None) is not None
        coefficient_kind = getattr(
            self,
            "_ablation_coefficient_kind",
            (
                "soft_prefix_model_probability"
                if isinstance(self, (SoftTargetDAPRO, SoftTargetCRCDAPRO))
                else "hard_realized_target_indicator"
            ),
        )
        support_kind = getattr(
            self,
            "_ablation_support_kind",
            (
                "prefix_grid"
                if isinstance(self, (SoftTargetDAPRO, SoftTargetCRCDAPRO))
                else "terminal_endpoint"
            ),
        )
        metadata.update({
            "ablation_study": (
                "dapro_metric_event_rate" if is_metric else "dapro_lpb"
            ),
            "ablation_task": "metric" if is_metric else "lpb",
            "ablation_target_definition": (
                f"1{{T<={self.metric_estimation_horizon}}}"
                if is_metric else "1{T<q_alpha(X)}"
            ),
            "ablation_kind": self.ablation_kind,
            "ablation_value": self.ablation_value,
            "ablation_label": self.ablation_label,
            "ablation_n1": self.n1,
            "ablation_crc_control_size": getattr(
                self, "budget_control_size", 0
            ),
            "ablation_uses_crc": int(
                getattr(self, "budget_control_mode", None) == "crc"
            ),
            "ablation_score_kind": self.ablation_score_kind,
            "ablation_score_is_causal": int(
                self.ablation_score_kind != "oracle_remaining_time"
            ),
            "ablation_score_noise_lambda": self.score_noise_lambda,
            "ablation_score_noise_mode": (
                "per_time_permuted_original_score_mixture"
            ),
            "ablation_score_noise_seed": self.score_noise_seed,
            "ablation_score_mean_timewise_pearson_correlation": (
                self._ablation_score_mean_timewise_correlation
            ),
            "ablation_score_original_k2_bin_agreement": (
                self._ablation_score_k2_bin_agreement
            ),
            "ablation_score_bin_count": self.score_bin_count,
            "ablation_continuous_score_map": int(
                getattr(self, "smooth_score_rank_map", False)
            ),
            "ablation_coefficient_kind": coefficient_kind,
            "ablation_support_kind": support_kind,
            "ablation_row_cost_cap_multiplier": (
                getattr(self, "row_cost_cap_multiplier", np.nan)
            ),
            "ablation_row_cost_cap_applied": int(
                getattr(self, "risk_candidate_row_cost_cap", None) is not None
            ),
        })
        return metadata


class AblationSoftTargetDAPRO(_DAPROAblationMixin, SoftTargetDAPRO):
    _ablation_coefficient_kind = "soft_model_probability"
    _ablation_support_kind = "prefix_grid"

    def __init__(self, *args, ablation_kind: str, ablation_value: float,
                 ablation_label: str = "", score_kind: str = "hazard",
                 score_noise_lambda: float = 0.0,
                 score_noise_seed: int = 314159, **kwargs):
        self._initialize_ablation(
            ablation_kind=ablation_kind, ablation_value=ablation_value,
            ablation_label=ablation_label, score_kind=score_kind,
            score_noise_lambda=score_noise_lambda,
            score_noise_seed=score_noise_seed,
        )
        super().__init__(*args, **kwargs)


class AblationSoftTargetCRCDAPRO(_DAPROAblationMixin, SoftTargetCRCDAPRO):
    _ablation_coefficient_kind = "soft_model_probability"
    _ablation_support_kind = "prefix_grid"

    def __init__(self, *args, ablation_kind: str, ablation_value: float,
                 ablation_label: str = "", score_kind: str = "hazard",
                 score_noise_lambda: float = 0.0,
                 score_noise_seed: int = 314159, **kwargs):
        self._initialize_ablation(
            ablation_kind=ablation_kind, ablation_value=ablation_value,
            ablation_label=ablation_label, score_kind=score_kind,
            score_noise_lambda=score_noise_lambda,
            score_noise_seed=score_noise_seed,
        )
        super().__init__(*args, **kwargs)


class AblationHardTargetDAPRO(_DAPROAblationMixin, DefinitiveDAPRO):
    _ablation_coefficient_kind = "hard_realized_target_indicator"
    _ablation_support_kind = "terminal_endpoint"

    def __init__(self, *args, ablation_kind: str, ablation_value: float,
                 ablation_label: str = "", score_kind: str = "hazard",
                 score_noise_lambda: float = 0.0,
                 score_noise_seed: int = 314159, **kwargs):
        self._initialize_ablation(
            ablation_kind=ablation_kind, ablation_value=ablation_value,
            ablation_label=ablation_label, score_kind=score_kind,
            score_noise_lambda=score_noise_lambda,
            score_noise_seed=score_noise_seed,
        )
        super().__init__(*args, **kwargs)


class AblationHardTargetCRCDAPRO(_DAPROAblationMixin, DefinitiveCRCDAPRO):
    _ablation_coefficient_kind = "hard_realized_target_indicator"
    _ablation_support_kind = "terminal_endpoint"

    def __init__(self, *args, ablation_kind: str, ablation_value: float,
                 ablation_label: str = "", score_kind: str = "hazard",
                 score_noise_lambda: float = 0.0,
                 score_noise_seed: int = 314159, **kwargs):
        self._initialize_ablation(
            ablation_kind=ablation_kind, ablation_value=ablation_value,
            ablation_label=ablation_label, score_kind=score_kind,
            score_noise_lambda=score_noise_lambda,
            score_noise_seed=score_noise_seed,
        )
        super().__init__(*args, **kwargs)


class _HardPrefixObjectiveMixin:
    """Represent the realized hard target on the full prefix-time grid.

    A realized event contributes a one-hot mass at its event prefix.  Hence
    this is algebraically equivalent to the hard terminal objective; keeping
    it as a separate, explicitly grid-valued implementation is useful in the
    four-cell coefficient/support ablation and verifies that changing only the
    internal representation does not change the fitted objective.
    """

    _ablation_coefficient_kind = "hard_realized_target_indicator"
    _ablation_support_kind = "prefix_grid_one_hot"

    def phase1_objective_masses(
            self,
            event_times: torch.Tensor,
            prior_q: torch.Tensor,
            quantile_est: torch.Tensor,
            conditional_grid: torch.Tensor,
    ) -> torch.Tensor:
        width = conditional_grid.shape[1]
        active_lengths = torch.minimum(
            event_times.reshape(-1).to(torch.long),
            prior_q.reshape(-1).to(torch.long),
        ).clamp(min=0, max=width)
        horizons, strict = self._task_target_horizons(quantile_est)
        times = event_times.reshape(-1).to(torch.float64)
        horizons = horizons.reshape(-1).to(
            device=times.device, dtype=torch.float64
        )
        target = (
            times < horizons if strict else times <= horizons
        ).to(torch.float64)
        regularization = float(getattr(self, "global_regularization", 0.0))
        weights = (target + regularization) / (1.0 + regularization)
        masses = torch.zeros(
            (len(times), width),
            dtype=torch.float64,
            device=times.device,
        )
        positive = active_lengths > 0
        rows = torch.arange(len(times), device=times.device)[positive]
        masses[rows, active_lengths[positive] - 1] = weights[positive]
        return masses


class _SoftTerminalObjectiveMixin:
    """Place an initial-prefix model target probability at one endpoint.

    This differs from soft-prefix DAPRO only in support: it aggregates the
    model's target-event probability at ``X_i0`` into a single row weight and
    lets the ordinary terminal backend attach that weight to the observed
    active endpoint.  It therefore isolates the benefit of updating and
    distributing soft coefficients over observed prefixes.
    """

    _ablation_coefficient_kind = "soft_initial_model_probability"
    _ablation_support_kind = "terminal_endpoint"

    def phase1_objective_masses(
            self,
            event_times: torch.Tensor,
            prior_q: torch.Tensor,
            quantile_est: torch.Tensor,
            conditional_grid: torch.Tensor,
    ) -> torch.Tensor:
        pmf = conditional_grid[:, 0, :].to(torch.float64).clamp_min(0.0)
        horizons, strict = self._task_target_horizons(quantile_est)
        horizons = horizons.reshape(-1).to(
            device=pmf.device, dtype=torch.float64
        )
        one_based_time = torch.arange(
            1, pmf.shape[1] + 1, device=pmf.device, dtype=torch.float64
        ).unsqueeze(0)
        target_mask = (
            one_based_time < horizons[:, None]
            if strict
            else one_based_time <= horizons[:, None]
        )
        target_probability = (pmf * target_mask).sum(dim=1).clamp(0.0, 1.0)
        regularization = float(getattr(self, "global_regularization", 0.0))
        weights = (
            target_probability + regularization
        ) / (1.0 + regularization)
        active_lengths = torch.minimum(
            event_times.reshape(-1).to(torch.long),
            prior_q.reshape(-1).to(torch.long),
        ).clamp(min=0, max=conditional_grid.shape[1])
        masses = torch.zeros(
            (len(event_times), conditional_grid.shape[1]),
            dtype=torch.float64,
            device=pmf.device,
        )
        positive = active_lengths > 0
        rows = torch.arange(len(event_times), device=pmf.device)[positive]
        masses[rows, active_lengths[positive] - 1] = weights[positive]
        return masses


class AblationHardPrefixDAPRO(
        _HardPrefixObjectiveMixin, _DAPROAblationMixin, DefinitiveDAPRO
):
    def __init__(self, *args, ablation_kind: str, ablation_value: float,
                 ablation_label: str = "", score_kind: str = "hazard",
                 score_noise_lambda: float = 0.0,
                 score_noise_seed: int = 314159, **kwargs):
        self._initialize_ablation(
            ablation_kind=ablation_kind, ablation_value=ablation_value,
            ablation_label=ablation_label, score_kind=score_kind,
            score_noise_lambda=score_noise_lambda,
            score_noise_seed=score_noise_seed,
        )
        super().__init__(*args, **kwargs)


class AblationHardPrefixCRCDAPRO(
        _HardPrefixObjectiveMixin, _DAPROAblationMixin, DefinitiveCRCDAPRO
):
    def __init__(self, *args, ablation_kind: str, ablation_value: float,
                 ablation_label: str = "", score_kind: str = "hazard",
                 score_noise_lambda: float = 0.0,
                 score_noise_seed: int = 314159, **kwargs):
        self._initialize_ablation(
            ablation_kind=ablation_kind, ablation_value=ablation_value,
            ablation_label=ablation_label, score_kind=score_kind,
            score_noise_lambda=score_noise_lambda,
            score_noise_seed=score_noise_seed,
        )
        super().__init__(*args, **kwargs)


class AblationSoftTerminalDAPRO(
        _SoftTerminalObjectiveMixin, _DAPROAblationMixin, DefinitiveDAPRO
):
    def __init__(self, *args, ablation_kind: str, ablation_value: float,
                 ablation_label: str = "", score_kind: str = "hazard",
                 score_noise_lambda: float = 0.0,
                 score_noise_seed: int = 314159, **kwargs):
        self._initialize_ablation(
            ablation_kind=ablation_kind, ablation_value=ablation_value,
            ablation_label=ablation_label, score_kind=score_kind,
            score_noise_lambda=score_noise_lambda,
            score_noise_seed=score_noise_seed,
        )
        super().__init__(*args, **kwargs)


class AblationSoftTerminalCRCDAPRO(
        _SoftTerminalObjectiveMixin, _DAPROAblationMixin, DefinitiveCRCDAPRO
):
    def __init__(self, *args, ablation_kind: str, ablation_value: float,
                 ablation_label: str = "", score_kind: str = "hazard",
                 score_noise_lambda: float = 0.0,
                 score_noise_seed: int = 314159, **kwargs):
        self._initialize_ablation(
            ablation_kind=ablation_kind, ablation_value=ablation_value,
            ablation_label=ablation_label, score_kind=score_kind,
            score_noise_lambda=score_noise_lambda,
            score_noise_seed=score_noise_seed,
        )
        super().__init__(*args, **kwargs)


class AblationUniformContinuationCRCDAPRO(
        BudgetAllocator
):
    """No-optimization control: one fixed continuation family plus CRC.

    The policy family is fixed before any trajectory labels are inspected:
    every eligible row and time uses the same core conditional continuation
    probability.  The only learned scalar is its aggressiveness, selected on
    an ``n1``-row CRC fold.  Consequently all fully observed Phase-I rows are
    budget-control rows and none are used to fit a score map or a DAPRO
    objective.  A row-level always-follow mixture preserves terminal reach
    probability ``terminal_pi_min`` and is included in the CRC costs.
    """

    def __init__(
            self,
            conditional_grid,
            budget_per_sample,
            taus_range,
            tau_prior,
            m_upper_bound,
            *,
            n1: int,
            terminal_pi_min: float = 0.005,
            budget_candidate_count: int = 401,
            row_cost_cap_multiplier: float = 2.0,
            ablation_kind: str = "optimization",
            ablation_value: float = 0.0,
            ablation_label: str = "Uniform continuation",
            reach_t_max_is_success: bool = False,
    ) -> None:
        if str(ablation_kind).lower() != "optimization":
            raise ValueError(
                "Uniform continuation is only defined for the optimization "
                "ablation."
            )
        self.ablation_kind = "optimization"
        self.ablation_value = float(ablation_value)
        self.ablation_label = str(ablation_label)
        self.n1 = int(n1)
        if self.n1 <= 0:
            raise ValueError("`n1` must be a positive CRC-fold size.")
        if not 0 < terminal_pi_min <= 1:
            raise ValueError("`terminal_pi_min` must lie in (0, 1].")
        if budget_candidate_count < 2:
            raise ValueError("`budget_candidate_count` must be at least two.")
        if not np.isfinite(row_cost_cap_multiplier) or row_cost_cap_multiplier <= 0:
            raise ValueError(
                "`row_cost_cap_multiplier` must be finite and positive."
            )
        super().__init__(budget_per_sample, taus_range, tau_prior)
        self.conditional_grid = conditional_grid
        self.m_upper_bound = int(m_upper_bound)
        self.phase1_size = self.n1
        self.budget_control_size = self.n1
        self.budget_control_mode = "crc"
        self.budget_candidate_count = int(budget_candidate_count)
        self.row_cost_cap_multiplier = float(row_cost_cap_multiplier)
        self.risk_candidate_row_cost_cap = min(
            self.row_cost_cap_multiplier * float(budget_per_sample),
            float(m_upper_bound),
        )
        self.terminal_pi_min = float(terminal_pi_min)
        self.min_pi = self.terminal_pi_min
        self.schedule_family = "constant"
        self.reach_t_max_is_success = bool(reach_t_max_is_success)

    @property
    def name(self) -> str:
        floor = _float_token(float(self.min_pi))
        cap = _float_token(self.row_cost_cap_multiplier)
        return (
            "uniform_continuation_crc_ablation_optimization"
            f"_floor_{floor}_row_cap_{cap}x_budget_v2_n1_{self.n1}"
        )

    def _cumulative_path(
            self, core_probability: float, width: int
    ) -> np.ndarray:
        times = np.arange(1, width + 1, dtype=np.float64)
        return self.min_pi + (1.0 - self.min_pi) * np.power(
            float(core_probability), times
        )

    def _largest_capped_core_probability(self, width: int) -> float:
        minimum_cost = float(self._cumulative_path(0.0, width).sum())
        if minimum_cost > self.risk_candidate_row_cost_cap + 1e-12:
            raise ValueError(
                "The terminal reach floor alone exceeds the requested "
                "candidate row-cost cap."
            )
        if width <= self.risk_candidate_row_cost_cap + 1e-12:
            return 1.0
        low, high = 0.0, 1.0
        for _ in range(80):
            middle = (low + high) / 2.0
            if (
                    self._cumulative_path(middle, width).sum()
                    <= self.risk_candidate_row_cost_cap
            ):
                low = middle
            else:
                high = middle
        return low

    def allocate_budget(
            self,
            probability_est: torch.Tensor,
            x: torch.Tensor,
            t: torch.Tensor,
            quantile_est: torch.Tensor,
    ) -> BudgetAllocationResult:
        del probability_est, x
        device = self.conditional_grid.device
        n_samples, width, _ = self.conditional_grid.shape
        if self.n1 >= n_samples:
            raise ValueError("The CRC fold must be smaller than the sample count.")
        prior = get_prior(
            quantile_est, self.taus_range, self.tau_prior
        ).to(torch.long).clamp(min=1, max=width)
        event_times = t.reshape(-1).to(torch.long)
        active_lengths = torch.minimum(event_times, prior).clamp(
            min=0, max=width
        )

        permutation = np.random.permutation(n_samples)
        control_indices = permutation[:self.n1]
        deployment_indices = permutation[self.n1:]
        control_lengths = active_lengths[control_indices].detach().cpu().numpy()
        deployment_lengths = active_lengths[
            deployment_indices
        ].detach().cpu().numpy()

        maximum_core_probability = self._largest_capped_core_probability(
            width
        )
        core_probabilities = np.linspace(
            maximum_core_probability,
            0.0,
            self.budget_candidate_count,
            dtype=np.float64,
        )
        cumulative_family = np.stack([
            self._cumulative_path(probability, width)
            for probability in core_probabilities
        ])
        control_candidate_costs = cumulative_policy_costs(
            cumulative_family, control_lengths
        )
        maximum_candidate_cost = float(cumulative_family[0].sum())
        selection = select_crc_budget_candidate(
            control_candidate_costs,
            control_lengths.astype(np.float64),
            total_budget_after_policy_fit=(
                float(self.budget_per_sample) * n_samples
            ),
            deployment_sample_count=len(deployment_indices),
            maximum_cost_per_sample=float(width),
            maximum_candidate_cost_per_sample=maximum_candidate_cost,
            maximum_pilot_cost_per_sample=float(width),
        )
        selected_index = selection.selected_index
        selected_core_probability = float(
            core_probabilities[selected_index]
        )
        selected_cumulative = cumulative_family[selected_index]
        previous_cumulative = np.concatenate([
            np.ones(1, dtype=np.float64), selected_cumulative[:-1]
        ])
        selected_conditionals = np.clip(
            selected_cumulative / previous_cumulative, 0.0, 1.0
        )

        deployment_count = len(deployment_indices)
        expected_remaining = torch.ones(
            (deployment_count, width),
            dtype=self.conditional_grid.dtype,
            device=device,
        )
        conditional_tensor = torch.as_tensor(
            selected_conditionals, dtype=torch.float64, device=device
        )

        def deployment_policy(_):
            return conditional_tensor.unsqueeze(0).expand(
                deployment_count, -1
            )

        uniforms = self.get_acquisition_uniforms(
            n_samples,
            width,
            device=device,
            dtype=torch.float64,
        )
        deployment_uniforms = (
            None if uniforms is None else uniforms[deployment_indices]
        )
        if uniforms is None:
            self.reset_acquisition_rng()
        deployment_c, deployment_pi, deployment_realized_cost = (
            simulate_process_vectorized(
                expected_remaining,
                prior[deployment_indices],
                event_times[deployment_indices],
                selected_core_probability,
                stochastic=True,
                reach_t_max_is_success=self.reach_t_max_is_success,
                pi_func=deployment_policy,
                terminal_pi_min=None,
                terminal_floor_mode="none",
                uniforms=deployment_uniforms,
            )
        )

        final_c = torch.empty(
            n_samples, dtype=prior.dtype, device=device
        )
        final_c[control_indices] = prior[control_indices]
        final_c[deployment_indices] = deployment_c.to(final_c.dtype)
        final_pi = torch.empty(
            n_samples, dtype=torch.float64, device=device
        )
        final_pi[control_indices] = 1.0
        final_pi[deployment_indices] = deployment_pi.to(torch.float64)

        control_pilot_cost = float(active_lengths[control_indices].sum().item())
        deployment_expected_costs = cumulative_policy_costs(
            selected_cumulative[None, :], deployment_lengths
        )[:, 0]
        deployment_expected_cost = float(deployment_expected_costs.sum())
        total_expected_cost = control_pilot_cost + deployment_expected_cost
        total_realized_cost = (
            control_pilot_cost + float(deployment_realized_cost)
        )
        crc_rho = self.n1 / deployment_count
        selected_control_costs = control_candidate_costs[:, selected_index]
        metrics = {
            "objective_kind": "uniform_continuation_no_optimization_crc",
            "ablation_study": "dapro_optimization_process",
            "ablation_kind": self.ablation_kind,
            "ablation_value": self.ablation_value,
            "ablation_label": self.ablation_label,
            "ablation_n1": self.n1,
            "ablation_crc_control_size": self.n1,
            "ablation_uses_crc": 1,
            "ablation_coefficient_kind": "none_no_objective_fit",
            "ablation_support_kind": "uniform_row_time_policy",
            "ablation_score_kind": "none",
            "ablation_score_is_causal": 1,
            "ablation_score_bin_count": np.nan,
            "ablation_continuous_score_map": 0,
            "ablation_row_cost_cap_multiplier": self.row_cost_cap_multiplier,
            "ablation_row_cost_cap_applied": 1,
            "policy_fit_label_count": 0,
            "crc_control_sample_count": self.n1,
            "optimization_process_enabled": 0,
            "uniform_continuation_policy": 1,
            "uniform_continuation_core_probability": (
                selected_core_probability
            ),
            "uniform_continuation_terminal_reach_floor": self.min_pi,
            "random_constant_continuation_probability": (
                selected_core_probability
            ),
            "random_constant_probability": selected_core_probability,
            "random_schedule_family": "constant",
            "random_policy_uses_phase2_event_times": 0,
            "terminal_pi_min": self.min_pi,
            "phase1_sample_count": self.n1,
            "phase2_sample_count": deployment_count,
            "phase1_realized_cost_total": control_pilot_cost,
            "phase1_realized_cost_per_sample": control_pilot_cost / self.n1,
            "phase2_expected_cost_total": deployment_expected_cost,
            "phase2_expected_cost_per_sample": (
                deployment_expected_cost / deployment_count
            ),
            "phase2_realized_cost_per_sample": (
                float(deployment_realized_cost) / deployment_count
            ),
            "risk_budget_control_enabled": 1,
            "risk_budget_control_mode": "crc",
            "risk_budget_control_size": self.n1,
            "risk_budget_policy_fit_size": 0,
            "risk_budget_candidate_count": self.budget_candidate_count,
            "risk_budget_maximum_candidate_cost_per_sample": (
                maximum_candidate_cost
            ),
            "risk_budget_maximum_pilot_cost_per_sample": float(width),
            "risk_budget_selected_index": selected_index,
            "risk_budget_selected_mixture_parameter": (
                selected_core_probability
            ),
            "risk_budget_empirical_control_cost_per_sample": (
                selection.empirical_expected_cost_per_sample
            ),
            "risk_budget_control_pilot_cost_total": control_pilot_cost,
            "risk_budget_control_pilot_cost_per_sample": (
                control_pilot_cost / self.n1
            ),
            "risk_budget_control_to_deployment_ratio": crc_rho,
            "risk_budget_deployment_target_per_sample": (
                selection.deployment_budget_per_sample
            ),
            "risk_budget_selector_left_side_per_sample": (
                selection.selector_left_side_per_sample
            ),
            "risk_budget_correction_per_sample": (
                selection.correction_per_sample
            ),
            "risk_budget_guarantee_kind": selection.guarantee_kind,
            "risk_budget_selector_valid": int(
                selection.selector_left_side_per_sample
                <= selection.deployment_budget_per_sample + 1e-12
            ),
            "risk_budget_crc_empirical_combined_loss_per_sample": float(
                selected_control_costs.mean()
                + crc_rho * control_lengths.mean()
            ),
            "risk_budget_row_cost_cap_enabled": 1,
            "risk_budget_row_cost_cap_per_sample": maximum_candidate_cost,
            "risk_budget_row_cost_cap_kind": (
                "fixed_uniform_continuation_family_horizon_cap"
            ),
            "budget_guarantee_kind": selection.guarantee_kind,
            "expected_budget_guarantee_kind": selection.guarantee_kind,
            **summarize_expected_budget(
                total_expected_cost,
                n_samples,
                self.budget_per_sample,
                cost_semantics=(
                    "crc_control_full_plus_uniform_continuation_event_stopped"
                ),
            ),
        }
        all_conditionals = torch.ones(
            (n_samples, width), dtype=torch.float64, device=device
        )
        all_conditionals[deployment_indices] = (
            conditional_tensor.unsqueeze(0).expand(deployment_count, -1)
        )
        return BudgetAllocationResult(
            quantile_est,
            final_c,
            final_pi,
            total_budget_used=total_realized_cost,
            mean_weight=float(final_pi.reciprocal().mean().item()),
            max_weight=float(final_pi.reciprocal().max().item()),
            additional_metrics=metrics,
            candidate_C_probs=candidate_reach_probabilities(
                all_conditionals,
                quantile_est,
                infinity_value=width + 1,
            ),
            continuation_probabilities=all_conditionals,
        )


__all__ = [
    "AblationHardPrefixCRCDAPRO", "AblationHardPrefixDAPRO",
    "AblationHardTargetCRCDAPRO", "AblationHardTargetDAPRO",
    "AblationSoftTerminalCRCDAPRO", "AblationSoftTerminalDAPRO",
    "AblationSoftTargetCRCDAPRO", "AblationSoftTargetDAPRO",
    "AblationUniformContinuationCRCDAPRO",
]
