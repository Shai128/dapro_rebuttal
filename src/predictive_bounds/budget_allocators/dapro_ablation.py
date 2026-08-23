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


_VALID_ABLATION_KINDS = {
    "n1", "score_noise", "budget", "hard_soft", "representation",
    "score", "attacker_shift",
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
        index = int(torch.argmin(torch.abs(taus - self.target_alpha)).item())
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
            "ablation_coefficient_kind": (
                "soft_prefix_model_probability"
                if isinstance(self, (SoftTargetDAPRO, SoftTargetCRCDAPRO))
                else "hard_realized_target_indicator"
            ),
        })
        return metadata


class AblationSoftTargetDAPRO(_DAPROAblationMixin, SoftTargetDAPRO):
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


__all__ = [
    "AblationHardTargetCRCDAPRO", "AblationHardTargetDAPRO",
    "AblationSoftTargetCRCDAPRO", "AblationSoftTargetDAPRO",
]
