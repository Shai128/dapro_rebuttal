"""Controlled LPB ablations for soft-prefix Generalized DAPRO.

The production allocator is intentionally left unchanged.  These wrappers add
only (i) stable experiment identity/metadata and (ii), for the score-quality
ablation, a deterministic degradation of the current-prefix hazard score.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from src.predictive_bounds.budget_allocators.DAPRO import (
    SoftTargetCRCDAPRO,
    SoftTargetDAPRO,
)


_VALID_ABLATION_KINDS = {"n1", "score_noise", "budget"}


def _float_token(value: float) -> str:
    """Return a compact filesystem-safe decimal token."""
    return np.format_float_positional(float(value), trim="-").replace(".", "p")


class _DAPROAblationMixin:
    """Identity, diagnostics, and optional score corruption for an ablation."""

    def _initialize_ablation(
            self,
            *,
            ablation_kind: str,
            ablation_value: float,
            score_noise_lambda: float = 0.0,
            score_noise_seed: int = 314159,
    ) -> None:
        kind = str(ablation_kind).lower()
        if kind not in _VALID_ABLATION_KINDS:
            raise ValueError(
                f"Unknown DAPRO ablation kind {ablation_kind!r}; expected one "
                f"of {sorted(_VALID_ABLATION_KINDS)}."
            )
        value = float(ablation_value)
        if not math.isfinite(value):
            raise ValueError("`ablation_value` must be finite.")
        noise = float(score_noise_lambda)
        if not 0.0 <= noise <= 1.0:
            raise ValueError("`score_noise_lambda` must lie in [0, 1].")
        self.ablation_kind = kind
        self.ablation_value = value
        self.score_noise_lambda = noise
        self.score_noise_seed = int(score_noise_seed)
        self._ablation_score_mean_timewise_correlation = np.nan
        self._ablation_score_k2_bin_agreement = np.nan

    @property
    def name(self) -> str:
        value = _float_token(self.ablation_value)
        return f"{super().name}_ablation_{self.ablation_kind}_{value}"

    def policy_scores(self, quantile_est: torch.Tensor) -> torch.Tensor:
        scores = super().policy_scores(quantile_est)
        lam = self.score_noise_lambda
        if lam == 0.0:
            self._ablation_score_mean_timewise_correlation = 1.0
            self._ablation_score_k2_bin_agreement = 1.0
            return scores

        # Each time column is independently permuted.  This retains the
        # original time-specific score distribution and numerical scale, so
        # lambda isolates ranking information instead of score calibration.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.score_noise_seed)
        corrupted = torch.empty_like(scores)
        for time_idx in range(scores.shape[1]):
            permutation = torch.randperm(
                scores.shape[0], generator=generator, device="cpu"
            ).to(scores.device)
            corrupted[:, time_idx] = scores[permutation, time_idx]
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
        original_median = scores.median(dim=0).values
        mixed_median = mixed.median(dim=0).values
        original_bin = scores >= original_median[None, :]
        mixed_bin = mixed >= mixed_median[None, :]
        self._ablation_score_k2_bin_agreement = float(
            (original_bin == mixed_bin).to(torch.float64).mean().detach().cpu()
        )
        return mixed

    def objective_metadata(self) -> dict:
        metadata = super().objective_metadata()
        metadata.update({
            "ablation_study": "dapro_lpb",
            "ablation_kind": self.ablation_kind,
            "ablation_value": self.ablation_value,
            "ablation_n1": self.n1,
            "ablation_crc_control_size": getattr(
                self, "budget_control_size", 0
            ),
            "ablation_uses_crc": int(
                getattr(self, "budget_control_mode", None) == "crc"
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
            "ablation_score_definition": "current_prefix_instantaneous_hazard",
            "ablation_score_bin_count": self.score_bin_count,
        })
        return metadata


class AblationSoftTargetDAPRO(_DAPROAblationMixin, SoftTargetDAPRO):
    """Raw, zero-margin soft-prefix Generalized DAPRO ablation."""

    def __init__(self, *args, ablation_kind: str, ablation_value: float,
                 score_noise_lambda: float = 0.0,
                 score_noise_seed: int = 314159, **kwargs):
        self._initialize_ablation(
            ablation_kind=ablation_kind,
            ablation_value=ablation_value,
            score_noise_lambda=score_noise_lambda,
            score_noise_seed=score_noise_seed,
        )
        super().__init__(*args, **kwargs)


class AblationSoftTargetCRCDAPRO(
        _DAPROAblationMixin,
        SoftTargetCRCDAPRO,
):
    """CRC-controlled soft-prefix Generalized DAPRO ablation."""

    def __init__(self, *args, ablation_kind: str, ablation_value: float,
                 score_noise_lambda: float = 0.0,
                 score_noise_seed: int = 314159, **kwargs):
        self._initialize_ablation(
            ablation_kind=ablation_kind,
            ablation_value=ablation_value,
            score_noise_lambda=score_noise_lambda,
            score_noise_seed=score_noise_seed,
        )
        super().__init__(*args, **kwargs)
