"""Estimator-only UPB ablations on a fixed budget-allocation policy."""

from __future__ import annotations

import hashlib
from typing import Dict

import torch

from src.predictive_bounds.budget_allocators.budget_allocator import (
    BudgetAllocator,
)
from src.predictive_bounds.calibration.survival_upb_calibration_with_known_weights import (
    SurvivalUPBCalibrationWithKnownWeights,
)


_ESTIMATOR_LABELS = {
    "ordinary_ht": "Ordinary HT",
    "terminal_residual": "Terminal AHT",
    "sequential": "Sequential AHT",
}


class UPBEstimatorAblationCalibration(
        SurvivalUPBCalibrationWithKnownWeights
):
    """Evaluate one UPB estimator without changing the acquisition path.

    The experiment driver resets the policy RNG and supplies common per-row
    acquisition uniforms before every calibration.  Consequently, separately
    instantiated calibrations with the same allocator specification produce
    the same fitted policy and realized acquisition path; this wrapper changes
    only the estimator used on that path.
    """

    def __init__(
            self,
            budget_allocator: BudgetAllocator,
            taus_range: torch.Tensor,
            tau_prior: float,
            *,
            estimator_kind: str,
            allocation_method: str,
            uses_crc: bool,
    ):
        if estimator_kind not in _ESTIMATOR_LABELS:
            raise ValueError(
                f"Unknown UPB estimator kind {estimator_kind!r}; expected "
                f"one of {sorted(_ESTIMATOR_LABELS)}."
            )
        if allocation_method not in {"static", "dapro"}:
            raise ValueError("`allocation_method` must be 'static' or 'dapro'.")
        if allocation_method == "static" and uses_crc:
            raise ValueError("Static has no CRC variant in this ablation.")
        if estimator_kind == "sequential" and allocation_method == "static":
            raise ValueError(
                "Static has one block-level reach indicator, so its sequential "
                "AHT algebraically equals terminal AHT and is not duplicated."
            )
        # The base calibrator reads this explicit contract from the allocator.
        budget_allocator.upb_estimator_kind = estimator_kind
        super().__init__(budget_allocator, taus_range, tau_prior)
        self.estimator_kind = estimator_kind
        self.allocation_method = allocation_method
        self.uses_crc = bool(uses_crc)

    @property
    def method_label(self) -> str:
        if self.allocation_method == "static":
            return "Static"
        return "DAPRO" if self.uses_crc else "DAPRO w/o CRC"

    @property
    def estimator_label(self) -> str:
        return _ESTIMATOR_LABELS[self.estimator_kind]

    @property
    def name(self) -> str:
        controller = (
            "static"
            if self.allocation_method == "static"
            else "dapro_crc"
            if self.uses_crc
            else "dapro_raw"
        )
        estimator = {
            "ordinary_ht": "ordinary_ht",
            "terminal_residual": "terminal_aht",
            "sequential": "sequential_aht",
        }[self.estimator_kind]
        return f"calibration_upb_estimator_ablation_{controller}_{estimator}"

    def compute_metrics(self, model_prediction, target_taus) -> Dict[str, float]:
        metrics = super().compute_metrics(model_prediction, target_taus)
        digest = hashlib.sha256()
        for value in (
                self.allocation_result.C,
                self.allocation_result.C_probs,
                self.allocation_result.continuation_probabilities,
        ):
            if value is None:
                digest.update(b"<none>")
                continue
            tensor = value.detach().to(device="cpu").contiguous()
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        metrics.update({
            "upb_estimator_ablation": 1,
            "upb_estimator_ablation_method": self.method_label,
            "upb_estimator_ablation_estimator": self.estimator_label,
            "upb_estimator_ablation_estimator_kind": self.estimator_kind,
            "upb_estimator_ablation_uses_crc": int(self.uses_crc),
            "upb_estimator_ablation_allocator_name": (
                self.budget_allocator.name
            ),
            "upb_estimator_ablation_paired_reference": (
                "Static + Terminal AHT"
            ),
            "upb_estimator_ablation_acquisition_sha256": digest.hexdigest(),
        })
        return metrics
