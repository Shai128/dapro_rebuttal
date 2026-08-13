"""IPW calibration of upper predictive bounds under sequential acquisition."""

from typing import Dict, Union

import numpy as np
import torch

from src.predictive_bounds.budget_allocators.budget_allocator import (
    BudgetAllocator,
    BudgetAllocationResult,
)
from src.predictive_bounds.calibration.abstract_calibration import (
    SurvivalUPBCalibration,
)
from src.predictive_bounds.calibration.calibration_utils import (
    get_prior,
    indexed_tensor_metrics,
    select_upb_calibration_positions,
)
from src.train_model.models.utils import ModelPrediction, SurvivalModelPrediction


UPB_INFINITY_VALUE = 201


def get_gamma(m_upper_bound: float, budget_per_sample: float):
    return m_upper_bound / budget_per_sample


def get_coverage_rate_deviation(gamma: float, cal_size: int, delta: float):
    return np.sqrt((2 * gamma ** 2 + 5) / cal_size * np.log(1 / delta))


class SurvivalUPBCalibrationWithKnownWeights(SurvivalUPBCalibration):
    """Calibrate a UPB from unbiased estimates of its coverage curve.

    For every finite candidate ``f`` the target is

        A_i(f) = 1{T_i <= f(X_i)}.

    A positive contribution is available exactly when the event was reached,
    and is reweighted by its logged terminal inclusion propensity.  The value
    201 is different: it denotes infinity/no event through turn 200, so its
    coverage contribution is one deterministically and is never reweighted.
    """

    def __init__(
            self,
            budget_allocator: BudgetAllocator,
            taus_range: torch.Tensor,
            tau_prior: float,
    ):
        super().__init__()
        self.budget_allocator = budget_allocator
        self.coverage = None
        self.miscoverage = None
        self.taus_range = taus_range
        self.allocation_result: Union[BudgetAllocationResult, None] = None
        self.t_cal = None
        self.tau_prior = tau_prior

    @staticmethod
    def _coverage_contributions(
            event_times: torch.Tensor,
            candidates: torch.Tensor,
            acquisition_horizons: torch.Tensor,
            terminal_propensities: torch.Tensor,
    ) -> torch.Tensor:
        times = event_times.reshape(-1, 1).to(candidates.device)
        horizons = acquisition_horizons.reshape(-1, 1).to(candidates.device)
        propensities = terminal_propensities.reshape(-1, 1).to(
            device=candidates.device,
            dtype=torch.float64,
        )
        if torch.any(propensities <= 0) or not torch.all(
                torch.isfinite(propensities)
        ):
            raise ValueError("UPB terminal propensities must be finite and positive.")
        infinite = candidates == UPB_INFINITY_VALUE
        finite_covered_event = (
            (candidates < UPB_INFINITY_VALUE)
            & (times <= candidates)
            & (times <= horizons)
        )
        return torch.where(
            infinite,
            torch.ones_like(candidates, dtype=torch.float64),
            finite_covered_event.to(torch.float64) / propensities,
        )

    def calibrate(
            self,
            x_cal: torch.Tensor,
            t_cal: torch.Tensor,
            model_prediction_cal: ModelPrediction,
    ):
        if not isinstance(model_prediction_cal, SurvivalModelPrediction):
            raise TypeError("UPB calibration needs survival predictions.")

        full_candidates = model_prediction_cal.quantile_est
        horizon = int(model_prediction_cal.probability_est.shape[1])
        # Ordinary allocators need only the executable 200-turn acquisition
        # horizon.  UPB-aware DAPRO explicitly opts into the unmodified
        # candidates because its soft target must distinguish 200 from 201;
        # its own acquisition prior is still clipped to the grid width.
        if getattr(
                self.budget_allocator,
                "requires_unclipped_upb_quantiles",
                False,
        ):
            allocation_candidates = full_candidates
        else:
            allocation_candidates = full_candidates.clamp(max=horizon)
        allocation_result = self.budget_allocator.allocate_budget(
            model_prediction_cal.probability_est,
            x_cal,
            t_cal,
            allocation_candidates,
        )
        # Calibration always acts on the original UPB candidates, including
        # the 201 infinity sentinel, independently of acquisition trimming.
        allocation_result.f = full_candidates
        self.allocation_result = allocation_result
        self.t_cal = t_cal
        contributions = self._coverage_contributions(
            t_cal,
            full_candidates,
            allocation_result.C,
            allocation_result.C_probs,
        )
        self.coverage = contributions.mean(dim=0)
        self.miscoverage = 1.0 - self.coverage

    def _selected_positions(self, target_coverages: torch.Tensor) -> torch.Tensor:
        return select_upb_calibration_positions(
            self.coverage,
            target_coverages,
        )

    def get_calibrated_upb(
            self,
            target_taus: torch.Tensor,
            x: torch.Tensor,
            model_prediction: ModelPrediction,
    ):
        del x
        if not isinstance(model_prediction, SurvivalModelPrediction):
            raise TypeError("UPB calibration needs survival predictions.")
        positions = self._selected_positions(target_taus)
        calibrated = model_prediction.quantile_est[:, positions]
        if hasattr(self.budget_allocator, "max_estimator"):
            # 201 is a semantic infinity marker, not a value to trim to the
            # acquisition horizon.  Finite values remain within the horizon.
            finite = calibrated < UPB_INFINITY_VALUE
            calibrated = torch.where(
                finite,
                torch.minimum(
                    calibrated,
                    torch.as_tensor(
                        self.budget_allocator.max_estimator,
                        dtype=calibrated.dtype,
                        device=calibrated.device,
                    ),
                ),
                calibrated,
            )
        return calibrated.squeeze()

    @property
    def name(self) -> str:
        return f"calibration_{self.budget_allocator.name}_allocation"

    def compute_metrics(self, model_prediction, target_taus) -> Dict[str, float]:
        del model_prediction
        t = self.t_cal.reshape(-1, 1)
        f = self.allocation_result.f
        C = self.allocation_result.C.reshape(-1, 1)
        C_probs = self.allocation_result.C_probs.reshape(-1).to(torch.float64)
        inverse_probability = C_probs.reciprocal()
        positions = self._selected_positions(target_taus)
        selected_f = f[:, positions]
        infinite = selected_f == UPB_INFINITY_VALUE
        finite_a = (
            (selected_f < UPB_INFINITY_VALUE)
            & (t.to(selected_f.device) <= selected_f)
        )
        weighted = finite_a.to(torch.float64) * inverse_probability[:, None]
        variance_proxy = finite_a.to(torch.float64) * (
            inverse_probability[:, None] - 1.0
        )
        observed_positive = finite_a & (t.to(C.device) <= C)

        mean_weight = self.allocation_result.mean_weight
        max_weight = self.allocation_result.max_weight
        if mean_weight is None:
            mean_weight = inverse_probability.mean().item()
        if max_weight is None:
            max_weight = inverse_probability.max().item()
        budget_used = self.allocation_result.total_budget_used
        if budget_used is None:
            budget_used = C.sum().item()

        f_prior = get_prior(f, self.taus_range, self.tau_prior)
        prior_infinite = f_prior == UPB_INFINITY_VALUE
        prior_covered = prior_infinite | (
            t.reshape(-1).to(f_prior.device) <= f_prior
        )
        prior_observable = prior_infinite | (
            t.reshape(-1).to(C.device) <= C.reshape(-1)
        )
        additional = dict(self.allocation_result.additional_metrics or {})
        indexed = indexed_tensor_metrics({
            "all_observed_jailbreaks": finite_a.to(f.dtype).sum(dim=0),
            "all_f_lower_c": (
                infinite | (selected_f <= C.to(selected_f.device))
            ).to(f.dtype).sum(dim=0),
            "all_observed_both": observed_positive.to(f.dtype).sum(dim=0),
            "alpha_hat_per_tau": self.miscoverage[positions],
            "coverage_hat_per_tau": self.coverage[positions],
            "mean_a_weighted_inverse_probability_minus_one": (
                variance_proxy.mean(dim=0)
            ),
            "mean_a_weighted_inverse_probability": weighted.mean(dim=0),
            "variance_a_weighted_inverse_probability": weighted.var(
                dim=0, unbiased=False
            ),
            "selected_infinite_bound_rate": infinite.to(f.dtype).mean(dim=0),
        })
        return {
            "coverage_deviation": None,
            "prior_observed_jailbreaks": prior_covered.to(f.dtype).sum().item(),
            "prior_observed_f_lower_c": prior_observable.to(f.dtype).sum().item(),
            "prior_observed_both": (
                prior_covered & prior_observable
            ).to(f.dtype).sum().item(),
            "n_observed_events": ((t <= C) & (t <= 200)).sum().item(),
            "n_achieved_q_prior1": prior_observable.to(f.dtype).sum().item(),
            "n_achieved_q_prior2": prior_observable.to(f.dtype).sum().item(),
            "budget_used": budget_used,
            "mean_weight": mean_weight,
            "max_weight": max_weight,
            "upb_infinity_value": UPB_INFINITY_VALUE,
            "upb_calibration_target": "coverage_event_t_le_f",
            **indexed,
            **additional,
        }
