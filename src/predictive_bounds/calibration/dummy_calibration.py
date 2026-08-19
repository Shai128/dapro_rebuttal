from typing import Dict

import torch

from src.predictive_bounds.calibration.abstract_calibration import SurvivalLPBCalibration, SurvivalUPBCalibration
from src.predictive_bounds.calibration.calibration_utils import (
    get_prior,
    indexed_tensor_metrics,
    select_calibration_positions,
    select_upb_calibration_positions,
)
from src.train_model.models.utils import ModelPrediction, SurvivalModelPrediction


class UncalibratedLPBSurvivalCalibration(SurvivalLPBCalibration):
    def __init__(self, taus_range: torch.Tensor, tau_prior: float | None = None):
        super().__init__()
        self.taus_range = taus_range
        self.miscoverage = taus_range
        self.tau_prior = (
            float(taus_range[-1]) if tau_prior is None else float(tau_prior)
        )
        self.t_cal = None
        self.f_cal = None

    def calibrate(self, x_cal: torch.Tensor, t_tilde_cal: torch.Tensor, model_prediction_cal: ModelPrediction):
        self.t_cal = t_tilde_cal
        self.f_cal = model_prediction_cal.quantile_est

    def get_calibrated_lpb(self, target_taus: torch.Tensor, x: torch.Tensor, model_prediction: ModelPrediction):
        if not isinstance(model_prediction, SurvivalModelPrediction):
            raise Exception("error")
        miscoverage = self.miscoverage
        quantile_est = model_prediction.quantile_est
        target_taus = target_taus.to(quantile_est.device)
        smallest_pos = select_calibration_positions(
            miscoverage,
            target_taus,
        )
        calibrated_test_quantile_est = quantile_est[:, smallest_pos].squeeze()
        return calibrated_test_quantile_est

    def compute_metrics(self, model_prediction, target_taus) -> Dict[str, float]:
        del model_prediction
        t = self.t_cal.reshape(-1)
        f = self.f_cal
        positions = select_calibration_positions(
            self.miscoverage.to(f.device), target_taus.to(f.device)
        )
        selected = f[:, positions]
        selected_a = (t[:, None] < selected).to(torch.float64)
        prior_a = (
            t < get_prior(f, self.taus_range, self.tau_prior)
        ).to(torch.float64)
        tau_point_one_a = (
            t < get_prior(f, self.taus_range, 0.10)
        ).to(torch.float64)
        return {
            **indexed_tensor_metrics({
                "mean_a_weighted_inverse_probability": selected_a.mean(dim=0),
                "mean_calibrated_a_weighted_inverse_probability": (
                    selected_a.mean(dim=0)
                ),
            }),
            "mean_prior_a_weighted_inverse_probability": prior_a.mean().item(),
            "mean_tau_0p10_a_weighted_inverse_probability": (
                tau_point_one_a.mean().item()
            ),
            "tau_0p10_target_a_rate": tau_point_one_a.mean().item(),
            "weight_semantics": "unit_weight_full_label_diagnostic",
        }

    @property
    def name(self) -> str:
        return "uncalibrated"


class UncalibratedUPBSurvivalCalibration(SurvivalUPBCalibration):
    def __init__(self, taus_range: torch.Tensor, tau_prior: float | None = None):
        super().__init__()
        self.taus_range = taus_range
        self.coverage = taus_range
        self.miscoverage = 1.0 - taus_range
        self.tau_prior = (
            float(taus_range[-1]) if tau_prior is None else float(tau_prior)
        )
        self.t_cal = None
        self.f_cal = None

    def calibrate(self, x_cal: torch.Tensor, t_tilde_cal: torch.Tensor, model_prediction_cal: ModelPrediction):
        self.t_cal = t_tilde_cal
        self.f_cal = model_prediction_cal.quantile_est

    def get_calibrated_upb(self, target_taus: torch.Tensor, x: torch.Tensor, model_prediction: ModelPrediction):
        if not isinstance(model_prediction, SurvivalModelPrediction):
            raise Exception("error")
        quantile_est = model_prediction.quantile_est
        target_taus = target_taus.to(quantile_est.device)
        smallest_pos = select_upb_calibration_positions(
            self.coverage,
            target_taus,
        )
        calibrated_test_quantile_est = quantile_est[:, smallest_pos].squeeze()
        return calibrated_test_quantile_est

    def compute_metrics(self, model_prediction, target_taus) -> Dict[str, float]:
        del model_prediction
        t = self.t_cal.reshape(-1)
        f = self.f_cal
        positions = select_upb_calibration_positions(
            self.coverage.to(f.device), target_taus.to(f.device)
        )
        selected = f[:, positions]
        selected_a = ((selected < 201) & (t[:, None] > selected)).to(
            torch.float64
        )
        f_prior = get_prior(f, self.taus_range, self.tau_prior)
        f_point_one = get_prior(f, self.taus_range, 0.10)
        prior_a = ((f_prior < 201) & (t > f_prior)).to(torch.float64)
        tau_point_one_a = (
            (f_point_one < 201) & (t > f_point_one)
        ).to(torch.float64)
        return {
            **indexed_tensor_metrics({
                "mean_a_weighted_inverse_probability": selected_a.mean(dim=0),
                "mean_calibrated_a_weighted_inverse_probability": (
                    selected_a.mean(dim=0)
                ),
            }),
            "mean_prior_a_weighted_inverse_probability": prior_a.mean().item(),
            "mean_tau_0p10_a_weighted_inverse_probability": (
                tau_point_one_a.mean().item()
            ),
            "tau_0p10_target_a_rate": tau_point_one_a.mean().item(),
            "weight_semantics": "unit_weight_full_label_diagnostic",
        }

    @property
    def name(self) -> str:
        return "uncalibrated"
