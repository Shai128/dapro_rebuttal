from typing import Dict, Union

import numpy as np
import torch

from src.predictive_bounds.budget_allocators.budget_allocator import BudgetAllocator, BudgetAllocationResult
from src.predictive_bounds.calibration.abstract_calibration import  SurvivalUPBCalibration
from src.predictive_bounds.calibration.calibration_utils import (
    get_prior,
    indexed_tensor_metrics,
)
from src.train_model.models.utils import ModelPrediction, SurvivalModelPrediction


def get_gamma(m_upper_bound: float, budget_per_sample: float):
    gamma = (m_upper_bound / budget_per_sample)
    return gamma


def get_coverage_rate_deviation(gamma: float, cal_size: int, delta: float):
    coverage_rate_bound = np.sqrt((2 * gamma ** 2 + 5) / cal_size * np.log(1 / delta))
    return coverage_rate_bound


class SurvivalUPBCalibrationWithKnownWeights(SurvivalUPBCalibration):

    def __init__(self, budget_allocator: BudgetAllocator, taus_range: torch.Tensor, tau_prior: float):
        super().__init__()
        self.budget_allocator = budget_allocator
        self.miscoverage = None
        self.taus_range = taus_range
        self.allocation_result: Union[BudgetAllocationResult, None] = None
        self.t_cal = None
        self.tau_prior = tau_prior

    def calibrate(self, x_cal: torch.Tensor, t_cal: torch.Tensor, model_prediction_cal: ModelPrediction):
        if not isinstance(model_prediction_cal, SurvivalModelPrediction):
            raise Exception("error")
        allocation_result = self.budget_allocator.allocate_budget(model_prediction_cal.probability_est, x_cal, t_cal,
                                                                  model_prediction_cal.quantile_est)
        self.allocation_result = allocation_result
        self.t_cal = t_cal
        f, C, C_probs = allocation_result.f, allocation_result.C, allocation_result.C_probs
        # tau_idx = 1200
        # gt_coverage_rate = ((t_cal <= f[:, tau_idx]) | (f[:, tau_idx] == 200)).float().mean()
        # est_coverage_rate = ((1.0 / C_probs)*( ((f[:, tau_idx] <= C)) & ((t_cal <= f[:, tau_idx]) | (f[:, tau_idx] == 200))).float()).mean()
        # print(f"tau: {self.taus_range[tau_idx]}, gt_coverage_rate: {gt_coverage_rate}, est_coverage_rate: {est_coverage_rate}")

        estimable_miscoverage = (
            (t_cal.reshape(-1, 1) > f)
            & (f <= C.reshape(-1, 1))
            & (f != 200)
        )
        # Broadcasting avoids materializing a repeated N-by-n_taus copy of
        # the inverse-probability vector before applying the same mask.
        self.miscoverage = (
            estimable_miscoverage
            * (1.0 / C_probs.reshape(-1, 1))
        ).mean(dim=0)


    def get_calibrated_upb(self, target_taus: torch.Tensor, x: torch.Tensor, model_prediction: ModelPrediction):
        if not isinstance(model_prediction, SurvivalModelPrediction):
            raise Exception("error")
        miscoverage = self.miscoverage
        quantile_est = model_prediction.quantile_est

        target_coverages = target_taus.to(quantile_est.device)
        alpha_levels = (1.0 - target_coverages).to(quantile_est.device)  # [n_targets]

        valid = (miscoverage[:, None] <= alpha_levels[None, :])  # decreasing → F..F,T..T

        hat_tau_idx = valid.float().argmax(dim=0)  # [n_targets]

        calibrated_test_quantile_est = quantile_est[:, hat_tau_idx].squeeze()

        # tau_diff = target_taus - miscoverage[:, np.newaxis]
        # smallest_pos = torch.where(tau_diff > 0, 1, -1.0 * np.inf).cumsum(dim=0).argmax(dim=0)
        # calibrated_test_quantile_est = quantile_est[:, smallest_pos].squeeze()
        if hasattr(self.budget_allocator, 'max_estimator'):
            max_estimator = self.budget_allocator.max_estimator
            calibrated_test_quantile_est = torch.min(calibrated_test_quantile_est,
                                                     max_estimator * torch.ones_like(calibrated_test_quantile_est))
        return calibrated_test_quantile_est

    @property
    def name(self) -> str:
        return f"calibration_{self.budget_allocator.name}_allocation"

    def compute_metrics(self, model_prediction, target_taus) -> Dict[str, float]:
        t = self.t_cal
        f = self.allocation_result.f
        C = self.allocation_result.C
        C_probs = self.allocation_result.C_probs
        mean_weight = self.allocation_result.mean_weight
        max_weight = self.allocation_result.max_weight
        inverse_probability = 1 / C_probs.reshape(-1).to(torch.float64)
        inverse_probability_minus_one = inverse_probability - 1
        if mean_weight is None:
            mean_weight = inverse_probability.mean().item()
        if max_weight is None:
            max_weight = inverse_probability.max().item()
        budget_used = self.allocation_result.total_budget_used
        if budget_used is None:
            budget_used = C.sum().item()
        f_prior = get_prior(f, self.taus_range, self.tau_prior)
        prior_observed_jailbreaks = (t.squeeze() <= f_prior.squeeze()).float().sum().item()
        prior_observed_f_lower_c = (f_prior.squeeze() <= C.squeeze()).float().sum().item()
        prior_observed_both = (
                (f_prior.squeeze() <= C.squeeze()) & (t.squeeze() <= f_prior.squeeze())).float().sum().item()
        n_observed_events = (C.squeeze() > t).float().sum().item()
        n_achieved_q_prior1 = (C.squeeze() >= f_prior).float().sum().item()
        n_achieved_q_prior2 = (C.squeeze() > f_prior).float().sum().item()

        miscoverage = self.miscoverage
        quantile_est = model_prediction.quantile_est
        target_taus = target_taus.to(quantile_est.device)
        alpha_levels = 1.0 - target_taus
        valid = miscoverage[:, None] <= alpha_levels[None, :]
        selected_positions = valid.float().argmax(dim=0)
        calibrated_test_quantile_est = f[:, selected_positions]
        alpha_hat_per_tau = miscoverage[selected_positions]

        latent_a = (
            (t.reshape(-1, 1) > calibrated_test_quantile_est)
            & (calibrated_test_quantile_est != 200)
        )
        a_variance_proxy = (
            latent_a.to(torch.float64)
            * inverse_probability_minus_one.reshape(-1, 1)
        )
        a_inverse_probability = (
            latent_a.to(torch.float64)
            * inverse_probability.reshape(-1, 1)
        )
        mean_a_variance_proxy = a_variance_proxy.mean(dim=0)
        mean_a_inverse_probability = a_inverse_probability.mean(dim=0)
        variance_a_inverse_probability = a_inverse_probability.var(
            dim=0,
            unbiased=False,
        )
        selected_f_observed = (
            calibrated_test_quantile_est <= C.reshape(-1, 1)
        )
        all_observed_jailbreaks = latent_a.float().sum(dim=0)
        all_f_lower_c = selected_f_observed.float().sum(dim=0)
        all_observed_both = (
            latent_a & selected_f_observed
        ).float().sum(dim=0)
        if hasattr(self.budget_allocator, "cal_size") and hasattr(self.budget_allocator, "max_estimator"):
            cal_size = self.budget_allocator.cal_size
            max_estimator = self.budget_allocator.max_estimator
            budget_per_sample = self.budget_allocator.budget_per_sample
            gamma = get_gamma(max_estimator, budget_per_sample)
            coverage_deviation = get_coverage_rate_deviation(gamma, cal_size, delta=0.1)
        else:
            coverage_deviation = None
        additional_metrics = self.allocation_result.additional_metrics # if self.allocation_result.additional_metrics else {}
        if additional_metrics is None:
            additional_metrics = {}
        indexed_metrics = indexed_tensor_metrics({
            "all_observed_jailbreaks": all_observed_jailbreaks,
            "all_f_lower_c": all_f_lower_c,
            "all_observed_both": all_observed_both,
            "alpha_hat_per_tau": alpha_hat_per_tau,
            "mean_a_weighted_inverse_probability_minus_one": (
                mean_a_variance_proxy
            ),
            "mean_a_weighted_inverse_probability": (
                mean_a_inverse_probability
            ),
            "variance_a_weighted_inverse_probability": (
                variance_a_inverse_probability
            ),
        })
        metrics = {
            'coverage_deviation': coverage_deviation,
            'prior_observed_jailbreaks': prior_observed_jailbreaks,
            'prior_observed_f_lower_c': prior_observed_f_lower_c,
            'prior_observed_both': prior_observed_both,
            'n_observed_events': n_observed_events,
            'n_achieved_q_prior1': n_achieved_q_prior1,
            'n_achieved_q_prior2': n_achieved_q_prior2,
            'budget_used': budget_used,
            'mean_weight': mean_weight,
            'max_weight': max_weight,
            **indexed_metrics,
            **additional_metrics
        }
        return metrics
