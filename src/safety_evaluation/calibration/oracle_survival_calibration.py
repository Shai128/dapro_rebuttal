from typing import Dict, Union

import numpy as np
import torch

from src.safety_evaluation.budget_allocators.budget_allocator import BudgetAllocationResult
from src.safety_evaluation.calibration.abstract_calibration import SurvivalLPBCalibration
from src.safety_evaluation.calibration.calibration_utils import get_prior
from src.train_model.models.utils import ModelPrediction, SurvivalModelPrediction



class OracleSurvivalCalibration(SurvivalLPBCalibration):
    """Calibrate survival quantiles when every calibration time is observable.

    The oracle has an infinite generation budget, so there is no budget
    censoring and therefore no inverse-probability reweighting.
    """

    def __init__(self, taus_range: torch.Tensor, tau_prior: float):
        super().__init__()
        self.miscoverage = None
        self.taus_range = taus_range
        self.allocation_result: Union[BudgetAllocationResult, None] = None
        self.t_cal = None
        self.tau_prior = tau_prior

    def calibrate(self, x_cal: torch.Tensor, t_cal: torch.Tensor, model_prediction_cal: ModelPrediction):
        if not isinstance(model_prediction_cal, SurvivalModelPrediction):
            raise Exception("error")

        # With an infinite oracle budget, f is observed against every t_cal.
        # In particular, allocation must not trim f, randomly censor samples,
        # or introduce inverse-probability weights.
        f = model_prediction_cal.quantile_est
        C = torch.full(
            (f.shape[0],),
            float("inf"),
            dtype=f.dtype,
            device=f.device,
        )
        C_probs = torch.ones(f.shape[0], dtype=f.dtype, device=f.device)
        allocation_result = BudgetAllocationResult(
            f=f,
            C=C,
            C_probs=C_probs,
            total_budget_used=0,
            mean_weight=1.0,
            max_weight=1.0,
        )
        self.allocation_result = allocation_result
        self.t_cal = t_cal

        self.miscoverage = (
            t_cal.to(f.device).reshape(-1, 1) < f
        ).to(f.dtype).mean(dim=0)

    def get_calibrated_lpb(self, target_taus: torch.Tensor, x: torch.Tensor, model_prediction: ModelPrediction):
        if not isinstance(model_prediction, SurvivalModelPrediction):
            raise Exception("error")
        miscoverage = self.miscoverage
        quantile_est = model_prediction.quantile_est
        target_taus = target_taus.to(quantile_est.device)
        tau_diff = target_taus - miscoverage[:, np.newaxis]
        smallest_pos = torch.where(tau_diff > 0, 1, -1.0 * np.inf).cumsum(dim=0).argmax(dim=0)
        calibrated_test_quantile_est = quantile_est[:, smallest_pos].squeeze()
        return calibrated_test_quantile_est

    @property
    def name(self) -> str:
        return "oracle_survival_calibration"

    def compute_metrics(self, model_prediction, target_taus) -> Dict[str, float]:
        t = self.t_cal
        f = self.allocation_result.f
        C = self.allocation_result.C
        C_probs = self.allocation_result.C_probs
        mean_weight = self.allocation_result.mean_weight
        max_weight = self.allocation_result.max_weight
        if mean_weight is None:
            mean_weight = (1 / C_probs).mean().item()
        if max_weight is None:
            max_weight = (1 / C_probs).max().item()
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
        tau_diff = target_taus - miscoverage[:, np.newaxis]
        smallest_pos = torch.where(tau_diff > 0, 1, -1.0 * np.inf).cumsum(dim=0).argmax(dim=0)
        calibrated_test_quantile_est = f[:, smallest_pos].squeeze()
        alpha_hat_per_tau = miscoverage[smallest_pos]

        all_observed_jailbreaks = (t.squeeze().unsqueeze(-1) <= calibrated_test_quantile_est.squeeze()).float().sum(
            dim=0)
        all_f_lower_c = (calibrated_test_quantile_est.squeeze() <= C.squeeze().unsqueeze(-1)).float().sum(dim=0)
        all_observed_both = ((calibrated_test_quantile_est.squeeze() <= C.squeeze().unsqueeze(-1)) & (
                t.squeeze().unsqueeze(-1) <= calibrated_test_quantile_est.squeeze())).float().sum(dim=0)
        coverage_deviation = None
        additional_metrics = self.allocation_result.additional_metrics # if self.allocation_result.additional_metrics else {}
        if additional_metrics is None:
            additional_metrics = {}
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
            **{f'all_observed_jailbreaks_{i}': all_observed_jailbreaks[i].item() for i in
               range(len(all_observed_jailbreaks))},
            **{f'all_f_lower_c_{i}': all_f_lower_c[i].item() for i in range(len(all_f_lower_c))},
            **{f'all_observed_both_{i}': all_observed_both[i].item() for i in range(len(all_observed_both))},
            **{f'alpha_hat_per_tau_{i}': alpha_hat_per_tau[i].item() for i in range(len(alpha_hat_per_tau))},
            **additional_metrics
        }
        return metrics
