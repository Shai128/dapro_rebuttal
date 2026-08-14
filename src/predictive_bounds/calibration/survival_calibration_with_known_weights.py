from typing import Dict, Union

import numpy as np
import torch

from src.predictive_bounds.budget_allocators.budget_allocator import BudgetAllocator, BudgetAllocationResult
from src.predictive_bounds.calibration.abstract_calibration import SurvivalLPBCalibration
from src.predictive_bounds.calibration.calibration_utils import (
    get_prior,
    indexed_tensor_metrics,
    select_calibration_positions,
)
from src.predictive_bounds.calibration.sequential_aht import (
    sequential_lower_curve,
    terminal_residual_lower_curve,
)
from src.train_model.models.utils import ModelPrediction, SurvivalModelPrediction


def get_gamma(m_upper_bound: float, budget_per_sample: float):
    gamma = (m_upper_bound / budget_per_sample)
    return gamma


def get_coverage_rate_deviation(gamma: float, cal_size: int, delta: float):
    coverage_rate_bound = np.sqrt((2 * gamma ** 2 + 5) / cal_size * np.log(1 / delta))
    return coverage_rate_bound


class SurvivalCalibrationWithKnownWeights(SurvivalLPBCalibration):

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

        # quantile_est = model_prediction_cal.quantile_est

        estimator_kind = getattr(
            self.budget_allocator, "aht_estimator_kind", "ordinary_ht"
        )
        if estimator_kind in {"sequential", "terminal_residual"}:
            prediction_grid = getattr(
                self.budget_allocator,
                "conditional_grid",
                model_prediction_cal.probability_est,
            )
            if allocation_result.candidate_C_probs is None:
                raise ValueError(
                    f"Allocator {self.budget_allocator.name!r} must expose "
                    "candidate-specific reach propensities for LPB AHT."
                )
            if estimator_kind == "sequential":
                if allocation_result.continuation_probabilities is None:
                    raise ValueError(
                        "Sequential LPB AHT requires executed continuation "
                        "probabilities."
                    )
                contributions = sequential_lower_curve(
                    t_cal,
                    f,
                    C,
                    allocation_result.continuation_probabilities,
                    prediction_grid,
                    strict=True,
                )
            else:
                contributions = terminal_residual_lower_curve(
                    t_cal,
                    f,
                    C,
                    C_probs,
                    allocation_result.candidate_C_probs,
                    prediction_grid,
                    strict=True,
                )
            self.miscoverage = contributions.mean(dim=0)
            if allocation_result.additional_metrics is None:
                allocation_result.additional_metrics = {}
            allocation_result.additional_metrics.update({
                "lpb_aht_estimator_path": estimator_kind,
                "lpb_sequential_prefix_updates": int(
                    estimator_kind == "sequential"
                ),
            })
        else:
            estimable_miscoverage = (
                (t_cal.reshape(-1, 1) < f)
                & (f <= C.reshape(-1, 1))
            )
            # Broadcasting avoids materializing a repeated N-by-n_taus copy
            # of the inverse-probability vector before applying the mask.
            self.miscoverage = (
                estimable_miscoverage
                * (1 / C_probs.reshape(-1, 1))
            ).mean(dim=0)
        # from predictive_bounds.calibration.calibration_utils import get_prior
        # prior_quantile_est = get_prior(f, self.taus_range, self.budget_allocator.tau_prior)
        # (prior_quantile_est <= C).float().mean()

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
        variance_weight = inverse_probability.var(unbiased=False).item()
        weight_quantiles = torch.quantile(
            inverse_probability,
            torch.tensor(
                [0.50, 0.90, 0.99],
                dtype=torch.float64,
                device=inverse_probability.device,
            ),
        )
        weight_square_sum = inverse_probability.square().sum()
        effective_sample_size_weight = (
            inverse_probability.sum().square()
            / weight_square_sum.clamp_min(torch.finfo(torch.float64).tiny)
        ).item()
        top_count = max(1, int(np.ceil(0.01 * len(inverse_probability))))
        top_weight_share = (
            torch.topk(inverse_probability, top_count).values.sum()
            / inverse_probability.sum().clamp_min(
                torch.finfo(torch.float64).tiny
            )
        ).item()
        if mean_weight is None:
            mean_weight = inverse_probability.mean().item()
        if max_weight is None:
            max_weight = inverse_probability.max().item()
        budget_used = self.allocation_result.total_budget_used
        if budget_used is None:
            budget_used = C.sum().item()
        f_prior = get_prior(f, self.taus_range, self.tau_prior)
        prior_observed_jailbreaks = (t.squeeze() < f_prior.squeeze()).float().sum().item()
        prior_observed_f_lower_c = (f_prior.squeeze() <= C.squeeze()).float().sum().item()
        prior_observed_both = (
                (f_prior.squeeze() <= C.squeeze()) & (t.squeeze() < f_prior.squeeze())).float().sum().item()
        n_observed_events = (C.squeeze() >= t.squeeze()).float().sum().item()
        n_achieved_q_prior1 = (C.squeeze() >= f_prior).float().sum().item()
        n_achieved_q_prior2 = (C.squeeze() > f_prior).float().sum().item()

        miscoverage = self.miscoverage
        quantile_est = model_prediction.quantile_est
        target_taus = target_taus.to(quantile_est.device)
        smallest_pos = select_calibration_positions(
            miscoverage,
            target_taus,
        )
        calibrated_test_quantile_est = f[:, smallest_pos]
        alpha_hat_per_tau = miscoverage[smallest_pos]

        latent_a = (
                t.reshape(-1, 1)
                < calibrated_test_quantile_est
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
        a_inverse_probability_sum = a_inverse_probability.sum(dim=0)
        a_effective_sample_size = (
            a_inverse_probability_sum.square()
            / a_inverse_probability.square().sum(dim=0).clamp_min(
                torch.finfo(torch.float64).tiny
            )
        )
        prior_estimable = (
                calibrated_test_quantile_est
                <= f_prior.reshape(-1, 1)
        )
        estimable_a_variance_proxy = (
                (latent_a & prior_estimable).to(torch.float64)
                * inverse_probability_minus_one.reshape(-1, 1)
        )
        mean_estimable_a_variance_proxy = (
                estimable_a_variance_proxy.mean(dim=0)
        )
        prior_a = (t.reshape(-1) < f_prior.reshape(-1)).to(torch.float64)
        mean_prior_a_variance_proxy = (
                prior_a * inverse_probability_minus_one
        ).mean().item()
        prior_a_inverse_probability = prior_a * inverse_probability
        mean_prior_a_inverse_probability = (
            prior_a_inverse_probability.mean().item()
        )

        selected_f_observed = (
            calibrated_test_quantile_est
            <= C.reshape(-1, 1)
        )
        all_observed_jailbreaks = latent_a.float().sum(dim=0)
        all_f_lower_c = selected_f_observed.float().sum(dim=0)
        all_observed_both = (
            selected_f_observed & latent_a
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
            "mean_a_weighted_inverse_probability_minus_one":
                mean_a_variance_proxy,
            "mean_a_weighted_inverse_probability":
                mean_a_inverse_probability,
            "variance_a_weighted_inverse_probability":
                variance_a_inverse_probability,
            "conditional_variance_of_ht_mean":
                mean_a_variance_proxy / len(inverse_probability),
            "a_weighted_effective_sample_size":
                a_effective_sample_size,
            "mean_estimable_a_weighted_inverse_probability_minus_one":
                mean_estimable_a_variance_proxy,
            "all_observed_jailbreaks": all_observed_jailbreaks,
            "all_f_lower_c": all_f_lower_c,
            "all_observed_both": all_observed_both,
            "alpha_hat_per_tau": alpha_hat_per_tau,
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
            'variance_weight': variance_weight,
            'median_weight': weight_quantiles[0].item(),
            'p90_weight': weight_quantiles[1].item(),
            'p99_weight': weight_quantiles[2].item(),
            'effective_sample_size_weight': effective_sample_size_weight,
            'top_1pct_weight_share': top_weight_share,
            'mean_inverse_probability_minus_one': inverse_probability_minus_one.mean().item(),
            'mean_prior_a_weighted_inverse_probability_minus_one': mean_prior_a_variance_proxy,
            'mean_prior_a_weighted_inverse_probability': (
                mean_prior_a_inverse_probability
            ),
            'variance_prior_a_weighted_inverse_probability': (
                prior_a_inverse_probability.var(unbiased=False).item()
            ),
            **indexed_metrics,
            **additional_metrics
        }
        return metrics
