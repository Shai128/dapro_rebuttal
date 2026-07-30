import torch

from src.safety_evaluation.budget_allocators.budget_allocator import (
    BudgetAllocationResult,
    BudgetAllocator,
    summarize_expected_budget,
)
from src.safety_evaluation.budget_allocators.utils import sample_c
from src.safety_evaluation.calibration.calibration_utils import get_prior
from src.safety_evaluation.utils.utils import geom_cdf_stable


class NaiveBudgetAllocator(BudgetAllocator):

    def __init__(self, budget_per_sample: float, taus_range: torch.Tensor, tau_prior: float):
        super().__init__(budget_per_sample, taus_range, tau_prior)

    def allocate_budget(self, probability_est: torch.Tensor, x: torch.Tensor, t: torch.Tensor,
                        quantile_est: torch.Tensor) -> BudgetAllocationResult:
        prior_quantile_est = get_prior(quantile_est, self.taus_range, self.tau_prior)

        p_C = 1 / self.budget_per_sample
        C_probs = 1 - geom_cdf_stable(p=p_C, k=prior_quantile_est.cpu().detach().numpy())
        C_probs = torch.Tensor(C_probs).to(prior_quantile_est.device)

        C = sample_c(C_probs, prior_quantile_est)
        total_budget_used = C.squeeze().sum().item()
        probabilities = C_probs.reshape(-1).to(torch.float64)
        horizons = prior_quantile_est.reshape(-1).to(torch.float64)
        expected_assigned_horizons = probabilities * horizons
        event_stopped_lengths = torch.minimum(
            t.reshape(-1).to(torch.float64),
            horizons,
        )
        expected_event_stopped_costs = (
            probabilities * event_stopped_lengths
        )
        additional_metrics = {
            **summarize_expected_budget(
                expected_assigned_horizons.sum().item(),
                len(probabilities),
                self.budget_per_sample,
                cost_semantics=(
                    "bernoulli_assigned_horizon_matching_legacy_budget_used"
                ),
            ),
            "static_expected_cost_total": (
                expected_assigned_horizons.sum().item()
            ),
            "static_expected_cost_per_sample": (
                expected_assigned_horizons.mean().item()
            ),
            "static_expected_assigned_horizon_total": (
                expected_assigned_horizons.sum().item()
            ),
            "static_expected_assigned_horizon_per_sample": (
                expected_assigned_horizons.mean().item()
            ),
            "static_expected_event_stopped_cost_total": (
                expected_event_stopped_costs.sum().item()
            ),
            "static_expected_event_stopped_cost_per_sample": (
                expected_event_stopped_costs.mean().item()
            ),
        }

        return BudgetAllocationResult(
            quantile_est,
            C,
            C_probs,
            total_budget_used=total_budget_used,
            additional_metrics=additional_metrics,
        )

    @property
    def name(self) -> str:
        return "naive"
