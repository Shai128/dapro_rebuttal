import torch

from src.predictive_bounds.budget_allocators.budget_allocator import (
    BudgetAllocationResult,
    BudgetAllocator,
    summarize_expected_budget,
)
from src.predictive_bounds.budget_allocators.utils import sample_c
from src.predictive_bounds.calibration.calibration_utils import get_prior


class BasicBudgetAllocator(BudgetAllocator):

    def __init__(self, budget_per_sample: float, taus_range: torch.Tensor, tau_prior: float):
        super().__init__(budget_per_sample, taus_range, tau_prior)

    def allocate_budget(self, probability_est: torch.Tensor, x: torch.Tensor, t: torch.Tensor,
                        quantile_est: torch.Tensor) -> BudgetAllocationResult:
        self.cal_size = len(probability_est)
        prior_quantile_est = get_prior(quantile_est, self.taus_range, self.tau_prior)
        C_probs = self.budget_per_sample / prior_quantile_est
        C_probs = torch.minimum(C_probs, torch.ones_like(C_probs))
        C = sample_c(C_probs, prior_quantile_est)
        total_budget_used = torch.minimum(
            t.reshape(-1),
            C.reshape(-1).to(t.dtype),
        ).sum().item()
        probabilities = C_probs.reshape(-1).to(torch.float64)
        horizons = prior_quantile_est.reshape(-1).to(torch.float64)
        active_lengths = torch.minimum(
            t.reshape(-1).to(torch.float64),
            horizons,
        )
        expected_costs = probabilities * active_lengths
        expected_assigned_horizons = probabilities * horizons
        additional_metrics = {
            **summarize_expected_budget(
                expected_costs.sum().item(),
                len(probabilities),
                self.budget_per_sample,
                cost_semantics=(
                    "bernoulli_horizon_with_event_stopping"
                ),
            ),
            "static_expected_cost_total": expected_costs.sum().item(),
            "static_expected_cost_per_sample": (
                expected_costs.mean().item()
            ),
            "static_expected_assigned_horizon_total": (
                expected_assigned_horizons.sum().item()
            ),
            "static_expected_assigned_horizon_per_sample": (
                expected_assigned_horizons.mean().item()
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
        return "basic"
