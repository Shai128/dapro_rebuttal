import torch

from src.predictive_bounds.budget_allocators.budget_allocator import (
    BudgetAllocationResult,
    BudgetAllocator,
    summarize_expected_budget,
)
from src.predictive_bounds.budget_allocators.utils import sample_c
from src.predictive_bounds.calibration.calibration_utils import get_prior, solve_optimization


class OptimizedBudgetAllocator(BudgetAllocator):

    def __init__(self, budget_per_sample: float, taus_range: torch.Tensor, tau_prior: float, max_estimator: int):
        super().__init__(budget_per_sample, taus_range, tau_prior)
        self.max_estimator = max_estimator

    def allocate_budget(self, probability_est: torch.Tensor, x: torch.Tensor, t: torch.Tensor,
                        quantile_est: torch.Tensor) -> BudgetAllocationResult:
        self.cal_size = len(probability_est)

        prior_quantile_est = get_prior(quantile_est, self.taus_range, self.tau_prior)

        device = probability_est.device
        trimmed_prior_quantile_est = torch.minimum(prior_quantile_est,
                                                   self.max_estimator * torch.ones_like(prior_quantile_est))
        trimmed_quantile_est = torch.minimum(quantile_est, self.max_estimator * torch.ones_like(quantile_est))
        C_probs, _ = solve_optimization(trimmed_prior_quantile_est.cpu().detach().numpy(),
                                        self.budget_per_sample * len(prior_quantile_est), tol=1e-8)
        C_probs = torch.Tensor(C_probs).to(device)

        C = sample_c(C_probs, prior_quantile_est)
        total_budget_used = torch.minimum(
            t.reshape(-1),
            C.reshape(-1).to(t.dtype),
        ).sum().item()
        probabilities = C_probs.reshape(-1).to(torch.float64)
        executed_horizons = prior_quantile_est.reshape(-1).to(torch.float64)
        solver_horizons = trimmed_prior_quantile_est.reshape(-1).to(
            torch.float64
        )
        active_lengths = torch.minimum(
            t.reshape(-1).to(torch.float64),
            executed_horizons,
        )
        expected_costs = probabilities * active_lengths
        expected_assigned_horizons = probabilities * executed_horizons
        solver_expected_assigned_horizons = probabilities * solver_horizons
        additional_metrics = {
            **summarize_expected_budget(
                expected_costs.sum().item(),
                len(probabilities),
                self.budget_per_sample,
                cost_semantics=(
                    "optimized_bernoulli_horizon_with_event_stopping"
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
            "optimization_constraint_expected_horizon_total": (
                solver_expected_assigned_horizons.sum().item()
            ),
            "optimization_to_executed_expected_horizon_gap": (
                expected_assigned_horizons.sum().item()
                - solver_expected_assigned_horizons.sum().item()
            ),
        }
        # store(t, C_probs, prior_quantile_est, C)
        return BudgetAllocationResult(
            trimmed_quantile_est,
            C,
            C_probs,
            total_budget_used=total_budget_used,
            additional_metrics=additional_metrics,
        )

    @property
    def name(self) -> str:
        return "optimized"

def store(t, C_probs, prior_quantile_est, C):

    curr_event_time = t.cpu()
    curr_prior_q = prior_quantile_est.cpu()
    curr_final_C = C.cpu()
    expected_c = (C_probs * prior_quantile_est).cpu()

    data_to_save = {
        'expected_c': expected_c,
        'event_times': curr_event_time,
        'prior_qs':curr_prior_q,
        'final_Cs': curr_final_C,
    }

    # 4. Save to a file
    torch.save(data_to_save, 'static_optimized_evaluation_plot_data.pt')
    print("stored static optimized")
