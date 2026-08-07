"""Full-observation reference allocator for fixed-benchmark metrics."""

import torch

from src.predictive_bounds.budget_allocators.budget_allocator import (
    BudgetAllocationResult,
    BudgetAllocator,
    summarize_expected_budget,
)


class FullBudgetOracleAllocator(BudgetAllocator):
    """Observe every row up to the benchmark horizon without re-weighting."""

    uses_full_benchmark = True

    def __init__(
            self,
            taus_range: torch.Tensor,
            tau_prior: float,
            m_upper_bound: int,
    ):
        super().__init__(float(m_upper_bound), taus_range, tau_prior)
        self.m_upper_bound = int(m_upper_bound)

    def allocate_budget(
            self,
            probability_est: torch.Tensor,
            x: torch.Tensor,
            t: torch.Tensor,
            quantile_est: torch.Tensor,
    ) -> BudgetAllocationResult:
        del probability_est, x
        n = len(t)
        horizons = torch.full(
            (n,),
            self.m_upper_bound,
            dtype=torch.long,
            device=t.device,
        )
        realized_costs = torch.minimum(t.reshape(-1).long(), horizons)
        total_budget_used = realized_costs.sum().item()
        return BudgetAllocationResult(
            f=quantile_est,
            C=horizons,
            C_probs=torch.ones(n, dtype=torch.float64, device=t.device),
            total_budget_used=total_budget_used,
            mean_weight=1.0,
            max_weight=1.0,
            additional_metrics={
                **summarize_expected_budget(
                    total_budget_used,
                    n,
                    self.m_upper_bound,
                    cost_semantics="full_observation_with_event_stopping",
                ),
                "is_full_budget_oracle": 1,
            },
        )

    @property
    def name(self) -> str:
        return "oracle_full_budget"
