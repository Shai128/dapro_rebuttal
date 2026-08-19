"""Zero-acquisition model plug-in baseline for metric estimation."""

import torch

from src.predictive_bounds.budget_allocators.budget_allocator import (
    BudgetAllocator,
)


class UncalibratedMetricAllocator(BudgetAllocator):
    """Identify the model-only metric baseline in the shared registry.

    This object intentionally never allocates trajectories.  The metric
    runner recognizes ``uses_model_only_metric_estimator`` and computes
    plug-in metrics directly from the initial-prefix PMF.
    """

    uses_model_only_metric_estimator = True
    uses_full_benchmark = False

    def __init__(
            self,
            conditional_grid: torch.Tensor,
            taus_range: torch.Tensor,
            tau_prior: float,
            m_upper_bound: int,
    ):
        super().__init__(0.0, taus_range, tau_prior)
        self.conditional_grid = conditional_grid
        self.m_upper_bound = int(m_upper_bound)

    def allocate_budget(self, *_args, **_kwargs):
        raise RuntimeError(
            "UncalibratedMetricAllocator is a model-only estimator and "
            "must not simulate trajectory acquisition."
        )

    @property
    def name(self) -> str:
        return "uncalibrated"
