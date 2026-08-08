from types import SimpleNamespace

import numpy as np
import torch

from src.evaluation.estimate import (
    AllocationEstimationDiagnostics,
    IPCWTrajectorySimulator,
    metric_experiment_name,
)
from src.predictive_bounds.budget_allocators.DAPRO import TargetAWeightedDAPRO
from src.predictive_bounds.budget_allocators.budget_allocator import (
    BudgetAllocationResult,
)
from src.predictive_bounds.budget_allocators.oracle_dapro_allocator import (
    GlobalOracleTargetADAPRO,
)
from src.predictive_bounds.utils.get_calibration_methods_utils import (
    get_metric_allocators,
)


def test_metric_registry_contains_exact_requested_comparison():
    taus = torch.arange(0.01, 1.0, 0.01)
    names = [
        allocator.name
        for allocator in get_metric_allocators(
            None,
            20,
            200,
            taus,
            0.56,
            "cpu",
            dapro_n1=200,
            crc_control_size=100,
        )
    ]
    assert names == [
        "UniformBudgetAllocator",
        "UnweightedUniformBudgetAllocator",
        "oracle_full_budget",
        "optimized",
        "random_adaptive_optimized_mixture_terminal_floor_0p005_crc",
        "projected_optimization_direct_bins_2_prob_n1_200",
        (
            "projected_optimization_direct_bins_2_prob_a_target_raw_"
            "alpha_0p10_n1_200"
        ),
        (
            "dapro_variance_aligned_bins_2_alpha_0p10_global_0p001_"
            "projection_margin_1p00_n1_200"
        ),
        (
            "projected_optimization_direct_bins_2_prob_budget_crc_control_"
            "100_row_cap_2p00x_budget_n1_200"
        ),
        (
            "projected_optimization_direct_bins_2_prob_a_target_raw_"
            "alpha_0p10_budget_crc_control_100_row_cap_2p00x_budget_n1_200"
        ),
        (
            "dapro_variance_aligned_bins_2_alpha_0p10_global_0p001_budget_"
            "crc_control_100_row_cap_2p00x_budget_n1_200"
        ),
        "oracle_target_a_dapro_alpha_0p10_n1_200",
        "oracle_target_a_dapro_alpha_0p10_crc_control_100_n1_200",
        "oracle_target_a_dapro_no_split_alpha_0p10",
    ]


def test_metric_experiment_name_normalizes_integer_budget():
    assert metric_experiment_name("data", "setup", 5.0, 200, 100, "v1") == (
        "data_setup_5_metric_estimation_n1_200_crc_100__v1"
    )


def test_target_a_uses_metric_event_instead_of_lpb_anchor():
    grid = torch.full((3, 4, 4), 0.25)
    taus = torch.tensor([0.10, 0.56])
    allocator = TargetAWeightedDAPRO(
        grid,
        2.0,
        taus,
        0.56,
        4,
        projection="direct_bins_2",
        score="prob",
        n1=1,
        anchor_kind="raw_alpha",
        target_alpha=0.10,
        metric_estimation_horizon=4,
    )
    event_times = torch.tensor([1.0, 4.0, 5.0])
    prior = torch.full((3,), 4.0)
    quantiles = torch.tensor([[1.0, 4.0], [2.0, 4.0], [3.0, 4.0]])

    weights = allocator.phase1_objective_weights(
        event_times,
        prior,
        quantiles,
    )

    assert weights.tolist() == [1.0, 1.0, 0.0]
    assert allocator.objective_kind == (
        "mean_metric_event_weighted_inverse_probability_minus_one"
    )
    assert allocator.objective_metadata()["target_metric_horizon"] == 4


def test_metric_oracle_uses_the_same_target_a_indicator():
    grid = torch.full((3, 4, 4), 0.25)
    taus = torch.tensor([0.10, 0.56])
    allocator = GlobalOracleTargetADAPRO(
        grid,
        2.0,
        taus,
        0.56,
        4,
        metric_estimation_horizon=4,
    )
    target_a, anchor = allocator._target_indicator(
        torch.tensor([1.0, 4.0, 5.0]),
        torch.full((3, 2), 4.0),
        torch.full((3,), 4.0),
    )
    assert target_a.tolist() == [1.0, 1.0, 0.0]
    assert anchor == -1


def test_metric_diagnostics_record_requested_seed_level_values():
    class FixedAllocator:
        name = "fixed"

        def allocate_budget(self, *_args):
            return BudgetAllocationResult(
                f=torch.empty(0),
                C=torch.tensor([2, 4, 1]),
                C_probs=torch.tensor([0.5, 0.25, 1.0]),
                total_budget_used=7,
                additional_metrics={"total_expected_budget": 6.5},
            )

    event_times = torch.tensor([2, 4, 5])
    prediction = SimpleNamespace(
        probability_est=torch.empty((3, 4)),
        quantile_est=torch.empty((3, 1)),
    )
    data = IPCWTrajectorySimulator.simulate(
        FixedAllocator(),
        None,
        prediction,
        event_times,
        max_time=4,
    )
    metrics = AllocationEstimationDiagnostics().compute(data)

    np.testing.assert_allclose(metrics["mean_weight"], (2 + 4 + 1) / 3)
    np.testing.assert_allclose(
        metrics["mean_a_weighted_weight"],
        (2 + 4) / 3,
    )
    assert metrics["metric_target_a_count"] == 2
    assert metrics["num_trajectories_fully_resolved"] == 2
    assert metrics["total_expected_budget"] == 6.5
