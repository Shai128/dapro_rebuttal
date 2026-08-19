from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from src.evaluation.estimate import (
    AllocationEstimationDiagnostics,
    IPCWTrajectorySimulator,
    RestrictedMeanTimeToUnsafeMetric,
    TotalBudgetUsed,
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
from src.evaluation import merge_results as metric_merge
from src.evaluation.summarize import _normalize_legacy_configuration_rows
from src.predictive_bounds.utils.utils import get_calibration_experiment_name


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
        "optimized",
        (
            "dapro_soft_prefix_bins_2_metric_horizon_200_global_0p001_"
            "projection_margin_1p00_n1_200"
        ),
        (
            "dapro_soft_prefix_bins_2_metric_horizon_200_global_0p001_budget_"
            "crc_control_100_row_cap_2p00x_budget_causal_shared_pav_v1_"
            "n1_200"
        ),
        "oracle_split_full_budget",
        "oracle_full_budget",
    ]


def test_legacy_registry_flags_do_not_change_the_production_matrix():
    taus = torch.arange(0.01, 1.0, 0.01)
    default_names = [allocator.name for allocator in get_metric_allocators(
        None,
        20,
        200,
        taus,
        0.56,
        "cpu",
        dapro_n1=200,
        crc_control_size=100,
    )]
    flagged_names = [allocator.name for allocator in get_metric_allocators(
        None,
        20,
        200,
        taus,
        0.56,
        "cpu",
        dapro_n1=200,
        crc_control_size=100,
        include_legacy_dapro=False,
        include_locally_adaptive=False,
    )]

    assert flagged_names == default_names


def test_metric_experiment_name_normalizes_integer_budget():
    assert metric_experiment_name("data", "setup", 5.0, 200, 100, "v1") == (
        "data_setup_5_m_v1"
    )
    assert metric_experiment_name("data", "setup", 5, 50, 25, "v1") == (
        "data_setup_5_m_v1"
    )


def test_bound_experiment_name_is_compact_and_configuration_independent():
    assert get_calibration_experiment_name(
        "data", "setup", 5.0, 3000, 0.56, 40.0, "lpb_v1"
    ) == "data_setup_5_calibration_lpb_v1"


def test_metric_merge_consolidates_configs_in_one_output(tmp_path, monkeypatch):
    def fake_registry(*args, dapro_n1, crc_control_size, **kwargs):
        del args, kwargs, crc_control_size
        return [
            SimpleNamespace(name="shared_baseline"),
            SimpleNamespace(name=f"dapro_n1_{dapro_n1}"),
        ]

    def fake_process(calibration, seed, experiments_name):
        del experiments_name
        return pd.DataFrame([{
            "seed": seed,
            "allocator_name": calibration.name,
            "calibration_name": calibration.name,
        }])

    monkeypatch.setattr(metric_merge, "get_metric_allocators", fake_registry)
    monkeypatch.setattr(metric_merge, "process_calibration", fake_process)
    monkeypatch.setattr(
        metric_merge,
        "get_merged_metric_calibration_result_path",
        lambda _name: str(tmp_path),
    )

    metric_merge.merge_results(
        experiments_name="compact",
        seeds=(0, 2),
        budget_per_sample=5,
        taus_range=torch.tensor([0.1]),
        tau_prior=0.56,
        m_upper_bound=200,
        device="cpu",
        dapro_configs=[(200, 100), (100, 50), (50, 25)],
    )

    merged = pd.read_csv(tmp_path / "all_df.csv")
    assert len(merged) == 8
    shared = merged[merged["allocator_name"] == "shared_baseline"]
    assert len(shared) == 2
    assert shared["configuration_scope"].eq("shared").all()
    assert shared["configured_dapro_n1"].isna().all()
    assert shared["configured_crc_control_size"].isna().all()
    assert set(shared["applicable_dapro_configs"]) == {
        "200:100|100:50|50:25"
    }

    specific = merged[merged["allocator_name"] != "shared_baseline"]
    assert specific["configuration_scope"].eq("specific").all()
    assert set(zip(
        specific["configured_dapro_n1"],
        specific["configured_crc_control_size"],
    )) == {(200, 100), (100, 50), (50, 25)}


def test_legacy_configuration_copies_are_normalized_without_data_loss():
    rows = []
    for n1, crc in ((200, 100), (100, 50), (50, 25)):
        rows.append({
            "seed": 7,
            "allocator_name": "shared",
            "estimated_cjr": 42.0,
            "configured_dapro_n1": n1,
            "configured_crc_control_size": crc,
        })
    rows.append({
        "seed": 7,
        "allocator_name": "dapro_n1_100",
        "estimated_cjr": 41.0,
        "configured_dapro_n1": 100,
        "configured_crc_control_size": 50,
    })

    normalized = _normalize_legacy_configuration_rows(pd.DataFrame(rows))

    assert len(normalized) == 2
    shared = normalized[normalized["allocator_name"] == "shared"].iloc[0]
    assert shared["configuration_scope"] == "shared"
    assert shared["applicable_dapro_configs"] == "50:25|100:50|200:100"
    assert pd.isna(shared["configured_dapro_n1"])
    specific = normalized[
        normalized["allocator_name"] == "dapro_n1_100"
    ].iloc[0]
    assert specific["configuration_scope"] == "specific"
    assert specific["configured_dapro_n1"] == 100


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
    np.testing.assert_allclose(
        metrics["mean_metric_target_a_weighted_inverse_probability"],
        (2 + 4) / 3,
    )
    assert metrics["metric_target_a_count"] == 2
    assert metrics["num_trajectories_fully_resolved"] == 2
    assert metrics["total_expected_budget"] == 6.5
    np.testing.assert_allclose(
        metrics[
            "estimated_conditional_variance_unsafe_event_rate_estimator"
        ],
        14 / 9,
    )


def test_reported_metric_budget_is_sum_c_not_event_stopped_cost():
    class FixedAllocator:
        name = "fixed"

        def allocate_budget(self, *_args):
            return BudgetAllocationResult(
                f=torch.empty(0),
                C=torch.tensor([4, 4, 1]),
                C_probs=torch.ones(3),
                # Allocator historically reported actual event-stopped cost.
                total_budget_used=7,
            )

    event_times = torch.tensor([2, 4, 5])
    prediction = SimpleNamespace(
        probability_est=torch.empty((3, 4)),
        quantile_est=torch.empty((3, 1)),
    )
    data = IPCWTrajectorySimulator.simulate(
        FixedAllocator(), None, prediction, event_times, max_time=4
    )
    metrics = TotalBudgetUsed().compute(data)

    assert metrics["total_budget_utilized"] == 9
    assert metrics["budget_per_sample"] == 3
    assert metrics["actual_event_stopped_budget_total"] == 7
    assert metrics["allocator_reported_budget_total"] == 7
    assert metrics["reported_budget_semantics"] == "sum_assigned_C_i"


def test_standard_restricted_mean_is_distinct_from_conditional_event_time():
    class FullHorizonAllocator:
        name = "full"

        def allocate_budget(self, *_args):
            return BudgetAllocationResult(
                f=torch.empty(0),
                C=torch.tensor([3, 3]),
                C_probs=torch.ones(2),
                total_budget_used=6,
            )

    event_times = torch.tensor([1, 4])
    prediction = SimpleNamespace(
        probability_est=torch.empty((2, 3)),
        quantile_est=torch.empty((2, 1)),
    )
    data = IPCWTrajectorySimulator.simulate(
        FullHorizonAllocator(), None, prediction, event_times, max_time=3
    )
    metrics = RestrictedMeanTimeToUnsafeMetric(
        oracle_rmttu=1.0,
        oracle_restricted_mean=2.0,
    ).compute(data)

    assert metrics["estimated_rmttu"] == 1.0
    assert metrics["estimated_restricted_mean_time_to_event"] == 2.0
    assert (
        metrics[
            "conditional_variance_restricted_mean_time_to_event_estimator"
        ]
        == 0.0
    )
    assert metrics["historical_rmttu_semantics"] == (
        "mean_event_time_conditional_on_event"
    )
