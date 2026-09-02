from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.paper_figures.bounds import BOUND_BOX_SPECS
from src.paper_figures.config import (
    METRIC_MAIN_METHOD_ORDER,
    RECOMMENDED_CONFIGURATIONS,
)
from src.paper_figures.data import (
    _discover_sources,
    _derive_restricted_mean,
    _paper_budget_used_per_sample,
)
from src.paper_figures.metrics import (
    METRIC_BOX_SPECS,
    METRIC_VARIANCE_SPECS,
)


def test_recommended_paper_cells_match_final_table():
    assert RECOMMENDED_CONFIGURATIONS["lpb"]["autoif"].budget_per_sample == 10
    assert RECOMMENDED_CONFIGURATIONS["lpb"]["toxicity"].n1 == 50
    assert RECOMMENDED_CONFIGURATIONS["lpb"]["red_team_qwen_judge"].budget_per_sample == 10
    assert RECOMMENDED_CONFIGURATIONS["upb"]["autoif"].target_coverage == 0.80
    assert RECOMMENDED_CONFIGURATIONS["upb"]["autoif"].n1 == 200
    assert RECOMMENDED_CONFIGURATIONS["upb"]["hallucination3"].n1 == 200
    assert RECOMMENDED_CONFIGURATIONS["metrics"]["autoif"].n1 == 100
    assert RECOMMENDED_CONFIGURATIONS["metrics"]["hallucination3"].n1 == 100


def test_paper_cells_use_common_task_level_hyperparameters():
    for task, recommendations in RECOMMENDED_CONFIGURATIONS.items():
        n1_by_budget: dict[float, set[int]] = {}
        for recommendation in recommendations.values():
            n1_by_budget.setdefault(
                recommendation.budget_per_sample, set()
            ).add(recommendation.n1)
        assert all(len(values) == 1 for values in n1_by_budget.values()), task

        assert (
            recommendations["red_team_qwen_judge"].budget_per_sample
            == recommendations["red_team_llama_guard"].budget_per_sample
        )

    upb_coverages = {
        recommendation.target_coverage
        for recommendation in RECOMMENDED_CONFIGURATIONS["upb"].values()
    }
    assert upb_coverages == {0.80}


def test_legacy_ordinary_ht_columns_reconstruct_exact_restricted_mean():
    frame = pd.DataFrame({
        "estimated_cjr": [25.0],
        "estimated_rmttu": [80.0],
        "estimated_restricted_mean_time_to_event": [np.nan],
        "full_benchmark_cjr": [20.0],
        "full_benchmark_rmttu": [50.0],
        "full_benchmark_restricted_mean_time_to_event": [np.nan],
        "unsafe_event_rate_estimator_kind": ["ordinary_ht"],
    })
    _derive_restricted_mean(frame, horizon=200.0)
    assert frame.loc[0, "estimated_restricted_mean"] == 170.0
    assert frame.loc[0, "full_benchmark_restricted_mean"] == 170.0
    assert frame.loc[0, "restricted_mean_source"].startswith("exact_")


def test_nonordinary_legacy_estimator_is_not_mislabeled_as_rmst():
    frame = pd.DataFrame({
        "estimated_cjr": [25.0],
        "estimated_rmttu": [80.0],
        "estimated_restricted_mean_time_to_event": [np.nan],
        "full_benchmark_cjr": [20.0],
        "full_benchmark_rmttu": [50.0],
        "full_benchmark_restricted_mean_time_to_event": [170.0],
        "unsafe_event_rate_estimator_kind": ["sequential"],
    })
    _derive_restricted_mean(frame, horizon=200.0)
    assert np.isnan(frame.loc[0, "estimated_restricted_mean"])
    assert frame.loc[0, "restricted_mean_source"] == "unavailable"


def test_bound_allocation_diagnostics_hide_nonallocation_methods():
    specifications = {
        metric: allocation_only
        for metric, _, _, allocation_only in BOUND_BOX_SPECS
    }
    for metric in (
        "budget_used_per_sample",
        "n_observed_events",
        "mean_weight",
        "mean_selected_a_weight",
        "mean_prior_a_weight",
        "mean_tau_0p10_a_weight",
    ):
        assert specifications[metric]
    assert not specifications["coverage_pct"]
    assert not specifications["size"]


def test_metric_paper_matrix_contains_rmst_and_filters_diagnostics():
    specifications = {
        metric: hidden
        for metric, _, _, hidden, _ in METRIC_BOX_SPECS
    }
    assert "estimated_restricted_mean" in specifications
    assert {
        metric for metric, _, _ in METRIC_VARIANCE_SPECS
    } == {"estimated_cjr", "estimated_restricted_mean"}
    for metric in (
        "budget_per_sample",
        "mean_metric_a_weight",
    ):
        assert set(specifications[metric]) == {"Uncalibrated", "Oracle"}
    assert "observed_events" not in specifications
    assert specifications["estimated_cjr"] == ()
    assert specifications["estimated_restricted_mean"] == ()


def test_metric_main_figures_use_crc_dapro_and_exclude_other_variants():
    assert METRIC_MAIN_METHOD_ORDER == (
        "Uncalibrated",
        "Static",
        "DAPRO",
    )
    assert "Oracle" not in METRIC_MAIN_METHOD_ORDER
    assert "DAPRO w/o CRC" not in METRIC_MAIN_METHOD_ORDER


def test_paper_budget_uses_assigned_static_and_realized_dapro_cost():
    frame = pd.DataFrame({
        "method": [
            "Static",
            "DAPRO",
            "DAPRO w/o CRC",
            "Uncalibrated",
            "Oracle",
        ]
    })
    assigned = pd.Series([20.0, 42.0, 39.0, 0.0, 200.0])
    realized = pd.Series([12.0, 19.5, 20.1, 0.0, 125.0])

    budget, semantics = _paper_budget_used_per_sample(
        frame, assigned=assigned, realized=realized
    )

    np.testing.assert_allclose(budget.iloc[:3], [20.0, 19.5, 20.1])
    assert budget.iloc[3:].isna().all()
    assert semantics.tolist() == [
        "assigned_sum_C_i_per_sample",
        "actual_event_stopped_turns_per_sample",
        "actual_event_stopped_turns_per_sample",
        "not_applicable",
        "not_applicable",
    ]


def test_metric_source_discovery_ignores_score_ablation_results(tmp_path: Path):
    setup = (
        "dataset_toxicity_attack_toxic_attack_qwen25_14b_instruct_"
        "lm_target_qwen25_14b_instruct_judge_detoxify_20_m"
    )
    canonical = tmp_path / f"{setup}_metric" / "all_df.csv"
    ablation = (
        tmp_path
        / f"{setup}_dapro_metric_ablation_v1_score"
        / "all_df.csv"
    )
    columns = {
        "reported_budget_semantics": ["actual_event_stopped_turns_per_sample"],
        "actual_event_stopped_budget_per_sample": [20.0],
        "estimated_restricted_mean_time_to_event": [150.0],
        "mean_metric_target_a_weighted_inverse_probability": [2.0],
    }
    canonical.parent.mkdir(parents=True)
    ablation.parent.mkdir(parents=True)
    pd.DataFrame(columns).to_csv(canonical, index=False)
    pd.DataFrame(columns).to_csv(ablation, index=False)

    selected = _discover_sources(tmp_path, "metrics")

    assert [item.path for item in selected] == [canonical]
