from pathlib import Path
import inspect

import numpy as np
import pandas as pd
import torch

from src.predictive_bounds.experiments.dapro_ablation import (
    summarize_paper as ablation_summarizer,
)
from src.predictive_bounds.budget_allocators.dapro_ablation import (
    AblationHardTargetDAPRO,
    AblationSoftTargetDAPRO,
)
from src.predictive_bounds.experiments.dapro_ablation.summarize_paper import (
    _discover,
    generate_ablation_figure,
    generate_metric_ablation_figure,
    load_ablation_data,
    load_metric_ablation_data,
)
from src.predictive_bounds.budget_allocators.optimization_solver_utils import (
    solve_binned_deployable_policy,
)
from src.predictive_bounds.utils.get_calibration_methods_utils import (
    get_dapro_ablation_calibrations,
    get_metric_dapro_ablation_allocators,
)
from src.predictive_bounds import construct_calibrated_bound


def _allocator(grid: torch.Tensor, noise: float) -> AblationSoftTargetDAPRO:
    return AblationSoftTargetDAPRO(
        grid,
        2.0,
        torch.arange(0.01, 0.5, 0.01),
        0.56,
        4,
        n1=2,
        target_alpha=0.10,
        projection_budget_margin=0.0,
        ablation_kind="score_noise",
        ablation_value=noise,
        score_noise_lambda=noise,
        score_noise_seed=17,
    )


def test_score_corruption_is_deterministic_and_preserves_column_marginals():
    grid = torch.zeros(8, 4, 5, dtype=torch.float64)
    for row in range(8):
        for time in range(4):
            grid[row, time, time] = 10 * time + row / 10
    quantiles = torch.zeros(8, 49)
    original_allocator = _allocator(grid, 0.0)
    random_allocator = _allocator(grid, 1.0)
    original = original_allocator.policy_scores(quantiles)
    random_a = random_allocator.policy_scores(quantiles)
    random_b = _allocator(grid, 1.0).policy_scores(quantiles)
    mixed = _allocator(grid, 0.5).policy_scores(quantiles)

    torch.testing.assert_close(random_a, random_b)
    torch.testing.assert_close(mixed, 0.5 * original + 0.5 * random_a)
    for time in range(original.shape[1]):
        torch.testing.assert_close(
            torch.sort(random_a[:, time]).values,
            torch.sort(original[:, time]).values,
        )
    assert not torch.equal(original, random_a)
    metadata = random_allocator.objective_metadata()
    assert 0.0 <= metadata["ablation_score_original_k2_bin_agreement"] <= 1.0
    assert np.isfinite(
        metadata["ablation_score_mean_timewise_pearson_correlation"]
    )


def test_ablation_registry_is_symmetric_between_raw_and_crc():
    taus = torch.arange(0.01, 0.5, 0.01)
    methods = get_dapro_ablation_calibrations(
        None,
        20,
        taus,
        0.56,
        200,
        ablation_kind="score_noise",
        dapro_n1_values=(100,),
        score_noise_lambdas=(0.0, 0.5, 1.0),
    )
    names = [method.name for method in methods]
    assert len(names) == 7
    assert len(set(names)) == len(names)
    assert sum("budget_crc" in name for name in names) == 3
    assert sum("projection_margin_0p00" in name for name in names) == 3
    assert all(
        "ablation_score_noise" in name
        for name in names
        if "dapro_soft_prefix" in name
    )


def test_summarizer_uses_assigned_static_and_realized_dapro_budget(tmp_path):
    taus = torch.arange(0.01, 0.5, 0.01)
    methods = get_dapro_ablation_calibrations(
        None,
        20,
        taus,
        0.56,
        200,
        ablation_kind="score_noise",
        dapro_n1_values=(100,),
        score_noise_lambdas=(0.0, 1.0),
    )
    rows = []
    for seed in (0, 1):
        for method in methods:
            name = method.name
            is_static = "optimized" in name
            is_crc = "budget_crc" in name
            value = np.nan
            if "ablation_score_noise_0_" in name:
                value = 0.0
            elif "ablation_score_noise_1_" in name:
                value = 1.0
            rows.append({
                "seed": seed,
                "calibration_name": name,
                "target_coverage": 0.90,
                "coverage": 0.895 + 0.002 * seed,
                "configured_budget_per_sample": 20,
                "reported_assigned_budget_per_sample": 20 if is_static else 30,
                "actual_event_stopped_budget_per_sample": 12 if is_static else 19,
                "mean_calibrated_a_weighted_inverse_probability": 1.2,
                "ablation_kind": np.nan if is_static else "score_noise",
                "ablation_value": value,
                "ablation_n1": np.nan if is_static else 100,
                "ablation_crc_control_size": 50 if is_crc else 0,
                "ablation_uses_crc": int(is_crc),
                "ablation_score_noise_lambda": value,
            })
    parent = tmp_path / "toxicity_20_calibration_test_score_noise"
    parent.mkdir()
    pd.DataFrame(rows).to_csv(parent / "all_df.csv", index=False)
    data, _ = load_ablation_data(
        tmp_path, experiment_prefix="test", kind="score_noise"
    )

    assert set(data["method"]) == {"Static", "DAPRO", "DAPRO w/o CRC"}
    assert set(data.loc[data["method"].eq("Static"), "budget_used_per_sample"]) == {20}
    assert set(data.loc[~data["method"].eq("Static"), "budget_used_per_sample"]) == {19}
    assert data.groupby(["factor_value", "method"])["seed"].nunique().eq(2).all()

    output = tmp_path / "figure.jpg"
    statistics = generate_ablation_figure(
        data, kind="score_noise", output_path=output, quality="low"
    )
    assert {"mean", "variance", "std", "count", "metric"}.issubset(
        statistics.columns
    )
    assert statistics["count"].eq(2).all()
    assert output.exists()
    assert output.stat().st_size < 120 * 1024


def test_server_launcher_exposes_slurm_cpu_and_parallel_controls():
    root = Path(__file__).resolve().parents[1]
    script = (
        root
        / "src/predictive_bounds/experiments/dapro_ablation/scripts/run.sh"
    ).read_text(encoding="utf-8")
    assert "--slurm" in script
    assert "--parallel-jobs" in script
    assert "--cpu" in script
    assert "--method-suite dapro_ablation" in script
    assert "SEED_END=50" in script
    assert "hard_soft" in script
    assert "representation" in script
    assert "attacker_shift" in script
    assert "--test-dataset-setup" in script
    assert "src.evaluation.estimate" in script
    assert "src.evaluation.merge_results" in script
    assert "METRIC_DAPRO_N1=50" in script
    assert "METRIC_CRC_CONTROL_SIZE=25" in script
    assert "ATTACKER_SHIFT_BUDGET=10" in script
    assert '--budget-per-sample "$ATTACKER_SHIFT_BUDGET"' in script
    assert "metric_score_noise" in script
    assert "metric_score" in script


def test_attacker_shift_setup_is_not_executed_before_main_parses_args():
    """Regression for the server-side NameError in main()."""
    main_source = inspect.getsource(construct_calibrated_bound.main)
    runner_source = inspect.getsource(
        construct_calibrated_bound.run_experiments
    )
    assert "if test_dataset_name is not None" not in main_source
    assert "test_dataset_name=args.test_dataset_name" in main_source
    assert "if test_dataset_name is not None" in runner_source


def test_additional_ablation_registry_has_static_and_paired_controllers():
    taus = torch.arange(0.01, 0.5, 0.01)
    expected_sizes = {
        "hard_soft": 5,
        "representation": 11,
        "score": 11,
        "attacker_shift": 3,
    }
    for kind, expected in expected_sizes.items():
        methods = get_dapro_ablation_calibrations(
            None,
            20,
            taus,
            0.56,
            200,
            ablation_kind=kind,
            dapro_n1_values=(50,),
        )
        names = [method.name for method in methods]
        assert len(names) == expected
        assert len(names) == len(set(names))
        assert sum("budget_crc" in name for name in names) == (expected - 1) // 2
        assert sum("projection_margin_0p00" in name for name in names) == (
            expected - 1
        ) // 2


def test_hard_soft_and_score_metadata_are_explicit():
    grid = torch.zeros(6, 4, 5, dtype=torch.float64)
    grid[:, :, 4] = 1.0
    taus = torch.arange(0.01, 0.5, 0.01)
    hard = AblationHardTargetDAPRO(
        grid, 2.0, taus, 0.56, 4,
        n1=2, target_alpha=.10, global_regularization=0.0,
        score_bin_count=2, projection_budget_margin=0.0,
        ablation_kind="hard_soft", ablation_value=0,
        ablation_label="Hard",
    )
    assert hard.objective_metadata()["ablation_coefficient_kind"] == (
        "hard_realized_target_indicator"
    )

    score_methods = get_dapro_ablation_calibrations(
        grid, 2.0, taus, 0.56, 4,
        ablation_kind="score", dapro_n1_values=(2,),
    )
    allocators = [
        method.budget_allocator for method in score_methods
        if hasattr(method, "budget_allocator")
        and hasattr(method.budget_allocator, "ablation_score_kind")
    ]
    assert {a.ablation_score_kind for a in allocators} == {
        "hazard", "remaining_quantile", "target_value", "random",
        "oracle_remaining_time",
    }
    oracle = next(
        a for a in allocators
        if a.ablation_score_kind == "oracle_remaining_time"
    )
    assert oracle.objective_metadata()["ablation_score_is_causal"] == 0


def test_continuous_representation_and_causal_target_value_scores():
    rng = torch.Generator().manual_seed(9)
    raw = torch.rand((8, 4, 5), generator=rng, dtype=torch.float64)
    grid = raw / raw.sum(dim=2, keepdim=True)
    taus = torch.arange(0.01, 0.5, 0.01)
    quantiles = torch.full((8, len(taus)), 4.0)
    methods = get_dapro_ablation_calibrations(
        grid, 2.0, taus, 0.56, 4,
        ablation_kind="representation", dapro_n1_values=(2,),
    )
    allocators = [
        method.budget_allocator for method in methods
        if hasattr(method, "budget_allocator")
        and hasattr(method.budget_allocator, "ablation_kind")
    ]
    assert {a.score_bin_count for a in allocators} == {1, 2, 4, 8}
    assert sum(a.smooth_score_rank_map for a in allocators) == 2

    target_methods = get_dapro_ablation_calibrations(
        grid, 2.0, taus, 0.56, 4,
        ablation_kind="score", dapro_n1_values=(2,),
    )
    target = next(
        method.budget_allocator for method in target_methods
        if hasattr(method, "budget_allocator")
        and hasattr(method.budget_allocator, "ablation_score_kind")
        and method.budget_allocator.ablation_score_kind == "target_value"
        and method.budget_allocator.budget_control_mode is None
    )
    before = target.policy_scores(quantiles)
    mutated = grid.clone()
    mutated[:, 2:, :] = torch.flip(mutated[:, 2:, :], dims=(2,))
    target.conditional_grid = mutated
    after = target.policy_scores(quantiles)
    torch.testing.assert_close(before[:, :2], after[:, :2])


def test_continuous_rank_lookup_is_monotone_and_not_a_hard_bin_table():
    fit_scores = torch.linspace(0, 1, 40, dtype=torch.float64)[:, None]
    deploy_scores = torch.linspace(0, 1, 101, dtype=torch.float64)[:, None]
    lengths = torch.ones(40, dtype=torch.long)
    masses = torch.linspace(.01, 1, 40, dtype=torch.float64)[:, None]
    _, _, deployed, diagnostics = solve_binned_deployable_policy(
        fit_scores,
        deploy_scores,
        lengths,
        budget_per_sample=.55,
        objective_weights=None,
        n_bins=4,
        objective_masses=masses,
        smooth_rank_lookup=True,
    )
    values = deployed[:, 0].numpy()
    assert np.all(np.diff(values) >= -1e-12)
    assert len(np.unique(np.round(values, 10))) > 4
    assert diagnostics["direct_score_smooth_rank_lookup"] == 1


def test_metric_registry_uses_event_rate_target_and_paired_controllers():
    taus = torch.arange(0.01, 0.5, 0.01)
    noise = get_metric_dapro_ablation_allocators(
        None, 20, taus, .56, 200,
        ablation_kind="score_noise", dapro_n1=50, crc_control_size=25,
        score_noise_lambdas=(0.0, .5, 1.0),
    )
    assert len(noise) == 7
    assert len({allocator.name for allocator in noise}) == 7
    assert sum("budget_crc" in allocator.name for allocator in noise) == 3
    score = get_metric_dapro_ablation_allocators(
        None, 20, taus, .56, 200,
        ablation_kind="score", dapro_n1=50, crc_control_size=25,
    )
    assert len(score) == 11
    dynamic = [
        allocator for allocator in score
        if hasattr(allocator, "ablation_score_kind")
    ]
    assert {allocator.ablation_score_kind for allocator in dynamic} == {
        "hazard", "remaining_quantile", "target_value", "random",
        "oracle_remaining_time",
    }
    assert all(allocator.metric_estimation_horizon == 200 for allocator in dynamic)
    assert all(allocator.score_bin_count == 4 for allocator in dynamic)
    assert all(
        allocator.objective_metadata()["ablation_target_definition"]
        == "1{T<=200}"
        for allocator in dynamic
    )


def test_metric_target_value_and_oracle_scores_use_non_strict_horizon():
    pmf = torch.tensor([.1, .2, .3, .1, .3], dtype=torch.float64)
    grid = pmf.repeat(4, 4, 1)
    taus = torch.arange(0.01, 0.5, 0.01)
    quantiles = torch.full((4, len(taus)), 4.0)
    target = AblationSoftTargetDAPRO(
        grid, 2.0, taus, .56, 4,
        n1=2, target_alpha=.10, metric_estimation_horizon=4,
        score_bin_count=4, projection_budget_margin=0.0,
        ablation_kind="score", ablation_value=2,
        ablation_label="Causal event-rate target value",
        score_kind="target_value",
    )
    scores = target.policy_scores(quantiles)
    assert np.isclose(scores[0, 0].item(), .7)
    assert np.isclose(scores[0, 2].item(), .4 / .7)

    oracle = AblationSoftTargetDAPRO(
        grid, 2.0, taus, .56, 4,
        n1=2, target_alpha=.10, metric_estimation_horizon=4,
        score_bin_count=4, projection_budget_margin=0.0,
        ablation_kind="score", ablation_value=4,
        ablation_label="Oracle remaining time",
        score_kind="oracle_remaining_time",
    )
    event_times = torch.tensor([4, 5, 2, 5], dtype=torch.float64)
    oracle_scores = oracle.policy_scores_for_allocation(quantiles, event_times)
    assert oracle_scores[0, 0] > 0  # T=4 is in the inclusive metric target.
    assert oracle_scores[1].eq(0).all()  # T=5 is outside horizon M=4.


def test_metric_summarizer_reports_error_and_across_split_variance(tmp_path):
    rows = []
    for seed, event_rate in enumerate((48.0, 52.0, 50.0)):
        for method, name in (
            ("Static", "calibration_optimized_allocation"),
            (
                "DAPRO w/o CRC",
                "calibration_dapro_soft_prefix_ablation_score_noise_raw_allocation",
            ),
            (
                "DAPRO",
                "calibration_dapro_soft_prefix_budget_crc_ablation_score_noise_allocation",
            ),
        ):
            for value in ((np.nan,) if method == "Static" else (0.0, 1.0)):
                rows.append({
                    "seed": seed,
                    "allocator_name": name,
                    "calibration_name": name,
                    "estimated_cjr": event_rate + (1 if method == "DAPRO" else 0),
                    "oracle_cjr": 50.0,
                    "full_benchmark_cjr": 50.0,
                    "abs_diff_cjr": abs(event_rate - 50.0),
                    "reported_assigned_budget_per_sample": 20.0,
                    "actual_event_stopped_budget_per_sample": 19.0,
                    "num_events_observed": 300 + seed,
                    "mean_metric_target_a_weighted_inverse_probability": 2.0,
                    "conditional_variance_unsafe_event_rate_estimator": .001,
                    "ablation_kind": np.nan if method == "Static" else "score_noise",
                    "ablation_value": np.nan if method == "Static" else value,
                    "ablation_label": np.nan if method == "Static" else f"lambda={value:g}",
                    "ablation_n1": np.nan if method == "Static" else 50,
                })
    parent = tmp_path / "toxicity_20_m_test_score_noise"
    parent.mkdir()
    pd.DataFrame(rows).to_csv(parent / "all_df.csv", index=False)
    data, _ = load_metric_ablation_data(
        tmp_path, experiment_prefix="test", kind="score_noise"
    )
    assert set(data["method"]) == {"Static", "DAPRO", "DAPRO w/o CRC"}
    assert data.groupby(["factor_value", "method"])["seed"].nunique().eq(3).all()
    # Static event rates are 48, 52, 50 in percent, so sample variance is 4.
    assert np.allclose(
        data.loc[
            data["method"].eq("Static"),
            "event_rate_across_split_variance_pp2",
        ],
        4.0,
    )
    output = tmp_path / "metric.jpg"
    statistics = generate_metric_ablation_figure(
        data, kind="score_noise", output_path=output, quality="low"
    )
    assert output.exists()
    assert {
        "event_rate_pct", "event_rate_abs_error_pp",
        "event_rate_across_split_variance_pp2", "budget_used_per_sample",
        "observed_events", "mean_target_a_weight",
    }.issubset(set(statistics["metric"]))


def test_hard_soft_and_attacker_shift_dispatch_to_boxplots(
        tmp_path, monkeypatch):
    rows = []
    for value, label in ((0.0, "Hard"), (1.0, "Soft")):
        for method in ("Static", "DAPRO", "DAPRO w/o CRC"):
            for seed in range(3):
                rows.append({
                    "factor_value": value,
                    "factor_label": label,
                    "method": method,
                    "coverage_pct": 90 + seed / 10,
                    "budget_used_per_sample": 19 + seed / 10,
                    "coverage_diff_pct": seed / 10,
                    "mean_target_a_weight": 1 + seed / 10,
                })
    data = pd.DataFrame(rows)
    calls = []

    def record_boxplot(*args, **kwargs):
        calls.append(kwargs)
        return kwargs["ax"]

    monkeypatch.setattr(ablation_summarizer.sns, "boxplot", record_boxplot)
    output = tmp_path / "hard_soft.jpg"
    generate_ablation_figure(
        data, kind="hard_soft", output_path=output, quality="low"
    )
    assert len(calls) == 4
    assert all(call["hue"] == "method" for call in calls)
    assert output.exists()


def test_discovery_accepts_long_downloaded_result_beside_merged_directory(
        tmp_path):
    results = tmp_path / "results"
    merged = results / "merged_calibration_dfs"
    merged.mkdir(parents=True)
    direct = results / (
        "dataset_red_team_very_long_configuration_"
        "dapro_lpb_ablation_v1_attacker_shift_red"
    )
    direct.mkdir()
    (direct / "all_df.csv").write_text("seed\n0\n", encoding="utf-8")
    paths = _discover(
        merged, "dapro_lpb_ablation_v1_attacker_shift"
    )
    assert paths == [direct / "all_df.csv"]


def test_attacker_shift_loader_uses_only_budget_ten(tmp_path):
    raw_name = (
        "calibration_dapro_soft_prefix_bins_2_lpb_alpha_0p10_"
        "projection_margin_0p00_n1_50_ablation_attacker_shift_0_allocation"
    )
    crc_name = (
        "calibration_dapro_soft_prefix_bins_2_lpb_alpha_0p10_"
        "budget_crc_control_25_n1_50_ablation_attacker_shift_0_allocation"
    )
    for budget in (10, 20):
        rows = []
        for name in (
            "calibration_optimized_allocation", raw_name, crc_name
        ):
            dynamic = "dapro_" in name
            rows.append({
                "seed": 0,
                "calibration_name": name,
                "target_coverage": 0.90,
                "coverage": 0.90,
                "configured_budget_per_sample": budget,
                "reported_assigned_budget_per_sample": float(budget),
                "actual_event_stopped_budget_per_sample": float(budget),
                "mean_calibrated_a_weighted_inverse_probability": 0.1,
                "ablation_kind": "attacker_shift" if dynamic else np.nan,
                "ablation_value": 0.0 if dynamic else np.nan,
                "ablation_n1": 50 if dynamic else np.nan,
                "attacker_shift_source_dataset_name": "dataset_red_team",
                "attacker_shift_source_dataset_setup": "gemma_source",
                "attacker_shift_test_dataset_setup": "qwen_test",
            })
        parent = tmp_path / (
            f"red_team_{budget}_calibration_"
            "test_attacker_shift_red"
        )
        parent.mkdir()
        pd.DataFrame(rows).to_csv(parent / "all_df.csv", index=False)

    data, inventory = load_ablation_data(
        tmp_path,
        experiment_prefix="test",
        kind="attacker_shift",
    )
    assert set(data["configured_budget_per_sample"]) == {10}
    assert set(data["budget_used_per_sample"]) == {10}
    assert len(inventory) == 1
