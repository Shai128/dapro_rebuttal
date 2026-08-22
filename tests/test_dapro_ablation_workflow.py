from pathlib import Path
import inspect

import numpy as np
import pandas as pd
import torch

from src.predictive_bounds.budget_allocators.dapro_ablation import (
    AblationHardTargetDAPRO,
    AblationSoftTargetDAPRO,
)
from src.predictive_bounds.experiments.dapro_ablation.summarize_paper import (
    generate_ablation_figure,
    load_ablation_data,
)
from src.predictive_bounds.budget_allocators.optimization_solver_utils import (
    solve_binned_deployable_policy,
)
from src.predictive_bounds.utils.get_calibration_methods_utils import (
    get_dapro_ablation_calibrations,
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
