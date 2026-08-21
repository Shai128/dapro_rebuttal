from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.predictive_bounds.budget_allocators.dapro_ablation import (
    AblationSoftTargetDAPRO,
)
from src.predictive_bounds.experiments.dapro_ablation.summarize_paper import (
    generate_ablation_figure,
    load_ablation_data,
)
from src.predictive_bounds.utils.get_calibration_methods_utils import (
    get_dapro_ablation_calibrations,
)


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
    generate_ablation_figure(
        data, kind="score_noise", output_path=output, quality="low"
    )
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
