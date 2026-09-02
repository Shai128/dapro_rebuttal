from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.predictive_bounds.experiments.dapro_ablation.summarize_paper import (
    generate_upb_estimator_ablation_figure,
    load_upb_estimator_ablation_data,
)
from src.predictive_bounds.utils.get_calibration_methods_utils import (
    get_upb_estimator_ablation_calibrations,
)
from src.train_model.models.utils import SurvivalModelPrediction
from src.utils.utils import set_seeds


def _toy_problem(n: int = 30, width: int = 4):
    grid = torch.zeros(n, width, width + 1, dtype=torch.float64)
    for row in range(n):
        for step in range(width):
            hazard = 0.10 + 0.05 * ((row + step) % 3)
            grid[row, step, step] = hazard
            grid[row, step, -1] = 1.0 - hazard
    candidates = torch.tensor([[2.0, 3.0, 201.0]] * n)
    times = torch.tensor(
        [1 + row % width if row % 5 else 201 for row in range(n)],
        dtype=torch.float64,
    )
    taus = torch.tensor([0.5, 0.7, 0.9])
    return grid, candidates, times, taus


def test_upb_estimator_registry_has_the_requested_eight_cells():
    grid, _, _, taus = _toy_problem()
    calibrations = get_upb_estimator_ablation_calibrations(
        grid,
        2.5,
        taus,
        0.9,
        4,
        dapro_n1_values=(8,),
        target_coverages=(0.7,),
    )
    cells = {(item.method_label, item.estimator_label) for item in calibrations}
    assert len(calibrations) == len({item.name for item in calibrations}) == 8
    assert cells == {
        ("Static", "Ordinary HT"),
        ("Static", "Terminal AHT"),
        ("DAPRO", "Ordinary HT"),
        ("DAPRO", "Terminal AHT"),
        ("DAPRO", "Sequential AHT"),
        ("DAPRO w/o CRC", "Ordinary HT"),
        ("DAPRO w/o CRC", "Terminal AHT"),
        ("DAPRO w/o CRC", "Sequential AHT"),
    }


def test_estimator_variants_reuse_the_same_realized_allocation_path():
    grid, candidates, times, taus = _toy_problem()
    prediction = SurvivalModelPrediction(candidates, grid)
    calibrations = get_upb_estimator_ablation_calibrations(
        grid,
        2.5,
        taus,
        0.9,
        4,
        dapro_n1_values=(8,),
        target_coverages=(0.7,),
    )
    uniforms = np.linspace(
        0.01, 0.99, len(grid) * grid.shape[1], endpoint=False
    ).reshape(len(grid), grid.shape[1])
    fingerprints = {}
    for calibration in calibrations:
        set_seeds(11)
        calibration.budget_allocator.set_acquisition_randomness(
            seed=13, uniforms=uniforms
        )
        calibration.calibrate(torch.zeros(len(grid), 1), times, prediction)
        metrics = calibration.compute_metrics(
            prediction, torch.tensor([0.7])
        )
        fingerprints.setdefault(calibration.method_label, set()).add(
            metrics["upb_estimator_ablation_acquisition_sha256"]
        )
        assert metrics["upb_calibration_estimator"] in {
            "ordinary_horvitz_thompson",
            "terminal_residual_augmented_ht",
            "sequential_augmented_horvitz_thompson",
        }
    assert all(len(values) == 1 for values in fingerprints.values())


def _synthetic_merged_rows(dataset: str) -> pd.DataFrame:
    cells = [
        ("Static", "Ordinary HT"),
        ("Static", "Terminal AHT"),
        ("DAPRO", "Ordinary HT"),
        ("DAPRO", "Terminal AHT"),
        ("DAPRO", "Sequential AHT"),
        ("DAPRO w/o CRC", "Ordinary HT"),
        ("DAPRO w/o CRC", "Terminal AHT"),
        ("DAPRO w/o CRC", "Sequential AHT"),
    ]
    rows = []
    for seed in range(4):
        for cell_index, (method, estimator) in enumerate(cells):
            controller = (
                "static" if method == "Static"
                else "dapro_crc" if method == "DAPRO"
                else "dapro_raw"
            )
            rows.append({
                "seed": seed,
                "calibration_name": f"{controller}_{estimator}",
                "target_coverage": 0.80,
                "coverage": 0.79 + 0.002 * seed + 0.0002 * cell_index,
                "size": 25.0 + seed + 0.2 * cell_index,
                "estimated_conditional_variance_upb_coverage_estimator": (
                    2.0 + 0.1 * seed + 0.2 * cell_index
                ),
                "upb_estimator_ablation": 1,
                "upb_estimator_ablation_method": method,
                "upb_estimator_ablation_estimator": estimator,
                "upb_estimator_ablation_estimator_kind": estimator,
                "upb_estimator_ablation_uses_crc": int(method == "DAPRO"),
                "upb_estimator_ablation_allocator_name": controller,
                "upb_estimator_ablation_paired_reference": (
                    "Static + Terminal AHT"
                ),
                "upb_estimator_ablation_acquisition_sha256": (
                    f"{dataset}-{seed}-{method}"
                ),
            })
    return pd.DataFrame(rows)


def test_upb_estimator_loader_and_three_by_two_figure(tmp_path: Path):
    for dataset in ("toxicity", "autoif"):
        result_dir = (
            tmp_path
            / f"dataset_{dataset}_qwen_20_calibration_upb_est_test"
        )
        result_dir.mkdir()
        _synthetic_merged_rows(dataset).to_csv(
            result_dir / "all_df.csv", index=False
        )
    data = load_upb_estimator_ablation_data(
        tmp_path,
        experiment_suffix="upb_est_test",
        target_coverage=0.80,
    )
    assert set(data["dataset_key"]) == {"toxicity", "autoif"}
    assert data.groupby(
        ["dataset_key", "method", "estimator"], observed=True
    )["seed"].nunique().eq(4).all()
    reference = data[
        data["method"].eq("Static")
        & data["estimator"].eq("Terminal AHT")
    ]
    assert np.allclose(reference["paired_estimator_variance_ratio"], 1.0)
    example = data[
        data["dataset_key"].eq("toxicity")
        & data["method"].eq("DAPRO")
        & data["estimator"].eq("Sequential AHT")
    ]
    assert example["coverage_variance_contribution_pp2"].mean() == pytest.approx(
        example["coverage_pct"].var()
    )

    for dataset in ("toxicity", "autoif"):
        output = tmp_path / f"{dataset}.jpg"
        generate_upb_estimator_ablation_figure(
            data,
            dataset_key=dataset,
            output_path=output,
            quality="low",
            target_coverage=0.80,
        )
        assert output.exists()
        assert output.stat().st_size > 10_000
