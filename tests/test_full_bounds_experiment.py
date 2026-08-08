"""Regression tests for the manuscript-wide construct/merge/plot matrix."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.predictive_bounds.experiments.full_bounds.config import (
    CRC_DAPRO_ORACLE,
    GLOBAL_DAPRO_ORACLE,
    LPB_ORACLE,
    LPB_DAPRO,
    LOCALLY_ADAPTIVE,
    POWER_REACH,
    UPB_DAPRO,
    UPB_ORACLE,
    SPLIT_DAPRO_ORACLE,
    all_experiment_configs,
    calibration_names,
)
from src.predictive_bounds.experiments.full_bounds.summarize import (
    LOW_QUALITY_MAX_BYTES,
    _compact_result_configurations,
    _method_n1,
    _prefer_latest_compact_lpb_results,
    _save_jpeg,
)


def test_full_bounds_matrix_covers_every_dataset_model_and_bound():
    configs = all_experiment_configs()
    assert len(configs) == 24
    assert sum(config.bound_type == "lpb" for config in configs) == 20
    assert sum(config.bound_type == "upb" for config in configs) == 4
    assert {config.target_model.key for config in configs} == {
        "qwen", "llama", "phi", "gemma",
    }
    assert {config.figure_dataset_name for config in configs} == {
        "toxicity",
        "red_team_qwen",
        "red_team_llama_guard",
        "hallucination3",
        "autoif",
        "autoif_upb",
    }


def test_full_bounds_method_profiles_are_exact_and_bound_specific():
    lpb = calibration_names("lpb")
    upb = calibration_names("upb")
    assert len(lpb) == 9
    assert len(upb) == 6
    assert LOCALLY_ADAPTIVE not in lpb and LOCALLY_ADAPTIVE not in upb
    assert LPB_DAPRO in lpb and LPB_DAPRO not in upb
    assert UPB_DAPRO in upb and UPB_DAPRO not in lpb
    assert POWER_REACH in lpb and POWER_REACH in upb
    assert LPB_ORACLE in lpb and LPB_ORACLE not in upb
    assert UPB_ORACLE in upb and UPB_ORACLE not in lpb
    for oracle in [
        SPLIT_DAPRO_ORACLE,
        CRC_DAPRO_ORACLE,
        GLOBAL_DAPRO_ORACLE,
    ]:
        assert oracle in lpb and oracle not in upb


def test_low_quality_jpeg_respects_the_hard_size_limit(tmp_path: Path):
    figure, axis = plt.subplots(figsize=(12.5, 6.5))
    for offset in range(40):
        axis.plot(
            [index / 100 for index in range(100)],
            [((index * (offset + 3)) % 97) / 97 for index in range(100)],
        )
    path = tmp_path / "plot.jpg"
    _save_jpeg(figure, path, "low")
    plt.close(figure)

    assert path.exists()
    assert path.stat().st_size <= LOW_QUALITY_MAX_BYTES


def test_compact_result_configurations_are_inferred_from_method_names():
    names = pd.Series([
        "uncalibrated_lpb",
        "calibration_projected_optimization_direct_bins_2_prob_n1_200_allocation",
        "calibration_projected_optimization_direct_bins_2_prob_allocation",
        "calibration_budget_crc_control_25_n1_50_allocation",
    ])

    assert _method_n1(names.iloc[0]) is None
    assert _method_n1(names.iloc[2]) == 100
    assert _compact_result_configurations(names) == [
        (200, 100),
        (100, 50),
        (50, 25),
    ]


def test_latest_compact_lpb_result_supersedes_stale_version():
    base = "dataset_x_attack_a_lm_target_m_judge_j_5_calibration_lpb"
    v1 = Path("results") / f"{base}_v1" / "all_df.csv"
    v2 = Path("results") / f"{base}_v2" / "all_df.csv"
    unrelated = Path("results") / "legacy-layout" / "all_df.csv"

    assert _prefer_latest_compact_lpb_results([v1, unrelated, v2]) == [
        v2,
        unrelated,
    ]
