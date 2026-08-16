"""Regression tests for the manuscript-wide construct/merge/plot matrix."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.predictive_bounds.experiments.full_bounds.config import (
    CRC_DAPRO_ORACLE,
    GLOBAL_DAPRO_ORACLE,
    LPB_ORACLE,
    LPB_DAPRO,
    GENERALIZED_LPB_CRC_DAPRO,
    GENERALIZED_LPB_DAPRO,
    GENERALIZED_UPB_CRC_DAPRO,
    GENERALIZED_UPB_DAPRO,
    LOCALLY_ADAPTIVE,
    POWER_REACH,
    UPB_DAPRO,
    UPB_ORACLE,
    SPLIT_DAPRO_ORACLE,
    all_experiment_configs,
    calibration_names,
    method_display_name,
)
from src.predictive_bounds.experiments.full_bounds.summarize import (
    LOW_QUALITY_MAX_BYTES,
    _compact_result_configurations,
    _method_n1,
    _method_palette,
    _prefer_latest_compact_lpb_results,
    _save_jpeg,
    load_upb_matrix,
)
from src.evaluation.result_matrix import parse_upb_result


def test_full_bounds_matrix_covers_every_dataset_model_and_bound():
    configs = all_experiment_configs()
    # 20 LPB configurations plus three UPB coverage targets for each of the
    # 20 dataset/model pairs.
    assert len(configs) == 80
    assert sum(config.bound_type == "lpb" for config in configs) == 20
    assert sum(config.bound_type == "upb" for config in configs) == 60
    assert {
        config.target_coverage
        for config in configs
        if config.bound_type == "upb"
    } == {0.70, 0.80, 0.90}
    assert {config.target_model.key for config in configs} == {
        "qwen", "llama", "phi", "gemma",
    }
    assert {config.figure_dataset_name for config in configs} == {
        "toxicity",
        "toxicity_upb",
        "red_team_qwen",
        "red_team_qwen_upb",
        "red_team_llama_guard",
        "red_team_llama_guard_upb",
        "hallucination3",
        "hallucination3_upb",
        "autoif",
        "autoif_upb",
    }


def test_full_bounds_method_profiles_are_exact_and_bound_specific():
    lpb = calibration_names("lpb")
    upb = calibration_names("upb")
    assert len(lpb) == 7
    assert len(upb) == 7
    assert LOCALLY_ADAPTIVE not in lpb and LOCALLY_ADAPTIVE not in upb
    assert LPB_DAPRO not in lpb and LPB_DAPRO not in upb
    assert GENERALIZED_LPB_DAPRO in lpb and GENERALIZED_LPB_DAPRO not in upb
    assert (
        GENERALIZED_LPB_CRC_DAPRO in lpb
        and GENERALIZED_LPB_CRC_DAPRO not in upb
    )
    assert UPB_DAPRO in upb and UPB_DAPRO not in lpb
    assert GENERALIZED_UPB_DAPRO in upb
    assert GENERALIZED_UPB_CRC_DAPRO in upb
    assert GENERALIZED_UPB_DAPRO not in lpb
    assert POWER_REACH in lpb and POWER_REACH in upb
    assert LPB_ORACLE in lpb and LPB_ORACLE not in upb
    assert UPB_ORACLE in upb and UPB_ORACLE not in lpb
    for oracle in [
        SPLIT_DAPRO_ORACLE,
        CRC_DAPRO_ORACLE,
        GLOBAL_DAPRO_ORACLE,
    ]:
        assert oracle not in lpb and oracle not in upb


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


def test_unified_lpb_methods_have_canonical_labels_and_complete_palette():
    names = [
        "calibration_dapro_soft_prefix_bins_2_lpb_alpha_0p10_global_0p001_projection_margin_0p00_n1_50_allocation",
        "calibration_dapro_soft_prefix_bins_2_lpb_alpha_0p10_global_0p001_budget_crc_control_25_n1_50_allocation",
        "calibration_dapro_information_gain_sequential_aht_lpb_c0p90_bins_2_raw_margin_0p00_n1_50_allocation",
        "calibration_dapro_residual_sequential_aht_lpb_c0p90_bins_2_raw_margin_0p00_budget_crc_control_25_n1_50_allocation",
        "calibration_endpoint_block_terminal_residual_aht_lpb_c0p90_crc_control_25_allocation",
    ]
    labels = [method_display_name(name) for name in names]

    assert labels == [
        "Soft-prefix DAPRO",
        "Soft-prefix DAPRO + CRC",
        "Information-gain + sequential AHT",
        "Residual + sequential AHT + CRC",
        "Endpoint/block + terminal residual AHT + CRC",
    ]
    assert set(_method_palette(labels)) == set(labels)
    assert _method_palette(["Future method"])["Future method"] == "#6b7280"


def test_upb_matrix_is_split_by_policy_coverage_and_n1(tmp_path: Path):
    result_dir = (
        tmp_path
        / "dataset_toxicity_attack_toxic_attack_qwen_lm_target_"
        "qwen25_14b_instruct_judge_detoxify_20_calibration_upb_test"
    )
    result_dir.mkdir(parents=True)
    path = result_dir / "all_df.csv"
    rows = []
    for coverage in (0.70, 0.80, 0.90):
        rows.extend([
            {
                "seed": 0,
                "target_coverage": coverage,
                "policy_target_coverage": None,
                "calibration_name": "calibration_optimized_allocation",
                "coverage": coverage,
            },
            {
                "seed": 0,
                "target_coverage": coverage,
                "policy_target_coverage": coverage,
                "calibration_name": (
                    "calibration_dapro_soft_prefix_bins_2_"
                    f"upb_endpoint_dynamic_aht_coverage_0p{int(100 * coverage)}_"
                    "global_0p001_projection_margin_0p00_n1_50_allocation"
                ),
                "coverage": coverage,
            },
            {
                "seed": 0,
                "target_coverage": coverage,
                "policy_target_coverage": coverage,
                "calibration_name": (
                    "calibration_dapro_soft_prefix_bins_2_"
                    f"upb_endpoint_dynamic_aht_coverage_0p{int(100 * coverage)}_"
                    "global_0p001_budget_crc_control_25_n1_50_allocation"
                ),
                "coverage": coverage,
            },
        ])
    pd.DataFrame(rows).to_csv(path, index=False)

    assert parse_upb_result(path) is not None
    frame, inventory = load_upb_matrix(
        tmp_path, experiment_suffix="test"
    )

    assert set(frame["target_coverage_pct"]) == {70.0, 80.0, 90.0}
    assert set(frame["dapro_n1"]) == {50}
    assert set(frame["crc_control_size"]) == {25}
    assert set(inventory["target_coverage"]) == {0.70, 0.80, 0.90}
    learned = frame[frame["policy_target_coverage"].notna()]
    assert (
        100 * learned["policy_target_coverage"]
        == learned["target_coverage_pct"]
    ).all()
