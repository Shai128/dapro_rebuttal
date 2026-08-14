from pathlib import Path

from src.evaluation.result_matrix import (
    method_display_name,
    numeric_label,
    parse_lpb_result,
    parse_metric_result,
)


SETUP = (
    "dataset_red_team_attack_default_attack_qwen25_14b_instruct_"
    "lm_target_mini_phi_4_instruct_judge_llama_guard"
)


def test_parse_lpb_budget_n1_matrix_path():
    path = Path("results/merged_calibration_dfs") / (
        f"{SETUP}_10.0_3000_0.56_20.0__"
        "lpb_all_methods_v1_n1_200_crc_100_budget_10"
    ) / "all_df.csv"
    result = parse_lpb_result(path)

    assert result is not None
    assert result.dataset_key == "red_team_llama_guard"
    assert result.target_model == "mini_phi_4_instruct"
    assert result.target_model_display == "Phi-4 Mini"
    assert result.budget_per_sample == 10
    assert result.dapro_n1 == 200
    assert result.crc_control_size == 100


def test_parse_metric_budget_n1_matrix_path():
    path = Path("results/merged_metric_calibration_dfs") / (
        f"{SETUP}_20_metric_estimation_n1_50_crc_25__metric_estimation_v1"
    ) / "all_df.csv"
    result = parse_metric_result(path)

    assert result is not None
    assert result.dataset_display == "Red Team (Llama-Guard)"
    assert result.budget_per_sample == 20
    assert result.dapro_n1 == 50
    assert result.crc_control_size == 25


def test_parse_compact_metric_result_path():
    path = Path("results/merged_metric_calibration_dfs") / (
        f"{SETUP}_20_metric_estimation_metric_v1"
    ) / "all_df.csv"
    result = parse_metric_result(path)

    assert result is not None
    assert result.budget_per_sample == 20
    assert result.dapro_n1 is None
    assert result.crc_control_size is None


def test_parse_compact_lpb_result_path():
    path = Path("results/merged_calibration_dfs") / (
        f"{SETUP}_10_calibration_lpb_v1"
    ) / "all_df.csv"
    result = parse_lpb_result(path)

    assert result is not None
    assert result.budget_per_sample == 10
    assert result.dapro_n1 is None


def test_unsuffixed_lpb_result_is_not_part_of_matrix():
    path = Path("results/merged_calibration_dfs") / (
        f"{SETUP}_20.0_3000_0.56_10.0"
    ) / "all_df.csv"
    assert parse_lpb_result(path) is None


def test_n1_dependent_method_names_have_stable_labels():
    assert method_display_name(
        "calibration_metric_optimal_pooled_time_model_budget_allocation"
    ) == "Pooled-Neyman schedule"
    assert method_display_name(
        "calibration_metric_optimal_pooled_time_crc_control_50_"
        "row_cap_2p00x_budget_allocation"
    ) == "Pooled-Neyman schedule + CRC"
    assert method_display_name(
        "calibration_metric_prefix_neyman_crc_control_50_"
        "row_cap_2p00x_budget_allocation"
    ) == "Prefix-Neyman + CRC"
    assert method_display_name(
        "calibration_projected_optimization_direct_bins_2_prob_"
        "a_target_raw_alpha_0p10_budget_crc_control_25_"
        "row_cap_2p00x_budget_causal_shared_pav_v1_n1_50_allocation"
    ) == "Target-A DAPRO + CRC"
    assert method_display_name(
        "calibration_dapro_variance_aligned_bins_2_alpha_0p10_"
        "global_0p001_projection_margin_1p00_n1_100_allocation"
    ) == "DAPRO (projection)"
    assert method_display_name(
        "calibration_dapro_soft_prefix_bins_2_metric_horizon_200_"
        "global_0p001_projection_margin_1p00_n1_50_allocation"
    ) == "Generalized DAPRO (soft metric)"
    assert method_display_name(
        "calibration_dapro_soft_prefix_bins_2_metric_horizon_200_"
        "global_0p001_budget_crc_control_25_row_cap_2p00x_budget_"
        "causal_shared_pav_v1_n1_50_allocation"
    ) == "Soft-prefix DAPRO + CRC"
    assert method_display_name(
        "calibration_dapro_soft_prefix_bins_2_lpb_alpha_0p10_"
        "global_0p001_projection_margin_1p00_n1_50_allocation"
    ) == "Generalized DAPRO (soft LPB)"
    assert method_display_name(
        "calibration_dapro_soft_prefix_bins_2_lpb_alpha_0p10_"
        "global_0p001_budget_crc_control_25_row_cap_2p00x_budget_"
        "causal_shared_pav_v1_n1_50_allocation"
    ) == "Generalized DAPRO (soft LPB) + CRC"
    assert method_display_name(
        "calibration_oracle_split_full_budget_allocation"
    ) == "Full budget (calibration)"
    assert method_display_name(
        "full_calibration_test_oracle_full_budget_allocation"
    ) == "Full budget (calibration+test)"
    assert numeric_label(10.0) == "10"
    assert numeric_label(2.5) == "2p5"
