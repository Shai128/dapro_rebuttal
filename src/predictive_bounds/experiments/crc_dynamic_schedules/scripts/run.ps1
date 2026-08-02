# Run the matched five-dataset CRC and dynamic-schedule audit.
param(
    [string]$PythonExe = $(if ($env:PYTHON) { $env:PYTHON } else { "python" }),
    [string]$Device = $(if ($env:DEVICE) { $env:DEVICE } else { "cuda:0" }),
    [string]$SeedStart = $(if ($env:SEED_START) { $env:SEED_START } else { "0" }),
    [string]$SeedEnd = $(if ($env:SEED_END) { $env:SEED_END } else { "10" }),
    [string]$Suffix = $(if ($env:EXPERIMENT_SUFFIX) { $env:EXPERIMENT_SUFFIX } else { "crc_dynamic_schedules_v1" })
)

$ErrorActionPreference = "Stop"

$Methods = @(
    "calibration_dapro_variance_aligned_bins_2_alpha_0p10_global_0p001_projection_margin_0p00_n1_200_allocation",
    "calibration_dapro_variance_aligned_bins_2_alpha_0p10_global_0p001_projection_margin_1p00_n1_200_allocation",
    "calibration_dapro_variance_aligned_bins_2_alpha_0p10_global_0p001_budget_crc_control_100_n1_200_allocation",
    "calibration_dapro_variance_aligned_bins_2_alpha_0p10_global_0p001_budget_crc_control_100_n1_300_allocation",
    "calibration_random_adaptive_optimized_allocation",
    "calibration_random_adaptive_optimized_crc_allocation",
    "calibration_random_schedule_complement_power_alpha_0p5_crc_allocation",
    "calibration_random_schedule_complement_power_alpha_1_crc_allocation",
    "calibration_random_schedule_complement_power_alpha_2_crc_allocation",
    "calibration_random_schedule_power_reach_alpha_0p5_crc_allocation",
    "calibration_random_schedule_power_reach_alpha_1_crc_allocation",
    "calibration_random_schedule_power_reach_alpha_2_crc_allocation"
) -join ","

$Datasets = @(
    @{Name="dataset_toxicity"; Setup="attack_toxic_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_detoxify"; Budget="20"; Gamma="10"},
    @{Name="dataset_autoif"; Setup="attack_autoif_helper_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_autoif"; Budget="20"; Gamma="10"},
    @{Name="dataset_hallucination3"; Setup="attack_hallucination_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_llm-judge_qwen25_14b_instruct"; Budget="10"; Gamma="20"},
    @{Name="dataset_red_team"; Setup="attack_default_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_llama_guard"; Budget="10"; Gamma="20"},
    @{Name="dataset_red_team"; Setup="attack_default_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_llm-judge_qwen25_14b_instruct"; Budget="20"; Gamma="10"}
)

foreach ($Dataset in $Datasets) {
    $Common = @(
        "--bound-type", "lpb", "--data-type", "real",
        "--dataset-name", $Dataset.Name, "--dataset-setup", $Dataset.Setup,
        "--cal-size", "3000", "--tau-prior", "0.56",
        "--budget-per-sample", $Dataset.Budget, "--gamma", $Dataset.Gamma,
        "--seed-start", $SeedStart, "--seed-end", $SeedEnd,
        "--device", $Device, "--experiment-suffix", $Suffix,
        "--dapro-n1-values", "200", "300",
        "--definitive-dapro-margins", "0", "1",
        "--calibration-names", $Methods
    )
    & $PythonExe -m src.predictive_bounds.construct_calibrated_bound @Common
    & $PythonExe -m src.predictive_bounds.merge_bounds_results @Common
}

& $PythonExe -m src.predictive_bounds.experiments.crc_dynamic_schedules.analyze --suffix $Suffix
