#!/usr/bin/env bash
# Run the matched five-dataset CRC and dynamic-schedule audit.
set -euo pipefail

PYTHON_EXE="${PYTHON:-python}"
DEVICE_VALUE="${DEVICE:-cuda:0}"
SEED_START_VALUE="${SEED_START:-0}"
SEED_END_VALUE="${SEED_END:-10}"
SUFFIX_VALUE="${EXPERIMENT_SUFFIX:-crc_dynamic_schedules_v1}"

METHODS="calibration_dapro_variance_aligned_bins_2_alpha_0p10_global_0p001_projection_margin_0p00_n1_200_allocation,calibration_dapro_variance_aligned_bins_2_alpha_0p10_global_0p001_projection_margin_1p00_n1_200_allocation,calibration_dapro_variance_aligned_bins_2_alpha_0p10_global_0p001_budget_crc_control_100_n1_200_allocation,calibration_dapro_variance_aligned_bins_2_alpha_0p10_global_0p001_budget_crc_control_100_n1_300_allocation,calibration_random_adaptive_optimized_allocation,calibration_random_adaptive_optimized_crc_allocation,calibration_random_schedule_complement_power_alpha_0p5_crc_allocation,calibration_random_schedule_complement_power_alpha_1_crc_allocation,calibration_random_schedule_complement_power_alpha_2_crc_allocation,calibration_random_schedule_power_reach_alpha_0p5_crc_allocation,calibration_random_schedule_power_reach_alpha_1_crc_allocation,calibration_random_schedule_power_reach_alpha_2_crc_allocation"

DATASETS=(
  "dataset_toxicity|attack_toxic_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_detoxify|20|10"
  "dataset_autoif|attack_autoif_helper_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_autoif|20|10"
  "dataset_hallucination3|attack_hallucination_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_llm-judge_qwen25_14b_instruct|10|20"
  "dataset_red_team|attack_default_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_llama_guard|10|20"
  "dataset_red_team|attack_default_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_llm-judge_qwen25_14b_instruct|20|10"
)

for specification in "${DATASETS[@]}"; do
  IFS='|' read -r dataset setup budget gamma <<< "$specification"
  common=(
    --bound-type lpb --data-type real
    --dataset-name "$dataset" --dataset-setup "$setup"
    --cal-size 3000 --tau-prior 0.56
    --budget-per-sample "$budget" --gamma "$gamma"
    --seed-start "$SEED_START_VALUE" --seed-end "$SEED_END_VALUE"
    --device "$DEVICE_VALUE" --experiment-suffix "$SUFFIX_VALUE"
    --dapro-n1-values 200 300 --definitive-dapro-margins 0 1
    --calibration-names "$METHODS"
  )
  "$PYTHON_EXE" -m src.predictive_bounds.construct_calibrated_bound "${common[@]}"
  "$PYTHON_EXE" -m src.predictive_bounds.merge_bounds_results "${common[@]}"
done

"$PYTHON_EXE" -m src.predictive_bounds.experiments.crc_dynamic_schedules.analyze \
  --suffix "$SUFFIX_VALUE"
