#!/usr/bin/env bash
# Reproduce the historical projection-DAPRO versus constant-policy study.
set -euo pipefail

PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"
SEED_START="${SEED_START:-0}"
SEED_END="${SEED_END:-50}"
EXPERIMENT_SUFFIX="${EXPERIMENT_SUFFIX:-definitive_dapro_v1}"

FINAL_NAME="calibration_dapro_variance_aligned_bins_2_alpha_0p10_global_0p001_projection_margin_1p00_n1_200_allocation"
RANDOM_NAME="calibration_random_adaptive_optimized_allocation"
METHODS="${FINAL_NAME},${RANDOM_NAME}"

run_dataset() {
  local dataset_name="$1"
  local dataset_setup="$2"
  local budget="$3"
  local gamma="$4"

  "$PYTHON" -m src.predictive_bounds.construct_calibrated_bound \
    --bound-type lpb --data-type real \
    --dataset-name "$dataset_name" --dataset-setup "$dataset_setup" \
    --cal-size 3000 --tau-prior 0.56 \
    --budget-per-sample "$budget" --gamma "$gamma" \
    --seed-start "$SEED_START" --seed-end "$SEED_END" \
    --device "$DEVICE" --experiment-suffix "$EXPERIMENT_SUFFIX" \
    --dapro-n1-values 200 --definitive-dapro-margins 1.0 \
    --calibration-names "$METHODS"

  "$PYTHON" -m src.predictive_bounds.merge_bounds_results \
    --bound-type lpb --data-type real \
    --dataset-name "$dataset_name" --dataset-setup "$dataset_setup" \
    --cal-size 3000 --tau-prior 0.56 \
    --budget-per-sample "$budget" --gamma "$gamma" \
    --seed-start "$SEED_START" --seed-end "$SEED_END" \
    --device "$DEVICE" --experiment-suffix "$EXPERIMENT_SUFFIX" \
    --dapro-n1-values 200 --definitive-dapro-margins 1.0 \
    --calibration-names "$METHODS"
}

run_dataset dataset_toxicity attack_toxic_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_detoxify 20 10
run_dataset dataset_autoif attack_autoif_helper_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_autoif 20 10
run_dataset dataset_hallucination3 attack_hallucination_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_llm-judge_qwen25_14b_instruct 10 20
run_dataset dataset_red_team attack_default_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_llama_guard 10 20
run_dataset dataset_red_team attack_default_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_llm-judge_qwen25_14b_instruct 20 10
