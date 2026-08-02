#!/usr/bin/env bash
# Train on a source domain; calibrate and test on one distinct target domain.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$ROOT"

if [[ -n "${SHIFT_CONFIG_FILE:-}" ]]; then
  python -m src.predictive_bounds.experiments.distribution_shift.run_matrix \
    --config-file "$SHIFT_CONFIG_FILE" \
    --shift-type train_calibration_test_shift \
    --seed-start "${SEED_START:-0}" --seed-end "${SEED_END:-50}" \
    --device "${DEVICE:-cuda:0}"
  exit 0
fi

DATASET_NAME="${DATASET_NAME:-dataset_red_team}"
MODEL_DATASET_NAME="${MODEL_DATASET_NAME:-$DATASET_NAME}"
CALIBRATION_DATASET_NAME="${CALIBRATION_DATASET_NAME:-$DATASET_NAME}"
TEST_DATASET_NAME="${TEST_DATASET_NAME:-$CALIBRATION_DATASET_NAME}"
MODEL_SETUP="${MODEL_SETUP:-attack_default_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_llm-judge_qwen25_14b_instruct}"
: "${TEST_SETUP:?Set TEST_SETUP to the held-out evaluation setup}"
CALIBRATION_SETUP="${CALIBRATION_SETUP:-$TEST_SETUP}"

python -m src.predictive_bounds.experiments.distribution_shift.run_all \
  --shift-type train_calibration_test_shift \
  --dataset-name "$DATASET_NAME" \
  --model-dataset-name "$MODEL_DATASET_NAME" \
  --calibration-dataset-name "$CALIBRATION_DATASET_NAME" \
  --test-dataset-name "$TEST_DATASET_NAME" \
  --model-dataset-setup "$MODEL_SETUP" \
  --calibration-dataset-setup "$CALIBRATION_SETUP" \
  --test-dataset-setup "$TEST_SETUP" \
  --cal-size "${CAL_SIZE:-3000}" --test-size "${TEST_SIZE:-0}" \
  --budget-per-sample "${BUDGET:-20}" --tau-prior "${TAU_PRIOR:-0.56}" \
  --target-coverage "${TARGET_COVERAGE:-0.90}" \
  --seed-start "${SEED_START:-0}" --seed-end "${SEED_END:-50}" \
  --device "${DEVICE:-cuda:0}"
