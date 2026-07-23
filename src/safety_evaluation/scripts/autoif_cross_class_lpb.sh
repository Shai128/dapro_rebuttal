#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_NAME="${DATASET_NAME:-dataset_autoif}"
DATASET_SETUP="${DATASET_SETUP:-attack_autoif_helper_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_autoif}"
AUTOIF_DATA_PATH="${AUTOIF_DATA_PATH:-src/multi_turn_data_generation/data/autoif_helper_dataset.csv}"
CLASSIFICATIONS_PATH="${CLASSIFICATIONS_PATH:-src/multi_turn_data_generation/data/classified_instructions.csv}"
CALIBRATION_CLASS="${CALIBRATION_CLASS:-Programming & Technology}"
TEST_CLASS="${TEST_CLASS:-Marketing & Social Media}"
BUDGET_PER_SAMPLE="${BUDGET_PER_SAMPLE:-20}"
CAL_SIZE="${CAL_SIZE:-800}"
TEST_SIZE="${TEST_SIZE:-100}"
TAU_PRIOR="${TAU_PRIOR:-0.56}"
GAMMA="${GAMMA:-10}"
SEED_START="${SEED_START:-0}"
SEED_END="${SEED_END:-50}"
ALLOCATIONS="${ALLOCATIONS:-one}"
DEVICE="${DEVICE:-cuda:0}"
MAX_WORKERS="${MAX_WORKERS:-1}"

COMMON_ARGS=(
  --bound-type lpb
  --seed-start "${SEED_START}"
  --seed-end "${SEED_END}"
  --dataset-name "${DATASET_NAME}"
  --dataset-setup "${DATASET_SETUP}"
  --calibration-class "${CALIBRATION_CLASS}"
  --test-class "${TEST_CLASS}"
  --budget-per-sample "${BUDGET_PER_SAMPLE}"
  --cal-size "${CAL_SIZE}"
  --test-size "${TEST_SIZE}"
  --tau-prior "${TAU_PRIOR}"
  --gamma "${GAMMA}"
)

"${PYTHON_BIN}" -m src.safety_evaluation.construct_autoif_cross_class_calibrated_bound \
  "${COMMON_ARGS[@]}" \
  --autoif-data-path "${AUTOIF_DATA_PATH}" \
  --classifications-path "${CLASSIFICATIONS_PATH}" \
  --allocations "${ALLOCATIONS}" \
  --device "${DEVICE}" \
  --max-workers "${MAX_WORKERS}"

"${PYTHON_BIN}" -m src.safety_evaluation.merge_autoif_cross_class_bounds_results \
  "${COMMON_ARGS[@]}"
