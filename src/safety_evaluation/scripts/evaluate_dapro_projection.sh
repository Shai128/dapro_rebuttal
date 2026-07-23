#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_TYPE="${DATA_TYPE:-real}"
DATASET_NAME="${DATASET_NAME:-dataset_red_team}"
DATASET_SETUP="${DATASET_SETUP:-attack_default_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_llm-judge_qwen25_14b_instruct}"
BUDGET_PER_SAMPLE="${BUDGET_PER_SAMPLE:-20}"
CAL_SIZE="${CAL_SIZE:-3000}"
TAU_PRIOR="${TAU_PRIOR:-0.56}"
M_UPPER_BOUND="${M_UPPER_BOUND:-200}"
SEED_START="${SEED_START:-0}"
SEED_END="${SEED_END:-50}"
DEVICE="${DEVICE:-cuda:0}"
PROJECTIONS="${PROJECTIONS:-platt beta}"
SCORES="${SCORES:-prob quantile}"
N1_VALUES="${N1_VALUES:-100}"

read -r -a PROJECTION_ARGS <<< "${PROJECTIONS}"
read -r -a SCORE_ARGS <<< "${SCORES}"
read -r -a N1_ARGS <<< "${N1_VALUES}"

COMMON_ARGS=(
  --seed-start "${SEED_START}"
  --seed-end "${SEED_END}"
  --data-type "${DATA_TYPE}"
  --dataset-name "${DATASET_NAME}"
  --dataset-setup "${DATASET_SETUP}"
  --budget-per-sample "${BUDGET_PER_SAMPLE}"
  --cal-size "${CAL_SIZE}"
  --tau-prior "${TAU_PRIOR}"
  --m-upper-bound "${M_UPPER_BOUND}"
)

"${PYTHON_BIN}" -m src.safety_evaluation.evaluate_dapro_projection \
  "${COMMON_ARGS[@]}" \
  --projections "${PROJECTION_ARGS[@]}" \
  --scores "${SCORE_ARGS[@]}" \
  --n1-values "${N1_ARGS[@]}" \
  --device "${DEVICE}"

"${PYTHON_BIN}" -m src.safety_evaluation.merge_dapro_projection_results \
  "${COMMON_ARGS[@]}"
