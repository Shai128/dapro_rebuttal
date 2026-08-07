#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_TYPE="${DATA_TYPE:-real}"
DATASET_NAME="${DATASET_NAME:-dataset_toxicity}"
DATASET_SETUP="${DATASET_SETUP:-attack_toxic_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_detoxify}"
BUDGET_PER_SAMPLE="${BUDGET_PER_SAMPLE:-20}"
CAL_SIZE="${CAL_SIZE:-3000}"
SEED_START="${SEED_START:-0}"
SEED_END="${SEED_END:-10}"
TAU_PRIOR="${TAU_PRIOR:-0.56}"
DEVICE="${DEVICE:-cuda:0}"

COMMON_ARGS=(
  --data-type "${DATA_TYPE}"
  --dataset-name "${DATASET_NAME}"
  --dataset-setup "${DATASET_SETUP}"
  --budget-per-sample "${BUDGET_PER_SAMPLE}"
  --cal-size "${CAL_SIZE}"
  --seed-start "${SEED_START}"
  --seed-end "${SEED_END}"
  --device "${DEVICE}"
)

"${PYTHON_BIN}" -m src.predictive_bounds.experiments.metrics.estimate \
  "${COMMON_ARGS[@]}"

"${PYTHON_BIN}" -m src.predictive_bounds.experiments.metrics.merge_results \
  "${COMMON_ARGS[@]}" \
  --tau-prior "${TAU_PRIOR}"
