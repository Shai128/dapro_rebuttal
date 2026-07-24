#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_NAME="${DATASET_NAME:-dataset_red_team}"
DATASET_SETUP="${DATASET_SETUP:-attack_default_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_llm-judge_qwen25_14b_instruct}"
BUDGET_PER_SAMPLE="${BUDGET_PER_SAMPLE:-20}"
CAL_SIZE="${CAL_SIZE:-3000}"
TAU_PRIOR="${TAU_PRIOR:-0.56}"
SEED_START="${SEED_START:-0}"
SEED_END="${SEED_END:-50}"
N1="${N1:-100}"
PROJECTION="${PROJECTION:-platt}"
SCORE="${SCORE:-prob}"
DEVICE="${DEVICE:-cuda:0}"

"${PYTHON_BIN}" -m src.safety_evaluation.phase1_optimization_ablation \
  --dataset-name "${DATASET_NAME}" \
  --dataset-setup "${DATASET_SETUP}" \
  --budget-per-sample "${BUDGET_PER_SAMPLE}" \
  --cal-size "${CAL_SIZE}" \
  --tau-prior "${TAU_PRIOR}" \
  --seed-start "${SEED_START}" \
  --seed-end "${SEED_END}" \
  --n1 "${N1}" \
  --projection "${PROJECTION}" \
  --score "${SCORE}" \
  --device "${DEVICE}"
