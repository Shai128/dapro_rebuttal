#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_NAME="${DATASET_NAME:-dataset_toxicity}"
DATASET_SETUP="${DATASET_SETUP:-attack_toxic_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_detoxify}"
CAL_SIZE="${CAL_SIZE:-3000}"
SEED_START="${SEED_START:-0}"
SEED_END="${SEED_END:-10}"
BUDGET_PER_SAMPLE="${BUDGET_PER_SAMPLE:-20}"
TAU_PRIOR="${TAU_PRIOR:-0.56}"
M_UPPER_BOUND="${M_UPPER_BOUND:-200}"
DEVICE="${DEVICE:-cuda:0}"

"${PYTHON_BIN}" -m src.predictive_bounds.ablations.random_floor \
  --dataset-name "${DATASET_NAME}" \
  --dataset-setup "${DATASET_SETUP}" \
  --cal-size "${CAL_SIZE}" \
  --seed-start "${SEED_START}" \
  --seed-end "${SEED_END}" \
  --budget-per-sample "${BUDGET_PER_SAMPLE}" \
  --tau-prior "${TAU_PRIOR}" \
  --m-upper-bound "${M_UPPER_BOUND}" \
  --device "${DEVICE}"
