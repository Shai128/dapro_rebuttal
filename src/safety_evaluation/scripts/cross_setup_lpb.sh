#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-srun -p galileo -A galileo -c4 --gres=gpu:0 python}"
#DATASET_NAME="${DATASET_NAME:-dataset_red_team}"

DATASET_NAME="dataset_toxicity"
MODEL_DATASET_SETUP="attack_toxic_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_detoxify"

EVALUATION_DATASET_SETUP="attack_toxic_attack_qwen25_14b_instruct_lm_target_gemma3_4b_it_judge_detoxify"


BUDGET_PER_SAMPLE="${BUDGET_PER_SAMPLE:-20}"
CAL_SIZE="${CAL_SIZE:-3000}"
TAU_PRIOR="${TAU_PRIOR:-0.56}"
GAMMA="${GAMMA:-10}"
SEED_START="${SEED_START:-0}"
SEED_END="${SEED_END:-50}"
ALLOCATIONS="${ALLOCATIONS:-one}"
DEVICE="${DEVICE:-cuda:0}"
MAX_WORKERS="${MAX_WORKERS:-1}"

COMMON_ARGS=(
  --bound-type lpb
  --data-type real
  --dataset-name "${DATASET_NAME}"
  --model-dataset-setup "${MODEL_DATASET_SETUP}"
  --evaluation-dataset-setup "${EVALUATION_DATASET_SETUP}"
  --budget-per-sample "${BUDGET_PER_SAMPLE}"
  --cal-size "${CAL_SIZE}"
  --tau-prior "${TAU_PRIOR}"
  --gamma "${GAMMA}"
  --seed-start "${SEED_START}"
  --seed-end "${SEED_END}"
)

${PYTHON_BIN} -m src.safety_evaluation.construct_cross_setup_calibrated_bound \
  "${COMMON_ARGS[@]}" \
  --allocations "${ALLOCATIONS}" \
  --device "${DEVICE}" \
  --max-workers "${MAX_WORKERS}"

${PYTHON_BIN} -m src.safety_evaluation.merge_cross_setup_bounds_results \
  "${COMMON_ARGS[@]}"
