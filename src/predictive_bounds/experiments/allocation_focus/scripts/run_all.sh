#!/usr/bin/env bash
# Per-sample allocation audit, strict merge, quartile plots, and LaTeX tables.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$ROOT"

if [[ "${ALL_CONFIGS:-0}" == "1" ]]; then
  EXTRA=()
  [[ "${AVAILABLE_ONLY:-0}" == "1" ]] && EXTRA+=(--available-only)
  python -m src.predictive_bounds.experiments.matrix_runner \
    --experiment allocation_focus \
    --seed-start "${SEED_START:-0}" --seed-end "${SEED_END:-50}" \
    --device "${DEVICE:-cuda:0}" "${EXTRA[@]}"
  exit 0
fi

python -m src.predictive_bounds.experiments.allocation_focus.run_all \
  --dataset-name "${DATASET_NAME:-dataset_toxicity}" \
  --dataset-setup "${DATASET_SETUP:-attack_toxic_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_detoxify}" \
  --cal-size "${CAL_SIZE:-3000}" --budget-per-sample "${BUDGET:-20}" \
  --tau-prior "${TAU_PRIOR:-0.56}" --target-coverage "${TARGET_COVERAGE:-0.90}" \
  --future-time "${FUTURE_TIME:-3}" \
  --seed-start "${SEED_START:-0}" --seed-end "${SEED_END:-50}" \
  --device "${DEVICE:-cuda:0}"
