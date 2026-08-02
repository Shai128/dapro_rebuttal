#!/usr/bin/env bash
# Calibration-only FN, FP, and combined judge-noise levels.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$ROOT"

if [[ "${ALL_CONFIGS:-0}" == "1" ]]; then
  EXTRA=()
  [[ "${AVAILABLE_ONLY:-0}" == "1" ]] && EXTRA+=(--available-only)
  python -m src.predictive_bounds.experiments.matrix_runner \
    --experiment judge_noise --noise-levels ${NOISE_LEVELS:-0.01 0.05 0.10 0.20} \
    --seed-start "${SEED_START:-0}" --seed-end "${SEED_END:-50}" \
    --device "${DEVICE:-cuda:0}" "${EXTRA[@]}"
  exit 0
fi

python -m src.predictive_bounds.experiments.judge_noise.run_all \
  --dataset-name "${DATASET_NAME:-dataset_red_team}" \
  --dataset-setup "${DATASET_SETUP:-attack_default_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_llm-judge_qwen25_14b_instruct}" \
  --noise-mode all --noise-levels ${NOISE_LEVELS:-0.01 0.05 0.10 0.20} \
  --cal-size "${CAL_SIZE:-3000}" --budget-per-sample "${BUDGET:-20}" \
  --tau-prior "${TAU_PRIOR:-0.56}" --target-coverage "${TARGET_COVERAGE:-0.90}" \
  --seed-start "${SEED_START:-0}" --seed-end "${SEED_END:-50}" \
  --device "${DEVICE:-cuda:0}"
