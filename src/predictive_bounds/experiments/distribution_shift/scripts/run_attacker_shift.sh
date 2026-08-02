#!/usr/bin/env bash
# Same target/judge, but source and held-out test trajectories use different attackers.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$ROOT"

if [[ -n "${SHIFT_CONFIG_FILE:-}" ]]; then
  python -m src.predictive_bounds.experiments.distribution_shift.run_matrix \
    --config-file "$SHIFT_CONFIG_FILE" --shift-type attacker_shift \
    --seed-start "${SEED_START:-0}" --seed-end "${SEED_END:-50}" \
    --device "${DEVICE:-cuda:0}"
  exit 0
fi

DATASET_NAME="${DATASET_NAME:-dataset_red_team}"
MODEL_SETUP="${MODEL_SETUP:-attack_default_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_llm-judge_qwen25_14b_instruct}"
: "${ATTACKER_TEST_SETUP:?Set ATTACKER_TEST_SETUP to the held-out attacker setup}"

python -m src.predictive_bounds.experiments.distribution_shift.run_all \
  --shift-type attacker_shift --dataset-name "$DATASET_NAME" \
  --model-dataset-name "$DATASET_NAME" \
  --calibration-dataset-name "$DATASET_NAME" \
  --test-dataset-name "$DATASET_NAME" \
  --model-dataset-setup "$MODEL_SETUP" \
  --calibration-dataset-setup "$MODEL_SETUP" \
  --test-dataset-setup "$ATTACKER_TEST_SETUP" \
  --cal-size "${CAL_SIZE:-3000}" --test-size "${TEST_SIZE:-0}" \
  --budget-per-sample "${BUDGET:-20}" --tau-prior "${TAU_PRIOR:-0.56}" \
  --target-coverage "${TARGET_COVERAGE:-0.90}" \
  --seed-start "${SEED_START:-0}" --seed-end "${SEED_END:-50}" \
  --device "${DEVICE:-cuda:0}"
