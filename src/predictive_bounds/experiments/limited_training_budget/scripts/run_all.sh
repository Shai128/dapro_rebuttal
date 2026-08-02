#!/usr/bin/env bash
# Train at 10% of 200 turns/sample, cache predictions, then run bounds/reporting.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$ROOT"

if [[ "${ALL_CONFIGS:-0}" == "1" ]]; then
  EXTRA=()
  [[ "${AVAILABLE_ONLY:-0}" == "1" ]] && EXTRA+=(--available-only)
  python -m src.predictive_bounds.experiments.matrix_runner \
    --experiment limited_training_budget \
    --seed-start "${SEED_START:-0}" --seed-end "${SEED_END:-50}" \
    --device "${DEVICE:-cuda:0}" "${EXTRA[@]}"
  exit 0
fi

DATASET_NAME="${DATASET_NAME:-dataset_toxicity}"
DATASET_SETUP="${DATASET_SETUP:-attack_toxic_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_detoxify}"
TRAIN_BUDGET_FRACTION="${TRAIN_BUDGET_FRACTION:-0.10}"
FRACTION_KEY="${TRAIN_BUDGET_FRACTION//./p}"
CACHE="${PREDICTION_CACHE:-alg_playground_model/limited_training_budget/${DATASET_NAME}/${DATASET_SETUP}/probability_est_cal_test_fraction_${FRACTION_KEY}.pt}"
FULL_CACHE="${FULL_PREDICTION_CACHE:-alg_playground_model/is_real_True_dataset_${DATASET_NAME}_dataset_${DATASET_SETUP}/probability_est_cal_test.pt}"

python -m src.train_model.train_model \
  --dataset-name "$DATASET_NAME" --dataset-setup "$DATASET_SETUP" \
  --acquisition-strategy naive --uniform-training-budget-fraction "$TRAIN_BUDGET_FRACTION" \
  --full-budget-per-sample "${FULL_BUDGET_PER_SAMPLE:-200}" \
  --prediction-cache-output "$CACHE" --total-budget 0 \
  --seed-start "${TRAIN_SEED:-0}" --seed-end "$(( ${TRAIN_SEED:-0} + 1 ))" \
  --device "${DEVICE:-cuda:0}"

python -m src.predictive_bounds.experiments.limited_training_budget.run_comparison \
  --dataset-name "$DATASET_NAME" --dataset-setup "$DATASET_SETUP" \
  --limited-prediction-cache "$CACHE" --full-prediction-cache "$FULL_CACHE" \
  --limited-training-budget-fraction "$TRAIN_BUDGET_FRACTION" \
  --cal-size "${CAL_SIZE:-3000}" --budget-per-sample "${BUDGET:-20}" \
  --seed-start "${SEED_START:-0}" --seed-end "${SEED_END:-50}" \
  --device "${DEVICE:-cuda:0}"
