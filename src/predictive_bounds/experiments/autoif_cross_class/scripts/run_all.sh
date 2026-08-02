#!/usr/bin/env bash
# AutoIF calibration subject -> held-out test subject, then merge/report.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$ROOT"

if [[ "${ALL_CONFIGS:-0}" == "1" ]]; then
  EXTRA=()
  [[ "${AVAILABLE_ONLY:-0}" == "1" ]] && EXTRA+=(--available-only)
  python -m src.predictive_bounds.experiments.matrix_runner \
    --experiment autoif_cross_class \
    --autoif-data-path "${AUTOIF_DATA_PATH:-src/multi_turn_data_generation/data/autoif_helper_dataset.csv}" \
    --classifications-path "${CLASSIFICATIONS_PATH:-src/multi_turn_data_generation/data/classified_instructions.csv}" \
    --calibration-class "${CALIBRATION_CLASS:-Programming & Technology}" \
    --test-class "${TEST_CLASS:-Marketing & Social Media}" \
    --cross-class-cal-size "${CAL_SIZE:-800}" \
    --cross-class-test-size "${TEST_SIZE:-100}" \
    --seed-start "${SEED_START:-0}" --seed-end "${SEED_END:-50}" \
    --device "${DEVICE:-cuda:0}" "${EXTRA[@]}"
  exit 0
fi

python -m src.predictive_bounds.experiments.autoif_cross_class.run_all \
  --dataset-name "${DATASET_NAME:-dataset_autoif}" \
  --dataset-setup "${DATASET_SETUP:-attack_autoif_helper_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_autoif}" \
  --autoif-data-path "${AUTOIF_DATA_PATH:-src/multi_turn_data_generation/data/autoif_helper_dataset.csv}" \
  --classifications-path "${CLASSIFICATIONS_PATH:-src/multi_turn_data_generation/data/classified_instructions.csv}" \
  --calibration-class "${CALIBRATION_CLASS:-Programming & Technology}" \
  --test-class "${TEST_CLASS:-Marketing & Social Media}" \
  --cal-size "${CAL_SIZE:-800}" --test-size "${TEST_SIZE:-100}" \
  --budget-per-sample "${BUDGET:-20}" \
  --tau-prior "${TAU_PRIOR:-0.56}" --gamma "${GAMMA:-10}" \
  --allocations "${ALLOCATIONS:-one}" --max-workers "${MAX_WORKERS:-1}" \
  --paper-methods-only \
  --seed-start "${SEED_START:-0}" --seed-end "${SEED_END:-50}" \
  --device "${DEVICE:-cuda:0}"
