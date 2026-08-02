#!/usr/bin/env bash
# One-factor and crossed data/policy/acquisition variance decomposition.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$ROOT"

if [[ "${ALL_CONFIGS:-0}" == "1" ]]; then
  EXTRA=()
  [[ "${AVAILABLE_ONLY:-0}" == "1" ]] && EXTRA+=(--available-only)
  python -m src.predictive_bounds.experiments.matrix_runner \
    --experiment variance_components \
    --n1-values ${N1_VALUES:-100 200 400 800} \
    --variance-replicates "${REPLICATES:-50}" \
    --crossed-groups "${CROSSED_GROUPS:-10}" \
    --seed-start "${SEED_START:-0}" --seed-end "${SEED_END:-50}" \
    --device "${DEVICE:-cuda:0}" "${EXTRA[@]}"
  exit 0
fi

ARGS=(
  --dataset-name "${DATASET_NAME:-dataset_toxicity}"
  --dataset-setup "${DATASET_SETUP:-attack_toxic_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_detoxify}"
  --cal-size "${CAL_SIZE:-3000}" --budget-per-sample "${BUDGET:-20}"
  --tau-prior "${TAU_PRIOR:-0.56}" --m-upper-bound "${M_UPPER_BOUND:-200}"
  --n1-values ${N1_VALUES:-100 200 400 800}
  --replicates "${REPLICATES:-50}" --crossed-groups "${CROSSED_GROUPS:-10}"
  --suffix-prefix "${SUFFIX_PREFIX:-variance_components_v1}"
  --device "${DEVICE:-cuda:0}"
)
python -m src.predictive_bounds.experiments.variance_components.run "${ARGS[@]}"
python -m src.predictive_bounds.experiments.variance_components.merge_results "${ARGS[@]}"
python -m src.predictive_bounds.experiments.variance_components.summarize "${ARGS[@]}"
