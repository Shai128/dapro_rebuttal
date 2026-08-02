#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../../.." && pwd)"
cd "${REPO_ROOT}"

#  "dataset_toxicity", dataset_red_team
#  "attack_toxic_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_detoxify",
# attack_default_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_llm-judge_qwen25_14b_instruct
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_TYPE="${DATA_TYPE:-real}"
DATASET_NAME="${DATASET_NAME:-dataset_toxicity}"
DATASET_SETUP="${DATASET_SETUP:-attack_toxic_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_detoxify}"
BUDGET_PER_SAMPLE="${BUDGET_PER_SAMPLE:-20}"
CAL_SIZE="${CAL_SIZE:-3000}"
TAU_PRIOR="${TAU_PRIOR:-0.56}"
M_UPPER_BOUND="${M_UPPER_BOUND:-200}"
DEVICE="${DEVICE:-cuda:0}"
PROJECTIONS="${PROJECTIONS:-platt beta}"
SCORES="${SCORES:-prob quantile}"
N1_VALUES="${N1_VALUES:-100}"


SEED_START="${SEED_START:-0}"
SEED_END="${SEED_END:-50}"
SEED_GROUP_SIZE="${SEED_GROUP_SIZE:-1}"


read -r -a PROJECTION_ARGS <<< "${PROJECTIONS}"
read -r -a SCORE_ARGS <<< "${SCORES}"
read -r -a N1_ARGS <<< "${N1_VALUES}"

COMMON_ARGS=(
  --data-type "${DATA_TYPE}"
  --dataset-name "${DATASET_NAME}"
  --dataset-setup "${DATASET_SETUP}"
  --budget-per-sample "${BUDGET_PER_SAMPLE}"
  --cal-size "${CAL_SIZE}"
  --tau-prior "${TAU_PRIOR}"
  --m-upper-bound "${M_UPPER_BOUND}"
)


mkdir -p logs

pids=()
job_descriptions=()

for ((group_start = SEED_START; group_start < SEED_END; group_start += SEED_GROUP_SIZE)); do
  group_end=$((group_start + SEED_GROUP_SIZE))

  # Do not exceed the requested overall seed end.
  if ((group_end > SEED_END)); then
    group_end="${SEED_END}"
  fi

  log_file="logs/seeds_${group_start}_${group_end}.log"

  echo "Launching seeds [${group_start}, ${group_end}) -> ${log_file}"

  "${PYTHON_BIN}" -m src.predictive_bounds.experiments.dapro_projection.evaluate \
    "${COMMON_ARGS[@]}" \
    --seed-start "${group_start}" \
    --seed-end "${group_end}" \
    --projections "${PROJECTION_ARGS[@]}" \
    --scores "${SCORE_ARGS[@]}" \
    --n1-values "${N1_ARGS[@]}" \
    --device "${DEVICE}" \
    >"${log_file}" 2>&1 &

  pids+=("$!")
  job_descriptions+=("seeds [${group_start}, ${group_end})")
done

echo "Launched ${#pids[@]} parallel jobs."

failed=0

for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  description="${job_descriptions[$i]}"

  if wait "${pid}"; then
    echo "Completed: ${description}"
  else
    status=$?
    echo "Failed: ${description}, exit status ${status}" >&2
    failed=1
  fi
done

if ((failed)); then
  echo "One or more seed groups failed. Check the logs directory." >&2
  exit 1
fi

echo "All seed groups completed successfully."


"${PYTHON_BIN}" -m src.predictive_bounds.experiments.dapro_projection.merge_results \
  "${COMMON_ARGS[@]}"
