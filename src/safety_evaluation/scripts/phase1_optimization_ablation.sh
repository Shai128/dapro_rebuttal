#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-srun -p galileo -A galileo -c4 --gres=gpu:0 python}"
DATASET_NAME="dataset_toxicity"
DATASET_SETUP="attack_toxic_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_detoxify"
BUDGET_PER_SAMPLE="${BUDGET_PER_SAMPLE:-20}"
CAL_SIZE="${CAL_SIZE:-3000}"
TAU_PRIOR="${TAU_PRIOR:-0.56}"
SEED_START="${SEED_START:-0}"
SEED_END="${SEED_END:-50}"
N1="${N1:-100}"
PROJECTION="${PROJECTION:-platt}"
SCORE="${SCORE:-prob}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT_DIR="${OUTPUT_DIR:-results/phase1_optimization_ablation}"
NUM_JOBS="50"

# Prevent each worker from creating its own large BLAS/OpenMP thread pool.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

if ! [[ "${NUM_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_JOBS must be a positive integer; got ${NUM_JOBS}" >&2
  exit 2
fi
if (( SEED_END <= SEED_START )); then
  echo "SEED_END must be greater than SEED_START." >&2
  exit 2
fi

SEED_COUNT=$((SEED_END - SEED_START))
JOB_COUNT="${NUM_JOBS}"
if (( JOB_COUNT > SEED_COUNT )); then
  JOB_COUNT="${SEED_COUNT}"
fi
if (( JOB_COUNT > 1 )) && [[ "${DEVICE}" == cuda* ]]; then
  echo "Warning: ${JOB_COUNT} workers will share ${DEVICE}; use DEVICE=cpu for CPU parallelism or reduce NUM_JOBS to avoid GPU-memory contention." >&2
fi

SHARD_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/phase1-optimization-ablation.XXXXXX")"
pids=()
cleanup() {
  for pid in "${pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
  if [[ -n "${SHARD_ROOT:-}" && -d "${SHARD_ROOT}" ]]; then
    rm -rf -- "${SHARD_ROOT}"
  fi
}
trap cleanup EXIT

COMMON_ARGS=(
  --dataset-name "${DATASET_NAME}"
  --dataset-setup "${DATASET_SETUP}"
  --budget-per-sample "${BUDGET_PER_SAMPLE}"
  --cal-size "${CAL_SIZE}"
  --tau-prior "${TAU_PRIOR}"
  --n1 "${N1}"
  --projection "${PROJECTION}"
  --score "${SCORE}"
  --device "${DEVICE}"
)

BASE_WIDTH=$((SEED_COUNT / JOB_COUNT))
REMAINDER=$((SEED_COUNT % JOB_COUNT))
cursor="${SEED_START}"
shard_dirs=()
log_paths=()

echo "Launching ${JOB_COUNT} parallel workers for seeds [${SEED_START}, ${SEED_END})."
for ((job = 0; job < JOB_COUNT; job++)); do
  width="${BASE_WIDTH}"
  if (( job < REMAINDER )); then
    width=$((width + 1))
  fi
  shard_start="${cursor}"
  shard_end=$((shard_start + width))
  cursor="${shard_end}"
  shard_dir="${SHARD_ROOT}/shard_${job}_${shard_start}_${shard_end}"
  log_path="${SHARD_ROOT}/worker_${job}.log"
  shard_dirs+=("${shard_dir}")
  log_paths+=("${log_path}")

  "${PYTHON_BIN}" -m src.safety_evaluation.phase1_optimization_ablation \
    "${COMMON_ARGS[@]}" \
    --seed-start "${shard_start}" \
    --seed-end "${shard_end}" \
    --output-dir "${shard_dir}" \
    --raw-only \
    >"${log_path}" 2>&1 &
  pids+=("$!")
  echo "  worker ${job}: seeds [${shard_start}, ${shard_end})"
done

failed=0
for ((job = 0; job < JOB_COUNT; job++)); do
  if ! wait "${pids[job]}"; then
    failed=1
    echo "Worker ${job} failed; output follows:" >&2
    sed "s/^/[worker ${job}] /" "${log_paths[job]}" >&2
  fi
done
if (( failed )); then
  exit 1
fi

echo "All workers completed; merging shards into ${OUTPUT_DIR}."
"${PYTHON_BIN}" -m src.safety_evaluation.phase1_optimization_ablation \
  "${COMMON_ARGS[@]}" \
  --seed-start "${SEED_START}" \
  --seed-end "${SEED_END}" \
  --output-dir "${OUTPUT_DIR}" \
  --merge-shards "${shard_dirs[@]}"

echo "Parallel phase-I ablation complete."
