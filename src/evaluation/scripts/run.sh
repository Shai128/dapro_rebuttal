#!/usr/bin/env bash
# Run, merge, and plot the complete fixed-benchmark metric comparison.
#
# Examples:
#   bash src/evaluation/scripts/run.sh --local --cpu --available-only
#   bash src/evaluation/scripts/run.sh --slurm --parallel-jobs 50
#   bash src/evaluation/scripts/run.sh --slurm --dry-run
set -euo pipefail

# ======================== EDITABLE CONFIGURATION ========================
RUN_MODE="local"                    # "local" or "slurm"
DEVICE="cuda:0"
PYTHON_BIN="${PYTHON_BIN:-python}"
AVAILABLE_ONLY=0
DRY_RUN=0
GENERATE_FIGURES=1

DATASETS=(
#  toxicity
#  red_team_qwen
#  red_team_llama_guard
  hallucination3
  autoif
)

TARGET_MODELS=(
  qwen25_14b_instruct
  llama_31_8B_instruct
  mini_phi_4_instruct
)

ATTACKER_MODEL="qwen25_14b_instruct"
CAL_SIZE=3000
SEED_START=0
SEED_END=50
TAU_PRIOR=0.56
EXPERIMENT_SUFFIX="metric_estimation_v1"

# Each entry is DAPRO_N1:CRC_CONTROL_SIZE.
DAPRO_CONFIGS=(
  "200:100"
  "100:50"
)

BUDGET_PER_SAMPLE_VALUES=(5 10 20)

SLURM_ACCOUNT="galileo"
SLURM_PARTITION="galileo"
SLURM_CPUS=4
SLURM_GRES="gpu:1"
SLURM_JOB_NAME="metricEst"
MAX_CONCURRENT_SRUNS=50
EXCLUDE_LIST=""
# ====================== END EDITABLE CONFIGURATION ======================

usage() {
  echo "Usage: $0 [--local | --slurm] [--cpu | --device DEVICE]"
  echo "          [--available-only] [--parallel-jobs N] [--seed-end N]"
  echo "          [--skip-figures] [--dry-run]"
}

while (( $# > 0 )); do
  case "$1" in
    --local) RUN_MODE="local" ;;
    --slurm) RUN_MODE="slurm" ;;
    --cpu) DEVICE="cpu"; SLURM_GRES="" ;;
    --device) DEVICE="$2"; shift ;;
    --available-only) AVAILABLE_ONLY=1 ;;
    --parallel-jobs) MAX_CONCURRENT_SRUNS="$2"; shift ;;
    --seed-end) SEED_END="$2"; shift ;;
    --skip-figures) GENERATE_FIGURES=0 ;;
    --dry-run) DRY_RUN=1 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if [[ "$RUN_MODE" != "local" && "$RUN_MODE" != "slurm" ]]; then
  echo "RUN_MODE must be local or slurm." >&2
  exit 2
fi
if ! [[ "$SEED_START" =~ ^[0-9]+$ && "$SEED_END" =~ ^[0-9]+$ ]] || (( SEED_END <= SEED_START )); then
  echo "Seed range must be nonnegative and nonempty." >&2
  exit 2
fi
if ! [[ "$MAX_CONCURRENT_SRUNS" =~ ^[1-9][0-9]*$ ]]; then
  echo "--parallel-jobs must be a positive integer." >&2
  exit 2
fi
for config in "${DAPRO_CONFIGS[@]}"; do
  if [[ ! "$config" =~ ^([0-9]+):([0-9]+)$ ]]; then
    echo "Invalid DAPRO configuration: $config" >&2
    exit 2
  fi
  if (( BASH_REMATCH[2] >= BASH_REMATCH[1] || BASH_REMATCH[1] >= CAL_SIZE )); then
    echo "Require CRC_CONTROL_SIZE < DAPRO_N1 < CAL_SIZE: $config" >&2
    exit 2
  fi
done
if [[ "$RUN_MODE" == "slurm" && -n "${SLURM_JOB_ID+x}" ]]; then
  echo "Run this launcher from a login node, not inside a Slurm allocation." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

run_module() {
  local module="$1"
  local job_suffix="$2"
  shift 2
  local command=("$PYTHON_BIN" -m "$module" "$@")
  if [[ "$RUN_MODE" == "slurm" ]]; then
    local prefix=(
      srun -A "$SLURM_ACCOUNT" -p "$SLURM_PARTITION" -c "$SLURM_CPUS"
      -J "${SLURM_JOB_NAME}_${job_suffix}"
    )
    if [[ -n "$SLURM_GRES" ]]; then prefix+=(--gres="$SLURM_GRES"); fi
    if [[ -n "$EXCLUDE_LIST" ]]; then prefix+=(--exclude="$EXCLUDE_LIST"); fi
    command=("${prefix[@]}" "${command[@]}")
  fi
  if (( DRY_RUN == 1 )); then print_command "${command[@]}"; else "${command[@]}"; fi
}

wait_batch() {
  local failed=0
  local pid
  for pid in "$@"; do
    if ! wait "$pid"; then failed=1; fi
  done
  return "$failed"
}

configure_dataset() {
  local key="$1"
  local target="$2"
  case "$key" in
    toxicity)
      DATASET_NAME="dataset_toxicity"
      DATASET_SETUP="attack_toxic_attack_${ATTACKER_MODEL}_lm_target_${target}_judge_detoxify"
      ;;
    red_team_qwen)
      DATASET_NAME="dataset_red_team"
      DATASET_SETUP="attack_default_attack_${ATTACKER_MODEL}_lm_target_${target}_judge_llm-judge_qwen25_14b_instruct"
      ;;
    red_team_llama_guard)
      DATASET_NAME="dataset_red_team"
      DATASET_SETUP="attack_default_attack_${ATTACKER_MODEL}_lm_target_${target}_judge_llama_guard"
      ;;
    hallucination3)
      DATASET_NAME="dataset_hallucination3"
      DATASET_SETUP="attack_hallucination_attack_${ATTACKER_MODEL}_lm_target_${target}_judge_llm-judge_qwen25_14b_instruct"
      ;;
    autoif)
      DATASET_NAME="dataset_autoif"
      DATASET_SETUP="attack_autoif_helper_${ATTACKER_MODEL}_lm_target_${target}_judge_autoif"
      ;;
    *) echo "Unknown dataset key: $key" >&2; exit 2 ;;
  esac
}

EXPERIMENT_NAMES=()
for config in "${DAPRO_CONFIGS[@]}"; do
  IFS=: read -r dapro_n1 crc_control_size <<< "$config"
  for budget in "${BUDGET_PER_SAMPLE_VALUES[@]}"; do
    for dataset in "${DATASETS[@]}"; do
      for target in "${TARGET_MODELS[@]}"; do
        configure_dataset "$dataset" "$target"
        cache="alg_playground_model/is_real_True_dataset_${DATASET_NAME}_dataset_${DATASET_SETUP}/probability_est_cal_test.pt"
        if (( AVAILABLE_ONLY == 1 )) && [[ ! -f "$cache" ]]; then
          echo "Skipping $dataset / $target: missing $cache"
          continue
        fi

        experiment_name="${DATASET_NAME}_${DATASET_SETUP}_${budget}_metric_estimation_n1_${dapro_n1}_crc_${crc_control_size}__${EXPERIMENT_SUFFIX}"
        EXPERIMENT_NAMES+=("$experiment_name")
        common=(
          --data-type real
          --dataset-name "$DATASET_NAME"
          --dataset-setup "$DATASET_SETUP"
          --budget-per-sample "$budget"
          --cal-size "$CAL_SIZE"
          --tau-prior "$TAU_PRIOR"
          --device "$DEVICE"
          --dapro-n1 "$dapro_n1"
          --crc-control-size "$crc_control_size"
          --experiment-suffix "$EXPERIMENT_SUFFIX"
        )
        key="${dataset}_${target}_n1_${dapro_n1}_crc_${crc_control_size}_b_${budget}"
        echo "[$key] estimate"

        if [[ "$RUN_MODE" == "slurm" && "$MAX_CONCURRENT_SRUNS" -gt 1 ]]; then
          jobs=()
          for (( seed=SEED_START; seed<SEED_END; seed++ )); do
            run_module src.evaluation.estimate "${key}_s${seed}" \
              "${common[@]}" --seed-start "$seed" --seed-end "$((seed + 1))" &
            jobs+=("$!")
            if (( ${#jobs[@]} == MAX_CONCURRENT_SRUNS )); then
              wait_batch "${jobs[@]}"
              jobs=()
            fi
          done
          if (( ${#jobs[@]} > 0 )); then wait_batch "${jobs[@]}"; fi
        else
          run_module src.evaluation.estimate "${key}_estimate" \
            "${common[@]}" --seed-start "$SEED_START" --seed-end "$SEED_END"
        fi

        echo "[$key] merge"
        run_module src.evaluation.merge_results "${key}_merge" \
          "${common[@]}" --seed-start "$SEED_START" --seed-end "$SEED_END"
      done
    done
  done
done

if (( GENERATE_FIGURES == 1 && ${#EXPERIMENT_NAMES[@]} > 0 )); then
  plot_command=("$PYTHON_BIN" -m src.evaluation.summarize)
  for experiment_name in "${EXPERIMENT_NAMES[@]}"; do
    plot_command+=(--experiment "$experiment_name")
  done
  echo "Generate figures"
  if (( DRY_RUN == 1 )); then print_command "${plot_command[@]}"; else "${plot_command[@]}"; fi
fi
