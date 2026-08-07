#!/usr/bin/env bash
# Construct and merge the complete LPB comparison for every configured dataset.
#
# Experiment matrix:
#   (DAPRO_N1=200, CRC_CONTROL_SIZE=100) x BUDGET_PER_SAMPLE={5,10,20}
#   (DAPRO_N1=100, CRC_CONTROL_SIZE=50)  x BUDGET_PER_SAMPLE={5,10,20}
#
# Typical invocations:
#   bash src/predictive_bounds/scripts/calibrate.sh --local
#   bash src/predictive_bounds/scripts/calibrate.sh --local --cpu --available-only
#   bash src/predictive_bounds/scripts/calibrate.sh --slurm
#   bash src/predictive_bounds/scripts/calibrate.sh --slurm --cpu --parallel-jobs 50
#   bash src/predictive_bounds/scripts/calibrate.sh --slurm --dry-run
#
# Edit only the configuration block below to change the experiment matrix.
set -euo pipefail

# ======================== EDITABLE CONFIGURATION ========================
RUN_MODE="local"                    # "local" or "slurm"
DEVICE="cuda:0"                     # Use "cpu" when no GPU is available.
AVAILABLE_ONLY=0                    # 1 skips configurations without cached predictions.
DRY_RUN=0                           # 1 prints commands without executing them.

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
#  gemma3_4b_it
)

ATTACKER_MODEL="qwen25_14b_instruct"
CAL_SIZE=3000
SEED_START=0
SEED_END=50
TAU_PRIOR=0.56
M_UPPER_BOUND=200

# Each entry is DAPRO_N1:CRC_CONTROL_SIZE.
DAPRO_CONFIGS=(
  "200:100"
  "100:50"
)

# Every budget is run for every DAPRO/CRC configuration above.
BUDGET_PER_SAMPLE_VALUES=(
  5
  10
  20
)

# Each N1/CRC/budget combination receives its own suffix to prevent merge
# outputs from overwriting one another.
BASE_EXPERIMENT_SUFFIX="lpb_all_methods_v1"
ARCHIVE_PATH="results/lpb_merged_${BASE_EXPERIMENT_SUFFIX}.tar.gz"

SLURM_ACCOUNT="galileo"
SLURM_PARTITION="galileo"
SLURM_CPUS=4
SLURM_GRES="gpu:1"                 # One GPU; Python addresses it as cuda:0.
SLURM_JOB_NAME="plsNoKil"
MAX_CONCURRENT_SRUNS=50             # 1 runs the full seed range in one srun.
EXCLUDE_LIST=""                    # Leave empty to use the automatic GPU filter.
AUTO_EXCLUDE_INCOMPATIBLE_GPUS=1
# ====================== END EDITABLE CONFIGURATION ======================

usage() {
  echo "Usage: $0 [--local | --slurm] [--cpu | --device DEVICE]"
  echo "          [--available-only] [--parallel-jobs N] [--dry-run]"
  echo "          [--seed-end N]"
  echo
  echo "The editable block at the top controls datasets, models, budgets,"
  echo "DAPRO/CRC configurations, Slurm resources, result isolation, and archive output."
}

while (( $# > 0 )); do
  case "$1" in
    --local)
      RUN_MODE="local"
      ;;
    --slurm)
      RUN_MODE="slurm"
      ;;
    --cpu)
      DEVICE="cpu"
      SLURM_GRES=""
      ;;
    --device)
      if (( $# < 2 )); then
        echo "--device requires a value." >&2
        exit 2
      fi
      DEVICE="$2"
      shift
      ;;
    --available-only)
      AVAILABLE_ONLY=1
      ;;
    --dry-run)
      DRY_RUN=1
      ;;
    --parallel-jobs)
      if (( $# < 2 )); then
        echo "--parallel-jobs requires a positive integer." >&2
        exit 2
      fi
      MAX_CONCURRENT_SRUNS="$2"
      shift
      ;;
    --seed-end)
      if (( $# < 2 )); then
        echo "--seed-end requires an integer." >&2
        exit 2
      fi
      SEED_END="$2"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if [[ "$RUN_MODE" != "local" && "$RUN_MODE" != "slurm" ]]; then
  echo "RUN_MODE must be either 'local' or 'slurm'." >&2
  exit 2
fi

if ! [[ "$SEED_START" =~ ^[0-9]+$ && "$SEED_END" =~ ^[0-9]+$ ]]; then
  echo "SEED_START and SEED_END must be nonnegative integers." >&2
  exit 2
fi
if (( SEED_END <= SEED_START )); then
  echo "SEED_END must be larger than SEED_START." >&2
  exit 2
fi
if ! [[ "$MAX_CONCURRENT_SRUNS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_CONCURRENT_SRUNS must be a positive integer." >&2
  exit 2
fi

for dapro_config in "${DAPRO_CONFIGS[@]}"; do
  if [[ ! "$dapro_config" =~ ^([0-9]+):([0-9]+)$ ]]; then
    echo "Invalid DAPRO_CONFIGS entry '$dapro_config'; expected DAPRO_N1:CRC_CONTROL_SIZE." >&2
    exit 2
  fi
  dapro_n1="${BASH_REMATCH[1]}"
  crc_control_size="${BASH_REMATCH[2]}"

  if (( dapro_n1 >= CAL_SIZE )); then
    echo "DAPRO_N1=$dapro_n1 must be smaller than CAL_SIZE=$CAL_SIZE." >&2
    exit 2
  fi
  if (( crc_control_size >= dapro_n1 )); then
    echo "CRC_CONTROL_SIZE=$crc_control_size must be smaller than DAPRO_N1=$dapro_n1." >&2
    exit 2
  fi
done

for budget_per_sample in "${BUDGET_PER_SAMPLE_VALUES[@]}"; do
  if ! [[ "$budget_per_sample" =~ ^[1-9][0-9]*$ ]]; then
    echo "BUDGET_PER_SAMPLE values must be positive integers; got '$budget_per_sample'." >&2
    exit 2
  fi
done

if [[ "$RUN_MODE" == "slurm" && -v SLURM_JOB_ID ]]; then
  echo "Do not start this launcher from inside srun, sbatch, or salloc." >&2
  echo "Run it on the login node so each child srun requests an independent job." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

if [[ "$RUN_MODE" == "slurm" && "$AUTO_EXCLUDE_INCOMPATIBLE_GPUS" == "1" && -z "$EXCLUDE_LIST" ]]; then
  if ! command -v sinfo >/dev/null 2>&1; then
    if (( DRY_RUN == 1 )); then
      echo "sinfo is unavailable; the dry run omits automatic GPU exclusions."
    else
      echo "sinfo is required for automatic Slurm GPU filtering." >&2
      exit 2
    fi
  else
    EXCLUDE_LIST="$(
      sinfo -N -h -o "%n %G" |
        awk '$2 !~ /A100|A40|A6000|6000ADA|L40|L4|A4000/ {print $1}' |
        paste -sd, -
    )"
  fi
fi

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

run_python_module() {
  local module="$1"
  local job_suffix="$2"
  shift 2

  local command=(python -m "$module" "$@")
  if [[ "$RUN_MODE" == "slurm" ]]; then
    local slurm_command=(
      srun
      -A "$SLURM_ACCOUNT"
      -p "$SLURM_PARTITION"
      -c "$SLURM_CPUS"
      -J "${SLURM_JOB_NAME}_${job_suffix}"
    )
    if [[ -n "$SLURM_GRES" ]]; then
      slurm_command+=(--gres="$SLURM_GRES")
    fi
    if [[ -n "$EXCLUDE_LIST" ]]; then
      slurm_command+=(--exclude="$EXCLUDE_LIST")
    fi
    command=("${slurm_command[@]}" "${command[@]}")
  fi

  if (( DRY_RUN == 1 )); then
    print_command "${command[@]}"
  else
    "${command[@]}"
  fi
}

wait_for_job_batch() {
  local failed=0
  local pid
  for pid in "$@"; do
    if ! wait "$pid"; then
      echo "Parallel construction job $pid failed." >&2
      failed=1
    fi
  done
  return "$failed"
}

configure_dataset() {
  local key="$1"
  local target_model="$2"

  case "$key" in
    toxicity)
      DATASET_NAME="dataset_toxicity"
      DATASET_SETUP="attack_toxic_attack_${ATTACKER_MODEL}_lm_target_${target_model}_judge_detoxify"
      ;;
    red_team_qwen)
      DATASET_NAME="dataset_red_team"
      DATASET_SETUP="attack_default_attack_${ATTACKER_MODEL}_lm_target_${target_model}_judge_llm-judge_qwen25_14b_instruct"
      ;;
    red_team_llama_guard)
      DATASET_NAME="dataset_red_team"
      DATASET_SETUP="attack_default_attack_${ATTACKER_MODEL}_lm_target_${target_model}_judge_llama_guard"
      ;;
    hallucination3)
      DATASET_NAME="dataset_hallucination3"
      DATASET_SETUP="attack_hallucination_attack_${ATTACKER_MODEL}_lm_target_${target_model}_judge_llm-judge_qwen25_14b_instruct"
      ;;
    autoif)
      DATASET_NAME="dataset_autoif"
      DATASET_SETUP="attack_autoif_helper_${ATTACKER_MODEL}_lm_target_${target_model}_judge_autoif"
      ;;
    *)
      echo "Unknown dataset key: $key" >&2
      exit 2
      ;;
  esac
}

build_methods() {
  local dapro_n1="$1"
  local crc_control_size="$2"

  # These are the exact calibration names emitted by the Python allocators.
  # The three DAPRO rows are repeated without and with independent CRC budget
  # control. The oracle observes every trajectory and therefore has no budget.
  METHODS=(
    # Raw prediction and infinite-observation reference.
    uncalibrated
    oracle_survival_calibration
    # Static baselines.
    calibration_optimized_allocation
    # Exact constant continuation (no propensity floor) and local adaptation.
    # CRC controls budget; the no-floor constant can still produce large IPW.
    calibration_random_adaptive_optimized_no_terminal_floor_crc_allocation
    calibration_adaptive_optimized_crc_allocation
    # DAPRO without a CRC budget wrapper: legacy, target-A, definitive best.
    "calibration_projected_optimization_direct_bins_2_prob_n1_${dapro_n1}_allocation"
    "calibration_projected_optimization_direct_bins_2_prob_a_target_raw_alpha_0p10_n1_${dapro_n1}_allocation"
    "calibration_dapro_variance_aligned_bins_2_alpha_0p10_global_0p001_projection_margin_1p00_n1_${dapro_n1}_allocation"
    # The same three DAPRO objectives with independent CRC budget control.
    "calibration_projected_optimization_direct_bins_2_prob_budget_crc_control_${crc_control_size}_row_cap_2p00x_budget_n1_${dapro_n1}_allocation"
    "calibration_projected_optimization_direct_bins_2_prob_a_target_raw_alpha_0p10_budget_crc_control_${crc_control_size}_row_cap_2p00x_budget_n1_${dapro_n1}_allocation"
    "calibration_dapro_variance_aligned_bins_2_alpha_0p10_global_0p001_budget_crc_control_${crc_control_size}_row_cap_2p00x_budget_n1_${dapro_n1}_allocation"
    # Full-trajectory Target-A oracle diagnostics. The split variants retain
    # DAPRO's pilot accounting; the global variant optimizes every row.
    "calibration_oracle_target_a_dapro_alpha_0p10_n1_${dapro_n1}_allocation"
    "calibration_oracle_target_a_dapro_alpha_0p10_crc_control_${crc_control_size}_n1_${dapro_n1}_allocation"
    calibration_oracle_target_a_dapro_no_split_alpha_0p10_allocation
  )
  METHOD_CSV="$(IFS=,; echo "${METHODS[*]}")"
}

run_configuration() {
  local dataset_key="$1"
  local target_model="$2"
  local dapro_n1="$3"
  local crc_control_size="$4"
  local budget_per_sample="$5"
  local experiment_suffix="$6"
  local cache_file
  local job_key

  configure_dataset "$dataset_key" "$target_model"
  build_methods "$dapro_n1" "$crc_control_size"

  cache_file="alg_playground_model/is_real_True_dataset_${DATASET_NAME}_dataset_${DATASET_SETUP}/probability_est_cal_test.pt"
  if (( AVAILABLE_ONLY == 1 )) && [[ ! -f "$cache_file" ]]; then
    echo "Skipping $dataset_key / $target_model (missing $cache_file)"
    return
  fi

  job_key="${dataset_key}_${target_model}_n1_${dapro_n1}_crc_${crc_control_size}_b_${budget_per_sample}"
  echo
  echo "[$dataset_key / $target_model | N1=$dapro_n1 | CRC=$crc_control_size | budget=$budget_per_sample] construct LPBs"
  local common_args=(
    --bound-type lpb
    --data-type real
    --dataset-name "$DATASET_NAME"
    --dataset-setup "$DATASET_SETUP"
    --budget-per-sample "$budget_per_sample"
    --cal-size "$CAL_SIZE"
    --tau-prior "$TAU_PRIOR"
    --m-upper-bound "$M_UPPER_BOUND"
    --device "$DEVICE"
    --allocations none
    --experiment-suffix "$experiment_suffix"
    --dapro-n1-values "$dapro_n1"
    --definitive-dapro-margins 1.0
    --calibration-names "$METHOD_CSV"
  )

  if [[ "$RUN_MODE" == "slurm" && "$MAX_CONCURRENT_SRUNS" -gt 1 ]]; then
    local seed
    local seed_end
    local seed_jobs=()
    local seed_count=$((SEED_END - SEED_START))
    echo "Submitting $seed_count independent srun jobs"
    echo "Maximum concurrently active srun commands: $MAX_CONCURRENT_SRUNS"
    for (( seed = SEED_START; seed < SEED_END; seed++ )); do
      seed_end=$((seed + 1))
      run_python_module \
        src.predictive_bounds.construct_calibrated_bound \
        "${job_key}_seed_${seed}_construct" \
        "${common_args[@]}" \
        --seed-start "$seed" \
        --seed-end "$seed_end" &
      seed_jobs+=("$!")
      echo "Submitted seed $seed (launcher PID $!)"

      if (( ${#seed_jobs[@]} == MAX_CONCURRENT_SRUNS )); then
        if ! wait_for_job_batch "${seed_jobs[@]}"; then
          return 1
        fi
        seed_jobs=()
      fi
    done
    if (( ${#seed_jobs[@]} > 0 )); then
      if ! wait_for_job_batch "${seed_jobs[@]}"; then
        return 1
      fi
    fi
  else
    run_python_module \
      src.predictive_bounds.construct_calibrated_bound \
      "${job_key}_construct" \
      "${common_args[@]}" \
      --seed-start "$SEED_START" \
      --seed-end "$SEED_END"
  fi

  echo "[$dataset_key / $target_model | N1=$dapro_n1 | CRC=$crc_control_size | budget=$budget_per_sample] merge LPBs"
  run_python_module \
    src.predictive_bounds.merge_bounds_results \
    "${job_key}_merge" \
    "${common_args[@]}" \
    --seed-start "$SEED_START" \
    --seed-end "$SEED_END"
}

echo "Mode: $RUN_MODE | device: $DEVICE | seeds: [$SEED_START, $SEED_END)"
echo "DAPRO/CRC configurations: ${DAPRO_CONFIGS[*]}"
echo "Budgets per sample: ${BUDGET_PER_SAMPLE_VALUES[*]}"
echo "Base experiment suffix: $BASE_EXPERIMENT_SUFFIX"
if [[ "$RUN_MODE" == "slurm" ]]; then
  echo "Maximum concurrent srun commands: $MAX_CONCURRENT_SRUNS"
fi

# Keep the exact suffixes used in this invocation so archiving only collects
# outputs from the requested experiment matrix.
EXPERIMENT_SUFFIXES=()

for dapro_config in "${DAPRO_CONFIGS[@]}"; do
  IFS=: read -r dapro_n1 crc_control_size <<< "$dapro_config"
  for budget_per_sample in "${BUDGET_PER_SAMPLE_VALUES[@]}"; do
    experiment_suffix="${BASE_EXPERIMENT_SUFFIX}_n1_${dapro_n1}_crc_${crc_control_size}_budget_${budget_per_sample}"
    EXPERIMENT_SUFFIXES+=("$experiment_suffix")

    for dataset_key in "${DATASETS[@]}"; do
      for target_model in "${TARGET_MODELS[@]}"; do
        run_configuration \
          "$dataset_key" \
          "$target_model" \
          "$dapro_n1" \
          "$crc_control_size" \
          "$budget_per_sample" \
          "$experiment_suffix"
      done
    done
  done
done

if (( DRY_RUN == 1 )); then
  echo
  echo "Dry run complete; no result archive was created."
  exit 0
fi

MERGED_FILES=()
for experiment_suffix in "${EXPERIMENT_SUFFIXES[@]}"; do
  while IFS= read -r merged_file; do
    MERGED_FILES+=("$merged_file")
  done < <(
    find results/merged_calibration_dfs \
      -type f \
      -path "*__${experiment_suffix}/all_df.csv" \
      -print 2>/dev/null | sort
  )
done

if (( ${#MERGED_FILES[@]} == 0 )); then
  echo "No merged CSV files were found for the requested experiment matrix." >&2
  exit 1
fi

mkdir -p "$(dirname "$ARCHIVE_PATH")"
tar -czf "$ARCHIVE_PATH" "${MERGED_FILES[@]}"
echo
echo "Archived ${#MERGED_FILES[@]} merged CSV files at $ARCHIVE_PATH"