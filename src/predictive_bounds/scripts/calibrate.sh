#!/usr/bin/env bash
# Construct and merge predictive bounds for every configured dataset.
#
# LPB experiment matrix:
#   (DAPRO_N1=200, CRC_CONTROL_SIZE=100) x BUDGET_PER_SAMPLE={5,10,20}
#   (DAPRO_N1=100, CRC_CONTROL_SIZE=50)  x BUDGET_PER_SAMPLE={5,10,20}
#   (DAPRO_N1=50,  CRC_CONTROL_SIZE=25)  x BUDGET_PER_SAMPLE={5,10,20}
# The only DAPRO family run is Soft-prefix Generalized DAPRO, with and without
# CRC.  The UPB specialization is model-only, so it has no N1 axis and runs
# once with CRC_CONTROL_SIZE=100 at each budget.  Its CRC deliberately uses the
# full 200-turn support bound and no shared-PAV row cap.  UPB value 201 is
# infinity/no event through turn 200.
#
# Typical invocations:
#   bash src/predictive_bounds/scripts/calibrate.sh --local
#   bash src/predictive_bounds/scripts/calibrate.sh --local --cpu --available-only
#   bash src/predictive_bounds/scripts/calibrate.sh --slurm
#  bash src/predictive_bounds/scripts/calibrate.sh --slurm --parallel-jobs 20
# #   bash src/predictive_bounds/scripts/calibrate.sh --slurm --dry-run
#
# Edit only the configuration block below to change the experiment matrix.
set -euo pipefail

# ======================== EDITABLE CONFIGURATION ========================
RUN_MODE="local"                    # "local" or "slurm"
BOUND_TYPE="upb"     # "upb" (default) or "lpb"
DEVICE="cuda:0"                     # Use "cpu" when no GPU is available.
AVAILABLE_ONLY=0                    # 1 skips configurations without cached predictions.
DRY_RUN=0                           # 1 prints commands without executing them.

DATASETS=(
  toxicity
  red_team_qwen
  red_team_llama_guard
  hallucination3
  autoif
)

TARGET_MODELS=(
  qwen25_14b_instruct
  llama_31_8B_instruct
  mini_phi_4_instruct
  gemma3_4b_it
)

ATTACKER_MODEL="qwen25_14b_instruct"
CAL_SIZE=3000
SEED_START=0
SEED_END=50
TAU_PRIOR="${TAU_PRIOR:-}"
M_UPPER_BOUND=200

# Each entry is DAPRO_N1:CRC_CONTROL_SIZE.
DAPRO_CONFIGS=(
  "200:100"
  "100:50"
  "50:25"
)

# Every budget is run for every DAPRO/CRC configuration above.
BUDGET_PER_SAMPLE_VALUES=(
  5
  10
  20
)

# All N1/CRC methods for one dataset/model/budget share one compact directory
# and are constructed and merged together.
BASE_EXPERIMENT_SUFFIX="${BASE_EXPERIMENT_SUFFIX:-}"
ARCHIVE_PATH=""

SLURM_ACCOUNT="galileo"
SLURM_PARTITION="galileo"
SLURM_CPUS=4
SLURM_GRES="gpu:1"                 # One GPU; Python addresses it as cuda:0.
SLURM_JOB_NAME="plsNoKil"
MAX_CONCURRENT_EXPERIMENTS=20       # Parallel full-seed experiment configurations (e.g. 10, 20, 50).
EXCLUDE_LIST=""                    # Leave empty to use the automatic GPU filter.
AUTO_EXCLUDE_INCOMPATIBLE_GPUS=1
# ====================== END EDITABLE CONFIGURATION ======================

usage() {
  echo "Usage: $0 [--local | --slurm] [--cpu | --device DEVICE]"
  echo "          [--bound-type upb|lpb]"
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
    --bound-type)
      if (( $# < 2 )); then
        echo "--bound-type requires upb or lpb." >&2
        exit 2
      fi
      BOUND_TYPE="$2"
      shift
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
      MAX_CONCURRENT_EXPERIMENTS="$2"
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

if [[ "$BOUND_TYPE" != "lpb" && "$BOUND_TYPE" != "upb" ]]; then
  echo "BOUND_TYPE must be either 'lpb' or 'upb'." >&2
  exit 2
fi
if [[ -z "$TAU_PRIOR" ]]; then
  if [[ "$BOUND_TYPE" == "upb" ]]; then
    TAU_PRIOR=0.97
  else
    TAU_PRIOR=0.56
  fi
fi
if [[ -z "$BASE_EXPERIMENT_SUFFIX" ]]; then
  if [[ "$BOUND_TYPE" == "upb" ]]; then
    BASE_EXPERIMENT_SUFFIX="upb_v2_soft_residual_aht"
  else
    BASE_EXPERIMENT_SUFFIX="lpb_v5_soft_prefix"
  fi
fi
ARCHIVE_PATH="results/${BOUND_TYPE}_merged_${BASE_EXPERIMENT_SUFFIX}.tar.gz"

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
if ! [[ "$MAX_CONCURRENT_EXPERIMENTS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_CONCURRENT_EXPERIMENTS must be a positive integer." >&2
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



wait_for_configuration_batch() {
  local -n pids_ref="$1"
  local -n labels_ref="$2"
  local i
  local failed_label

  for i in "${!pids_ref[@]}"; do
    if ! wait "${pids_ref[$i]}"; then
      failed_label="${labels_ref[$i]}"
      FAILED_CONFIGURATIONS+=("$failed_label")
      echo "WARNING: $failed_label failed; continuing with the remaining experiments." >&2
    fi
  done

  pids_ref=()
  labels_ref=()
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
  local oracle_name
  if [[ "$BOUND_TYPE" == "upb" ]]; then
    oracle_name="oracle_survival_upb_calibration"
  else
    oracle_name="oracle_survival_calibration"
  fi
  METHODS=(
    # Raw prediction and infinite-observation reference.
    uncalibrated
    "$oracle_name"
    # Static baselines.
    calibration_optimized_allocation
    # Constant continuation with an always-follow mixture that bounds IPW;
    # CRC accounts for the mixture while controlling the expected budget.
    calibration_random_adaptive_optimized_mixture_terminal_floor_0p005_crc_allocation
    calibration_random_schedule_power_reach_alpha_2_crc_allocation
  )

  DAPRO_N1_ARGS=()
  local dapro_config dapro_n1 crc_control_size
  if [[ "$BOUND_TYPE" == "upb" ]]; then
    # Compatibility value consumed by the Python constructor; the UPB policy
    # fits no labels, and only the independent CRC control size is operative.
    DAPRO_N1_ARGS=(200)
    METHODS+=(
      "calibration_dapro_soft_prefix_bins_2_upb_residual_aht_coverage_0p70_model_anchor_global_0p001_projection_margin_1p00_allocation"
      "calibration_dapro_soft_prefix_bins_2_upb_residual_aht_coverage_0p70_model_anchor_global_0p001_budget_crc_control_100_allocation"
    )
    METHOD_CSV="$(IFS=,; echo "${METHODS[*]}")"
    return
  fi
  for dapro_config in "${DAPRO_CONFIGS[@]}"; do
    IFS=: read -r dapro_n1 crc_control_size <<< "$dapro_config"
    DAPRO_N1_ARGS+=("$dapro_n1")
    METHODS+=(
      "calibration_dapro_soft_prefix_bins_2_lpb_alpha_0p10_global_0p001_projection_margin_1p00_n1_${dapro_n1}_allocation"
      "calibration_dapro_soft_prefix_bins_2_lpb_alpha_0p10_global_0p001_budget_crc_control_${crc_control_size}_row_cap_2p00x_budget_causal_shared_pav_v1_n1_${dapro_n1}_allocation"
    )
  done
  METHOD_CSV="$(IFS=,; echo "${METHODS[*]}")"
}

run_configuration() {
  local dataset_key="$1"
  local target_model="$2"
  local budget_per_sample="$3"
  local experiment_suffix="$4"
  local cache_file
  local job_key

  configure_dataset "$dataset_key" "$target_model"
  build_methods

  cache_file="alg_playground_model/is_real_True_dataset_${DATASET_NAME}_dataset_${DATASET_SETUP}/probability_est_cal_test.pt"
  if (( AVAILABLE_ONLY == 1 )) && [[ ! -f "$cache_file" ]]; then
    echo "Skipping $dataset_key / $target_model (missing $cache_file)"
    return
  fi

  job_key="${dataset_key}_${target_model}_b_${budget_per_sample}"
  echo
  echo "[$dataset_key / $target_model | all DAPRO configs | budget=$budget_per_sample] construct ${BOUND_TYPE^^}s"
  local common_args=(
    --bound-type "$BOUND_TYPE"
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
    --dapro-n1-values "${DAPRO_N1_ARGS[@]}"
    --definitive-dapro-margins 1.0
    --calibration-names "$METHOD_CSV"
  )

  # Run the complete seed range in one Python process / one srun.
  # With SEED_START=0 and SEED_END=50, the Python script handles seeds 0,...,49.
  if ! run_python_module \
    src.predictive_bounds.construct_calibrated_bound \
    "${job_key}_construct" \
    "${common_args[@]}" \
    --seed-start "$SEED_START" \
    --seed-end "$SEED_END"; then
    echo "ERROR: construction failed for $job_key; skipping merge and continuing." >&2
    return 1
  fi

  echo "[$dataset_key / $target_model | all DAPRO configs | budget=$budget_per_sample] merge ${BOUND_TYPE^^}s"
  if ! run_python_module \
    src.predictive_bounds.merge_bounds_results \
    "${job_key}_merge" \
    "${common_args[@]}" \
    --seed-start "$SEED_START" \
    --seed-end "$SEED_END"; then
    echo "ERROR: merge failed for $job_key; continuing with the next configuration." >&2
    return 1
  fi

  return 0
}

echo "Mode: $RUN_MODE | device: $DEVICE | seeds: [$SEED_START, $SEED_END)"
if [[ "$BOUND_TYPE" == "upb" ]]; then
  echo "DAPRO/CRC configuration: model-only UPB target, CRC control=100"
else
  echo "DAPRO/CRC configurations: ${DAPRO_CONFIGS[*]}"
fi
echo "Budgets per sample: ${BUDGET_PER_SAMPLE_VALUES[*]}"
echo "Base experiment suffix: $BASE_EXPERIMENT_SUFFIX"

FAILED_CONFIGURATIONS=()
CONFIG_PIDS=()
CONFIG_LABELS=()

echo "Maximum parallel full-seed experiment configurations: $MAX_CONCURRENT_EXPERIMENTS"

for budget_per_sample in "${BUDGET_PER_SAMPLE_VALUES[@]}"; do
  experiment_suffix="${BASE_EXPERIMENT_SUFFIX}"
  for dataset_key in "${DATASETS[@]}"; do
    for target_model in "${TARGET_MODELS[@]}"; do
      config_label="${dataset_key}/${target_model}/budget=${budget_per_sample}/all-dapro-configs"

      # Each background job constructs and merges all N1/CRC methods for one
      # dataset/model/budget and the complete seed range.
      run_configuration \
        "$dataset_key" \
        "$target_model" \
        "$budget_per_sample" \
        "$experiment_suffix" &

      CONFIG_PIDS+=("$!")
      CONFIG_LABELS+=("$config_label")
      echo "Submitted full-seed experiment: $config_label (launcher PID $!)"

      # Run experiments in batches of at most MAX_CONCURRENT_EXPERIMENTS.
      if (( ${#CONFIG_PIDS[@]} >= MAX_CONCURRENT_EXPERIMENTS )); then
        wait_for_configuration_batch CONFIG_PIDS CONFIG_LABELS
      fi
    done
  done
done

# Wait for the final partial batch.
if (( ${#CONFIG_PIDS[@]} > 0 )); then
  wait_for_configuration_batch CONFIG_PIDS CONFIG_LABELS
fi

if (( DRY_RUN == 1 )); then
  echo
  echo "Dry run complete; no result archive was created."
  exit 0
fi

MERGED_FILES=()
if [[ "$BOUND_TYPE" == "upb" ]]; then
  MERGED_RESULTS_DIR="results/merged_upb_calibration_dfs"
else
  MERGED_RESULTS_DIR="results/merged_calibration_dfs"
fi
while IFS= read -r merged_file; do
  MERGED_FILES+=("$merged_file")
done < <(
  find "$MERGED_RESULTS_DIR" \
    -type f \
    -path "*_${BASE_EXPERIMENT_SUFFIX}/all_df.csv" \
    -print 2>/dev/null | sort
)

if (( ${#MERGED_FILES[@]} == 0 )); then
  echo "No merged CSV files were found for the requested experiment matrix." >&2
  exit 1
fi

mkdir -p "$(dirname "$ARCHIVE_PATH")"
tar -czf "$ARCHIVE_PATH" "${MERGED_FILES[@]}"
echo
echo "Archived ${#MERGED_FILES[@]} merged CSV files at $ARCHIVE_PATH"

if (( ${#FAILED_CONFIGURATIONS[@]} > 0 )); then
  echo
  echo "Completed with ${#FAILED_CONFIGURATIONS[@]} failed configuration(s):" >&2
  for failed_key in "${FAILED_CONFIGURATIONS[@]}"; do
    echo "  - $failed_key" >&2
  done
else
  echo "All configurations completed successfully."
fi
