#!/usr/bin/env bash
# Run and merge the production fixed-benchmark metric comparison.
#
# The registry contains only:
#   weighted/unweighted Uniform, naive Static, Constant+CRC,
#   Metric-optimal PMF without CRC, Generalized DAPRO with/without CRC,
#   full-budget calibration, and full-budget calibration+test.
#
# Parallelism is across complete experiment configurations, not across seeds.
# Every estimate.py invocation processes the full configured seed range.
#
# Examples:
#   bash src/evaluation/scripts/run.sh --local --cpu --available-only
#   bash src/evaluation/scripts/run.sh --slurm --parallel-jobs 10
#   bash src/evaluation/scripts/run.sh --slurm --parallel-jobs 20
#   bash src/evaluation/scripts/run.sh --slurm --parallel-jobs 50
#   bash src/evaluation/scripts/run.sh --slurm --dry-run
set -euo pipefail

# ======================== EDITABLE CONFIGURATION ========================
RUN_MODE="local"                    # "local" or "slurm"
DEVICE="cuda:0"
PYTHON_BIN="${PYTHON_BIN:-python}"
AVAILABLE_ONLY=0
DRY_RUN=0
GENERATE_FIGURES=0

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
TAU_PRIOR=0.56
EXPERIMENT_SUFFIX="metric_v1"

# Each entry is DAPRO_N1:CRC_CONTROL_SIZE.
DAPRO_CONFIGS=(
  "200:100"
  "100:50"
  "50:25"
)

# Every budget is run for every DAPRO/CRC configuration.
BUDGET_PER_SAMPLE_VALUES=(
  20
  10
  5
)

SLURM_ACCOUNT="galileo"
SLURM_PARTITION="galileo"
SLURM_CPUS=4
SLURM_GRES="gpu:1"
SLURM_JOB_NAME="metricEst"

# Number of COMPLETE experiment configurations allowed to run concurrently.
# Each one independently runs estimate over [SEED_START, SEED_END), then merge.
MAX_CONCURRENT_EXPERIMENTS=50

EXCLUDE_LIST=""
# ====================== END EDITABLE CONFIGURATION ======================

usage() {
  echo "Usage: $0 [--local | --slurm] [--cpu | --device DEVICE]"
  echo "          [--available-only] [--parallel-jobs N] [--seed-end N]"
  echo "          [--figures | --skip-figures] [--dry-run]"
  echo
  echo "--parallel-jobs controls the number of complete experiment configurations"
  echo "that run concurrently. It does not split seeds into separate jobs."
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
    --skip-figures)
      GENERATE_FIGURES=0
      ;;
    --figures)
      GENERATE_FIGURES=1
      ;;
    --dry-run)
      DRY_RUN=1
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
  echo "RUN_MODE must be local or slurm." >&2
  exit 2
fi

if ! [[ "$SEED_START" =~ ^[0-9]+$ && "$SEED_END" =~ ^[0-9]+$ ]] ||
   (( SEED_END <= SEED_START )); then
  echo "Seed range must be nonnegative and nonempty." >&2
  exit 2
fi

if ! [[ "$MAX_CONCURRENT_EXPERIMENTS" =~ ^[1-9][0-9]*$ ]]; then
  echo "--parallel-jobs must be a positive integer." >&2
  exit 2
fi

for config in "${DAPRO_CONFIGS[@]}"; do
  if [[ ! "$config" =~ ^([0-9]+):([0-9]+)$ ]]; then
    echo "Invalid DAPRO configuration: $config" >&2
    exit 2
  fi

  dapro_n1="${BASH_REMATCH[1]}"
  crc_control_size="${BASH_REMATCH[2]}"

  if (( crc_control_size >= dapro_n1 || dapro_n1 >= CAL_SIZE )); then
    echo "Require CRC_CONTROL_SIZE < DAPRO_N1 < CAL_SIZE: $config" >&2
    exit 2
  fi
  if (( crc_control_size != dapro_n1 / 2 )); then
    echo "Require CRC_CONTROL_SIZE = DAPRO_N1 // 2: $config" >&2
    exit 2
  fi
done

for budget in "${BUDGET_PER_SAMPLE_VALUES[@]}"; do
  if ! [[ "$budget" =~ ^[1-9][0-9]*$ ]]; then
    echo "BUDGET_PER_SAMPLE values must be positive integers; got '$budget'." >&2
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
      srun
      -A "$SLURM_ACCOUNT"
      -p "$SLURM_PARTITION"
      -c "$SLURM_CPUS"
      -J "${SLURM_JOB_NAME}_${job_suffix}"
    )

    if [[ -n "$SLURM_GRES" ]]; then
      prefix+=(--gres="$SLURM_GRES")
    fi
    if [[ -n "$EXCLUDE_LIST" ]]; then
      prefix+=(--exclude="$EXCLUDE_LIST")
    fi

    command=("${prefix[@]}" "${command[@]}")
  fi

  if (( DRY_RUN == 1 )); then
    print_command "${command[@]}"
  else
    "${command[@]}"
  fi
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
    *)
      echo "Unknown dataset key: $key" >&2
      return 2
      ;;
  esac
}

run_configuration() {
  local dataset="$1"
  local target="$2"
  local budget="$3"

  configure_dataset "$dataset" "$target" || return 1

  local cache
  local key
  cache="alg_playground_model/is_real_True_dataset_${DATASET_NAME}_dataset_${DATASET_SETUP}/probability_est_cal_test.pt"

  if (( AVAILABLE_ONLY == 1 )) && [[ ! -f "$cache" ]]; then
    echo "Skipping $dataset / $target: missing $cache"
    return 3
  fi

  local shared_args=(
    --data-type real
    --dataset-name "$DATASET_NAME"
    --dataset-setup "$DATASET_SETUP"
    --budget-per-sample "$budget"
    --cal-size "$CAL_SIZE"
    --tau-prior "$TAU_PRIOR"
    --device "$DEVICE"
    --experiment-suffix "$EXPERIMENT_SUFFIX"
  )

  key="${dataset}_${target}_b_${budget}"
  local config
  local dapro_n1
  local crc_control_size
  local merge_config_args=()
  local failed_dapro_configs=()

  # All N1/CRC methods share one compact experiment directory. Run the
  # configurations sequentially inside this job so common baseline files are
  # never written concurrently, then merge the complete method set once.
  for config in "${DAPRO_CONFIGS[@]}"; do
    IFS=: read -r dapro_n1 crc_control_size <<< "$config"
    echo
    echo "[$key | N1=$dapro_n1 | CRC=$crc_control_size] estimate seeds [$SEED_START, $SEED_END)"
    if run_module \
      src.evaluation.estimate \
      "${key}_n1_${dapro_n1}_estimate" \
      "${shared_args[@]}" \
      --dapro-n1 "$dapro_n1" \
      --crc-control-size "$crc_control_size" \
      --seed-start "$SEED_START" \
      --seed-end "$SEED_END"; then
      merge_config_args+=(--dapro-config "$config")
    else
      failed_dapro_configs+=("$config")
      echo "WARNING: estimate failed for $key / N1=$dapro_n1 / CRC=$crc_control_size; continuing with the next N1." >&2
    fi
  done

  if (( ${#merge_config_args[@]} == 0 )); then
    echo "ERROR: every N1/CRC configuration failed for $key." >&2
    return 1
  fi

  echo "[$key] merge"

  if ! run_module \
    src.evaluation.merge_results \
    "${key}_merge" \
    "${shared_args[@]}" \
    "${merge_config_args[@]}" \
    --seed-start "$SEED_START" \
    --seed-end "$SEED_END"; then
    echo "ERROR: merge failed for $key; continuing with the remaining experiments." >&2
    return 1
  fi

  if (( ${#failed_dapro_configs[@]} > 0 )); then
    echo "[$key] merged the successful configurations; failed: ${failed_dapro_configs[*]}"
  else
    echo "[$key] completed successfully"
  fi
  return 0
}

# Wait for a batch of complete experiment configurations.
#
# Exit codes from run_configuration:
#   0 = estimate + merge succeeded
#   3 = skipped because AVAILABLE_ONLY cache was missing
#   other = failed
#
# Successful experiment names are retained for figure generation.
wait_configuration_batch() {
  local -n pids_ref="$1"
  local -n labels_ref="$2"
  local -n names_ref="$3"

  local i
  local status
  local label
  local experiment_name

  for i in "${!pids_ref[@]}"; do
    label="${labels_ref[$i]}"
    experiment_name="${names_ref[$i]}"

    if wait "${pids_ref[$i]}"; then
      SUCCESSFUL_EXPERIMENT_NAMES+=("$experiment_name")
    else
      status=$?
      if (( status == 3 )); then
        SKIPPED_CONFIGURATIONS+=("$label")
      else
        FAILED_CONFIGURATIONS+=("$label")
        echo "WARNING: $label failed; continuing." >&2
      fi
    fi
  done

  pids_ref=()
  labels_ref=()
  names_ref=()
}

echo "Mode: $RUN_MODE | device: $DEVICE | seeds: [$SEED_START, $SEED_END)"
echo "DAPRO/CRC configurations: ${DAPRO_CONFIGS[*]}"
echo "Budgets per sample: ${BUDGET_PER_SAMPLE_VALUES[*]}"
echo "Maximum parallel complete experiments: $MAX_CONCURRENT_EXPERIMENTS"
echo "Experiment suffix: $EXPERIMENT_SUFFIX"
echo "Figures during server run: $GENERATE_FIGURES"

CONFIG_PIDS=()
CONFIG_LABELS=()
CONFIG_EXPERIMENT_NAMES=()

SUCCESSFUL_EXPERIMENT_NAMES=()
FAILED_CONFIGURATIONS=()
SKIPPED_CONFIGURATIONS=()

for budget in "${BUDGET_PER_SAMPLE_VALUES[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for target in "${TARGET_MODELS[@]}"; do
      configure_dataset "$dataset" "$target"

      experiment_name="${DATASET_NAME}_${DATASET_SETUP}_${budget}_metric_estimation_${EXPERIMENT_SUFFIX}"
      config_label="${dataset}/${target}/budget=${budget}/all-dapro-configs"

      # Parallelism is across complete dataset/model/budget experiments.
      # Each job runs every N1/CRC configuration, then performs one merge.
      if [[ "$RUN_MODE" == "slurm" && "$MAX_CONCURRENT_EXPERIMENTS" -gt 1 ]]; then
        run_configuration \
          "$dataset" \
          "$target" \
          "$budget" &

        CONFIG_PIDS+=("$!")
        CONFIG_LABELS+=("$config_label")
        CONFIG_EXPERIMENT_NAMES+=("$experiment_name")

        echo "Submitted full-seed experiment: $config_label (launcher PID $!)"

        if (( ${#CONFIG_PIDS[@]} >= MAX_CONCURRENT_EXPERIMENTS )); then
          wait_configuration_batch \
            CONFIG_PIDS \
            CONFIG_LABELS \
            CONFIG_EXPERIMENT_NAMES
        fi
      else
        if run_configuration \
          "$dataset" \
          "$target" \
          "$budget"; then
          SUCCESSFUL_EXPERIMENT_NAMES+=("$experiment_name")
        else
          status=$?
          if (( status == 3 )); then
            SKIPPED_CONFIGURATIONS+=("$config_label")
          else
            FAILED_CONFIGURATIONS+=("$config_label")
            echo "WARNING: $config_label failed; continuing." >&2
          fi
        fi
      fi
    done
  done
done

# Wait for the final partial batch of Slurm jobs.
if (( ${#CONFIG_PIDS[@]} > 0 )); then
  wait_configuration_batch \
    CONFIG_PIDS \
    CONFIG_LABELS \
    CONFIG_EXPERIMENT_NAMES
fi

if (( GENERATE_FIGURES == 1 && ${#SUCCESSFUL_EXPERIMENT_NAMES[@]} > 0 )); then
  plot_command=("$PYTHON_BIN" -m src.evaluation.summarize)

  for experiment_name in "${SUCCESSFUL_EXPERIMENT_NAMES[@]}"; do
    plot_command+=(--experiment "$experiment_name")
  done

  echo
  echo "Generate figures from ${#SUCCESSFUL_EXPERIMENT_NAMES[@]} successful experiment(s)"

  if (( DRY_RUN == 1 )); then
    print_command "${plot_command[@]}"
  elif ! "${plot_command[@]}"; then
    echo "WARNING: figure generation failed; experiment launcher still completed." >&2
  fi
elif (( GENERATE_FIGURES == 1 )); then
  echo
  echo "No successful experiments available for figure generation." >&2
fi

echo
echo "Completed experiment launcher."
echo "Successful configurations: ${#SUCCESSFUL_EXPERIMENT_NAMES[@]}"
echo "Skipped configurations:    ${#SKIPPED_CONFIGURATIONS[@]}"
echo "Failed configurations:     ${#FAILED_CONFIGURATIONS[@]}"

if (( ${#FAILED_CONFIGURATIONS[@]} > 0 )); then
  echo
  echo "Failed configurations:" >&2
  for label in "${FAILED_CONFIGURATIONS[@]}"; do
    echo "  - $label" >&2
  done
fi

if (( ${#SKIPPED_CONFIGURATIONS[@]} > 0 )); then
  echo
  echo "Skipped configurations:"
  for label in "${SKIPPED_CONFIGURATIONS[@]}"; do
    echo "  - $label"
  done
fi

echo
echo "Merged CSVs are under: results/merged_metric_calibration_dfs"
echo "After downloading that directory, generate local figures with:"
echo "  python -m src.evaluation.summarize --input-dir results/merged_metric_calibration_dfs --output-dir figures/metric_estimation --quality high --experiment-suffix $EXPERIMENT_SUFFIX"

# Intentionally succeed even if individual Python experiments failed.
# Failures are reported above and do not crash the overall launcher.
exit 0
