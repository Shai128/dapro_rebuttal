#!/usr/bin/env bash
# LPB and metric ablations for Generalized DAPRO on Toxicity/Qwen.
#
# Examples:
#   bash src/predictive_bounds/experiments/dapro_ablation/scripts/run.sh --local --cpu
#   bash src/predictive_bounds/experiments/dapro_ablation/scripts/run.sh --slurm --parallel-jobs 8 --cpu
#   bash src/predictive_bounds/experiments/dapro_ablation/scripts/run.sh --slurm --parallel-jobs 8
#
# Each configuration runs 50 paired calibration/test splits, then merges its
# own rows. Every non-shift ablation is run for both LPB construction and
# event-rate metric estimation. A failed configuration is reported without
# cancelling the rest.
# Canonical cells use soft-prefix coefficients, current hazard, K=2, global
# regularization 0.001, raw margin 0, and the same capped CRC controller.  The
# representation study alone varies K; the Cmax study alone varies the CRC
# row-cost cap; every other study changes only its named x-axis factor.
set -uo pipefail

# ======================== EDITABLE CONFIGURATION ========================
RUN_MODE="local"                 # local or slurm
DEVICE="cuda:0"
PARALLEL_JOBS=8
DRY_RUN=0
AVAILABLE_ONLY=0
SEED_START=0
SEED_END=50
CAL_SIZE=3000
TAU_PRIOR=0.56
M_UPPER_BOUND=200
BASE_EXPERIMENT_SUFFIX="lpb_ablv1"
METRIC_EXPERIMENT_SUFFIX="m_ablv1"
PYTHON_BIN="${PYTHON_BIN:-python}"

N1_VALUES=(50 100 200 300 400)
SCORE_REFERENCE_N1=50
METRIC_DAPRO_N1=50
METRIC_CRC_CONTROL_SIZE=25
ATTACKER_SHIFT_BUDGET=10
SCORE_NOISE_LAMBDAS=(0 0.1 0.25 0.5 0.75 1)
BUDGET_VALUES=(5 10 20 30 40 50)
# Matching entries: lower budgets receive smaller Phase-I samples.
BUDGET_N1_VALUES=(25 50 50 100 150 200)

DATASET_NAME="dataset_toxicity"
DATASET_SETUP="attack_toxic_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_detoxify"

RED_SHIFT_GEMMA="attack_default_attack_gemma3_12b_it_lm_target_llama_31_8B_instruct_judge_llm-judge_gemma3_12b_it"
RED_SHIFT_QWEN="attack_default_attack_qwen25_14b_instruct_lm_target_llama_31_8B_instruct_judge_llm-judge_qwen25_14b_instruct"
TOX_SHIFT_GEMMA="attack_toxic_attack_gemma3_12b_it_lm_target_llama_31_8B_instruct_judge_detoxify"
TOX_SHIFT_QWEN="attack_toxic_attack_qwen25_14b_instruct_lm_target_llama_31_8B_instruct_judge_detoxify"

SLURM_ACCOUNT="galileo"
SLURM_PARTITION="galileo"
SLURM_CPUS=4
SLURM_GRES="gpu:1"
SLURM_JOB_NAME="daproAbl"
# ====================== END EDITABLE CONFIGURATION ======================

usage() {
  echo "Usage: $0 [--local|--slurm] [--cpu|--device DEVICE]"
  echo "          [--parallel-jobs N] [--seed-start N] [--seed-end N]"
  echo "          [--experiment-suffix NAME] [--metric-experiment-suffix NAME]"
  echo "          [--available-only] [--dry-run]"
}

while (( $# > 0 )); do
  case "$1" in
    --local) RUN_MODE="local" ;;
    --slurm) RUN_MODE="slurm" ;;
    --cpu) DEVICE="cpu"; SLURM_GRES="" ;;
    --device) DEVICE="$2"; shift ;;
    --parallel-jobs) PARALLEL_JOBS="$2"; shift ;;
    --seed-start) SEED_START="$2"; shift ;;
    --seed-end) SEED_END="$2"; shift ;;
    --experiment-suffix) BASE_EXPERIMENT_SUFFIX="$2"; shift ;;
    --metric-experiment-suffix) METRIC_EXPERIMENT_SUFFIX="$2"; shift ;;
    --available-only) AVAILABLE_ONLY=1 ;;
    --dry-run) DRY_RUN=1 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if [[ "$RUN_MODE" != "local" && "$RUN_MODE" != "slurm" ]]; then
  echo "RUN_MODE must be local or slurm." >&2; exit 2
fi
if ! [[ "$PARALLEL_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "--parallel-jobs must be a positive integer." >&2; exit 2
fi
if (( ${#BUDGET_VALUES[@]} != ${#BUDGET_N1_VALUES[@]} )); then
  echo "BUDGET_VALUES and BUDGET_N1_VALUES must have equal length." >&2; exit 2
fi
if [[ "$RUN_MODE" == "slurm" && -v SLURM_JOB_ID ]]; then
  echo "Start this launcher on a login node, not inside an existing Slurm job." >&2
  exit 2
fi

SCRIPT_PATH="$0"
if [[ "$SCRIPT_PATH" != */* ]]; then SCRIPT_PATH="./$SCRIPT_PATH"; fi
SCRIPT_DIR="$(cd "${SCRIPT_PATH%/*}" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
cd "$REPO_ROOT"

CACHE_FILE="alg_playground_model/is_real_True_dataset_${DATASET_NAME}_dataset_${DATASET_SETUP}/probability_est_cal_test.pt"
if (( AVAILABLE_ONLY == 1 )) && [[ ! -f "$CACHE_FILE" ]]; then
  echo "Missing cached predictions: $CACHE_FILE" >&2
  exit 0
fi

print_command() { printf '  '; printf '%q ' "$@"; printf '\n'; }

run_module() {
  local job_name="$1"; shift
  local command=("$PYTHON_BIN" -m "$@")
  if [[ "$RUN_MODE" == "slurm" ]]; then
    local prefix=(srun -A "$SLURM_ACCOUNT" -p "$SLURM_PARTITION" -c "$SLURM_CPUS" -J "${SLURM_JOB_NAME}_${job_name}")
    if [[ -n "$SLURM_GRES" ]]; then prefix+=(--gres="$SLURM_GRES"); fi
    command=("${prefix[@]}" "${command[@]}")
  fi
  if (( DRY_RUN == 1 )); then print_command "${command[@]}"; else "${command[@]}"; fi
}

run_metric_configuration() {
  local kind="$1" budget="$2" n1="$3" control="$4" suffix="$5"
  local -a noise_args common
  noise_args=("${SCORE_NOISE_LAMBDAS[@]}")
  common=(
    --data-type real
    --dataset-name "$DATASET_NAME" --dataset-setup "$DATASET_SETUP"
    --budget-per-sample "$budget" --cal-size "$CAL_SIZE"
    --tau-prior "$TAU_PRIOR" --device "$DEVICE"
    --experiment-suffix "$suffix" --method-suite dapro_ablation
    --dapro-ablation-kind "$kind"
    --dapro-n1 "$n1"
    --crc-control-size "$control"
    --score-noise-lambdas "${noise_args[@]}"
    --score-noise-seed 314159
    --seed-start "$SEED_START" --seed-end "$SEED_END"
  )
  echo "[metric $kind | B=$budget | N1=$n1 | CRC=$control] estimating event rate"
  if ! run_module "metric_${kind}_b${budget}_n${n1}_estimate" src.evaluation.estimate "${common[@]}"; then
    echo "ERROR: metric estimation failed for $kind; merge skipped." >&2
    return 1
  fi
  if ! run_module "metric_${kind}_b${budget}_n${n1}_merge" src.evaluation.merge_results "${common[@]}"; then
    echo "ERROR: metric merge failed for $kind." >&2
    return 1
  fi
}

run_configuration() {
  local kind="$1" budget="$2" n1_csv="$3" suffix="$4"
  local -a n1_args noise_args common
  IFS=',' read -r -a n1_args <<< "$n1_csv"
  noise_args=("${SCORE_NOISE_LAMBDAS[@]}")
  common=(
    --bound-type lpb --data-type real
    --dataset-name "$DATASET_NAME" --dataset-setup "$DATASET_SETUP"
    --budget-per-sample "$budget" --cal-size "$CAL_SIZE"
    --tau-prior "$TAU_PRIOR" --m-upper-bound "$M_UPPER_BOUND"
    --device "$DEVICE" --allocations none
    --experiment-suffix "$suffix" --method-suite dapro_ablation
    --dapro-ablation-kind "$kind" --dapro-n1-values "${n1_args[@]}"
    --score-noise-lambdas "${noise_args[@]}" --score-noise-seed 314159
    --target-coverages 0.90 --definitive-dapro-margins 0.0
    --seed-start "$SEED_START" --seed-end "$SEED_END"
  )
  echo "[$kind | B=$budget | N1=${n1_args[*]}] constructing 50-split LPB ablation"
  if ! run_module "${kind}_b${budget}_construct" src.predictive_bounds.construct_calibrated_bound "${common[@]}"; then
    echo "ERROR: construction failed for $kind/B=$budget; merge skipped." >&2
    return 1
  fi
  if ! run_module "${kind}_b${budget}_merge" src.predictive_bounds.merge_bounds_results "${common[@]}"; then
    echo "ERROR: merge failed for $kind/B=$budget." >&2
    return 1
  fi
}

run_shift_configuration() {
  local label="$1" dataset="$2" source_setup="$3" test_setup="$4" suffix="$5"
  local source_cache="alg_playground_model/is_real_True_dataset_${dataset}_dataset_${source_setup}/probability_est_cal_test.pt"
  local test_cache="alg_playground_model/is_real_True_dataset_${dataset}_dataset_${test_setup}/probability_est_cal_test.pt"
  if (( AVAILABLE_ONLY == 1 )) && [[ ! -f "$source_cache" || ! -f "$test_cache" ]]; then
    echo "[$label] skipped: source/test prediction cache unavailable"
    return 0
  fi
  local -a common construct
  common=(
    --bound-type lpb --data-type real
    --dataset-name "$dataset" --dataset-setup "$source_setup"
    --budget-per-sample "$ATTACKER_SHIFT_BUDGET" --cal-size "$CAL_SIZE"
    --tau-prior "$TAU_PRIOR" --m-upper-bound "$M_UPPER_BOUND"
    --device "$DEVICE" --allocations none
    --experiment-suffix "$suffix" --method-suite dapro_ablation
    --dapro-ablation-kind attacker_shift --dapro-n1-values 50
    --target-coverages 0.90 --definitive-dapro-margins 0.0
    --seed-start "$SEED_START" --seed-end "$SEED_END"
  )
  construct=("${common[@]}" --test-dataset-name "$dataset" --test-dataset-setup "$test_setup")
  echo "[attacker shift $label | B=$ATTACKER_SHIFT_BUDGET] source=$source_setup -> test=$test_setup"
  if ! run_module "shift_${label}_construct" src.predictive_bounds.construct_calibrated_bound "${construct[@]}"; then
    echo "ERROR: construction failed for attacker shift $label; merge skipped." >&2
    return 1
  fi
  if ! run_module "shift_${label}_merge" src.predictive_bounds.merge_bounds_results "${common[@]}"; then
    echo "ERROR: merge failed for attacker shift $label." >&2
    return 1
  fi
}

FAILED=()
PIDS=()
LABELS=()
wait_batch() {
  local i
  for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then FAILED+=("${LABELS[$i]}"); fi
  done
  PIDS=(); LABELS=()
}
submit() {
  local label="$1"; shift
  run_configuration "$@" &
  PIDS+=("$!"); LABELS+=("$label")
  if (( ${#PIDS[@]} >= PARALLEL_JOBS )); then wait_batch; fi
}
submit_metric() {
  local label="$1"; shift
  run_metric_configuration "$@" &
  PIDS+=("$!"); LABELS+=("$label")
  if (( ${#PIDS[@]} >= PARALLEL_JOBS )); then wait_batch; fi
}

N1_CSV="$(IFS=,; echo "${N1_VALUES[*]}")"
submit "N1" n1 20 "$N1_CSV" "${BASE_EXPERIMENT_SUFFIX}_n1"
submit "score_noise" score_noise 20 "$SCORE_REFERENCE_N1" "${BASE_EXPERIMENT_SUFFIX}_score_noise"
submit "hard_soft" hard_soft 20 50 "${BASE_EXPERIMENT_SUFFIX}_hard_soft"
submit "representation" representation 20 50 "${BASE_EXPERIMENT_SUFFIX}_representation"
submit "score" score 20 50 "${BASE_EXPERIMENT_SUFFIX}_score"
submit "cmax" cmax 20 50 "${BASE_EXPERIMENT_SUFFIX}_cmax"

# Metric estimation: the same Toxicity/Qwen cache and paired outer splits.
# Attacker shift is intentionally LPB-only. Every other factor includes
# Static, raw DAPRO, and CRC DAPRO.
for metric_n1 in "${N1_VALUES[@]}"; do
  submit_metric "metric_n1=${metric_n1}" n1 20 "$metric_n1" \
    "$((metric_n1 / 2))" \
    "${METRIC_EXPERIMENT_SUFFIX}_n1_n${metric_n1}"
done
submit_metric "metric_score_noise" score_noise 20 \
  "$METRIC_DAPRO_N1" "$METRIC_CRC_CONTROL_SIZE" \
  "${METRIC_EXPERIMENT_SUFFIX}_score_noise"
submit_metric "metric_hard_soft" hard_soft 20 \
  "$METRIC_DAPRO_N1" "$METRIC_CRC_CONTROL_SIZE" \
  "${METRIC_EXPERIMENT_SUFFIX}_hard_soft"
submit_metric "metric_representation" representation 20 \
  "$METRIC_DAPRO_N1" "$METRIC_CRC_CONTROL_SIZE" \
  "${METRIC_EXPERIMENT_SUFFIX}_representation"
submit_metric "metric_score" score 20 \
  "$METRIC_DAPRO_N1" "$METRIC_CRC_CONTROL_SIZE" \
  "${METRIC_EXPERIMENT_SUFFIX}_score"
submit_metric "metric_cmax" cmax 20 \
  "$METRIC_DAPRO_N1" "$METRIC_CRC_CONTROL_SIZE" \
  "${METRIC_EXPERIMENT_SUFFIX}_cmax"
for i in "${!BUDGET_VALUES[@]}"; do
  submit "budget=${BUDGET_VALUES[$i]}" budget "${BUDGET_VALUES[$i]}" \
    "${BUDGET_N1_VALUES[$i]}" "${BASE_EXPERIMENT_SUFFIX}_budget"
  metric_n1="${BUDGET_N1_VALUES[$i]}"
  submit_metric "metric_budget=${BUDGET_VALUES[$i]}" budget \
    "${BUDGET_VALUES[$i]}" "$metric_n1" "$((metric_n1 / 2))" \
    "${METRIC_EXPERIMENT_SUFFIX}_budget"
done
run_shift_configuration "red_gemma_to_qwen" "dataset_red_team" "$RED_SHIFT_GEMMA" "$RED_SHIFT_QWEN" \
  "${BASE_EXPERIMENT_SUFFIX}_attacker_shift_red" &
PIDS+=("$!"); LABELS+=("attacker_shift_red_gemma_to_qwen")
if (( ${#PIDS[@]} >= PARALLEL_JOBS )); then wait_batch; fi
run_shift_configuration "red_qwen_to_gemma" "dataset_red_team" "$RED_SHIFT_QWEN" "$RED_SHIFT_GEMMA" \
  "${BASE_EXPERIMENT_SUFFIX}_attacker_shift_red_reverse" &
PIDS+=("$!"); LABELS+=("attacker_shift_red_qwen_to_gemma")
if (( ${#PIDS[@]} >= PARALLEL_JOBS )); then wait_batch; fi
run_shift_configuration "toxicity_gemma_to_qwen" "dataset_toxicity" "$TOX_SHIFT_GEMMA" "$TOX_SHIFT_QWEN" \
  "${BASE_EXPERIMENT_SUFFIX}_attacker_shift_toxicity" &
PIDS+=("$!"); LABELS+=("attacker_shift_toxicity_gemma_to_qwen")
if (( ${#PIDS[@]} >= PARALLEL_JOBS )); then wait_batch; fi
run_shift_configuration "toxicity_qwen_to_gemma" "dataset_toxicity" "$TOX_SHIFT_QWEN" "$TOX_SHIFT_GEMMA" \
  "${BASE_EXPERIMENT_SUFFIX}_attacker_shift_toxicity_reverse" &
PIDS+=("$!"); LABELS+=("attacker_shift_toxicity_qwen_to_gemma")
if (( ${#PIDS[@]} >= PARALLEL_JOBS )); then wait_batch; fi
if (( ${#PIDS[@]} > 0 )); then wait_batch; fi

if (( DRY_RUN == 1 )); then echo "Dry run complete."; exit 0; fi

ARCHIVE="results/${BASE_EXPERIMENT_SUFFIX}.tar.gz"
mapfile -t MERGED < <(
  find results/merged_calibration_dfs -type f \
    -path "*_${BASE_EXPERIMENT_SUFFIX}_*/all_df.csv"
  find results/merged_metric_calibration_dfs -type f \
    -path "*_${METRIC_EXPERIMENT_SUFFIX}_*/all_df.csv"
)
if (( ${#MERGED[@]} > 0 )); then
  tar -czf "$ARCHIVE" "${MERGED[@]}"
  echo "Archived ${#MERGED[@]} merged files at $ARCHIVE"
else
  echo "No merged ablation files found." >&2
fi

if (( ${#FAILED[@]} > 0 )); then
  echo "Completed with ${#FAILED[@]} failed configuration(s): ${FAILED[*]}" >&2
else
  echo "All DAPRO ablation configurations completed successfully."
fi
