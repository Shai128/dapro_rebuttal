#!/usr/bin/env bash
# Construct, merge, plot, and tabulate the complete manuscript matrix.
set -euo pipefail

PYTHON_EXE="${PYTHON:-python}"
DEVICE_VALUE="${DEVICE:-cuda:0}"
SEED_START_VALUE="${SEED_START:-0}"
SEED_END_VALUE="${SEED_END:-50}"
SUFFIX_VALUE="${EXPERIMENT_SUFFIX:-full_bounds_v5_soft_upb_aht}"
QUALITY_VALUE="${FIGURE_QUALITY:-high}"

extra=()
if [[ "${AVAILABLE_ONLY:-0}" == "1" ]]; then
  extra+=(--available-only)
fi

"$PYTHON_EXE" -m src.predictive_bounds.experiments.full_bounds.run_all \
  --seed-start "$SEED_START_VALUE" \
  --seed-end "$SEED_END_VALUE" \
  --device "$DEVICE_VALUE" \
  --suffix "$SUFFIX_VALUE" \
  --quality "$QUALITY_VALUE" \
  "${extra[@]}" \
  "$@"
