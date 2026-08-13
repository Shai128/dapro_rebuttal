#!/usr/bin/env bash
# Construct, merge, plot, and tabulate only the complete UPB matrix.
# Value 201 is the infinity/no-event-through-turn-200 UPB sentinel.
set -euo pipefail

PYTHON_EXE="${PYTHON:-python}"
DEVICE_VALUE="${DEVICE:-cuda:0}"
SEED_START_VALUE="${SEED_START:-0}"
SEED_END_VALUE="${SEED_END:-50}"
SUFFIX_VALUE="${EXPERIMENT_SUFFIX:-full_bounds_v4_soft_upb}"
QUALITY_VALUE="${FIGURE_QUALITY:-high}"

extra=()
if [[ "${AVAILABLE_ONLY:-0}" == "1" ]]; then
  extra+=(--available-only)
fi

"$PYTHON_EXE" -m src.predictive_bounds.experiments.full_bounds.run_all \
  --bound-type upb \
  --seed-start "$SEED_START_VALUE" \
  --seed-end "$SEED_END_VALUE" \
  --device "$DEVICE_VALUE" \
  --suffix "$SUFFIX_VALUE" \
  --quality "$QUALITY_VALUE" \
  "${extra[@]}" \
  "$@"
