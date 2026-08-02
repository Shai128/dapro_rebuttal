#!/usr/bin/env bash
# Regenerate the single copy-paste-ready LaTeX table file from merged results.
set -euo pipefail

PYTHON_EXE="${PYTHON:-python}"
SUFFIX_VALUE="${EXPERIMENT_SUFFIX:-full_bounds_v1}"
extra=()
if [[ "${AVAILABLE_ONLY:-0}" == "1" ]]; then
  extra+=(--available-only)
fi

"$PYTHON_EXE" -m src.predictive_bounds.experiments.full_bounds.make_tables \
  --suffix "$SUFFIX_VALUE" \
  "${extra[@]}" \
  "$@"

