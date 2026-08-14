#!/usr/bin/env bash
# Construct, merge, summarize, and tabulate the 90%-coverage LPB comparison.
set -euo pipefail

PYTHON_EXE="${PYTHON:-python}"
DEVICE_VALUE="${DEVICE:-cuda:0}"
SEED_START_VALUE="${SEED_START:-0}"
SEED_END_VALUE="${SEED_END:-50}"
SUFFIX_VALUE="${EXPERIMENT_SUFFIX:-lpb_unified_aht_v1}"
QUALITY_VALUE="${FIGURE_QUALITY:-high}"

extra=()
if [[ "${AVAILABLE_ONLY:-0}" == "1" ]]; then
  extra+=(--available-only)
fi

"$PYTHON_EXE" -m src.predictive_bounds.experiments.full_bounds.run_all \
  --bound-type lpb \
  --seed-start "$SEED_START_VALUE" \
  --seed-end "$SEED_END_VALUE" \
  --device "$DEVICE_VALUE" \
  --suffix "$SUFFIX_VALUE" \
  --quality "$QUALITY_VALUE" \
  --dapro-n1-values 200 100 50 \
  "${extra[@]}" \
  "$@"

echo "Merged LPB CSVs: results/merged_calibration_dfs"
echo "Local figures: python -m src.predictive_bounds.experiments.full_bounds.summarize --bound-type lpb --experiment-suffix $SUFFIX_VALUE --input-dir results/merged_calibration_dfs --output-dir figures/lpb --quality high"
