#!/usr/bin/env bash
# Construct, merge, plot, and tabulate the 70/80/90%-coverage UPB matrix.
# Value 201 is the infinity/no-event-through-turn-200 UPB sentinel.
set -euo pipefail

PYTHON_EXE="${PYTHON:-python}"
DEVICE_VALUE="cpu"
SEED_START_VALUE="${SEED_START:-0}"
SEED_END_VALUE="${SEED_END:-50}"
SUFFIX_VALUE="${EXPERIMENT_SUFFIX:-upb}"
QUALITY_VALUE="${FIGURE_QUALITY:-low}"

extra=()
if [[ "${AVAILABLE_ONLY:-0}" == "1" ]]; then
  extra+=(--available-only)
fi

srun -A galileo -p galileo -c4 --gres=gpu:1   --mem=20G  python  -m src.predictive_bounds.experiments.full_bounds.run_all \
  --bound-type upb \
  --seed-start "$SEED_START_VALUE" \
  --seed-end "$SEED_END_VALUE" \
  --device "$DEVICE_VALUE" \
  --suffix "$SUFFIX_VALUE" \
  --quality "$QUALITY_VALUE" \
  --dapro-n1-values 200 100 50 \
  "${extra[@]}" \
  "$@"

echo "Merged UPB CSVs: results/merged_upb_calibration_dfs"
echo "Local figures: python -m src.predictive_bounds.experiments.full_bounds.summarize --bound-type upb --experiment-suffix $SUFFIX_VALUE --input-dir results/merged_upb_calibration_dfs --output-dir figures/upb --quality high"
