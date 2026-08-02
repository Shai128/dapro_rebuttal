#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../../.." && pwd)"
cd "${REPO_ROOT}"

: "${MERGED_CSV:?Set MERGED_CSV to the merged result CSV.}"
: "${EXPERIMENT_SUFFIX:?Set EXPERIMENT_SUFFIX to the run suffix.}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/a_weighted_dapro}"

"${PYTHON_BIN}" -m src.predictive_bounds.experiments.analysis.analyze_a_weighted_dapro_results \
  --merged-csv "${MERGED_CSV}" --output-dir "${OUTPUT_DIR}" \
  --experiment-suffix "${EXPERIMENT_SUFFIX}" "$@"
