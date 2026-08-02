#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../../.." && pwd)"
cd "${REPO_ROOT}"

: "${INPUT:?Set INPUT to one DATASET=PATH value; pass additional --input values as arguments.}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/optimal_dapro}"

"${PYTHON_BIN}" -m src.predictive_bounds.experiments.analysis.analyze_optimal_dapro_results \
  --input "${INPUT}" --output-dir "${OUTPUT_DIR}" "$@"
