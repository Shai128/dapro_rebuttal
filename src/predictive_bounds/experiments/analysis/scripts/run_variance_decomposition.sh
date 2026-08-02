#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MERGED_ROOT="${MERGED_ROOT:-results/merged_calibration_dfs}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/variance_decomposition}"

"${PYTHON_BIN}" -m src.predictive_bounds.experiments.analysis.analyze_variance_decomposition \
  --merged-root "${MERGED_ROOT}" --output-dir "${OUTPUT_DIR}"
