#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
AUDIT_DIR="${AUDIT_DIR:-outputs/final_extended_100_all_v3_audit}"
OUTPUT_DIR="${OUTPUT_DIR:-${AUDIT_DIR}/allocation_focus}"
SEED_START="${SEED_START:-0}"
SEED_END="${SEED_END:-100}"

"${PYTHON_BIN}" -m src.predictive_bounds.experiments.analysis.analyze_allocation_focus \
  --audit-dir "${AUDIT_DIR}" --output-dir "${OUTPUT_DIR}" \
  --seed-start "${SEED_START}" --seed-end "${SEED_END}"
