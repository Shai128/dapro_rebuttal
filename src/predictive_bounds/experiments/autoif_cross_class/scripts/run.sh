#!/usr/bin/env bash
# Backward-compatible entry point; the new runner also generates paper outputs.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_all.sh"
