#!/usr/bin/env bash
# Canonical UPB launcher. The shared engine provides local/Slurm scheduling,
# parallel complete experiments, construction, merging, and result archiving.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/calibrate.sh" "$@" --bound-type upb
