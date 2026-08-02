#!/usr/bin/env bash
# Run all eight requested experiments, strict merges, figures, and LaTeX tables.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Shift setups are repository/server-specific. Supply either a JSON matrix or
# one setup for each of the two shift families.
if [[ -z "${SHIFT_CONFIG_FILE:-}" ]]; then
  : "${TEST_SETUP:?Set TEST_SETUP or SHIFT_CONFIG_FILE for training-domain shift}"
  : "${ATTACKER_TEST_SETUP:?Set ATTACKER_TEST_SETUP or SHIFT_CONFIG_FILE for attacker shift}"
fi
export ALL_CONFIGS="${ALL_CONFIGS:-1}"

bash "$HERE/../distribution_shift/scripts/run_train_calibration_test_shift.sh"
bash "$HERE/../distribution_shift/scripts/run_attacker_shift.sh"
bash "$HERE/../autoif_cross_class/scripts/run_all.sh"
bash "$HERE/../allocation_focus/scripts/run_all.sh"
bash "$HERE/../variance_components/scripts/run_all.sh"
bash "$HERE/../budget_distribution/scripts/run_all.sh"
bash "$HERE/../limited_training_budget/scripts/run_all.sh"
bash "$HERE/../judge_noise/scripts/run_all.sh"
