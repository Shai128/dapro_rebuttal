#!/usr/bin/env bash
# Reuse the full benchmark shards, merge them, then report budget concentration.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$ROOT"

ARGS=(--seed-start "${SEED_START:-0}" --seed-end "${SEED_END:-50}" \
      --suffix "${SUFFIX:-full_bounds_v1}" --device "${DEVICE:-cuda:0}")
[[ "${AVAILABLE_ONLY:-0}" == "1" ]] && ARGS+=(--available-only)
python -m src.predictive_bounds.experiments.budget_distribution.run "${ARGS[@]}"
python -m src.predictive_bounds.experiments.budget_distribution.merge_results "${ARGS[@]}"
EXTRA=()
[[ "${AVAILABLE_ONLY:-0}" == "1" ]] && EXTRA+=(--available-only)
python -m src.predictive_bounds.experiments.budget_distribution.summarize \
  --suffix "${SUFFIX:-full_bounds_v1}" "${EXTRA[@]}"
