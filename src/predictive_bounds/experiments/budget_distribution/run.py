"""Construct the shared full-bound shards used by budget concentration."""

from __future__ import annotations

import sys

from src.predictive_bounds.experiments.full_bounds.run_all import main as full_main


if __name__ == "__main__":
    # Reuse the definitive matrix rather than maintaining a second method list.
    if "--stage" not in sys.argv:
        sys.argv.extend(["--stage", "construct"])
    full_main()

