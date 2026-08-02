"""Merge all full-bound shards needed by budget concentration."""

from __future__ import annotations

import sys

from src.predictive_bounds.experiments.full_bounds.run_all import main as full_main


if __name__ == "__main__":
    if "--stage" not in sys.argv:
        sys.argv.extend(["--stage", "merge"])
    full_main()

