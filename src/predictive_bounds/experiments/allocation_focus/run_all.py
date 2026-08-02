"""Construct, merge, plot, and tabulate allocation-focus diagnostics."""

import argparse
from pathlib import Path

from src.predictive_bounds.experiments.allocation_focus import construct, merge_results, summarize
from src.predictive_bounds.experiments.common.results import stable_experiment_name


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output-dir", type=Path, default=Path("figures/ablations/allocation_focus"))
    runner, remaining = parser.parse_known_args(argv)
    construct.main(remaining)
    merge_results.main(remaining)
    args = construct.parse_args(remaining)
    name = stable_experiment_name("allocation_focus", construct.metadata(args))
    merged = Path("results/experiments/allocation_focus") / name / "all_df.csv"
    summarize.main([str(merged), "--output-dir", str(runner.output_dir)])


if __name__ == "__main__":
    main()

