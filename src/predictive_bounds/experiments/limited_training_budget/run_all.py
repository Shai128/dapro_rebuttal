"""Construct, merge, plot, and tabulate one limited-training cache."""

import argparse
from pathlib import Path

from src.predictive_bounds.experiments.common.results import result_roots, stable_experiment_name
from src.predictive_bounds.experiments.limited_training_budget import construct_calibrated_bound, merge_results, summarize


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output-dir", type=Path, default=Path("figures/ablations/limited_training_budget"))
    runner, remaining = parser.parse_known_args(argv)
    construct_calibrated_bound.main(remaining)
    merge_results.main(remaining)
    args = construct_calibrated_bound.parse_args(remaining)
    name = stable_experiment_name("limited_training_budget", construct_calibrated_bound.metadata(args))
    _, merged = result_roots(args.bound_type, name)
    summarize.main([
        str(merged / "all_df.csv"), "--output-dir", str(runner.output_dir),
        "--target-coverage", str(args.target_coverage),
    ])


if __name__ == "__main__":
    main()

