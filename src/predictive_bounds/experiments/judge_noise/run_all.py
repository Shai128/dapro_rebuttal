"""Construct, merge, plot, and tabulate every requested judge-noise level."""

import argparse
from pathlib import Path

from src.predictive_bounds.experiments.common.results import result_roots, stable_experiment_name
from src.predictive_bounds.experiments.judge_noise import construct_calibrated_bound, merge_results, summarize


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output-dir", type=Path, default=Path("figures/ablations/judge_noise"))
    runner, remaining = parser.parse_known_args(argv)
    construct_calibrated_bound.main(remaining)
    merge_results.main(remaining)
    args = construct_calibrated_bound.parse_args(remaining)
    paths = []
    for noise_type, fn_rate, fp_rate in construct_calibrated_bound.noise_configurations(args):
        current = construct_calibrated_bound.metadata(args, noise_type, fn_rate, fp_rate)
        name = stable_experiment_name("judge_noise", current)
        _, merged = result_roots(args.bound_type, name)
        paths.append(str(merged / "all_df.csv"))
    summarize.main([
        *paths, "--output-dir", str(runner.output_dir),
        "--target-coverage", str(args.target_coverage),
    ])


if __name__ == "__main__":
    main()

