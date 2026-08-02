"""Construct, merge, plot, and tabulate one AutoIF cross-subject shift."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.predictive_bounds.experiments.autoif_cross_class import (
    construct_calibrated_bound,
    merge_bounds_results,
    summarize,
)
from src.predictive_bounds.experiments.autoif_cross_class.utils import (
    get_autoif_cross_class_experiment_name,
)


def main(argv=None):
    output_parser = argparse.ArgumentParser(add_help=False)
    output_parser.add_argument("--output-dir", type=Path, default=Path("figures/ablations/autoif_cross_class"))
    runner, remaining = output_parser.parse_known_args(argv)
    args = construct_calibrated_bound.parse_args(remaining)
    construct_calibrated_bound.main(remaining)
    test_size = None if args.test_size == 0 else args.test_size
    m_upper_bound = (
        args.m_upper_bound if args.m_upper_bound is not None else
        (args.gamma * args.budget_per_sample if args.gamma is not None else 200.0)
    )
    tau_prior = args.tau_prior if args.tau_prior is not None else (0.56 if args.bound_type == "lpb" else 0.98)
    name = get_autoif_cross_class_experiment_name(
        args.dataset_setup, args.calibration_class, args.test_class,
        args.budget_per_sample, args.cal_size, test_size, tau_prior,
        m_upper_bound / args.budget_per_sample,
    )
    merge_bounds_results.merge_autoif_cross_class_results(
        name, (args.seed_start, args.seed_end), args.bound_type,
        args.dataset_name, args.dataset_setup, args.calibration_class,
        args.test_class, args.cal_size, test_size,
    )
    kind = "merged_calibration_dfs" if args.bound_type == "lpb" else "merged_upb_calibration_dfs"
    summarize.main([
        str(Path("results") / kind / name / "all_df.csv"),
        "--output-dir", str(runner.output_dir),
        "--target-coverage", str(0.90 if args.bound_type == "lpb" else 0.70),
    ])


if __name__ == "__main__":
    main()
