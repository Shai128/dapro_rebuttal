"""Compare a limited-budget model cache with the ordinary full-data model."""

from __future__ import annotations

import argparse

from src.predictive_bounds.experiments.common.results import result_roots, stable_experiment_name
from src.predictive_bounds.experiments.limited_training_budget import (
    construct_calibrated_bound,
    merge_results,
    summarize,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limited-prediction-cache", required=True)
    parser.add_argument("--full-prediction-cache", required=True)
    parser.add_argument("--limited-training-budget-fraction", type=float, default=0.10)
    parser.add_argument("--dataset-name", default="dataset_toxicity")
    parser.add_argument("--dataset-setup", required=True)
    parser.add_argument("--bound-type", choices=["lpb", "upb"], default="lpb")
    parser.add_argument("--cal-size", type=int, default=3000)
    parser.add_argument("--budget-per-sample", type=float, default=20.0)
    parser.add_argument("--tau-prior", type=float, default=0.56)
    parser.add_argument("--target-coverage", type=float, default=0.90)
    parser.add_argument("--m-upper-bound", type=float, default=200.0)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="figures/ablations/limited_training_budget")
    args = parser.parse_args(argv)
    shared = [
        "--dataset-name", args.dataset_name,
        "--dataset-setup", args.dataset_setup,
        "--bound-type", args.bound_type,
        "--cal-size", str(args.cal_size),
        "--budget-per-sample", str(args.budget_per_sample),
        "--tau-prior", str(args.tau_prior),
        "--target-coverage", str(args.target_coverage),
        "--m-upper-bound", str(args.m_upper_bound),
        "--seed-start", str(args.seed_start),
        "--seed-end", str(args.seed_end),
        "--device", args.device,
    ]
    result_paths = []
    for cache, label, fraction in (
        (args.full_prediction_cache, "Full training data", 1.0),
        (
            args.limited_prediction_cache,
            f"{args.limited_training_budget_fraction:.0%} uniform training budget",
            args.limited_training_budget_fraction,
        ),
    ):
        current_argv = [
            *shared, "--prediction-cache", cache,
            "--training-budget-label", label,
            "--training-budget-fraction", str(fraction),
        ]
        construct_calibrated_bound.main(current_argv)
        merge_results.main(current_argv)
        parsed = construct_calibrated_bound.parse_args(current_argv)
        name = stable_experiment_name(
            "limited_training_budget", construct_calibrated_bound.metadata(parsed)
        )
        _, merged = result_roots(parsed.bound_type, name)
        result_paths.append(str(merged / "all_df.csv"))
    summarize.main([
        *result_paths,
        "--output-dir", args.output_dir,
        "--target-coverage", str(args.target_coverage),
    ])


if __name__ == "__main__":
    main()
