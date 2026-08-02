"""Generate figures and LaTeX tables across training-budget conditions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.predictive_bounds.experiments.common.reporting import (
    add_coverage_variance, prepare_bound_results, render_latex_summary,
    render_metric_boxplots,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("merged_csvs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("figures/ablations/limited_training_budget"))
    parser.add_argument("--target-coverage", type=float, default=0.90)
    args = parser.parse_args(argv)
    frame = pd.concat([pd.read_csv(path) for path in args.merged_csvs], ignore_index=True)
    frame = prepare_bound_results(frame, target_coverage=args.target_coverage)
    frame = add_coverage_variance(frame, ["training_budget_label"])
    render_metric_boxplots(
        frame, x="training_budget_label", output_dir=args.output_dir,
        prefix="limited_training_budget", x_label="Survival-model training data",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "limited_training_budget_tables.tex").write_text(
        render_latex_summary(
            frame, group_columns=["training_budget_label"],
            caption_prefix="Limited survival-model training budget",
        ), encoding="utf-8",
    )


if __name__ == "__main__":
    main()

