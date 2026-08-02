"""Generate box plots and LaTeX tables for AutoIF cross-subject shift."""

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
    parser.add_argument("merged_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("figures/ablations/autoif_cross_class"))
    parser.add_argument("--target-coverage", type=float, default=0.90)
    args = parser.parse_args(argv)
    frame = prepare_bound_results(pd.read_csv(args.merged_csv), target_coverage=args.target_coverage)
    frame["class_shift"] = frame["calibration_class"] + " $\\to$ " + frame["test_class"]
    frame = add_coverage_variance(frame, ["class_shift"])
    render_metric_boxplots(
        frame, x="class_shift", output_dir=args.output_dir,
        prefix="autoif_cross_class", x_label="Calibration subject to test subject",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "autoif_cross_class_tables.tex").write_text(
        render_latex_summary(
            frame, group_columns=["class_shift"], caption_prefix="AutoIF cross-subject shift"
        ), encoding="utf-8",
    )


if __name__ == "__main__":
    main()

