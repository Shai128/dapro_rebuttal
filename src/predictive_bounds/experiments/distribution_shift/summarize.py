"""Generate paper figures and LaTeX tables for distribution shift."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.predictive_bounds.experiments.common.reporting import (
    add_coverage_variance,
    prepare_bound_results,
    render_latex_summary,
    render_metric_boxplots,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("merged_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("figures/ablations/distribution_shift"))
    parser.add_argument("--target-coverage", type=float, default=0.90)
    args = parser.parse_args(argv)
    frame = prepare_bound_results(
        pd.read_csv(args.merged_csv), target_coverage=args.target_coverage
    )
    frame["shift"] = frame["experiment_type"].str.replace("_", " ")
    frame = add_coverage_variance(frame, ["shift"])
    render_metric_boxplots(
        frame, x="shift", output_dir=args.output_dir,
        prefix="distribution_shift", x_label="Shift design",
    )
    (args.output_dir / "distribution_shift_tables.tex").write_text(
        render_latex_summary(
            frame, group_columns=["shift"], caption_prefix="Distribution shift"
        ), encoding="utf-8",
    )


if __name__ == "__main__":
    main()

