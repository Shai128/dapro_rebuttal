"""Generate box plots and Overleaf tables for the judge-noise ablation."""

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
    parser.add_argument("--output-dir", type=Path, default=Path("figures/ablations/judge_noise"))
    parser.add_argument("--target-coverage", type=float, default=0.90)
    args = parser.parse_args(argv)
    frame = pd.concat([pd.read_csv(path) for path in args.merged_csvs], ignore_index=True)
    frame = prepare_bound_results(frame, target_coverage=args.target_coverage)
    frame["noise_condition"] = frame.apply(
        lambda row: (
            "clean" if row["noise_type"] == "clean" else
            f"{row['noise_type'].replace('_', ' ')} {max(row['false_negative_rate'], row['false_positive_rate']):.0%}"
        ), axis=1,
    )
    frame = add_coverage_variance(frame, ["noise_condition"])
    render_metric_boxplots(
        frame, x="noise_condition", output_dir=args.output_dir,
        prefix="judge_noise", x_label="Calibration-label noise",
    )
    (args.output_dir / "judge_noise_tables.tex").write_text(
        render_latex_summary(
            frame, group_columns=["noise_condition"], caption_prefix="Judge-noise ablation"
        ), encoding="utf-8",
    )


if __name__ == "__main__":
    main()

