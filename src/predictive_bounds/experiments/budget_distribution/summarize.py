"""Plot realized-budget histograms and tabulate overrun probabilities."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.predictive_bounds.experiments.full_bounds.config import select_configs
from src.predictive_bounds.experiments.full_bounds.summarize import (
    ROOT,
    load_comparison_data,
)


def concentration_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    selected = frame.dropna(subset=["budget_used_per_sample"])
    for keys, group in selected.groupby(
            ["configuration", "dataset_display", "target_model", "method"],
            observed=True, sort=False):
        values = group["budget_used_per_sample"].to_numpy(dtype=float)
        target = float(group["target_budget"].iloc[0])
        rows.append({
            "configuration": keys[0],
            "dataset": keys[1],
            "target_model": keys[2],
            "method": keys[3],
            "target_budget": target,
            "mean_budget": float(np.mean(values)),
            "std_budget": float(np.std(values, ddof=1)),
            "p05_budget": float(np.quantile(values, 0.05)),
            "p50_budget": float(np.quantile(values, 0.50)),
            "p95_budget": float(np.quantile(values, 0.95)),
            "p99_budget": float(np.quantile(values, 0.99)),
            "max_budget": float(np.max(values)),
            "overrun_probability": float(np.mean(values > target)),
            "overrun_5pct_probability": float(np.mean(values > 1.05 * target)),
            "overrun_10pct_probability": float(np.mean(values > 1.10 * target)),
            "max_relative_overrun": float(max(0, np.max(values) / target - 1)),
        })
    return pd.DataFrame(rows)


def latex_tables(summary: pd.DataFrame) -> str:
    blocks = ["% Requires booktabs and graphicx. Oracle rows are never bolded."]
    for configuration, group in summary.groupby("configuration", sort=False):
        eligible = group[~group["method"].isin(["Infinite-Budget Oracle"])]
        best = None if eligible.empty else eligible.loc[
            eligible["overrun_10pct_probability"].idxmin(), "method"
        ]
        blocks.extend([
            r"\begin{table*}[t]", r"\centering", r"\small",
            rf"\caption{{Realized-budget concentration: {configuration.replace('_', r'\_')}.}}",
            r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
            r"Method & Mean & SD & P50 & P95 & P99 & Max & $P(B>b)$ & $P(B>1.1b)$ \\",
            r"\midrule",
        ])
        for row in group.itertuples(index=False):
            cells = [
                f"{row.mean_budget:.3f}",
                f"{row.std_budget:.3f}",
                f"{row.p50_budget:.3f}",
                f"{row.p95_budget:.3f}",
                f"{row.p99_budget:.3f}",
                f"{row.max_budget:.3f}",
                f"{row.overrun_probability:.3f}",
                f"{row.overrun_10pct_probability:.3f}",
            ]
            if row.method == best:
                cells = [rf"\textbf{{{cell}}}" for cell in cells]
            blocks.append(
                str(row.method).replace("_", r"\_")
                + " & " + " & ".join(cells) + r" \\"
            )
        blocks.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(blocks)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suffix", default="full_bounds_v1")
    parser.add_argument("--output-dir", type=Path, default=Path("figures/ablations/budget_distribution"))
    parser.add_argument("--available-only", action="store_true")
    parser.add_argument("--config", action="append", dest="configs")
    args = parser.parse_args(argv)
    configs = select_configs(
        ROOT, keys=set(args.configs or []), available_only=args.available_only
    )
    frame = load_comparison_data(configs, args.suffix)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = concentration_table(frame)
    summary.to_csv(args.output_dir / "budget_concentration.csv", index=False)
    for configuration, group in frame.dropna(subset=["budget_used_per_sample"]).groupby(
            "configuration", observed=True, sort=False):
        figure, axis = plt.subplots(figsize=(10.5, 5.8))
        sns.histplot(
            data=group, x="budget_used_per_sample", hue="method",
            element="step", stat="density", common_norm=False,
            bins="auto", alpha=0.15, ax=axis,
        )
        target = float(group["target_budget"].iloc[0])
        axis.axvline(target, color="#b91c1c", linestyle="--", linewidth=1.5)
        axis.set_xlabel("Realized budget per calibration sample")
        axis.set_ylabel("Density")
        figure.tight_layout()
        figure.savefig(
            args.output_dir / f"{configuration}_budget_histogram.pdf",
            bbox_inches="tight",
        )
        plt.close(figure)
        figure, axis = plt.subplots(figsize=(10.5, 5.8))
        sns.boxplot(
            data=group, x="method", y="budget_used_per_sample",
            fliersize=2, ax=axis,
        )
        axis.axhline(target, color="#b91c1c", linestyle="--", linewidth=1.5)
        axis.set_xlabel("Method")
        axis.set_ylabel("Realized budget per calibration sample")
        axis.tick_params(axis="x", rotation=25)
        figure.tight_layout()
        figure.savefig(
            args.output_dir / f"{configuration}_budget_boxplot.pdf",
            bbox_inches="tight",
        )
        plt.close(figure)
    (args.output_dir / "budget_concentration_tables.tex").write_text(
        latex_tables(summary), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
