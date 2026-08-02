"""Plot and tabulate budget allocation by difficulty quartile."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.predictive_bounds.experiments.allocation_focus.construct import STRATIFIERS
from src.predictive_bounds.experiments.full_bounds.config import METHOD_DISPLAY


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["method_display"] = frame["method"].map(METHOD_DISPLAY).fillna(frame["method"])
    rows = []
    for stratifier in STRATIFIERS:
        quartile = f"{stratifier}_quartile"
        summary = frame.groupby(
            ["method_display", quartile], observed=True, as_index=False
        ).agg(
            mean_realized_budget=("realized_budget", "mean"),
            mean_terminal_inclusion_probability=("terminal_inclusion_probability", "mean"),
            mean_inverse_probability=("inverse_probability", "mean"),
            target_event_rate=("target_event", "mean"),
            target_event_observation_rate=("target_event_observed", "mean"),
            samples=("sample_id", "count"),
        )
        summary["stratifier"] = stratifier
        summary = summary.rename(columns={quartile: "quartile"})
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def latex_table(summary: pd.DataFrame) -> str:
    blocks = ["% Requires booktabs."]
    for stratifier, group in summary.groupby("stratifier", sort=False):
        blocks.extend([
            r"\begin{table}[t]", r"\centering", r"\small",
            rf"\caption{{Budget allocation stratified by {stratifier.replace('_', ' ')}.}}",
            r"\begin{tabular}{llrrrr}", r"\toprule",
            r"Method & Quartile & Budget & $\pi$ & $1/\pi$ & Target obs. \\",
            r"\midrule",
        ])
        for row in group.itertuples(index=False):
            blocks.append(
                f"{str(row.method_display).replace('_', r'\_')} & {row.quartile} & "
                f"{row.mean_realized_budget:.3f} & {row.mean_terminal_inclusion_probability:.3f} & "
                f"{row.mean_inverse_probability:.3f} & {row.target_event_observation_rate:.3f} \\\\"
            )
        blocks.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(blocks)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("merged_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("figures/ablations/allocation_focus"))
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.merged_csv)
    raw["method_display"] = raw["method"].map(METHOD_DISPLAY).fillna(raw["method"])
    summary = summarize(raw)
    summary.to_csv(args.output_dir / "allocation_focus_summary.csv", index=False)
    for stratifier, group in summary.groupby("stratifier", sort=False):
        figure, axis = plt.subplots(figsize=(10.5, 5.8))
        sns.barplot(
            data=group, x="quartile", y="mean_realized_budget",
            hue="method_display", order=["Q1", "Q2", "Q3", "Q4"], ax=axis,
        )
        axis.set_xlabel(f"{stratifier.replace('_', ' ').title()} quartile")
        axis.set_ylabel("Mean realized budget per calibration sample")
        axis.legend(title="Method", bbox_to_anchor=(0.5, 1.02), loc="lower center", ncol=3, frameon=False)
        figure.tight_layout(rect=(0, 0, 1, 0.88))
        figure.savefig(args.output_dir / f"allocation_focus_{stratifier}.pdf", bbox_inches="tight")
        plt.close(figure)
        figure, axis = plt.subplots(figsize=(10.5, 5.8))
        sns.boxplot(
            data=raw, x=f"{stratifier}_quartile", y="realized_budget",
            hue="method_display", order=["Q1", "Q2", "Q3", "Q4"],
            fliersize=1.5, ax=axis,
        )
        axis.set_xlabel(f"{stratifier.replace('_', ' ').title()} quartile")
        axis.set_ylabel("Realized budget per calibration sample")
        axis.legend(title="Method", bbox_to_anchor=(0.5, 1.02), loc="lower center", ncol=3, frameon=False)
        figure.tight_layout(rect=(0, 0, 1, 0.88))
        figure.savefig(
            args.output_dir / f"allocation_focus_{stratifier}_boxplot.pdf",
            bbox_inches="tight",
        )
        plt.close(figure)
    (args.output_dir / "allocation_focus_tables.tex").write_text(
        latex_table(summary), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
