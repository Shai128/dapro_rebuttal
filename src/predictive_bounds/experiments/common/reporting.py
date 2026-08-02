"""Paper-ready box plots and LaTeX summaries for merged bound experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.predictive_bounds.experiments.full_bounds.config import (
    METHOD_COLORS,
    METHOD_DISPLAY,
    METHOD_ORDER,
)


STANDARD_METRICS = (
    ("coverage_pct", "Coverage (\\%)", "min_target_distance"),
    ("coverage_diff_pct", "Coverage error (pp)", "min"),
    ("coverage_variance_pp2", "Coverage variance (pp$^2$)", "min"),
    ("mean_weight", "Mean $1/\\pi$", "min"),
    ("mean_a_weight", "Mean $A/\\pi$", "min"),
    ("budget_used_per_sample", "Budget/sample", "max_feasible"),
)
NONCOMPETITIVE_METHODS = {"Uncalibrated", "Infinite-Budget Oracle"}


def prepare_bound_results(
        frame: pd.DataFrame,
        *,
        target_coverage: float,
        cal_size: int | str = "configured_cal_size",
        budget_per_sample: float | str = "configured_budget_per_sample",
) -> pd.DataFrame:
    """Select one target level and derive the standard manuscript metrics."""
    selected = frame[np.isclose(
        pd.to_numeric(frame["target_coverage"], errors="coerce"),
        target_coverage,
        atol=5e-7,
    )].copy()
    if selected.empty:
        raise ValueError(f"No rows match target coverage {target_coverage}.")
    selected["method"] = selected["calibration_name"].map(METHOD_DISPLAY)
    selected["method"] = selected["method"].fillna(selected["calibration_name"])
    for column in (
        "coverage", "mean_weight",
        "mean_a_weighted_inverse_probability", "budget_used",
    ):
        if column not in selected:
            selected[column] = np.nan
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    selected["coverage_pct"] = 100 * selected["coverage"]
    selected["target_coverage_pct"] = 100 * float(target_coverage)
    selected["coverage_diff_pct"] = (
        selected["coverage_pct"] - selected["target_coverage_pct"]
    ).abs()
    selected["mean_a_weight"] = selected[
        "mean_a_weighted_inverse_probability"
    ]
    denominator = (
        pd.to_numeric(selected[cal_size], errors="coerce")
        if isinstance(cal_size, str)
        else float(cal_size)
    )
    selected["budget_used_per_sample"] = selected["budget_used"] / denominator
    selected["target_budget"] = (
        pd.to_numeric(selected[budget_per_sample], errors="coerce")
        if isinstance(budget_per_sample, str)
        else float(budget_per_sample)
    )
    selected.loc[selected["method"].eq("Uncalibrated"), [
        "mean_weight", "mean_a_weight", "budget_used_per_sample",
    ]] = np.nan
    return selected


def add_coverage_variance(
        frame: pd.DataFrame,
        group_columns: Sequence[str],
) -> pd.DataFrame:
    keys = [*group_columns, "method"]
    variance = (
        frame.groupby(keys, observed=True)["coverage_pct"]
        .var(ddof=1)
        .rename("coverage_variance_pp2")
        .reset_index()
    )
    return frame.merge(variance, on=keys, how="left", validate="many_to_one")


def render_metric_boxplots(
        frame: pd.DataFrame,
        *,
        x: str,
        output_dir: Path,
        prefix: str,
        x_label: str,
        hue: str = "method",
        metric_labels: Mapping[str, str] | None = None,
) -> list[Path]:
    """Render all standard metrics, using bars for across-seed variance."""
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = {metric: label for metric, label, _ in STANDARD_METRICS}
    labels.update(metric_labels or {})
    hue_order = [m for m in METHOD_ORDER if m in set(frame[hue].dropna())]
    hue_order.extend(sorted(set(frame[hue].dropna()) - set(hue_order)))
    palette = {name: METHOD_COLORS.get(name, "#6b7280") for name in hue_order}
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    paths = []
    for metric, _, _ in STANDARD_METRICS:
        if metric not in frame or frame[metric].notna().sum() == 0:
            continue
        data = frame.dropna(subset=[x, hue, metric]).copy()
        figure, axis = plt.subplots(figsize=(11.5, 6.2))
        plotter = sns.barplot if metric == "coverage_variance_pp2" else sns.boxplot
        kwargs = dict(data=data, x=x, y=metric, hue=hue, hue_order=hue_order,
                      palette=palette, ax=axis)
        if metric == "coverage_variance_pp2":
            kwargs["errorbar"] = None
        else:
            kwargs.update(showmeans=True, fliersize=2)
        plotter(**kwargs)
        if metric == "coverage_pct":
            axis.axhline(
                float(data["target_coverage_pct"].iloc[0]),
                color="#b91c1c", linestyle="--", linewidth=1.4,
            )
        if metric == "budget_used_per_sample" and data["target_budget"].nunique() == 1:
            axis.axhline(
                float(data["target_budget"].iloc[0]),
                color="#b91c1c", linestyle="--", linewidth=1.4,
            )
        axis.set_xlabel(x_label)
        axis.set_ylabel(labels[metric])
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(
            title="Method", loc="lower center", bbox_to_anchor=(0.5, 1.01),
            ncol=min(4, max(1, len(hue_order))), frameon=False,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.90))
        path = output_dir / f"{prefix}_{metric}.pdf"
        figure.savefig(path, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)
    return paths


def render_latex_summary(
        frame: pd.DataFrame,
        *,
        group_columns: Sequence[str],
        caption_prefix: str,
) -> str:
    """Return one compact LaTeX table per requested experimental condition."""
    blocks = [
        "% Generated experiment tables; requires booktabs and graphicx.",
        "% Oracle and uncalibrated rows are never eligible for boldface.",
        "",
    ]
    for key, group in frame.groupby(list(group_columns), observed=True, sort=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        target = float(group["target_coverage_pct"].iloc[0])
        budget = float(group["target_budget"].iloc[0])
        rows = []
        for method, method_frame in group.groupby("method", observed=True):
            row = {"method": method}
            for metric, _, _ in STANDARD_METRICS:
                values = pd.to_numeric(method_frame[metric], errors="coerce")
                row[metric] = values.mean()
                row[f"{metric}_std"] = values.std(ddof=1)
            rows.append(row)
        summary = pd.DataFrame(rows)
        eligible = summary[~summary["method"].isin(NONCOMPETITIVE_METHODS)]
        best: dict[str, str | None] = {}
        for metric, _, rule in STANDARD_METRICS:
            valid = eligible[np.isfinite(eligible[metric])]
            if valid.empty:
                best[metric] = None
            elif rule == "min_target_distance":
                best[metric] = valid.loc[
                    (valid[metric] - target).abs().idxmin(), "method"
                ]
            elif rule == "max_feasible":
                feasible = valid[valid[metric] <= budget + 1e-9]
                chosen = feasible if not feasible.empty else valid
                index = (
                    chosen[metric].idxmax() if not feasible.empty
                    else (chosen[metric] - budget).abs().idxmin()
                )
                best[metric] = chosen.loc[index, "method"]
            else:
                best[metric] = valid.loc[valid[metric].idxmin(), "method"]
        escaped_key = ", ".join(str(value).replace("_", r"\_") for value in key_tuple)
        blocks.extend([
            r"\begin{table*}[t]", r"\centering", r"\small",
            rf"\caption{{{caption_prefix}: {escaped_key}.}}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{lrrrrrr}", r"\toprule",
            "Method & " + " & ".join(label for _, label, _ in STANDARD_METRICS) + r" \\",
            r"\midrule",
        ])
        order = [m for m in METHOD_ORDER if m in set(summary["method"])]
        order.extend(sorted(set(summary["method"]) - set(order)))
        summary = summary.set_index("method").reindex(order).reset_index()
        for _, row in summary.iterrows():
            cells = []
            for metric, _, _ in STANDARD_METRICS:
                mean = float(row[metric])
                std = float(row[f"{metric}_std"])
                if not np.isfinite(mean):
                    cell = "--"
                elif metric == "coverage_variance_pp2" or not np.isfinite(std):
                    cell = f"{mean:.3f}"
                else:
                    cell = f"{mean:.3f} $\\pm$ {std:.3f}"
                if best[metric] == row["method"] and cell != "--":
                    cell = rf"\textbf{{{cell}}}"
                cells.append(cell)
            method = str(row["method"]).replace("_", r"\_")
            blocks.append(method + " & " + " & ".join(cells) + r" \\")
        blocks.extend([
            r"\bottomrule", r"\end{tabular}", r"}%", r"\end{table*}", "",
        ])
    return "\n".join(blocks)
