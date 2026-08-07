"""Create metric-estimation boxplots from merged per-seed results.

Every boxplot excludes the trivial full-budget oracle distribution and draws
its value as a horizontal dashed reference line instead.  Figures and summary
tables are written below ``figures/metric_estimation`` by default.
"""

from __future__ import annotations
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = ROOT / "results" / "merged_metric_calibration_dfs"
DEFAULT_OUTPUT_DIR = ROOT / "figures" / "metric_estimation"
ORACLE_NAME = "oracle_full_budget"

METRICS = {
    "estimated_cjr": "Estimated unsafe-event rate (%)",
    "abs_diff_cjr": "Absolute unsafe-event-rate error (pp)",
    "estimated_rmttu": "Estimated restricted mean time to unsafe",
    "abs_diff_rmttu": "Absolute RMTTU error",
    "budget_per_sample": "Realized budget per sample",
    "total_budget_utilized": "Total realized budget",
    "mean_weight": "Mean inverse-probability weight",
    "mean_a_weighted_weight": (
        r"Mean metric-event-weighted weight $A_i/\pi_i$"
    ),
    "num_events_observed": "Number of unsafe events observed",
    "conditional_variance_unsafe_event_rate_estimator": (
        "Conditional variance proxy for unsafe-event-rate estimator"
    ),
    "metric_a_weighted_effective_sample_size": (
        "Metric-event-weighted effective sample size"
    ),
    "fraction_trajectories_fully_resolved": (
        "Fraction of trajectories fully resolved"
    ),
}


def _display_name(name: str) -> str:
    if name == "UniformBudgetAllocator":
        return "Uniform + reweighting"
    if name == "UnweightedUniformBudgetAllocator":
        return "Uniform (unweighted)"
    if name == "optimized":
        return "Static"
    if name == "random_adaptive_optimized_no_terminal_floor_crc":
        return "Constant + CRC"
    if name == ORACLE_NAME:
        return "Full-budget oracle"
    if name.startswith("oracle_target_a_dapro_no_split"):
        return "Target-A oracle (global)"
    if "oracle_target_a_dapro" in name and "crc_control" in name:
        return "Target-A oracle + CRC"
    if name.startswith("oracle_target_a_dapro"):
        return "Target-A oracle (split)"
    if name.startswith("dapro_variance_aligned") and "budget_crc" in name:
        return "DAPRO definitive + CRC"
    if name.startswith("dapro_variance_aligned"):
        return "DAPRO definitive"
    if "a_target_raw" in name and "budget_crc" in name:
        return "DAPRO Target-A + CRC"
    if "a_target_raw" in name:
        return "DAPRO Target-A"
    if name.startswith("projected_optimization") and "budget_crc" in name:
        return "DAPRO legacy + CRC"
    if name.startswith("projected_optimization"):
        return "DAPRO legacy"
    return name.replace("_", " ").title()


def _safe_filename(metric: str) -> str:
    return metric.replace("_", "-") + "-boxplot.png"


def _oracle_reference(frame: pd.DataFrame, metric: str) -> float:
    values = pd.to_numeric(
        frame.loc[frame["allocator_name"] == ORACLE_NAME, metric],
        errors="coerce",
    ).dropna()
    if values.empty:
        raise ValueError(f"No finite full-budget oracle value for {metric}.")
    if not np.allclose(values, values.iloc[0], rtol=1e-10, atol=1e-10):
        raise ValueError(
            f"Full-budget oracle values vary across seeds for {metric}: "
            f"{values.tolist()}"
        )
    return float(values.iloc[0])


def _plot_metric(
        frame: pd.DataFrame,
        metric: str,
        ylabel: str,
        output_path: Path,
) -> None:
    reference = _oracle_reference(frame, metric)
    plot_frame = frame[frame["allocator_name"] != ORACLE_NAME].copy()
    plot_frame[metric] = pd.to_numeric(plot_frame[metric], errors="coerce")
    plot_frame = plot_frame.dropna(subset=[metric])
    if plot_frame.empty:
        raise ValueError(f"No finite non-oracle values for {metric}.")

    method_order = list(dict.fromkeys(plot_frame["method_display"]))
    width = max(12.0, 0.8 * len(method_order))
    figure, axis = plt.subplots(figsize=(width, 6.2))
    sns.boxplot(
        data=plot_frame,
        x="method_display",
        y=metric,
        order=method_order,
        hue="method_display",
        hue_order=method_order,
        palette="colorblind",
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": 4,
        },
        linewidth=1.0,
        fliersize=2.5,
        ax=axis,
    )
    if axis.legend_ is not None:
        axis.legend_.remove()
    axis.axhline(
        reference,
        color="#c62828",
        linestyle="--",
        linewidth=1.8,
        label=f"Full-budget oracle ({reference:.4g})",
    )
    axis.set_xlabel("")
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=35)
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")
    axis.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.55)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, loc="best")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def summarize_experiment(csv_path: Path, output_dir: Path) -> list[Path]:
    frame = pd.read_csv(csv_path)
    required = {"seed", "allocator_name"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns {sorted(missing)}")
    if ORACLE_NAME not in set(frame["allocator_name"]):
        raise ValueError(f"{csv_path} has no {ORACLE_NAME} rows.")
    frame["method_display"] = frame["allocator_name"].map(_display_name)

    experiment_dir = output_dir / csv_path.parent.name
    generated = []
    available_metrics = []
    for metric, ylabel in METRICS.items():
        if metric not in frame:
            print(f"Skipping absent metric {metric} in {csv_path}")
            continue
        path = experiment_dir / _safe_filename(metric)
        _plot_metric(frame, metric, ylabel, path)
        generated.append(path)
        available_metrics.append(metric)

    summary = (
        frame.groupby(["allocator_name", "method_display"], as_index=False)[
            available_metrics
        ]
        .agg(["count", "mean", "std", "var", "median"])
    )
    summary.columns = [
        "__".join(str(part) for part in column if part)
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    summary.to_csv(experiment_dir / "across-seed-summary.csv", index=False)
    frame.to_csv(experiment_dir / "seed-level-plot-data.csv", index=False)
    return generated


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--experiment",
        action="append",
        help="Merged experiment directory name; repeat to select several.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = sorted(args.input_dir.rglob("all_df.csv"))
    if args.experiment:
        requested = set(args.experiment)
        paths = [path for path in paths if path.parent.name in requested]
    if not paths:
        raise FileNotFoundError(
            f"No merged all_df.csv files below {args.input_dir}")
    for csv_path in paths:
        generated = summarize_experiment(csv_path, args.output_dir)
        print(f"{csv_path}: generated {len(generated)} figures")


if __name__ == "__main__":
    main()
