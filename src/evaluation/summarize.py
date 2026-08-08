"""Create metric-estimation boxplots and across-seed variance plots.

The trivial full-budget oracle distribution is excluded.  Its value is drawn
only on plots of the estimated metrics themselves; the realized budget plot
instead shows the configured target budget.  Figures and summary tables are
written below ``figures/metric_estimation`` by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.evaluation.result_matrix import (
    DAPRO_ORACLE_METHODS,
    METHOD_COLORS,
    METHOD_ORDER,
    method_display_name,
    numeric_label,
    parse_metric_result,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = ROOT / "results" / "merged_metric_calibration_dfs"
DEFAULT_OUTPUT_DIR = ROOT / "figures" / "metric_estimation"
ORACLE_NAME = "oracle_full_budget"
LOG_SCALE_METRICS = {"mean_weight", "mean_a_weighted_weight"}
ORACLE_REFERENCE_METRICS = {"estimated_cjr", "estimated_rmttu"}
TARGET_BUDGET_REFERENCE_METRICS = {"budget_per_sample"}

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
    return method_display_name(name)


def _safe_filename(metric: str) -> str:
    return metric.replace("_", "-") + "-boxplot.png"


def _variance_filename(metric: str) -> str:
    return metric.replace("_", "-") + "-variance-barplot.png"


def _save_figure(figure, output_path: Path, quality: str) -> None:
    dpi = 220 if quality == "high" else 110
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")


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
        quality: str,
) -> None:
    plot_frame = frame[frame["allocator_name"] != ORACLE_NAME].copy()
    plot_frame = plot_frame[
        ~plot_frame["method_display"].isin(DAPRO_ORACLE_METHODS)
    ]
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
        showfliers=False,
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": 4,
        },
        linewidth=1.0,
        ax=axis,
    )
    if axis.legend_ is not None:
        axis.legend_.remove()
    if metric in ORACLE_REFERENCE_METRICS:
        reference = _oracle_reference(frame, metric)
        axis.axhline(
            reference,
            color="#c62828",
            linestyle="--",
            linewidth=1.8,
            label=f"Full-budget oracle ({reference:.4g})",
        )
    elif metric in TARGET_BUDGET_REFERENCE_METRICS:
        target = pd.to_numeric(frame["target_budget"], errors="coerce").dropna()
        if target.empty or target.nunique() != 1:
            raise ValueError("Expected one configured target budget.")
        axis.axhline(
            float(target.iloc[0]),
            color="#c62828",
            linestyle="--",
            linewidth=1.8,
            label=f"Target budget ({target.iloc[0]:g})",
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
    if metric in ORACLE_REFERENCE_METRICS | TARGET_BUDGET_REFERENCE_METRICS:
        axis.legend(frameon=False, loc="best")
    figure.tight_layout()
    _save_figure(figure, output_path, quality)
    plt.close(figure)


def summarize_experiment(
        csv_path: Path, output_dir: Path, quality: str) -> list[Path]:
    frame = pd.read_csv(csv_path)
    required = {"seed", "allocator_name"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns {sorted(missing)}")
    if ORACLE_NAME not in set(frame["allocator_name"]):
        raise ValueError(f"{csv_path} has no {ORACLE_NAME} rows.")
    frame["method_display"] = frame["allocator_name"].map(_display_name)
    frame = frame[
        ~frame["method_display"].isin(DAPRO_ORACLE_METHODS)
    ].copy()
    metadata = parse_metric_result(csv_path)
    if metadata is not None:
        frame["target_budget"] = metadata.budget_per_sample

    experiment_dir = output_dir / csv_path.parent.name
    generated = []
    available_metrics = []
    for metric, ylabel in METRICS.items():
        if metric not in frame:
            print(f"Skipping absent metric {metric} in {csv_path}")
            continue
        path = experiment_dir / _safe_filename(metric)
        _plot_metric(frame, metric, ylabel, path, quality)
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


def load_metric_matrix(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load every metric result and attach budget/N1/dataset metadata."""
    frames = []
    inventory_rows = []
    skipped = []
    for path in sorted(input_dir.rglob("all_df.csv")):
        metadata = parse_metric_result(path)
        if metadata is None:
            skipped.append(path)
            continue
        frame = pd.read_csv(path)
        required = {"seed", "allocator_name", "calibration_name"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns {sorted(missing)}")
        if ORACLE_NAME not in set(frame["allocator_name"]):
            raise ValueError(f"{path} has no {ORACLE_NAME} rows.")
        frame["method_display"] = frame["calibration_name"].map(
            method_display_name
        )
        frame = frame[
            ~frame["method_display"].isin(DAPRO_ORACLE_METHODS)
        ].copy()
        frame["target_model"] = metadata.target_model_display
        frame["target_model_key"] = metadata.target_model
        frame["dataset_key"] = metadata.dataset_key
        frame["dataset_display"] = metadata.dataset_display
        frame["target_budget"] = metadata.budget_per_sample
        if {
            "configured_dapro_n1",
            "configured_crc_control_size",
        }.issubset(frame.columns):
            frame["dapro_n1"] = pd.to_numeric(
                frame["configured_dapro_n1"], errors="raise"
            ).astype(int)
            frame["crc_control_size"] = pd.to_numeric(
                frame["configured_crc_control_size"], errors="raise"
            ).astype(int)
        elif metadata.dapro_n1 is not None:
            frame["dapro_n1"] = metadata.dapro_n1
            frame["crc_control_size"] = metadata.crc_control_size
        else:
            raise ValueError(
                f"{path} uses a compact name but lacks configured N1/CRC columns."
            )
        frame["source_file"] = str(path)
        frame["plot_context"] = frame["dapro_n1"].map(
            lambda n1: (
                f"budget={metadata.budget_per_sample:g}, DAPRO N1={int(n1)}"
            )
        )
        frames.append(frame)
        for (n1, crc), config_frame in frame.groupby(
            ["dapro_n1", "crc_control_size"], sort=True
        ):
            inventory_rows.append({
                "dataset": metadata.dataset_key,
                "target_model": metadata.target_model,
                "budget_per_sample": metadata.budget_per_sample,
                "dapro_n1": int(n1),
                "crc_control_size": int(crc),
                "seed_count": config_frame["seed"].nunique(),
                "method_count": config_frame["allocator_name"].nunique(),
                "source_file": str(path),
            })
    if skipped:
        print(f"Ignored {len(skipped)} unrecognized metric result files.")
    if not frames:
        raise FileNotFoundError(
            f"No metric-estimation matrix results below {input_dir}."
        )
    return pd.concat(frames, ignore_index=True), pd.DataFrame(inventory_rows)


def _ordered_present(values: pd.Series, order: tuple[str, ...]) -> list[str]:
    present = set(values.dropna().unique())
    return [value for value in order if value in present] + sorted(
        present - set(order)
    )


def _plot_grouped_metric(
        frame: pd.DataFrame,
        metric: str,
        ylabel: str,
        output_path: Path,
        quality: str,
) -> None:
    plot_frame = frame[frame["allocator_name"] != ORACLE_NAME].copy()
    plot_frame = plot_frame[
        ~plot_frame["method_display"].isin(DAPRO_ORACLE_METHODS)
    ]
    plot_frame[metric] = pd.to_numeric(plot_frame[metric], errors="coerce")
    plot_frame = plot_frame.dropna(subset=[metric, "method_display"])
    if plot_frame.empty:
        raise ValueError(f"No finite non-oracle values for {metric}.")

    target_order = list(dict.fromkeys(plot_frame["target_model"]))
    method_order = _ordered_present(plot_frame["method_display"], METHOD_ORDER)
    palette = {
        method: METHOD_COLORS.get(method, "#777777")
        for method in method_order
    }
    figure, axis = plt.subplots(figsize=(13.5, 6.8))
    sns.boxplot(
        data=plot_frame,
        x="target_model",
        y=metric,
        hue="method_display",
        order=target_order,
        hue_order=method_order,
        palette=palette,
        showfliers=False,
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": 3.5,
        },
        linewidth=1.0,
        ax=axis,
    )

    if metric in ORACLE_REFERENCE_METRICS:
        oracle = frame[frame["allocator_name"] == ORACLE_NAME].copy()
        oracle[metric] = pd.to_numeric(oracle[metric], errors="coerce")
        for target_index, target in enumerate(target_order):
            values = oracle.loc[
                oracle["target_model"] == target, metric
            ].dropna()
            if values.empty:
                raise ValueError(f"Missing oracle {metric} for {target}.")
            if not np.allclose(
                    values, values.iloc[0], rtol=1e-10, atol=1e-10):
                raise ValueError(
                    f"Oracle {metric} varies across seeds for {target}.")
            axis.hlines(
                float(values.iloc[0]),
                target_index - 0.42,
                target_index + 0.42,
                color="#c62828",
                linestyle="--",
                linewidth=1.8,
                zorder=5,
                label="Full-budget oracle" if target_index == 0 else None,
            )
    elif metric in TARGET_BUDGET_REFERENCE_METRICS:
        targets = pd.to_numeric(
            frame["target_budget"], errors="coerce"
        ).dropna()
        if targets.empty or targets.nunique() != 1:
            raise ValueError("Expected one configured target budget.")
        target_budget = float(targets.iloc[0])
        axis.axhline(
            target_budget,
            color="#c62828",
            linestyle="--",
            linewidth=1.8,
            zorder=5,
            label=f"Target budget ({target_budget:g})",
        )

    if metric in LOG_SCALE_METRICS:
        positive = plot_frame.loc[plot_frame[metric] > 0, metric]
        if not positive.empty:
            axis.set_yscale("log")

    axis.set_xlabel("Target model")
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.55)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    dataset = str(frame["dataset_display"].iloc[0])
    context = str(frame["plot_context"].iloc[0])
    figure.suptitle(f"{dataset}: {ylabel} ({context})", y=0.995)
    handles, labels = axis.get_legend_handles_labels()
    if axis.legend_ is not None:
        axis.legend_.remove()
    figure.legend(
        handles,
        labels,
        title="Method",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=4,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.73))
    _save_figure(figure, output_path, quality)
    plt.close(figure)


def _plot_grouped_variance(
        frame: pd.DataFrame,
        metric: str,
        ylabel: str,
        output_path: Path,
        quality: str,
) -> pd.DataFrame:
    """Plot sample variance across calibration/test splits for each method."""
    plot_frame = frame[frame["allocator_name"] != ORACLE_NAME].copy()
    plot_frame = plot_frame[
        ~plot_frame["method_display"].isin(DAPRO_ORACLE_METHODS)
    ]
    plot_frame[metric] = pd.to_numeric(plot_frame[metric], errors="coerce")
    plot_frame = plot_frame.dropna(
        subset=[metric, "method_display", "target_model"]
    )
    variance = (
        plot_frame.groupby(
            ["target_model", "method_display"],
            observed=True,
            as_index=False,
        )[metric]
        .var(ddof=1)
        .rename(columns={metric: "across_seed_variance"})
        .dropna(subset=["across_seed_variance"])
    )
    if variance.empty:
        raise ValueError(f"No across-seed variance available for {metric}.")

    target_order = list(dict.fromkeys(plot_frame["target_model"]))
    method_order = _ordered_present(variance["method_display"], METHOD_ORDER)
    palette = {
        method: METHOD_COLORS.get(method, "#777777")
        for method in method_order
    }
    figure, axis = plt.subplots(figsize=(13.5, 6.8))
    sns.barplot(
        data=variance,
        x="target_model",
        y="across_seed_variance",
        hue="method_display",
        order=target_order,
        hue_order=method_order,
        palette=palette,
        ci=None,
        ax=axis,
    )
    positive = variance.loc[
        variance["across_seed_variance"] > 0, "across_seed_variance"
    ]
    if not positive.empty and positive.max() / positive.min() >= 1_000:
        axis.set_yscale("log")
    axis.set_xlabel("Target model")
    axis.set_ylabel(f"Across-seed variance of {ylabel.lower()}")
    axis.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.55)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    dataset = str(frame["dataset_display"].iloc[0])
    context = str(frame["plot_context"].iloc[0])
    figure.suptitle(
        f"{dataset}: across-seed variance of {ylabel.lower()} ({context})",
        y=0.995,
    )
    handles, labels = axis.get_legend_handles_labels()
    if axis.legend_ is not None:
        axis.legend_.remove()
    figure.legend(
        handles,
        labels,
        title="Method",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=3,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.67))
    _save_figure(figure, output_path, quality)
    plt.close(figure)
    variance.insert(0, "metric", metric)
    return variance


def generate_metric_matrix_figures(
        frame: pd.DataFrame,
        inventory: pd.DataFrame,
        output_dir: Path,
        quality: str,
) -> list[Path]:
    """Generate boxplots and variance bars for every budget/N1 combination."""
    generated = []
    index_rows = []
    group_columns = ["target_budget", "dapro_n1", "crc_control_size"]
    for (budget, n1, crc), combination in frame.groupby(
            group_columns, sort=True, observed=True
    ):
        combination_dir = (
            output_dir
            / f"budget_{numeric_label(budget)}"
            / f"n1_{int(n1)}"
            / f"crc_{int(crc)}"
        )
        for dataset_key, dataset_frame in combination.groupby(
                "dataset_key", sort=True, observed=True
        ):
            dataset_dir = combination_dir / dataset_key
            available_metrics = []
            variance_frames = []
            for metric, ylabel in METRICS.items():
                if metric not in dataset_frame:
                    continue
                path = dataset_dir / _safe_filename(metric)
                _plot_grouped_metric(
                    dataset_frame, metric, ylabel, path, quality
                )
                generated.append(path)
                variance_path = dataset_dir / _variance_filename(metric)
                variance_frames.append(_plot_grouped_variance(
                    dataset_frame,
                    metric,
                    ylabel,
                    variance_path,
                    quality,
                ))
                generated.append(variance_path)
                available_metrics.append(metric)
            summary = (
                dataset_frame.groupby(
                    ["target_model", "allocator_name", "method_display"],
                    as_index=False,
                )[available_metrics]
                .agg(["count", "mean", "std", "var", "median"])
            )
            summary.columns = [
                "__".join(str(part) for part in column if part)
                if isinstance(column, tuple)
                else str(column)
                for column in summary.columns
            ]
            dataset_dir.mkdir(parents=True, exist_ok=True)
            summary.to_csv(dataset_dir / "across-seed-summary.csv", index=False)
            dataset_frame.to_csv(
                dataset_dir / "seed-level-plot-data.csv", index=False
            )
            pd.concat(variance_frames, ignore_index=True).to_csv(
                dataset_dir / "across-seed-variances.csv", index=False
            )
            index_rows.append({
                "budget_per_sample": budget,
                "dapro_n1": int(n1),
                "crc_control_size": int(crc),
                "dataset": dataset_key,
                "target_model_count": dataset_frame["target_model"].nunique(),
                "source_file_count": dataset_frame["source_file"].nunique(),
                "figure_count": 2 * len(available_metrics),
                "directory": str(dataset_dir.relative_to(output_dir)),
            })
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(index_rows).to_csv(output_dir / "matrix-index.csv", index=False)
    inventory.to_csv(output_dir / "result-inventory.csv", index=False)
    return generated


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--quality", choices=["low", "high"], default="low",
        help="Low quality uses 110 DPI for faster regeneration.",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        help="Merged experiment directory name; repeat to select several.",
    )
    parser.add_argument(
        "--legacy-per-experiment",
        action="store_true",
        help="Generate the original one-target-model-per-directory figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.legacy_per_experiment:
        frame, inventory = load_metric_matrix(args.input_dir)
        print(
            f"Loaded {len(frame):,} metric rows from "
            f"{frame['source_file'].nunique()} files, covering "
            f"{frame['target_budget'].nunique()} budgets and "
            f"{frame['dapro_n1'].nunique()} N1 values."
        )
        generated = generate_metric_matrix_figures(
            frame, inventory, args.output_dir, args.quality
        )
        print(
            f"Generated {len(generated)} metric figures below "
            f"{args.output_dir}."
        )
        return
    paths = sorted(args.input_dir.rglob("all_df.csv"))
    if args.experiment:
        requested = set(args.experiment)
        paths = [path for path in paths if path.parent.name in requested]
    if not paths:
        raise FileNotFoundError(
            f"No merged all_df.csv files below {args.input_dir}")
    for csv_path in paths:
        generated = summarize_experiment(
            csv_path, args.output_dir, args.quality
        )
        print(f"{csv_path}: generated {len(generated)} figures")


if __name__ == "__main__":
    main()
