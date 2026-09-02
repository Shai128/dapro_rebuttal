"""Paper plot generation for population-level metric estimation."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import seaborn as sns

from src.paper_figures.common import (
    FULL_DATA_REFERENCE_COLOR,
    FULL_DATA_REFERENCE_LINESTYLE,
    TARGET_REFERENCE_COLOR,
    TARGET_REFERENCE_LINESTYLE,
    plot_grouped_boxplot,
    plot_grouped_variance,
    plot_shared_legend,
    save_jpeg,
)
from src.paper_figures.config import (
    DATASET_FILE_STEMS,
    DATASET_ORDER,
    MAIN_TARGET_MODEL_LABELS,
    MAIN_TARGET_MODEL_ORDER,
    METRIC_MAIN_METHOD_ORDER,
    METHOD_COLORS,
    METHOD_ORDER,
    TARGET_MODEL_ORDER,
)


METRIC_BOX_SPECS = (
    (
        "estimated_cjr",
        "estimated_cjr_boxplot.jpg",
        "Estimated event rate\n(%)",
        (),
        "event_truth",
    ),
    (
        "estimated_restricted_mean",
        "estimated_rmttu_boxplot.jpg",
        "Restricted mean\n(turns)",
        (),
        "restricted_mean_truth",
    ),
    (
        "budget_per_sample",
        "budget_per_sample_boxplot.jpg",
        "Budget used",
        ("Uncalibrated", "Oracle"),
        None,
    ),
    (
        "mean_metric_a_weight",
        "mean_a_weight_boxplot.jpg",
        "Weighted error",
        ("Uncalibrated", "Oracle"),
        None,
    ),
)

METRIC_TITLES = {
    "mean_metric_a_weight": r"Mean weighted error $A_i/\pi_i$",
}

METRIC_VARIANCE_SPECS = (
    (
        "estimated_cjr",
        "estimated_cjr_variance_barplot.jpg",
        "Event-rate variance\n" + r"(pp$^2$)",
    ),
    (
        "estimated_restricted_mean",
        "estimated_rmttu_variance_barplot.jpg",
        "Restricted-mean\nvariance\n" + r"(turns$^2$)",
    ),
)


def _reference_map(
    frame: pd.DataFrame, column: str
) -> dict[str, float]:
    references = {}
    for target_model, group in frame.groupby("target_model", observed=True):
        values = pd.to_numeric(group[column], errors="coerce").dropna().unique()
        if len(values) == 1:
            references[str(target_model)] = float(values[0])
    return references


def _record(
    rows: list[dict],
    *,
    dataset_key: str,
    metric: str,
    path: Path,
    generated: bool,
    scope: str,
) -> None:
    rows.append({
        "task": scope,
        "dataset_key": dataset_key,
        "metric": metric,
        "generated": bool(generated),
        "path": str(path),
    })


def _metric_grid_style(
    axis,
    *,
    title: str,
    ylabel: str,
    show_xlabel: bool = True,
) -> None:
    """Apply large, paper-readable typography to a metric-grid panel."""
    axis.set_title(title, fontsize=15.0, pad=7)
    axis.set_xlabel(
        "Target model" if show_xlabel else "",
        fontsize=13.5,
        labelpad=4,
    )
    axis.set_ylabel(ylabel, fontsize=13.5, labelpad=8)
    axis.tick_params(axis="both", labelsize=11.8)
    for label in axis.get_xticklabels():
        label.set_rotation(35)
        label.set_horizontalalignment("right")
    axis.grid(axis="y", linestyle=":", linewidth=0.75, alpha=0.48)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.yaxis.set_major_locator(MaxNLocator(nbins="auto", prune="upper"))
    if axis.legend_ is not None:
        axis.legend_.remove()


def _metric_grid_boxplot(
    axis,
    frame: pd.DataFrame,
    *,
    metric: str,
    title: str,
    ylabel: str,
    methods: tuple[str, ...],
    hide_methods: tuple[str, ...] = (),
    reference_by_x: dict[str, float] | None = None,
    reference: float | None = None,
    show_xlabel: bool = True,
) -> None:
    selected_methods = [
        method for method in methods
        if method not in hide_methods
        and method in set(frame["method"].dropna().astype(str))
    ]
    panel = frame[
        frame["method"].isin(selected_methods)
    ].dropna(subset=["target_model", metric]).copy()
    sns.boxplot(
        data=panel,
        x="target_model",
        y=metric,
        hue="method",
        order=list(TARGET_MODEL_ORDER),
        hue_order=selected_methods,
        palette={method: METHOD_COLORS[method] for method in selected_methods},
        linewidth=0.9,
        fliersize=1.8,
        ax=axis,
    )
    if reference_by_x:
        for position, target_model in enumerate(TARGET_MODEL_ORDER):
            value = reference_by_x.get(target_model)
            if value is not None and np.isfinite(value):
                axis.hlines(
                    value,
                    position - 0.45,
                    position + 0.45,
                    color=FULL_DATA_REFERENCE_COLOR,
                    linestyle=FULL_DATA_REFERENCE_LINESTYLE,
                    linewidth=1.25,
                    zorder=4,
                )
    if reference is not None and np.isfinite(reference):
        axis.axhline(
            reference,
            color=TARGET_REFERENCE_COLOR,
            linestyle=TARGET_REFERENCE_LINESTYLE,
            linewidth=1.25,
            zorder=4,
        )
    _metric_grid_style(
        axis,
        title=title,
        ylabel=ylabel,
        show_xlabel=show_xlabel,
    )


def _metric_grid_variance(
    axis,
    frame: pd.DataFrame,
    *,
    metric: str,
    title: str,
    ylabel: str,
    show_xlabel: bool = True,
) -> None:
    variance = (
        frame.dropna(subset=["target_model", "method", metric])
        .groupby(["target_model", "method"], observed=True, as_index=False)[
            metric
        ]
        .var(ddof=1)
        .rename(columns={metric: "variance"})
    )
    present_methods = [
        method for method in METHOD_ORDER
        if method in set(variance["method"].dropna().astype(str))
    ]
    sns.barplot(
        data=variance,
        x="target_model",
        y="variance",
        hue="method",
        order=list(TARGET_MODEL_ORDER),
        hue_order=present_methods,
        palette={method: METHOD_COLORS[method] for method in present_methods},
        errorbar=None,
        ax=axis,
    )
    _metric_grid_style(
        axis,
        title=title,
        ylabel=ylabel,
        show_xlabel=show_xlabel,
    )


def generate_metric_grid_figure(
    frame: pd.DataFrame,
    *,
    output_path: Path,
    quality: str,
) -> bool:
    """Generate the six population-metric panels as a 3-by-2 grid."""
    if frame.empty:
        output_path.unlink(missing_ok=True)
        return False
    event_reference = _reference_map(frame, "full_benchmark_cjr")
    rmst_reference = _reference_map(
        frame, "full_benchmark_restricted_mean"
    )
    target_budget = float(frame["target_budget"].iloc[0])
    figure, axes = plt.subplots(3, 2, figsize=(8.6, 9.0))
    _metric_grid_boxplot(
        axes[0, 0], frame,
        metric="estimated_cjr",
        title="Estimated Event Rate",
        ylabel="Event rate (%)",
        methods=METHOD_ORDER,
        reference_by_x=event_reference,
        show_xlabel=False,
    )
    _metric_grid_variance(
        axes[0, 1], frame,
        metric="estimated_cjr",
        title="Event-Rate Variance",
        ylabel=r"Variance (pp$^2$)",
        show_xlabel=False,
    )
    _metric_grid_boxplot(
        axes[1, 0], frame,
        metric="estimated_restricted_mean",
        title="Estimated Restricted Mean",
        ylabel="Restricted mean (turns)",
        methods=METHOD_ORDER,
        reference_by_x=rmst_reference,
        show_xlabel=False,
    )
    _metric_grid_variance(
        axes[1, 1], frame,
        metric="estimated_restricted_mean",
        title="Restricted-Mean Variance",
        ylabel=r"Variance (turns$^2$)",
        show_xlabel=False,
    )
    _metric_grid_boxplot(
        axes[2, 0], frame,
        metric="budget_per_sample",
        title="Budget Used per Sample",
        ylabel="Budget used",
        methods=METHOD_ORDER,
        hide_methods=("Uncalibrated", "Oracle"),
        reference=target_budget,
    )
    _metric_grid_boxplot(
        axes[2, 1], frame,
        metric="mean_metric_a_weight",
        title=r"Mean Weighted Error $A_i/\pi_i$",
        ylabel="Mean weighted error",
        methods=METHOD_ORDER,
        hide_methods=("Uncalibrated", "Oracle"),
    )
    handles = [
        Patch(
            facecolor=METHOD_COLORS[method],
            edgecolor="none",
            label=method,
        )
        for method in METHOD_ORDER
    ]
    handles.extend([
        Line2D(
            [0], [0], color=FULL_DATA_REFERENCE_COLOR,
            linestyle=FULL_DATA_REFERENCE_LINESTYLE, linewidth=1.35,
            label="Full-data value",
        ),
        Line2D(
            [0], [0], color=TARGET_REFERENCE_COLOR,
            linestyle=TARGET_REFERENCE_LINESTYLE, linewidth=1.35,
            label="Target budget",
        ),
    ])
    figure.legend(
        handles=handles,
        title="Method / reference",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.006),
        ncol=4,
        frameon=False,
        fontsize=12.0,
        title_fontsize=12.5,
        handlelength=1.55,
        columnspacing=1.25,
    )
    figure.subplots_adjust(
        left=0.11,
        right=0.985,
        top=0.95,
        bottom=0.205,
        wspace=0.36,
        hspace=0.52,
    )
    save_jpeg(
        figure,
        output_path,
        quality,
        tight=False,
        panel_count=6,
    )
    plt.close(figure)
    return True


def generate_metric_appendix_figures(
    frame: pd.DataFrame,
    *,
    output_dir: Path,
    quality: str,
) -> pd.DataFrame:
    """Generate six diagnostics and their paper-ready 3-by-2 grid."""
    rows: list[dict] = []
    for dataset_key in DATASET_ORDER:
        dataset = frame[frame["dataset_key"].eq(dataset_key)].copy()
        if dataset.empty:
            continue
        stem = DATASET_FILE_STEMS[dataset_key]
        dataset_dir = output_dir / f"dataset_{stem}"
        event_reference = _reference_map(dataset, "full_benchmark_cjr")
        rmst_reference = _reference_map(
            dataset, "full_benchmark_restricted_mean"
        )
        reference_maps = {
            "event_truth": event_reference,
            "restricted_mean_truth": rmst_reference,
        }
        for metric, filename, ylabel, hidden, reference_key in METRIC_BOX_SPECS:
            references = reference_maps.get(reference_key)
            path = dataset_dir / filename
            generated = plot_grouped_boxplot(
                dataset,
                metric=metric,
                output_path=path,
                ylabel=ylabel,
                quality=quality,
                x_order=TARGET_MODEL_ORDER,
                method_order=METHOD_ORDER,
                reference=(
                    float(dataset["target_budget"].iloc[0])
                    if metric == "budget_per_sample"
                    else None
                ),
                reference_by_x=references,
                reference_label=(
                    "Full-data value"
                    if references is not None
                    else "Target budget"
                ),
                hide_methods=hidden,
                figsize=(8.4, 2.65),
                font_scale=1.45,
                title=METRIC_TITLES.get(metric),
            )
            _record(
                rows,
                dataset_key=dataset_key,
                metric=metric,
                path=path,
                generated=generated,
                scope="metrics",
            )

        for metric, filename, ylabel in METRIC_VARIANCE_SPECS:
            path = dataset_dir / filename
            generated, _ = plot_grouped_variance(
                dataset,
                metric=metric,
                output_path=path,
                ylabel=ylabel,
                quality=quality,
                x_order=TARGET_MODEL_ORDER,
                method_order=METHOD_ORDER,
                figsize=(8.4, 2.65),
                font_scale=1.45,
            )
            _record(
                rows,
                dataset_key=dataset_key,
                metric=f"{metric}_variance",
                path=path,
                generated=generated,
                scope="metrics",
            )
        grid_path = dataset_dir / "metric_estimation_3x2.jpg"
        generated = generate_metric_grid_figure(
            dataset,
            output_path=grid_path,
            quality=quality,
        )
        _record(
            rows,
            dataset_key=dataset_key,
            metric="metric_estimation_3x2",
            path=grid_path,
            generated=generated,
            scope="metrics",
        )
    return pd.DataFrame(rows)


def generate_metric_main_figures(
    frame: pd.DataFrame, *, output_dir: Path, quality: str
) -> pd.DataFrame:
    """Generate Red-Team/Llama-Guard event-rate panels and one legend."""
    dataset_key = "red_team_llama_guard"
    dataset = frame[frame["dataset_key"].eq(dataset_key)].copy()
    dataset["target_model"] = dataset["target_model"].replace(
        MAIN_TARGET_MODEL_LABELS
    )
    rows: list[dict] = []
    path = output_dir / "red_team_llama_guard_event_rate_boxplot.jpg"
    generated = plot_grouped_boxplot(
        dataset,
        metric="estimated_cjr",
        output_path=path,
        ylabel="Event rate (%)",
        quality=quality,
        x_order=MAIN_TARGET_MODEL_ORDER,
        method_order=METRIC_MAIN_METHOD_ORDER,
        reference_by_x=_reference_map(
            dataset, "full_benchmark_cjr"
        ),
        reference_label="Full-data value",
        hide_methods=("Oracle",),
        figsize=(3.65, 2.65),
        show_legend=False,
        font_scale=1.30,
    )
    _record(
        rows,
        dataset_key=dataset_key,
        metric="estimated_cjr",
        path=path,
        generated=generated,
        scope="metrics-main",
    )
    path = output_dir / "red_team_llama_guard_event_rate_variance_barplot.jpg"
    generated, _ = plot_grouped_variance(
        dataset,
        metric="estimated_cjr",
        output_path=path,
        ylabel="Event-rate variance\n" + r"(pp$^2$)",
        quality=quality,
        x_order=MAIN_TARGET_MODEL_ORDER,
        method_order=METRIC_MAIN_METHOD_ORDER,
        hide_methods=("Oracle",),
        figsize=(3.65, 2.65),
        show_legend=False,
        font_scale=1.30,
    )
    _record(
        rows,
        dataset_key=dataset_key,
        metric="estimated_cjr_variance",
        path=path,
        generated=generated,
        scope="metrics-main",
    )
    legend_path = output_dir / "red_team_llama_guard_metric_legend.jpg"
    generated = plot_shared_legend(
        output_path=legend_path,
        quality=quality,
        methods=METRIC_MAIN_METHOD_ORDER,
        reference_label="Full-data value",
        figsize=(1.7, 2.65),
        font_scale=1.26,
    )
    _record(
        rows,
        dataset_key=dataset_key,
        metric="shared_legend",
        path=legend_path,
        generated=generated,
        scope="metrics-main",
    )
    return pd.DataFrame(rows)
