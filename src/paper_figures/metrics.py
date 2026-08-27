"""Paper plot generation for population-level metric estimation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.paper_figures.common import (
    plot_grouped_boxplot,
    plot_grouped_variance,
    plot_shared_legend,
)
from src.paper_figures.config import (
    DATASET_FILE_STEMS,
    DATASET_ORDER,
    MAIN_TARGET_MODEL_LABELS,
    MAIN_TARGET_MODEL_ORDER,
    METRIC_MAIN_METHOD_ORDER,
    METHOD_ORDER,
    TARGET_MODEL_ORDER,
)


METRIC_BOX_SPECS = (
    (
        "estimated_cjr",
        "estimated_cjr_boxplot.jpg",
        "Estimated event rate (%)",
        (),
        "event_truth",
    ),
    (
        "estimated_restricted_mean",
        "estimated_rmttu_boxplot.jpg",
        r"Estimated restricted mean $E[\min(T,200)]$",
        (),
        "restricted_mean_truth",
    ),
    (
        "budget_per_sample",
        "budget_per_sample_boxplot.jpg",
        "Budget Used per Sample",
        ("Uncalibrated", "Oracle"),
        None,
    ),
    (
        "observed_events",
        "observed_jailbreaks_boxplot.jpg",
        "Number of observed events",
        ("Uncalibrated", "Oracle"),
        None,
    ),
    (
        "mean_metric_a_weight",
        "mean_a_weight_boxplot.jpg",
        "Mean weighted error\n"
        r"$A_i/\pi_i$",
        ("Uncalibrated", "Oracle"),
        None,
    ),
)

METRIC_VARIANCE_SPECS = (
    (
        "estimated_cjr",
        "estimated_cjr_variance_barplot.jpg",
        r"Event-rate estimator variance (pp$^2$)",
    ),
    (
        "estimated_restricted_mean",
        "estimated_rmttu_variance_barplot.jpg",
        "Restricted-mean estimator variance\n(turns squared)",
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


def generate_metric_appendix_figures(
    frame: pd.DataFrame,
    *,
    output_dir: Path,
    quality: str,
) -> pd.DataFrame:
    """Generate the requested seven-panel metric set for each dataset."""
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
                    "Full-benchmark truth"
                    if references is not None
                    else "Target budget"
                ),
                hide_methods=hidden,
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
            )
            _record(
                rows,
                dataset_key=dataset_key,
                metric=f"{metric}_variance",
                path=path,
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
        ylabel="Estimated event rate (%)",
        quality=quality,
        x_order=MAIN_TARGET_MODEL_ORDER,
        method_order=METRIC_MAIN_METHOD_ORDER,
        reference_by_x=_reference_map(
            dataset, "full_benchmark_cjr"
        ),
        reference_label="Full-benchmark truth",
        hide_methods=("Oracle",),
        figsize=(3.65, 2.65),
        show_legend=False,
        font_scale=1.12,
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
        ylabel=r"Event-rate variance (pp$^2$)",
        quality=quality,
        x_order=MAIN_TARGET_MODEL_ORDER,
        method_order=METRIC_MAIN_METHOD_ORDER,
        hide_methods=("Oracle",),
        figsize=(3.65, 2.65),
        show_legend=False,
        font_scale=1.12,
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
        font_scale=1.08,
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
