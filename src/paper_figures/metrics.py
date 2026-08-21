"""Paper plot generation for population-level metric estimation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.paper_figures.common import (
    plot_grouped_boxplot,
    plot_grouped_variance,
)
from src.paper_figures.config import (
    DATASET_FILE_STEMS,
    DATASET_ORDER,
    MAIN_METHOD_ORDER,
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
        r"Mean metric-target weight $A_i/\pi_i$",
        ("Uncalibrated", "Oracle"),
        None,
    ),
)

METRIC_VARIANCE_SPECS = (
    (
        "estimated_cjr",
        "estimated_cjr_variance_barplot.jpg",
        "Event-rate estimator variance (squared pp)",
    ),
    (
        "estimated_restricted_mean",
        "estimated_rmttu_variance_barplot.jpg",
        "Restricted-mean estimator variance (turns squared)",
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
    """Generate Red-Team/Qwen-judge restricted-mean main-text figures."""
    dataset_key = "red_team_qwen_judge"
    dataset = frame[frame["dataset_key"].eq(dataset_key)].copy()
    rows: list[dict] = []
    path = output_dir / "red_team_qwen_estimated_rmttu_boxplot.jpg"
    generated = plot_grouped_boxplot(
        dataset,
        metric="estimated_restricted_mean",
        output_path=path,
        ylabel=r"Estimated restricted mean $E[\min(T,200)]$",
        quality=quality,
        x_order=TARGET_MODEL_ORDER,
        method_order=MAIN_METHOD_ORDER,
        reference_by_x=_reference_map(
            dataset, "full_benchmark_restricted_mean"
        ),
        reference_label="Full-benchmark truth",
        figsize=(7.0, 3.25),
    )
    _record(
        rows,
        dataset_key=dataset_key,
        metric="estimated_restricted_mean",
        path=path,
        generated=generated,
        scope="metrics-main",
    )
    path = output_dir / "red_team_qwen_estimated_rmttu_variance_barplot.jpg"
    generated, _ = plot_grouped_variance(
        dataset,
        metric="estimated_restricted_mean",
        output_path=path,
        ylabel="Restricted-mean variance (turns squared)",
        quality=quality,
        x_order=TARGET_MODEL_ORDER,
        method_order=MAIN_METHOD_ORDER,
        figsize=(7.0, 3.25),
    )
    _record(
        rows,
        dataset_key=dataset_key,
        metric="estimated_restricted_mean_variance",
        path=path,
        generated=generated,
        scope="metrics-main",
    )
    return pd.DataFrame(rows)
