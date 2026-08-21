"""Paper plot generation for lower and upper predictive bounds."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

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


BOUND_BOX_SPECS = (
    ("coverage_pct", "coverage_boxplot.jpg", "Coverage rate (%)", False),
    ("size", "size_boxplot.jpg", "Predictive-bound value", False),
    (
        "coverage_diff_pct",
        "coverage_diff_boxplot.jpg",
        "|Coverage - target| (pp)",
        False,
    ),
    (
        "budget_used_per_sample",
        "budget_used_boxplot.jpg",
        "Budget Used per Sample",
        True,
    ),
    (
        "n_observed_events",
        "n_observed_events_boxplot.jpg",
        "Number of observed events",
        True,
    ),
    (
        "mean_weight",
        "mean_weight_boxplot.jpg",
        r"Mean inverse-probability weight $1/\pi_i$",
        True,
    ),
    (
        "mean_selected_a_weight",
        "mean_calibrated_a_weight_boxplot.jpg",
        r"Mean selected-target weight $A_i(q_{\hat\tau})/\pi_i$",
        True,
    ),
    (
        "mean_prior_a_weight",
        "mean_prior_a_weight_boxplot.jpg",
        r"Mean prior-target weight $A_i(q_{\tau_{prior}})/\pi_i$",
        True,
    ),
    (
        "mean_tau_0p10_a_weight",
        "mean_tau_0p10_a_weight_boxplot.jpg",
        r"Mean $\alpha=0.1$ target weight $A_i(q_{0.1})/\pi_i$",
        True,
    ),
)


def _record(
    rows: list[dict],
    *,
    task: str,
    dataset_key: str,
    metric: str,
    path: Path,
    generated: bool,
) -> None:
    rows.append({
        "task": task,
        "dataset_key": dataset_key,
        "metric": metric,
        "generated": bool(generated),
        "path": str(path),
    })


def generate_bound_appendix_figures(
    frame: pd.DataFrame,
    *,
    task: str,
    output_dir: Path,
    quality: str,
) -> pd.DataFrame:
    """Generate one single-configuration appendix figure set per dataset."""
    manifest: list[dict] = []
    for dataset_key in DATASET_ORDER:
        dataset = frame[frame["dataset_key"].eq(dataset_key)].copy()
        if dataset.empty:
            continue
        stem = DATASET_FILE_STEMS[dataset_key]
        if task == "upb":
            stem += "_upb"
        target = float(dataset["target_coverage_pct"].iloc[0])
        for metric, filename, ylabel, allocation_only in BOUND_BOX_SPECS:
            path = output_dir / f"{stem}_{filename}"
            generated = plot_grouped_boxplot(
                dataset,
                metric=metric,
                output_path=path,
                ylabel=ylabel,
                quality=quality,
                x_order=TARGET_MODEL_ORDER,
                method_order=METHOD_ORDER,
                reference=(
                    target if metric == "coverage_pct"
                    else float(dataset["target_budget"].iloc[0])
                    if metric == "budget_used_per_sample"
                    else None
                ),
                reference_label=(
                    "Target coverage"
                    if metric == "coverage_pct"
                    else "Target budget"
                ),
                hide_methods=(
                    ("Uncalibrated", "Oracle") if allocation_only else ()
                ),
            )
            _record(
                manifest,
                task=task,
                dataset_key=dataset_key,
                metric=metric,
                path=path,
                generated=generated,
            )

        variance_path = output_dir / f"{stem}_coverage_variance_barplot.jpg"
        generated, _ = plot_grouped_variance(
            dataset,
            metric="coverage_pct",
            output_path=variance_path,
            ylabel="Coverage variance across splits (squared pp)",
            quality=quality,
            x_order=TARGET_MODEL_ORDER,
            method_order=METHOD_ORDER,
        )
        _record(
            manifest,
            task=task,
            dataset_key=dataset_key,
            metric="coverage_variance",
            path=variance_path,
            generated=generated,
        )
        size_path = output_dir / f"{stem}_size_variance_barplot.jpg"
        generated, _ = plot_grouped_variance(
            dataset,
            metric="size",
            output_path=size_path,
            ylabel="Bound-size variance (Static normalized to 1)",
            quality=quality,
            x_order=TARGET_MODEL_ORDER,
            method_order=METHOD_ORDER,
            normalize_to_static=True,
        )
        _record(
            manifest,
            task=task,
            dataset_key=dataset_key,
            metric="normalized_size_variance",
            path=size_path,
            generated=generated,
        )
    return pd.DataFrame(manifest)


def generate_lpb_main_figures(
    frame: pd.DataFrame, *, output_dir: Path, quality: str
) -> pd.DataFrame:
    """Generate the two requested toxicity LPB main-text panels."""
    toxicity = frame[frame["dataset_key"].eq("toxicity")].copy()
    rows = []
    specifications = (
        (
            "coverage_diff_pct",
            "toxicity_coverage_diff_boxplot.jpg",
            "|Coverage - target| (pp)",
            (),
            None,
        ),
        (
            "budget_used_per_sample",
            "toxicity_budget_used_boxplot.jpg",
            "Budget Used per Sample",
            ("Uncalibrated",),
            float(toxicity["target_budget"].iloc[0]) if not toxicity.empty else None,
        ),
    )
    for metric, filename, ylabel, hidden, reference in specifications:
        path = output_dir / filename
        generated = plot_grouped_boxplot(
            toxicity,
            metric=metric,
            output_path=path,
            ylabel=ylabel,
            quality=quality,
            x_order=TARGET_MODEL_ORDER,
            method_order=MAIN_METHOD_ORDER,
            reference=reference,
            reference_label=(
                "Target budget"
                if metric == "budget_used_per_sample"
                else "Reference"
            ),
            hide_methods=hidden,
            figsize=(7.0, 3.25),
        )
        _record(
            rows,
            task="lpb-main",
            dataset_key="toxicity",
            metric=metric,
            path=path,
            generated=generated,
        )
    return pd.DataFrame(rows)


def generate_autoif_main_figures(
    lpb_frame: pd.DataFrame,
    upb_frame: pd.DataFrame,
    *,
    output_dir: Path,
    quality: str,
    target_model: str = "Qwen2.5",
) -> pd.DataFrame:
    """Generate the combined LPB/UPB AutoIF main-text comparison."""
    combined = pd.concat(
        [
            lpb_frame[
                lpb_frame["dataset_key"].eq("autoif")
                & lpb_frame["target_model"].eq(target_model)
            ],
            upb_frame[
                upb_frame["dataset_key"].eq("autoif")
                & upb_frame["target_model"].eq(target_model)
            ],
        ],
        ignore_index=True,
    )
    rows = []
    specifications = (
        (
            "coverage_diff_pct",
            "autoif_coverage_diff_boxplot.jpg",
            "|Coverage - target| (pp)",
            (),
            None,
        ),
        (
            "budget_used_per_sample",
            "autoif_budget_used_boxplot.jpg",
            "Budget Used per Sample",
            ("Uncalibrated",),
            None,
        ),
    )
    for metric, filename, ylabel, hidden, reference in specifications:
        path = output_dir / filename
        generated = plot_grouped_boxplot(
            combined,
            metric=metric,
            output_path=path,
            ylabel=ylabel,
            quality=quality,
            x="bound_type",
            xlabel="Bound type",
            x_order=("LPB", "UPB"),
            method_order=MAIN_METHOD_ORDER,
            reference=reference,
            hide_methods=hidden,
            figsize=(6.2, 3.1),
        )
        _record(
            rows,
            task="bounds-main",
            dataset_key="autoif",
            metric=metric,
            path=path,
            generated=generated,
        )
    return pd.DataFrame(rows)
