"""Generate all manuscript box plots for the definitive method comparison.

Each ordinary metric is a seed-level box plot grouped by target model with
method as hue.  Coverage variance is the across-seed sample variance in
squared percentage points and is therefore shown as a grouped bar plot, as in
the manuscript.  Low-quality mode guarantees every JPEG is at most 100 KiB.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns

from src.predictive_bounds.experiments.full_bounds.config import (
    METHOD_COLORS,
    METHOD_DISPLAY,
    METHOD_ORDER,
    TARGET_MODELS,
    UNCALIBRATED,
    ExperimentConfig,
    calibration_names,
    select_configs,
)
from src.predictive_bounds.utils.utils import (
    get_calibration_experiment_name,
    get_merged_calibration_result_path,
    get_merged_upb_calibration_result_path,
)


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_DIR = ROOT / "figures" / "full"
LOW_QUALITY_MAX_BYTES = 100 * 1024

TARGET_MODEL_ORDER = tuple(model.display_name for model in TARGET_MODELS)

BOX_METRICS = {
    "mean_weight": {
        "filename": "mean_weight_boxplot.jpg",
        "ylabel": "Mean inverse-probability weight",
        "allocation_only": True,
    },
    "mean_a_weight": {
        "filename": "mean_a_weight_boxplot.jpg",
        "ylabel": r"Mean target weight $A_i/\pi_i$",
        "allocation_only": True,
    },
    "coverage_pct": {
        "filename": "coverage_boxplot.jpg",
        "ylabel": "Coverage rate (%)",
        "reference": "target_coverage_pct",
    },
    "coverage_diff_pct": {
        "filename": "coverage_diff_boxplot.jpg",
        "ylabel": "Absolute coverage difference (pp)",
    },
    "budget_used_per_sample": {
        "filename": "budget_used_boxplot.jpg",
        "ylabel": "Budget used per sample",
        "reference": "target_budget",
        "allocation_only": True,
    },
}


def _long_io_path(path: Path) -> str:
    absolute = str(path.resolve())
    if os.name == "nt" and not absolute.startswith("\\\\?\\"):
        return f"\\\\?\\{absolute}"
    return absolute


def experiment_name(config: ExperimentConfig, suffix: str) -> str:
    return get_calibration_experiment_name(
        config.dataset_name,
        config.dataset_setup,
        config.budget_per_sample,
        config.cal_size,
        config.tau_prior,
        config.m_upper_bound / config.budget_per_sample,
        suffix,
    )


def merged_result_path(config: ExperimentConfig, suffix: str) -> Path:
    name = experiment_name(config, suffix)
    resolver = (
        get_merged_calibration_result_path
        if config.bound_type == "lpb"
        else get_merged_upb_calibration_result_path
    )
    return ROOT / resolver(name) / "all_df.csv"


def load_comparison_data(
        configs: tuple[ExperimentConfig, ...],
        suffix: str,
        *,
        allow_missing: bool = False,
) -> pd.DataFrame:
    """Load the one requested coverage level from every merged result."""
    frames = []
    missing_paths = []
    missing_methods = []
    for config in configs:
        path = merged_result_path(config, suffix)
        if not os.path.exists(_long_io_path(path)):
            missing_paths.append(path)
            continue
        frame = pd.read_csv(_long_io_path(path))
        requested = set(calibration_names(config.bound_type))
        available = set(frame["calibration_name"].dropna().unique())
        absent = sorted(requested - available)
        if absent:
            missing_methods.append((config.key, absent))
        frame = frame[frame["calibration_name"].isin(requested)].copy()
        target_coverage = pd.to_numeric(
            frame["target_coverage"], errors="coerce"
        )
        frame = frame[np.isclose(
            target_coverage,
            config.target_coverage,
            atol=5e-7,
        )].copy()
        if frame.empty:
            missing_methods.append((
                config.key,
                [f"target coverage {config.target_coverage:.2f}"],
            ))
            continue

        for column in [
            "seed",
            "coverage",
            "mean_weight",
            "mean_a_weighted_inverse_probability",
            "budget_used",
        ]:
            if column not in frame:
                frame[column] = np.nan
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        frame["method"] = frame["calibration_name"].map(METHOD_DISPLAY)
        frame["target_model"] = config.target_model.display_name
        frame["target_model_key"] = config.target_model.key
        frame["configuration"] = config.key
        frame["dataset_key"] = config.figure_dataset_name
        frame["dataset_display"] = config.display_dataset_name
        frame["bound_type"] = config.bound_type.upper()
        frame["target_coverage_pct"] = 100 * config.target_coverage
        frame["target_budget"] = config.budget_per_sample
        frame["coverage_pct"] = 100 * frame["coverage"]
        frame["coverage_diff_pct"] = (
            frame["coverage_pct"] - frame["target_coverage_pct"]
        ).abs()
        frame["mean_a_weight"] = frame[
            "mean_a_weighted_inverse_probability"
        ]
        frame["budget_used_per_sample"] = (
            frame["budget_used"] / config.cal_size
        )

        uncalibrated = frame["calibration_name"] == UNCALIBRATED
        frame.loc[uncalibrated, [
            "mean_weight",
            "mean_a_weight",
            "budget_used_per_sample",
        ]] = np.nan
        frames.append(frame)

    if (missing_paths or missing_methods) and not allow_missing:
        details = [f"missing result: {path}" for path in missing_paths]
        details.extend(
            f"{key}: missing {methods}"
            for key, methods in missing_methods
        )
        raise FileNotFoundError(
            "The comparison is incomplete:\n" + "\n".join(details)
        )
    if not frames:
        raise FileNotFoundError("No complete merged comparison results found.")
    return pd.concat(frames, ignore_index=True)


def coverage_variance_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(
            [
                "dataset_key",
                "dataset_display",
                "bound_type",
                "target_model",
                "method",
                "target_coverage_pct",
                "target_budget",
            ],
            observed=True,
            as_index=False,
        )["coverage_pct"]
        .var(ddof=1)
        .rename(columns={"coverage_pct": "coverage_variance_pp2"})
    )


def _ordered_present(values: pd.Series, order: tuple[str, ...]) -> list[str]:
    present = set(values.dropna().unique())
    return [value for value in order if value in present]


def _style_axis(axis, ylabel: str) -> None:
    axis.set_xlabel("Target model")
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.55)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _place_legend(axis) -> None:
    handles, labels = axis.get_legend_handles_labels()
    if axis.legend_ is not None:
        axis.legend_.remove()
    if handles:
        axis.legend(
            handles,
            labels,
            title="Method",
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=4,
            frameon=False,
        )


def _save_jpeg(figure, path: Path, quality: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if quality == "high":
        figure.savefig(
            path,
            format="jpg",
            dpi=300,
            bbox_inches="tight",
            pil_kwargs={"quality": 95, "optimize": True},
        )
        return

    figure.savefig(
        path,
        format="jpg",
        dpi=145,
        bbox_inches="tight",
        pil_kwargs={"quality": 82, "optimize": True, "progressive": True},
    )
    jpeg_quality = 78
    while path.stat().st_size > LOW_QUALITY_MAX_BYTES:
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            if jpeg_quality < 48:
                image = image.resize(
                    (
                        max(640, int(image.width * 0.86)),
                        max(360, int(image.height * 0.86)),
                    ),
                    Image.Resampling.LANCZOS,
                )
            image.save(
                path,
                format="JPEG",
                quality=max(38, jpeg_quality),
                optimize=True,
                progressive=True,
            )
        jpeg_quality -= 8
        if jpeg_quality < 30 and path.stat().st_size > LOW_QUALITY_MAX_BYTES:
            raise RuntimeError(
                f"Could not compress {path} below 100 KiB without making it "
                "unreadable."
            )


def _plot_box_metric(
        frame: pd.DataFrame,
        metric: str,
        specification: dict,
        path: Path,
        quality: str,
) -> None:
    plot_frame = frame.dropna(subset=[metric, "method"]).copy()
    if specification.get("allocation_only"):
        plot_frame = plot_frame[plot_frame["method"] != "Uncalibrated"]
    hue_order = _ordered_present(plot_frame["method"], METHOD_ORDER)
    target_order = _ordered_present(
        plot_frame["target_model"], TARGET_MODEL_ORDER
    )
    figure, axis = plt.subplots(figsize=(12.5, 6.5))
    sns.boxplot(
        data=plot_frame,
        x="target_model",
        y=metric,
        hue="method",
        order=target_order,
        hue_order=hue_order,
        palette=METHOD_COLORS,
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": 4,
        },
        linewidth=1.1,
        fliersize=2.5,
        ax=axis,
    )
    reference = specification.get("reference")
    if reference:
        values = plot_frame[reference].dropna().unique()
        if len(values) == 1:
            axis.axhline(
                values[0], color="#c62828", linestyle="--", linewidth=1.8,
                label="Target",
            )
    _style_axis(axis, specification["ylabel"])
    _place_legend(axis)
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    _save_jpeg(figure, path, quality)
    plt.close(figure)


def _plot_coverage_variance(
        frame: pd.DataFrame,
        path: Path,
        quality: str,
) -> pd.DataFrame:
    variance = coverage_variance_frame(frame).dropna(
        subset=["coverage_variance_pp2", "method"]
    )
    hue_order = _ordered_present(variance["method"], METHOD_ORDER)
    target_order = _ordered_present(
        variance["target_model"], TARGET_MODEL_ORDER
    )
    figure, axis = plt.subplots(figsize=(12.5, 6.5))
    sns.barplot(
        data=variance,
        x="target_model",
        y="coverage_variance_pp2",
        hue="method",
        order=target_order,
        hue_order=hue_order,
        palette=METHOD_COLORS,
        errorbar=None,
        ax=axis,
    )
    _style_axis(axis, "Coverage variance (squared pp)")
    _place_legend(axis)
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    _save_jpeg(figure, path, quality)
    plt.close(figure)
    return variance


def generate_all_figures(
        frame: pd.DataFrame,
        output_dir: Path,
        quality: str,
) -> list[Path]:
    if quality not in {"high", "low"}:
        raise ValueError("quality must be 'high' or 'low'.")
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.25)
    generated = []
    variance_frames = []
    for dataset_key, dataset_frame in frame.groupby(
            "dataset_key", sort=False, observed=True
    ):
        for metric, specification in BOX_METRICS.items():
            path = output_dir / (
                f"{dataset_key}_{specification['filename']}"
            )
            _plot_box_metric(
                dataset_frame,
                metric,
                specification,
                path,
                quality,
            )
            generated.append(path)
        variance_path = output_dir / (
            f"{dataset_key}_coverage_variance_barplot.jpg"
        )
        variance_frames.append(_plot_coverage_variance(
            dataset_frame,
            variance_path,
            quality,
        ))
        generated.append(variance_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "full_bounds_plot_data.csv", index=False)
    pd.concat(variance_frames, ignore_index=True).to_csv(
        output_dir / "full_bounds_coverage_variances.csv", index=False
    )
    return generated


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suffix", default="full_bounds_v1")
    parser.add_argument(
        "--quality", choices=["high", "low"], default="high"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument("--config", action="append", dest="configs")
    parser.add_argument(
        "--target-model",
        action="append",
        choices=[model.key for model in TARGET_MODELS],
    )
    parser.add_argument("--available-only", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    configs = select_configs(
        ROOT,
        keys=set(args.configs or []),
        target_models=set(args.target_model or []),
        available_only=args.available_only,
    )
    if not configs:
        raise SystemExit("No configurations matched the requested filters.")
    frame = load_comparison_data(
        configs,
        args.suffix,
        allow_missing=args.allow_missing,
    )
    paths = generate_all_figures(frame, args.output_dir, args.quality)
    for path in paths:
        print(f"{path}: {path.stat().st_size / 1024:.1f} KiB")


if __name__ == "__main__":
    main()
