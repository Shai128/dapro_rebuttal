"""Generate a navigable, low-resolution summary of merged bound results.

The script discovers every ``all_df.csv`` under ``results/merged``, reads only
the columns required for plotting, and groups outputs by dataset and purpose.
Seed-level metrics are box plots by target model and method; coverage variance
is the across-seed sample variance in squared percentage points. Low-quality
mode is the default and guarantees every JPEG is at most 100 KiB.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
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
    all_experiment_configs,
    calibration_names,
)
from src.evaluation.result_matrix import (
    DAPRO_ORACLE_METHODS,
    method_display_name as matrix_method_display_name,
    numeric_label,
    parse_lpb_result,
)


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT_DIR = ROOT / "results" / "merged_calibration_dfs"
DEFAULT_OUTPUT_DIR = ROOT / "figures" / "full" / "merged"
LOW_QUALITY_MAX_BYTES = 100 * 1024

TARGET_MODEL_ORDER = tuple(model.display_name for model in TARGET_MODELS)

BOX_METRICS = {
    "mean_weight": {
        "filename": "mean-weight.jpg",
        "ylabel": "Mean inverse-probability weight",
        "allocation_only": True,
        "log_scale": True,
        "group": "core",
    },
    "mean_a_weight": {
        "filename": "target-weight.jpg",
        "ylabel": r"Mean target weight $A_i/\pi_i$",
        "allocation_only": True,
        "log_scale": True,
        "group": "core",
    },
    "coverage_pct": {
        "filename": "coverage.jpg",
        "ylabel": "Coverage rate (%)",
        "reference": "target_coverage_pct",
        "group": "core",
    },
    "coverage_diff_pct": {
        "filename": "coverage-gap.jpg",
        "ylabel": "Absolute coverage difference (pp)",
        "group": "core",
    },
    "budget_used_per_sample": {
        "filename": "realized-budget.jpg",
        "ylabel": "Budget used per sample",
        "reference": "target_budget",
        "allocation_only": True,
        "exclude_oracle": True,
        "group": "core",
    },
    "total_expected_budget_per_sample": {
        "filename": "expected-budget.jpg",
        "ylabel": "Expected budget per sample",
        "reference": "target_budget",
        "allocation_only": True,
        "group": "diagnostics",
    },
    "max_weight": {
        "filename": "max-weight.jpg",
        "ylabel": "Maximum inverse-probability weight",
        "allocation_only": True,
        "log_scale": True,
        "group": "diagnostics",
    },
    "a_weighted_effective_sample_size": {
        "filename": "target-ess.jpg",
        "ylabel": "Target-weight effective sample size",
        "allocation_only": True,
        "group": "diagnostics",
    },
    "method_runtime_seconds": {
        "filename": "runtime.jpg",
        "ylabel": "Runtime per seed (seconds)",
        "log_scale": True,
        "group": "diagnostics",
    },
}


def _long_io_path(path: Path) -> str:
    absolute = str(path.resolve())
    if os.name == "nt" and not absolute.startswith("\\\\?\\"):
        return f"\\\\?\\{absolute}"
    return absolute


def experiment_name(config: ExperimentConfig, suffix: str) -> str:
    from src.predictive_bounds.utils.utils import get_calibration_experiment_name

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
    from src.predictive_bounds.utils.utils import (
        get_merged_calibration_result_path,
        get_merged_upb_calibration_result_path,
    )

    name = experiment_name(config, suffix)
    resolver = (
        get_merged_calibration_result_path
        if config.bound_type == "lpb"
        else get_merged_upb_calibration_result_path
    )
    return ROOT / resolver(name) / "all_df.csv"


PLOT_COLUMNS = {
    "seed",
    "calibration_name",
    "target_coverage",
    "coverage",
    "mean_weight",
    "mean_a_weighted_inverse_probability",
    "budget_used",
    "max_weight",
    "total_expected_budget_per_sample",
    "a_weighted_effective_sample_size",
    "method_runtime_seconds",
    "configured_cal_size",
}

_METHOD_N1_RE = re.compile(r"(?:^|_)n1_(?P<n1>\d+)(?:_|$)")
_METHOD_CRC_RE = re.compile(r"(?:^|_)control_(?P<crc>\d+)(?:_|$)")
_COMPACT_LPB_VERSION_RE = re.compile(
    r"^(?P<base>.+_calibration_lpb)_v(?P<version>\d+)$"
)
EXCLUDED_DISPLAY_METHODS = DAPRO_ORACLE_METHODS | frozenset({
    "Legacy DAPRO",
    "Legacy DAPRO + CRC",
    "Local + CRC",
})


def _method_n1(calibration_name: str) -> int | None:
    """Extract N1, including historical method names that omit N1=100."""
    match = _METHOD_N1_RE.search(str(calibration_name))
    if match is not None:
        return int(match.group("n1"))
    if "projected_optimization" in str(calibration_name):
        return 100
    return None


def _compact_result_configurations(
        calibration_names: pd.Series,
) -> list[tuple[int, int]]:
    """Discover the N1/CRC pairs stored in one compact merged result."""
    pairs = set()
    for name in calibration_names.dropna().astype(str).unique():
        n1 = _method_n1(name)
        if n1 is None:
            continue
        match = _METHOD_CRC_RE.search(name)
        if match is not None:
            pairs.add((n1, int(match.group("crc"))))
    # A compact file can contain only non-CRC DAPRO rows after a partial run.
    # Retain those N1 values using the registry's configured half-split rule.
    discovered_n1 = {
        n1
        for name in calibration_names.dropna().astype(str).unique()
        if (n1 := _method_n1(name)) is not None
    }
    paired_n1 = {n1 for n1, _ in pairs}
    for n1 in discovered_n1 - paired_n1:
        pairs.add((n1, min(100, n1 // 2)))
    return sorted(pairs, reverse=True)


def _prefer_latest_compact_lpb_results(paths: list[Path]) -> list[Path]:
    """Ignore stale compact result versions when a newer version exists."""
    passthrough = []
    latest_by_experiment: dict[str, tuple[int, Path]] = {}
    for path in paths:
        match = _COMPACT_LPB_VERSION_RE.match(path.parent.name)
        if match is None:
            passthrough.append(path)
            continue
        key = match.group("base")
        candidate = (int(match.group("version")), path)
        previous = latest_by_experiment.get(key)
        if previous is None or candidate[0] > previous[0]:
            latest_by_experiment[key] = candidate
    return sorted(
        passthrough
        + [candidate[1] for candidate in latest_by_experiment.values()]
    )


def _read_plot_columns(path: Path) -> pd.DataFrame:
    """Read only plotting columns; merged files contain hundreds of metrics."""
    return pd.read_csv(
        _long_io_path(path),
        usecols=lambda column: column in PLOT_COLUMNS,
    )


def _fallback_method_name(calibration_name: str) -> str:
    """Create a readable label if a future method is not in the registry."""
    cleaned = calibration_name
    for token in ("calibration_", "_allocation", "survival_calibration"):
        cleaned = cleaned.replace(token, "")
    return cleaned.replace("_", " ").strip().title()


def _prepare_frame(
        frame: pd.DataFrame,
        config: ExperimentConfig,
        source_path: Path,
) -> pd.DataFrame:
    for column in PLOT_COLUMNS:
        if column not in frame:
            frame[column] = np.nan
    numeric_columns = PLOT_COLUMNS - {"calibration_name"}
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame[np.isclose(
        frame["target_coverage"],
        config.target_coverage,
        atol=5e-7,
    )].copy()
    if frame.empty:
        return frame

    frame["method"] = frame["calibration_name"].map(METHOD_DISPLAY)
    unknown = frame["method"].isna() & frame["calibration_name"].notna()
    frame.loc[unknown, "method"] = frame.loc[
        unknown, "calibration_name"
    ].map(_fallback_method_name)
    frame = frame[
        ~frame["method"].isin(EXCLUDED_DISPLAY_METHODS)
    ].copy()
    frame["target_model"] = config.target_model.display_name
    frame["target_model_key"] = config.target_model.key
    frame["configuration"] = config.key
    frame["dataset_key"] = config.figure_dataset_name
    frame["dataset_display"] = config.display_dataset_name
    frame["bound_type"] = config.bound_type.upper()
    frame["source_file"] = str(source_path)
    frame["target_coverage_pct"] = 100 * config.target_coverage
    frame["target_budget"] = config.budget_per_sample
    frame["coverage_pct"] = 100 * frame["coverage"]
    frame["coverage_diff_pct"] = (
        frame["coverage_pct"] - frame["target_coverage_pct"]
    ).abs()
    frame["mean_a_weight"] = frame[
        "mean_a_weighted_inverse_probability"
    ]
    calibration_size = frame["configured_cal_size"].fillna(config.cal_size)
    frame["budget_used_per_sample"] = frame["budget_used"] / calibration_size

    raw = frame["calibration_name"] == UNCALIBRATED
    frame.loc[raw, [
        "mean_weight",
        "mean_a_weight",
        "budget_used_per_sample",
        "max_weight",
        "total_expected_budget_per_sample",
        "a_weighted_effective_sample_size",
    ]] = np.nan
    return frame


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
        frame = _read_plot_columns(path)
        requested = set(calibration_names(config.bound_type))
        available = set(frame["calibration_name"].dropna().unique())
        absent = sorted(requested - available)
        if absent:
            missing_methods.append((config.key, absent))
        frame = frame[frame["calibration_name"].isin(requested)].copy()
        frame = _prepare_frame(frame, config, path)
        if frame.empty:
            missing_methods.append((
                config.key,
                [f"target coverage {config.target_coverage:.2f}"],
            ))
            continue

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


def _match_result_config(path: Path) -> ExperimentConfig | None:
    directory_name = path.parent.name
    matches = []
    for config in all_experiment_configs():
        base = experiment_name(config, "")
        if directory_name == base or directory_name.startswith(f"{base}__"):
            matches.append((len(base), config))
    if not matches:
        return None
    return max(matches, key=lambda item: item[0])[1]


def load_merged_directory(
        input_dir: Path,
        *,
        configuration_keys: set[str] | None = None,
        target_models: set[str] | None = None,
) -> pd.DataFrame:
    """Discover and load every recognized ``all_df.csv`` below a directory."""
    paths = sorted(input_dir.rglob("all_df.csv"))
    if not paths:
        raise FileNotFoundError(f"No all_df.csv files found below {input_dir}.")

    frames = []
    unrecognized = []
    for path in paths:
        config = _match_result_config(path)
        if config is None:
            unrecognized.append(path)
            continue
        if configuration_keys and config.key not in configuration_keys:
            continue
        if target_models and config.target_model.key not in target_models:
            continue
        frame = _prepare_frame(_read_plot_columns(path), config, path)
        if not frame.empty:
            frames.append(frame)

    if unrecognized:
        print(f"Skipped {len(unrecognized)} unrecognized result directories.")
        for path in unrecognized:
            print(f"  {path}")
    if not frames:
        raise FileNotFoundError(
            f"No recognized results with the requested coverage were found "
            f"below {input_dir}."
        )
    return pd.concat(frames, ignore_index=True)


def load_lpb_matrix(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load every suffixed all-method LPB result for every budget and N1."""
    frames = []
    inventory_rows = []
    skipped = []
    discovered_paths = sorted(input_dir.rglob("all_df.csv"))
    selected_paths = _prefer_latest_compact_lpb_results(discovered_paths)
    superseded_count = len(discovered_paths) - len(selected_paths)
    if superseded_count:
        print(
            f"Ignored {superseded_count} superseded compact LPB result "
            "file(s)."
        )
    for path in selected_paths:
        metadata = parse_lpb_result(path)
        if metadata is None:
            skipped.append(path)
            continue
        frame = _read_plot_columns(path)
        for column in PLOT_COLUMNS:
            if column not in frame:
                frame[column] = np.nan
        for column in PLOT_COLUMNS - {"calibration_name"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame[np.isclose(
            frame["target_coverage"], 0.90, atol=5e-7
        )].copy()
        if frame.empty:
            skipped.append(path)
            continue

        frame["method"] = frame["calibration_name"].map(
            matrix_method_display_name
        )
        frame = frame[
            ~frame["method"].isin(EXCLUDED_DISPLAY_METHODS)
        ].copy()
        method_n1 = frame["calibration_name"].map(_method_n1)
        configurations = (
            [(metadata.dapro_n1, metadata.crc_control_size)]
            if metadata.dapro_n1 is not None
            else _compact_result_configurations(frame["calibration_name"])
        )
        if not configurations:
            skipped.append(path)
            continue
        common_rows = method_n1.isna()
        frame["target_model"] = metadata.target_model_display
        frame["target_model_key"] = metadata.target_model
        frame["configuration"] = metadata.dataset_key
        frame["dataset_key"] = metadata.dataset_key
        frame["dataset_display"] = metadata.dataset_display
        frame["bound_type"] = "LPB"
        frame["source_file"] = str(path)
        frame["target_coverage_pct"] = 90.0
        frame["target_budget"] = metadata.budget_per_sample
        frame["coverage_pct"] = 100 * frame["coverage"]
        frame["coverage_diff_pct"] = (
            frame["coverage_pct"] - frame["target_coverage_pct"]
        ).abs()
        frame["mean_a_weight"] = frame[
            "mean_a_weighted_inverse_probability"
        ]
        calibration_size = frame["configured_cal_size"].fillna(3000)
        frame["budget_used_per_sample"] = (
            frame["budget_used"] / calibration_size
        )
        raw = frame["calibration_name"] == UNCALIBRATED
        frame.loc[raw, [
            "mean_weight",
            "mean_a_weight",
            "budget_used_per_sample",
            "max_weight",
            "total_expected_budget_per_sample",
            "a_weighted_effective_sample_size",
        ]] = np.nan
        for n1, crc in configurations:
            config_frame = frame[common_rows | (method_n1 == n1)].copy()
            config_frame["dapro_n1"] = n1
            config_frame["crc_control_size"] = crc
            config_frame["plot_context"] = (
                f"budget={metadata.budget_per_sample:g}, DAPRO N1={n1}"
            )
            frames.append(config_frame)
            inventory_rows.append({
                "dataset": metadata.dataset_key,
                "target_model": metadata.target_model,
                "budget_per_sample": metadata.budget_per_sample,
                "dapro_n1": n1,
                "crc_control_size": crc,
                "seed_count": config_frame["seed"].nunique(),
                "method_count": config_frame["calibration_name"].nunique(),
                "source_file": str(path),
            })
    if skipped:
        print(
            f"Ignored {len(skipped)} LPB result files that are not suffixed "
            "all-method N1/budget experiments."
        )
    if not frames:
        raise FileNotFoundError(
            f"No suffixed LPB matrix results found below {input_dir}."
        )
    return (
        pd.concat(frames, ignore_index=True),
        pd.DataFrame(inventory_rows),
    )


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
    known = [value for value in order if value in present]
    return known + sorted(present - set(known))


def _style_axis(axis, ylabel: str) -> None:
    axis.set_xlabel("Target model")
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.55)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _place_legend(axis, figure) -> None:
    handles, labels = axis.get_legend_handles_labels()
    if axis.legend_ is not None:
        axis.legend_.remove()
    if handles:
        figure.legend(
            handles,
            labels,
            title="Method",
            loc="upper center",
            bbox_to_anchor=(0.5, 0.94),
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
        dpi=110,
        bbox_inches="tight",
        pil_kwargs={"quality": 76, "optimize": True, "progressive": True},
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
                    getattr(Image, "Resampling", Image).LANCZOS,
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
    plot_frame = plot_frame[
        ~plot_frame["method"].isin(EXCLUDED_DISPLAY_METHODS)
    ]
    if specification.get("allocation_only"):
        plot_frame = plot_frame[plot_frame["method"] != "Raw"]
    if specification.get("exclude_oracle"):
        plot_frame = plot_frame[plot_frame["method"] != "Oracle"]
    hue_order = _ordered_present(plot_frame["method"], METHOD_ORDER)
    target_order = _ordered_present(
        plot_frame["target_model"], TARGET_MODEL_ORDER
    )
    figure, axis = plt.subplots(figsize=(11.5, 5.8))
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
    if specification.get("log_scale"):
        positive = plot_frame.loc[plot_frame[metric] > 0, metric]
        if not positive.empty:
            axis.set_yscale("log")
    reference = specification.get("reference")
    if reference:
        values = plot_frame[reference].dropna().unique()
        if len(values) == 1:
            axis.axhline(
                values[0], color="#c62828", linestyle="--", linewidth=1.8,
                label="Target",
            )
    _style_axis(axis, specification["ylabel"])
    dataset_name = str(plot_frame["dataset_display"].iloc[0])
    context = ""
    if "plot_context" in plot_frame:
        values = plot_frame["plot_context"].dropna().unique()
        if len(values) == 1:
            context = f" ({values[0]})"
    figure.suptitle(
        f"{dataset_name}: {specification['ylabel']}{context}", y=0.995
    )
    _place_legend(axis, figure)
    figure.tight_layout(rect=(0, 0, 1, 0.72))
    _save_jpeg(figure, path, quality)
    plt.close(figure)


def _plot_coverage_variance(
        frame: pd.DataFrame,
        path: Path,
        quality: str,
) -> pd.DataFrame:
    variance = coverage_variance_frame(
        frame[~frame["method"].isin(EXCLUDED_DISPLAY_METHODS)]
    ).dropna(
        subset=["coverage_variance_pp2", "method"]
    )
    hue_order = _ordered_present(variance["method"], METHOD_ORDER)
    target_order = _ordered_present(
        variance["target_model"], TARGET_MODEL_ORDER
    )
    figure, axis = plt.subplots(figsize=(11.5, 5.8))
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
    dataset_name = str(variance["dataset_display"].iloc[0])
    context = ""
    if "plot_context" in frame:
        values = frame["plot_context"].dropna().unique()
        if len(values) == 1:
            context = f" ({values[0]})"
    figure.suptitle(
        f"{dataset_name}: coverage variance{context}", y=0.995
    )
    _place_legend(axis, figure)
    figure.tight_layout(rect=(0, 0, 1, 0.72))
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
    manifest_rows = []
    variance_frames = []
    for dataset_key, dataset_frame in frame.groupby(
            "dataset_key", sort=False, observed=True
    ):
        for metric, specification in BOX_METRICS.items():
            group = specification["group"]
            path = output_dir / dataset_key / group / specification["filename"]
            _plot_box_metric(
                dataset_frame,
                metric,
                specification,
                path,
                quality,
            )
            generated.append(path)
            manifest_rows.append({
                "dataset": dataset_key,
                "group": group,
                "metric": metric,
                "label": specification["ylabel"].replace("$", ""),
                "path": path.relative_to(output_dir).as_posix(),
            })
        variance_path = output_dir / dataset_key / "core" / "coverage-variance.jpg"
        variance_frames.append(_plot_coverage_variance(
            dataset_frame,
            variance_path,
            quality,
        ))
        generated.append(variance_path)
        manifest_rows.append({
            "dataset": dataset_key,
            "group": "core",
            "metric": "coverage_variance_pp2",
            "label": "Coverage variance",
            "path": variance_path.relative_to(output_dir).as_posix(),
        })

    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(data_dir / "plot-data.csv", index=False)
    pd.concat(variance_frames, ignore_index=True).to_csv(
        data_dir / "coverage-variances.csv", index=False
    )
    manifest = pd.DataFrame(manifest_rows)
    manifest["size_kib"] = [
        round((output_dir / relative).stat().st_size / 1024, 1)
        for relative in manifest["path"]
    ]
    manifest.to_csv(output_dir / "figure-index.csv", index=False)
    _write_figure_readme(output_dir, manifest)
    return generated


def generate_lpb_matrix_figures(
        frame: pd.DataFrame,
        inventory: pd.DataFrame,
        output_dir: Path,
        quality: str,
) -> list[Path]:
    """Generate one complete figure tree per budget/N1/CRC combination."""
    generated = []
    combination_rows = []
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
        paths = generate_all_figures(combination, combination_dir, quality)
        generated.extend(paths)
        combination_rows.append({
            "budget_per_sample": budget,
            "dapro_n1": int(n1),
            "crc_control_size": int(crc),
            "dataset_count": combination["dataset_key"].nunique(),
            "target_model_count": combination["target_model_key"].nunique(),
            "source_file_count": combination["source_file"].nunique(),
            "figure_count": len(paths),
            "directory": str(combination_dir.relative_to(output_dir)),
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(combination_rows).to_csv(
        output_dir / "matrix-index.csv", index=False
    )
    inventory.to_csv(output_dir / "result-inventory.csv", index=False)
    return generated


def _write_figure_readme(output_dir: Path, manifest: pd.DataFrame) -> None:
    lines = [
        "# Predictive-bound figures",
        "",
        "Figures are grouped first by dataset and then as core comparisons or diagnostics.",
        "Method labels are shared across every plot; the oracle has infinite budget.",
        "",
    ]
    for dataset, dataset_rows in manifest.groupby("dataset", sort=False):
        lines.extend([f"## {dataset}", ""])
        for group, group_rows in dataset_rows.groupby("group", sort=False):
            lines.extend([f"### {group.title()}", ""])
            for row in group_rows.itertuples(index=False):
                lines.append(
                    f"- [{row.label}]({row.path}) ({row.size_kib:.1f} KiB)"
                )
            lines.append("")
    lines.extend([
        "## Data",
        "",
        "- [Seed-level plot data](data/plot-data.csv)",
        "- [Coverage variances](data/coverage-variances.csv)",
        "- [Machine-readable figure index](figure-index.csv)",
        "",
    ])
    (output_dir / "README.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quality", choices=["high", "low"], default="low"
    )
    parser.add_argument(
        "--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
        help="Directory recursively containing merged all_df.csv files.",
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
    parser.add_argument(
        "--legacy-config-mode",
        action="store_true",
        help="Use the original manuscript-config discovery instead of the "
        "budget/N1 result matrix.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.legacy_config_mode:
        frame, inventory = load_lpb_matrix(args.input_dir)
        print(
            f"Loaded {len(frame):,} LPB rows from "
            f"{frame['source_file'].nunique()} files, covering "
            f"{frame['target_budget'].nunique()} budgets and "
            f"{frame['dapro_n1'].nunique()} N1 values."
        )
        paths = generate_lpb_matrix_figures(
            frame, inventory, args.output_dir, args.quality
        )
        print(f"Generated {len(paths)} LPB figures below {args.output_dir}.")
        return
    frame = load_merged_directory(
        args.input_dir,
        configuration_keys=set(args.configs or []),
        target_models=set(args.target_model or []),
    )
    print(
        f"Loaded {len(frame):,} rows from "
        f"{frame['source_file'].nunique()} merged files; "
        f"{frame['method'].nunique()} methods."
    )
    paths = generate_all_figures(frame, args.output_dir, args.quality)
    for path in paths:
        print(f"{path}: {path.stat().st_size / 1024:.1f} KiB")


if __name__ == "__main__":
    main()
