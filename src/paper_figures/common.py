"""Shared plotting, validation, and low-resolution export helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns

from src.paper_figures.config import METHOD_COLORS, METHOD_ORDER


LOW_QUALITY_MAX_BYTES = 120 * 1024


def ordered_present(values: Iterable[str], order: Sequence[str]) -> list[str]:
    present = {str(value) for value in values if pd.notna(value)}
    return [value for value in order if value in present]


def save_jpeg(figure, path: Path, quality: str) -> None:
    """Save a paper-ready JPEG, aggressively compressing low-quality output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if quality == "high":
        figure.savefig(
            path,
            format="jpg",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.16,
            pil_kwargs={"quality": 95, "optimize": True},
        )
        return

    figure.savefig(
        path,
        format="jpg",
        dpi=120,
        bbox_inches="tight",
        pad_inches=0.16,
        pil_kwargs={"quality": 78, "optimize": True, "progressive": True},
    )
    jpeg_quality = 76
    while path.stat().st_size > LOW_QUALITY_MAX_BYTES:
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            if jpeg_quality < 48:
                image = image.resize(
                    (
                        max(640, int(image.width * 0.88)),
                        max(360, int(image.height * 0.88)),
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
            raise RuntimeError(f"Could not compress {path} below 120 KiB.")


def _style_axis(axis, *, xlabel: str, ylabel: str) -> None:
    axis.set_xlabel(xlabel)
    # A number of the paper diagnostics have necessarily descriptive labels.
    # Give them explicit breathing room so Matplotlib's tight bounding box does
    # not place the first glyph on the JPEG boundary (the long variance label
    # used to lose its first words in low-quality exports).
    ylabel_fontsize = 9 if len(ylabel) >= 42 else 10
    axis.set_ylabel(ylabel, fontsize=ylabel_fontsize, labelpad=9)
    axis.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.48)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _legend_outside(axis, figure, *, columns: int = 3) -> None:
    handles, labels = axis.get_legend_handles_labels()
    if axis.legend_ is not None:
        axis.legend_.remove()
    if handles:
        figure.legend(
            handles,
            labels,
            title="Method",
            loc="center left",
            bbox_to_anchor=(0.985, 0.5),
            frameon=False,
            ncol=1 if columns <= 3 else 2,
        )


def plot_grouped_boxplot(
    frame: pd.DataFrame,
    *,
    metric: str,
    output_path: Path,
    ylabel: str,
    quality: str,
    x: str = "target_model",
    xlabel: str = "Target model",
    x_order: Sequence[str] | None = None,
    method_order: Sequence[str] = METHOD_ORDER,
    reference: float | None = None,
    reference_by_x: dict[str, float] | None = None,
    reference_label: str = "Reference",
    hide_methods: Sequence[str] = (),
    log_scale: bool = False,
    figsize: tuple[float, float] = (7.4, 3.5),
) -> bool:
    plot_frame = frame.loc[
        ~frame["method"].isin(hide_methods), [x, "method", metric]
    ].dropna(subset=[x, "method", metric]).copy()
    methods = ordered_present(plot_frame["method"], method_order)
    if not methods:
        output_path.unlink(missing_ok=True)
        return False
    plot_frame = plot_frame[plot_frame["method"].isin(methods)]
    categories = (
        ordered_present(plot_frame[x], x_order)
        if x_order is not None
        else list(dict.fromkeys(plot_frame[x].astype(str)))
    )
    figure, axis = plt.subplots(figsize=figsize)
    sns.boxplot(
        data=plot_frame,
        x=x,
        y=metric,
        hue="method",
        order=categories,
        hue_order=methods,
        palette={method: METHOD_COLORS[method] for method in methods},
        linewidth=0.9,
        fliersize=2.0,
        showmeans=False,
        ax=axis,
    )
    if log_scale and bool((plot_frame[metric] > 0).any()):
        axis.set_yscale("log")
    if reference is not None and np.isfinite(reference):
        axis.axhline(
            reference,
            color="#D62728",
            linestyle="--",
            linewidth=1.25,
            label=f"{reference_label} ({reference:g})",
        )
    if reference_by_x:
        labelled = False
        for position, category in enumerate(categories):
            value = reference_by_x.get(str(category))
            if value is None or not np.isfinite(value):
                continue
            axis.hlines(
                value,
                position - 0.46,
                position + 0.46,
                color="#D62728",
                linestyle="--",
                linewidth=1.25,
                label=reference_label if not labelled else None,
            )
            labelled = True
    _style_axis(axis, xlabel=xlabel, ylabel=ylabel)
    _legend_outside(axis, figure, columns=len(methods))
    figure.tight_layout(rect=(0, 0, 0.83, 1))
    save_jpeg(figure, output_path, quality)
    plt.close(figure)
    return True


def plot_grouped_variance(
    frame: pd.DataFrame,
    *,
    metric: str,
    output_path: Path,
    ylabel: str,
    quality: str,
    x: str = "target_model",
    xlabel: str = "Target model",
    x_order: Sequence[str] | None = None,
    method_order: Sequence[str] = METHOD_ORDER,
    hide_methods: Sequence[str] = (),
    normalize_to_static: bool = False,
    figsize: tuple[float, float] = (7.4, 3.5),
) -> tuple[bool, pd.DataFrame]:
    plot_frame = frame.loc[
        ~frame["method"].isin(hide_methods), [x, "method", metric]
    ].dropna(subset=[x, "method", metric]).copy()
    variance = (
        plot_frame.groupby([x, "method"], observed=True, as_index=False)[metric]
        .var(ddof=1)
        .rename(columns={metric: "variance"})
    )
    if normalize_to_static:
        static = variance.loc[
            variance["method"].eq("Static"), [x, "variance"]
        ].rename(columns={"variance": "static_variance"})
        variance = variance.merge(static, on=x, how="left", validate="many_to_one")
        variance["variance"] = np.where(
            variance["static_variance"] > 0,
            variance["variance"] / variance["static_variance"],
            np.nan,
        )
    variance = variance.dropna(subset=["variance"])
    methods = ordered_present(variance["method"], method_order)
    if not methods:
        output_path.unlink(missing_ok=True)
        return False, variance
    variance = variance[variance["method"].isin(methods)]
    categories = (
        ordered_present(variance[x], x_order)
        if x_order is not None
        else list(dict.fromkeys(variance[x].astype(str)))
    )
    figure, axis = plt.subplots(figsize=figsize)
    sns.barplot(
        data=variance,
        x=x,
        y="variance",
        hue="method",
        order=categories,
        hue_order=methods,
        palette={method: METHOD_COLORS[method] for method in methods},
        errorbar=None,
        ax=axis,
    )
    _style_axis(axis, xlabel=xlabel, ylabel=ylabel)
    _legend_outside(axis, figure, columns=len(methods))
    figure.tight_layout(rect=(0, 0, 0.83, 1))
    save_jpeg(figure, output_path, quality)
    plt.close(figure)
    return True, variance


def write_readme(output_root: Path, lines: Sequence[str]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "README.md").write_text(
        "\n".join(lines).rstrip() + "\n", encoding="utf-8"
    )
