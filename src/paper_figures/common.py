"""Shared plotting, validation, and low-resolution export helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns

from src.paper_figures.config import METHOD_COLORS, METHOD_ORDER


LOW_QUALITY_MAX_BYTES = 120 * 1024
FULL_DATA_REFERENCE_COLOR = "#800020"
FULL_DATA_REFERENCE_LINESTYLE = "-."
TARGET_REFERENCE_COLOR = "#D62728"
TARGET_REFERENCE_LINESTYLE = "--"


def reference_line_style(label: str) -> tuple[str, str]:
    """Return the paper-wide style for a named horizontal reference."""
    if "full-data" in label.casefold():
        return FULL_DATA_REFERENCE_COLOR, FULL_DATA_REFERENCE_LINESTYLE
    return TARGET_REFERENCE_COLOR, TARGET_REFERENCE_LINESTYLE


def _validate_figure_text(figure, path: Path) -> None:
    """Fail fast when visible text is clipped or tick labels overlap."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    width, height = figure.canvas.get_width_height()
    tolerance = 2.0
    clipped: list[str] = []
    for artist in figure.findobj(match=matplotlib.text.Text):
        label = artist.get_text().strip()
        if not label or not artist.get_visible():
            continue
        bounds = artist.get_window_extent(renderer=renderer)
        coordinates = np.asarray(
            [bounds.x0, bounds.y0, bounds.x1, bounds.y1], dtype=float
        )
        if not bool(np.isfinite(coordinates).all()):
            continue
        if (
            bounds.x0 < -tolerance
            or bounds.y0 < -tolerance
            or bounds.x1 > width + tolerance
            or bounds.y1 > height + tolerance
        ):
            clipped.append(
                label.replace("\n", " / ")
                + f" @ ({bounds.x0:.1f}, {bounds.y0:.1f}, "
                + f"{bounds.x1:.1f}, {bounds.y1:.1f}) in {width}x{height}"
            )

    overlaps: list[str] = []
    for axis in figure.axes:
        for labels in (axis.get_xticklabels(), axis.get_yticklabels()):
            visible = [
                label for label in labels
                if label.get_visible() and label.get_text().strip()
            ]
            boxes = [
                label.get_window_extent(renderer=renderer)
                for label in visible
            ]
            for index, left in enumerate(boxes):
                for other_index in range(index + 1, len(boxes)):
                    right = boxes[other_index]
                    intersection_width = min(left.x1, right.x1) - max(
                        left.x0, right.x0
                    )
                    intersection_height = min(left.y1, right.y1) - max(
                        left.y0, right.y0
                    )
                    if intersection_width > 1.0 and intersection_height > 1.0:
                        overlaps.append(
                            visible[index].get_text().replace("\n", " / ")
                            + " <> "
                            + visible[other_index].get_text().replace("\n", " / ")
                        )
    if clipped or overlaps:
        details = []
        if clipped:
            details.append("clipped=" + repr(sorted(set(clipped))))
        if overlaps:
            details.append("overlapping_ticks=" + repr(sorted(set(overlaps))))
        raise ValueError(
            f"Figure text-layout validation failed for {path}: "
            + "; ".join(details)
        )


def ordered_present(values: Iterable[str], order: Sequence[str]) -> list[str]:
    present = {str(value) for value in values if pd.notna(value)}
    return [value for value in order if value in present]


def save_jpeg(
    figure,
    path: Path,
    quality: str,
    *,
    tight: bool = True,
    panel_count: int = 1,
    resolution_scale: float = 1.0,
) -> None:
    """Save a paper-ready JPEG with an area-aware preview resolution.

    ``panel_count`` scales the low-quality pixel and byte budgets with the
    amount of information in the canvas.  Multiplying DPI by the number of
    panels would scale the total pixel count quadratically; instead, the
    multi-panel canvas itself supplies most of the extra area and we apply a
    modest DPI increase for readable draft exports.  High-quality exports are
    always written at the publication-standard 300 DPI.
    """
    panel_count = max(1, int(panel_count))
    resolution_scale = max(1.0, float(resolution_scale))
    path.parent.mkdir(parents=True, exist_ok=True)
    _validate_figure_text(figure, path)
    bbox_inches = "tight" if tight else None
    pad_inches = 0.16 if tight else 0.0
    if quality == "high":
        figure.savefig(
            path,
            format="jpg",
            dpi=round(300 * resolution_scale),
            bbox_inches=bbox_inches,
            pad_inches=pad_inches,
            pil_kwargs={"quality": 95, "optimize": True},
        )
        return

    preview_dpi = round(
        min(240, 120 + 20 * (panel_count - 1)) * resolution_scale
    )
    maximum_bytes = round(
        LOW_QUALITY_MAX_BYTES * panel_count * resolution_scale**2
    )
    figure.savefig(
        path,
        format="jpg",
        dpi=preview_dpi,
        bbox_inches=bbox_inches,
        pad_inches=pad_inches,
        pil_kwargs={"quality": 78, "optimize": True, "progressive": True},
    )
    jpeg_quality = 76
    while path.stat().st_size > maximum_bytes:
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
        if jpeg_quality < 30 and path.stat().st_size > maximum_bytes:
            raise RuntimeError(
                f"Could not compress {path} below "
                f"{maximum_bytes // 1024} KiB."
            )


def _style_axis(
    axis,
    *,
    xlabel: str,
    ylabel: str,
    font_scale: float = 1.0,
) -> None:
    axis.set_xlabel(xlabel, fontsize=10.0 * font_scale, labelpad=5)
    # A number of the paper diagnostics have necessarily descriptive labels.
    # Give them explicit breathing room so Matplotlib's tight bounding box does
    # not place the first glyph on the JPEG boundary (the long variance label
    # used to lose its first words in low-quality exports).
    ylabel_fontsize = (9 if len(ylabel) >= 42 else 10) * font_scale
    axis.set_ylabel(ylabel, fontsize=ylabel_fontsize, labelpad=9)
    axis.tick_params(axis="both", labelsize=8.8 * font_scale)
    axis.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.48)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if axis.get_yscale() == "linear":
        axis.yaxis.set_major_locator(MaxNLocator(nbins="auto", prune="upper"))


def _legend_outside(
    axis,
    figure,
    *,
    columns: int = 3,
    font_scale: float = 1.0,
) -> None:
    handles, labels = axis.get_legend_handles_labels()
    if axis.legend_ is not None:
        axis.legend_.remove()
    if handles:
        figure.legend(
            handles,
            labels,
            title="Method",
            loc="center left",
            # Keep the legend inside a fixed reserved column.  This makes
            # every exported diagnostic exactly the same pixel size instead
            # of allowing a tight bounding box to grow with label length.
            # Reserve enough horizontal room for labels such as
            # ``Full-data value``; the former 0.785 anchor clipped that
            # label in the metric-estimation figures.
            bbox_to_anchor=(0.67, 0.5),
            frameon=False,
            ncol=1,
            fontsize=8.8 * font_scale,
            title_fontsize=9.2 * font_scale,
        )


def plot_shared_legend(
    *,
    output_path: Path,
    quality: str,
    methods: Sequence[str],
    reference_label: str | None = None,
    figsize: tuple[float, float] = (1.25, 2.55),
    font_scale: float = 1.0,
) -> bool:
    """Export one compact, publication-sized legend shared by main panels."""
    present = [method for method in methods if method in METHOD_COLORS]
    if not present and reference_label is None:
        output_path.unlink(missing_ok=True)
        return False
    handles = [
        Patch(facecolor=METHOD_COLORS[method], edgecolor="none", label=method)
        for method in present
    ]
    if reference_label is not None:
        reference_color, reference_linestyle = reference_line_style(
            reference_label
        )
        handles.append(Line2D(
            [0], [0],
            color=reference_color,
            linestyle=reference_linestyle,
            linewidth=1.35,
            label=reference_label,
        ))
    figure = plt.figure(figsize=figsize)
    figure.legend(
        handles=handles,
        labels=[handle.get_label() for handle in handles],
        title="Method",
        loc="center",
        frameon=False,
        ncol=1,
        fontsize=8.8 * font_scale,
        title_fontsize=9.2 * font_scale,
        handlelength=1.45,
        handletextpad=0.55,
        labelspacing=0.65,
        borderaxespad=0.0,
    )
    save_jpeg(figure, output_path, quality, tight=False)
    plt.close(figure)
    return True


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
    show_legend: bool = True,
    font_scale: float = 1.0,
    title: str | None = None,
    resolution_scale: float = 1.0,
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
        reference_color, reference_linestyle = reference_line_style(
            reference_label
        )
        axis.axhline(
            reference,
            color=reference_color,
            linestyle=reference_linestyle,
            linewidth=1.25,
            label=f"{reference_label} ({reference:g})",
        )
    if reference_by_x:
        reference_color, reference_linestyle = reference_line_style(
            reference_label
        )
        labelled = False
        for position, category in enumerate(categories):
            value = reference_by_x.get(str(category))
            if value is None or not np.isfinite(value):
                continue
            axis.hlines(
                value,
                position - 0.46,
                position + 0.46,
                color=reference_color,
                linestyle=reference_linestyle,
                linewidth=1.25,
                label=reference_label if not labelled else None,
            )
            labelled = True
    _style_axis(
        axis,
        xlabel=xlabel,
        ylabel=ylabel,
        font_scale=font_scale,
    )
    if title:
        axis.set_title(title, fontsize=11.2 * font_scale, pad=7)
    if show_legend:
        _legend_outside(
            axis,
            figure,
            columns=len(methods),
            font_scale=font_scale,
        )
    elif axis.legend_ is not None:
        axis.legend_.remove()
    compact = figsize[0] < 5.0
    short_wide = figsize[0] >= 5.0 and figsize[1] <= 2.8
    figure.subplots_adjust(
        left=0.255 if compact else 0.19,
        right=(0.65 if show_legend else 0.97),
        bottom=0.23 if compact else 0.29 if short_wide else 0.18,
        top=(
            0.82 if title and short_wide
            else 0.90 if short_wide
            else 0.86 if title
            else 0.95
        ),
    )
    save_jpeg(
        figure,
        output_path,
        quality,
        tight=False,
        resolution_scale=resolution_scale,
    )
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
    show_legend: bool = True,
    font_scale: float = 1.0,
    title: str | None = None,
    resolution_scale: float = 1.0,
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
    _style_axis(
        axis,
        xlabel=xlabel,
        ylabel=ylabel,
        font_scale=font_scale,
    )
    if title:
        axis.set_title(title, fontsize=11.2 * font_scale, pad=7)
    if show_legend:
        _legend_outside(
            axis,
            figure,
            columns=len(methods),
            font_scale=font_scale,
        )
    elif axis.legend_ is not None:
        axis.legend_.remove()
    compact = figsize[0] < 5.0
    short_wide = figsize[0] >= 5.0 and figsize[1] <= 2.8
    figure.subplots_adjust(
        left=0.255 if compact else 0.19,
        right=(0.65 if show_legend else 0.97),
        bottom=0.23 if compact else 0.29 if short_wide else 0.18,
        top=(
            0.82 if title and short_wide
            else 0.90 if short_wide
            else 0.86 if title
            else 0.95
        ),
    )
    save_jpeg(
        figure,
        output_path,
        quality,
        tight=False,
        resolution_scale=resolution_scale,
    )
    plt.close(figure)
    return True, variance


def write_readme(output_root: Path, lines: Sequence[str]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "README.md").write_text(
        "\n".join(lines).rstrip() + "\n", encoding="utf-8"
    )
