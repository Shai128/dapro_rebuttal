"""Create paper-style figures for the Toxicity/Qwen LPB DAPRO ablations."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.paper_figures.common import save_jpeg  # noqa: E402
from src.paper_figures.config import METHOD_COLORS  # noqa: E402
from src.predictive_bounds.experiments.full_bounds.config import (  # noqa: E402
    method_display_name,
)


METHOD_ORDER = ("Static", "DAPRO", "DAPRO w/o CRC")
FACTOR_SPECS = {
    "n1": {
        "xlabel": r"Phase-I sample size $N_1$",
        "title": r"Phase-I sample-size ablation",
    },
    "score_noise": {
        "xlabel": r"Score-noise strength $\lambda$",
        "title": r"Score-quality ablation",
    },
    "budget": {
        "xlabel": r"Target budget per sample (paired $N_1$ shown)",
        "title": r"Budget ablation",
    },
}
SOURCE_COLUMNS = {
    "seed",
    "calibration_name",
    "target_coverage",
    "coverage",
    "configured_budget_per_sample",
    "reported_assigned_budget_per_sample",
    "actual_event_stopped_budget_per_sample",
    "mean_calibrated_a_weighted_inverse_probability",
    "mean_a_weighted_inverse_probability",
    "ablation_kind",
    "ablation_value",
    "ablation_n1",
    "ablation_crc_control_size",
    "ablation_uses_crc",
    "ablation_score_noise_lambda",
    "ablation_score_mean_timewise_pearson_correlation",
    "ablation_score_original_k2_bin_agreement",
}


def _read_available(path: Path) -> pd.DataFrame:
    available = set(pd.read_csv(path, nrows=0).columns)
    frame = pd.read_csv(path, usecols=sorted(SOURCE_COLUMNS & available))
    for column in SOURCE_COLUMNS:
        if column not in frame:
            frame[column] = np.nan
    return frame


def _discover(input_dir: Path, suffix: str) -> list[Path]:
    paths = [
        path for path in input_dir.rglob("all_df.csv")
        if path.parent.name.endswith(f"_{suffix}")
    ]
    if not paths:
        raise FileNotFoundError(
            f"No merged ablation results ending in _{suffix} below {input_dir}."
        )
    return sorted(paths)


def _first_available(frame: pd.DataFrame, *columns: str) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        result = result.where(result.notna(), values)
    return result


def load_ablation_data(
        input_dir: Path,
        *,
        experiment_prefix: str,
        kind: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load seed-level rows and replicate Static at every factor value."""
    suffix = f"{experiment_prefix}_{kind}"
    frames: list[pd.DataFrame] = []
    inventory: list[dict] = []
    for path in _discover(input_dir, suffix):
        frame = _read_available(path)
        frame = frame[np.isclose(
            pd.to_numeric(frame["target_coverage"], errors="coerce"),
            0.90,
            atol=5e-7,
        )].copy()
        frame["method"] = frame["calibration_name"].map(method_display_name)
        frame = frame[frame["method"].isin(METHOD_ORDER)].copy()
        dynamic = frame["method"].isin({"DAPRO", "DAPRO w/o CRC"})
        dynamic_rows = frame[dynamic].copy()
        dynamic_rows = dynamic_rows[
            dynamic_rows["ablation_kind"].astype(str).eq(kind)
        ]
        values = sorted(
            pd.to_numeric(dynamic_rows["ablation_value"], errors="coerce")
            .dropna().unique()
        )
        if not values:
            continue
        static = frame[frame["method"].eq("Static")].copy()
        expanded_static = []
        for value in values:
            copy = static.copy()
            copy["ablation_value"] = value
            matching_n1 = pd.to_numeric(
                dynamic_rows.loc[
                    np.isclose(
                        pd.to_numeric(
                            dynamic_rows["ablation_value"], errors="coerce"
                        ),
                        value,
                    ),
                    "ablation_n1",
                ],
                errors="coerce",
            ).dropna()
            if not matching_n1.empty:
                copy["ablation_n1"] = float(matching_n1.iloc[0])
            expanded_static.append(copy)
        selected = pd.concat(
            [dynamic_rows, *expanded_static], ignore_index=True, sort=False
        )
        selected["source_file"] = str(path)
        frames.append(selected)
        inventory.append({
            "kind": kind,
            "source_file": str(path),
            "factor_values": " ".join(f"{value:g}" for value in values),
            "seed_count": int(selected["seed"].nunique()),
            "row_count": len(selected),
        })
    if not frames:
        raise ValueError(f"No usable {kind} ablation rows were found.")

    data = pd.concat(frames, ignore_index=True, sort=False)
    data["factor_value"] = pd.to_numeric(
        data["ablation_value"], errors="coerce"
    )
    data["coverage_pct"] = 100 * pd.to_numeric(
        data["coverage"], errors="coerce"
    )
    data["coverage_diff_pct"] = (data["coverage_pct"] - 90.0).abs()
    assigned = pd.to_numeric(
        data["reported_assigned_budget_per_sample"], errors="coerce"
    )
    realized = pd.to_numeric(
        data["actual_event_stopped_budget_per_sample"], errors="coerce"
    )
    data["budget_used_per_sample"] = np.where(
        data["method"].eq("Static"), assigned, realized
    )
    data["mean_target_a_weight"] = _first_available(
        data,
        "mean_calibrated_a_weighted_inverse_probability",
        "mean_a_weighted_inverse_probability",
    )
    if kind == "n1":
        data["factor_label"] = data["factor_value"].round().astype("Int64").astype(str)
    elif kind == "budget":
        budget_label = data["factor_value"].round().astype("Int64").astype(str)
        n1_label = pd.to_numeric(
            data["ablation_n1"], errors="coerce"
        ).round().astype("Int64").astype(str)
        data["factor_label"] = budget_label + "\n" + r"($N_1$=" + n1_label + ")"
    else:
        data["factor_label"] = data["factor_value"].map(lambda value: f"{value:g}")

    duplicate_key = ["source_file", "seed", "method", "factor_value"]
    duplicates = data.duplicated(duplicate_key, keep=False)
    if bool(duplicates.any()):
        raise ValueError(
            "Duplicate ablation rows detected:\n"
            + data.loc[duplicates, duplicate_key + ["calibration_name"]]
            .to_string(index=False)
        )
    return data, pd.DataFrame(inventory)


def _style(axis, *, xlabel: str, ylabel: str) -> None:
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel, labelpad=8, fontsize=9 if len(ylabel) > 42 else 10)
    axis.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.48)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def summarize_line_statistics(
        data: pd.DataFrame,
        *,
        metric: str,
) -> pd.DataFrame:
    """Return the mean, variance, and SD plotted for every paired cell."""
    return (
        data.dropna(subset=["factor_value", "method", metric])
        .groupby(["factor_value", "method"], observed=True)[metric]
        .agg(mean="mean", variance="var", std="std", count="count")
        .reset_index()
    )


def _draw_line(
        axis,
        data: pd.DataFrame,
        *,
        metric: str,
        ylabel: str,
        xlabel: str,
        values: list[float],
) -> pd.DataFrame:
    statistics = summarize_line_statistics(data, metric=metric)
    line_styles = {
        "Static": "--",
        "DAPRO": "-",
        "DAPRO w/o CRC": "-.",
    }
    markers = {"Static": "x", "DAPRO": "o", "DAPRO w/o CRC": "s"}
    for method in METHOD_ORDER:
        method_stats = statistics[statistics["method"].eq(method)].copy()
        if method_stats.empty:
            continue
        method_stats = method_stats.set_index("factor_value").reindex(values)
        x_values = np.asarray(values, dtype=float)
        means = method_stats["mean"].to_numpy(dtype=float)
        stds = method_stats["std"].fillna(0.0).to_numpy(dtype=float)
        color = METHOD_COLORS[method]
        axis.plot(
            x_values,
            means,
            color=color,
            linestyle=line_styles[method],
            marker=markers[method],
            markersize=4.0,
            linewidth=1.45,
            label=method,
            zorder=3,
        )
        axis.fill_between(
            x_values,
            means - stds,
            means + stds,
            color=color,
            alpha=0.14,
            linewidth=0,
            zorder=1,
        )
    axis.set_xticks(values)
    axis.set_xticklabels([f"{value:g}" for value in values])
    _style(axis, xlabel=xlabel, ylabel=ylabel)
    return statistics


def generate_ablation_figure(
        data: pd.DataFrame,
        *,
        kind: str,
        output_path: Path,
        quality: str,
) -> pd.DataFrame:
    spec = FACTOR_SPECS[kind]
    values = sorted(data["factor_value"].dropna().unique())
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.1))
    panels = (
        ("coverage_pct", "Coverage Rate", "Coverage rate (%)"),
        (
            "budget_used_per_sample",
            "Budget Used per Sample",
            "Budget Used per Sample",
        ),
        (
            "coverage_diff_pct",
            "Coverage Difference",
            "|Coverage - target| (pp)",
        ),
        (
            "mean_target_a_weight",
            "Mean Target-A Weight",
            r"Mean selected-target weight $A_i(q_{\hat\tau})/\pi_i$",
        ),
    )
    all_statistics = []
    for axis, (metric, panel_title, ylabel) in zip(axes.flat, panels):
        statistics = _draw_line(
            axis,
            data,
            metric=metric,
            ylabel=ylabel,
            xlabel=str(spec["xlabel"]),
            values=values,
        )
        statistics["metric"] = metric
        all_statistics.append(statistics)
        axis.set_title(panel_title, fontsize=11)
        if metric == "coverage_pct":
            axis.axhline(90.0, color="#D62728", linestyle="--", linewidth=1.2)
        elif metric == "budget_used_per_sample":
            if kind == "budget":
                axis.plot(
                    values,
                    values,
                    color="#D62728",
                    linestyle="--",
                    linewidth=1.2,
                    zorder=2,
                )
            else:
                axis.axhline(20.0, color="#D62728", linestyle="--", linewidth=1.2)
    # Keep the shared legend inside one panel to save horizontal space.
    axes.flat[1].legend(
        title="Method",
        loc="best",
        frameon=True,
        framealpha=0.88,
        fontsize=9,
        title_fontsize=9,
    )
    for axis in (axes.flat[0], axes.flat[2], axes.flat[3]):
        legend = axis.get_legend()
        if legend is not None:
            legend.remove()
    figure.suptitle(
        f"{spec['title']} (mean $\\pm$ 1 SD across splits)",
        y=0.995,
        fontsize=13,
    )
    bottom = 0.075 if kind == "budget" else 0.02
    if kind == "budget":
        mapping = (
            data[["factor_value", "ablation_n1"]]
            .dropna()
            .drop_duplicates("factor_value")
            .sort_values("factor_value")
        )
        mapping_text = ", ".join(
            f"{row.factor_value:g}$\\to${int(row.ablation_n1)}"
            for row in mapping.itertuples(index=False)
        )
        figure.text(
            0.5,
            0.012,
            r"Paired budget $\to N_1$: " + mapping_text,
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    figure.tight_layout(rect=(0.0, bottom, 1.0, 0.97), h_pad=2.0, w_pad=1.7)
    save_jpeg(figure, output_path, quality)
    plt.close(figure)
    return pd.concat(all_statistics, ignore_index=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=_ROOT / "results" / "merged_calibration_dfs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_ROOT / "figures" / "paper" / "ablations",
    )
    parser.add_argument(
        "--experiment-prefix", default="dapro_lpb_ablation_v1"
    )
    parser.add_argument(
        "--kinds",
        nargs="+",
        choices=tuple(FACTOR_SPECS),
        default=list(FACTOR_SPECS),
        help="Ablation panels to generate (default: all three).",
    )
    parser.add_argument("--quality", choices=("low", "high"), default="low")
    parser.add_argument(
        "--strict-seeds",
        action="store_true",
        help="Require exactly 50 random split rows for every method/factor cell.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_data = []
    all_inventory = []
    all_statistics = []
    manifest = []
    for kind in args.kinds:
        data, inventory = load_ablation_data(
            args.input_dir,
            experiment_prefix=args.experiment_prefix,
            kind=kind,
        )
        counts = data.groupby(
            ["factor_value", "method"], observed=True
        )["seed"].nunique()
        if args.strict_seeds and not bool(counts.eq(50).all()):
            raise ValueError(
                f"{kind} does not contain 50 splits in every cell:\n{counts}"
            )
        path = args.output_dir / f"dapro_{kind}_ablation.jpg"
        statistics = generate_ablation_figure(
            data, kind=kind, output_path=path, quality=args.quality
        )
        statistics["ablation_kind"] = kind
        data["ablation_kind"] = kind
        inventory["ablation_kind"] = kind
        all_data.append(data)
        all_inventory.append(inventory)
        all_statistics.append(statistics)
        manifest.append({
            "ablation_kind": kind,
            "figure": str(path),
            "generated": path.exists(),
        })
        print(f"Generated {path}")
    pd.concat(all_data, ignore_index=True).to_csv(
        args.output_dir / "dapro_ablation_plot_data.csv", index=False
    )
    pd.concat(all_inventory, ignore_index=True).to_csv(
        args.output_dir / "dapro_ablation_result_inventory.csv", index=False
    )
    pd.concat(all_statistics, ignore_index=True).to_csv(
        args.output_dir / "dapro_ablation_mean_variance.csv", index=False
    )
    pd.DataFrame(manifest).to_csv(
        args.output_dir / "dapro_ablation_figure_manifest.csv", index=False
    )


if __name__ == "__main__":
    main()
