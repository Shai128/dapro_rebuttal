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
import seaborn as sns

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


def _draw_box(
        axis,
        data: pd.DataFrame,
        *,
        metric: str,
        ylabel: str,
        xlabel: str,
        order: list[str],
) -> None:
    sns.boxplot(
        data=data,
        x="factor_label",
        y=metric,
        hue="method",
        order=order,
        hue_order=list(METHOD_ORDER),
        palette={method: METHOD_COLORS[method] for method in METHOD_ORDER},
        linewidth=0.9,
        fliersize=2.0,
        ax=axis,
    )
    _style(axis, xlabel=xlabel, ylabel=ylabel)
    if axis.legend_ is not None:
        axis.legend_.remove()


def generate_ablation_figure(
        data: pd.DataFrame,
        *,
        kind: str,
        output_path: Path,
        quality: str,
) -> None:
    spec = FACTOR_SPECS[kind]
    values = sorted(data["factor_value"].dropna().unique())
    label_by_value = (
        data[["factor_value", "factor_label"]]
        .drop_duplicates("factor_value")
        .set_index("factor_value")["factor_label"]
        .to_dict()
    )
    labels = [str(label_by_value[value]) for value in values]
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.1))
    panels = (
        ("coverage_pct", "Coverage rate (%)"),
        ("budget_used_per_sample", "Budget Used per Sample"),
        ("coverage_diff_pct", "|Coverage - target| (pp)"),
        (
            "mean_target_a_weight",
            r"Mean selected-target weight $A_i(q_{\hat\tau})/\pi_i$",
        ),
    )
    for axis, (metric, ylabel) in zip(axes.flat, panels):
        _draw_box(
            axis,
            data,
            metric=metric,
            ylabel=ylabel,
            xlabel=str(spec["xlabel"]),
            order=labels,
        )
        if metric == "coverage_pct":
            axis.axhline(90.0, color="#D62728", linestyle="--", linewidth=1.2)
        elif metric == "budget_used_per_sample":
            if kind == "budget":
                for position, value in enumerate(values):
                    axis.hlines(
                        value, position - 0.46, position + 0.46,
                        color="#D62728", linestyle="--", linewidth=1.2,
                    )
            else:
                axis.axhline(20.0, color="#D62728", linestyle="--", linewidth=1.2)
    handles = [
        plt.Line2D([0], [0], color=METHOD_COLORS[method], lw=7)
        for method in METHOD_ORDER
    ]
    figure.legend(
        handles,
        METHOD_ORDER,
        title="Method",
        loc="center left",
        bbox_to_anchor=(0.86, 0.5),
        frameon=False,
    )
    figure.suptitle(str(spec["title"]), y=0.995, fontsize=13)
    figure.tight_layout(rect=(0.0, 0.0, 0.85, 0.97), h_pad=2.0, w_pad=1.7)
    save_jpeg(figure, output_path, quality)
    plt.close(figure)


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
        generate_ablation_figure(
            data, kind=kind, output_path=path, quality=args.quality
        )
        data["ablation_kind"] = kind
        inventory["ablation_kind"] = kind
        all_data.append(data)
        all_inventory.append(inventory)
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
    pd.DataFrame(manifest).to_csv(
        args.output_dir / "dapro_ablation_figure_manifest.csv", index=False
    )


if __name__ == "__main__":
    main()
