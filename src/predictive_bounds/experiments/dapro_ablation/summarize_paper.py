"""Create paper-style Toxicity/Qwen LPB and metric DAPRO ablations."""

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
ATTACKER_SHIFT_BUDGET = 10.0
SCORE_DISPLAY_LABELS = {
    "Current hazard": "Current\nhazard",
    "Remaining-time quantile": "Remaining-time\nquantile",
    "Causal target value": "Causal target\nvalue",
    "Causal event-rate target value": "Causal target\nvalue",
    "Random": "Random",
    "Oracle remaining time": "Oracle remaining\ntime",
}
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
    "hard_soft": {
        "xlabel": "Target coefficient",
        "title": "Hard-vs-soft target ablation",
    },
    "representation": {
        "xlabel": "Score-map representation",
        "title": "Representation-capacity ablation",
    },
    "score": {
        "xlabel": "Score definition",
        "title": "Score-function ablation",
    },
    "attacker_shift": {
        "xlabel": "Calibration attacker to test attacker",
        "title": "Attacker-shift ablation",
    },
}
SOURCE_COLUMNS = {
    "seed",
    "allocator_name",
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
    "ablation_label",
    "ablation_n1",
    "ablation_crc_control_size",
    "ablation_uses_crc",
    "ablation_score_noise_lambda",
    "ablation_score_mean_timewise_pearson_correlation",
    "ablation_score_original_k2_bin_agreement",
    "ablation_score_kind",
    "ablation_score_is_causal",
    "ablation_score_bin_count",
    "global_regularization",
    "ablation_continuous_score_map",
    "ablation_coefficient_kind",
    "phase1_oracle_optimized_objective",
    "phase1_projected_mean_objective_variance_proxy",
    "soft_mass_phase1_raw_policy_fit_mean_variance_proxy",
    "phase2_mean_objective_variance_proxy",
    "soft_mass_phase2_frozen_policy_mean_variance_proxy",
    "attacker_shift_enabled",
    "attacker_shift_source_dataset_name",
    "attacker_shift_source_dataset_setup",
    "attacker_shift_test_dataset_name",
    "attacker_shift_test_dataset_setup",
    "estimated_cjr",
    "oracle_cjr",
    "full_benchmark_cjr",
    "abs_diff_cjr",
    "num_events_observed",
    "mean_metric_target_a_weighted_inverse_probability",
    "conditional_variance_unsafe_event_rate_estimator",
    "estimated_conditional_variance_unsafe_event_rate_estimator",
}


def _read_available(path: Path) -> pd.DataFrame:
    available = set(pd.read_csv(path, nrows=0).columns)
    frame = pd.read_csv(path, usecols=sorted(SOURCE_COLUMNS & available))
    for column in SOURCE_COLUMNS:
        if column not in frame:
            frame[column] = np.nan
    return frame


def _discover(input_dir: Path, suffix: str) -> list[Path]:
    # Long Windows experiment names may fit directly below ``results`` but
    # exceed MAX_PATH after adding ``merged_calibration_dfs``.  Accept those
    # downloaded merged directories in the parent as a short-path fallback;
    # the canonical nested location remains preferred and duplicate paths are
    # removed below.
    candidates = list(input_dir.rglob("all_df.csv"))
    if input_dir.name in {
            "merged_calibration_dfs", "merged_metric_calibration_dfs"}:
        candidates.extend(input_dir.parent.glob("*/all_df.csv"))
    paths = [
        path for path in candidates
        if (
            path.parent.name.endswith(f"_{suffix}")
            or (
                suffix.endswith("_attacker_shift")
                and f"_{suffix}_" in path.parent.name
            )
        )
    ]
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(
            f"No merged ablation results ending in _{suffix} below {input_dir}."
        )
    return paths


def _first_available(frame: pd.DataFrame, *columns: str) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        result = result.where(result.notna(), values)
    return result


def _ablation_method_name(calibration_name: str) -> str | None:
    name = str(calibration_name)
    if name == "calibration_optimized_allocation":
        return "Static"
    if "_ablation_" in name and "dapro_" in name:
        return "DAPRO" if "budget_crc" in name else "DAPRO w/o CRC"
    return method_display_name(name)


def _attacker_name(dataset_setup: object) -> str:
    """Return the attacker family encoded in an experiment setup string."""
    setup = str(dataset_setup).lower()
    if "attack_gemma3" in setup:
        return "Gemma"
    if "attack_qwen25" in setup:
        return "Qwen"
    raise ValueError(
        "Attacker-shift setup must encode a Gemma3 or Qwen2.5 attacker; "
        f"got {dataset_setup!r}."
    )


def _attacker_shift_plot_identity(row: pd.Series) -> tuple[int, int, str]:
    """Return stable task/direction order and the displayed shift label."""
    dataset = str(row["attacker_shift_source_dataset_name"])
    if "red_team" in dataset:
        task_order, task_label = 0, "Red team"
    elif "toxicity" in dataset:
        task_order, task_label = 1, "Toxicity"
    else:
        raise ValueError(f"Unsupported attacker-shift dataset {dataset!r}.")
    source = _attacker_name(row["attacker_shift_source_dataset_setup"])
    target = _attacker_name(row["attacker_shift_test_dataset_setup"])
    if source == target:
        raise ValueError(
            "Attacker-shift source and test attackers must differ; "
            f"both are {source}."
        )
    direction_order = 0 if (source, target) == ("Gemma", "Qwen") else 1
    return task_order, direction_order, (
        task_label + f"\n{source} $\\to$ {target}"
    )


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
        # Attacker-shift experiments use B=10.  The result root may still
        # contain historical B=20 files with the same experiment suffix, so
        # filter by recorded configuration before combining paths.  This
        # prevents silently pooling two different budget regimes.
        if kind == "attacker_shift":
            configured_budget = pd.to_numeric(
                frame["configured_budget_per_sample"], errors="coerce"
            )
            frame = frame[np.isclose(
                configured_budget,
                ATTACKER_SHIFT_BUDGET,
                atol=5e-7,
            )].copy()
        frame["method"] = frame["calibration_name"].map(
            _ablation_method_name
        )
        frame = frame[frame["method"].isin(METHOD_ORDER)].copy()
        dynamic = frame["method"].isin({"DAPRO", "DAPRO w/o CRC"})
        dynamic_rows = frame[dynamic].copy()
        dynamic_rows = dynamic_rows[
            dynamic_rows["ablation_kind"].astype(str).eq(kind)
        ]
        # Do not silently combine pre-standardization K4/regularization-zero
        # files with the canonical paper ablations.  Representation is the
        # sole study allowed to vary K.
        bins = pd.to_numeric(
            dynamic_rows["ablation_score_bin_count"], errors="coerce"
        )
        if kind != "representation" and bins.notna().any():
            dynamic_rows = dynamic_rows[np.isclose(
                bins, 2.0, atol=5e-7
            )].copy()
        regularization = pd.to_numeric(
            dynamic_rows["global_regularization"], errors="coerce"
        )
        if regularization.notna().any():
            dynamic_rows = dynamic_rows[np.isclose(
                regularization.loc[dynamic_rows.index], 0.001, atol=5e-10
            )].copy()
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
        detail = (
            f" at configured budget {ATTACKER_SHIFT_BUDGET:g}"
            if kind == "attacker_shift"
            else ""
        )
        raise ValueError(
            f"No usable {kind} ablation rows{detail} were found."
        )

    data = pd.concat(frames, ignore_index=True, sort=False)
    data["factor_value"] = pd.to_numeric(
        data["ablation_value"], errors="coerce"
    )
    if kind == "attacker_shift":
        dynamic = data[~data["method"].eq("Static")].copy()
        setup_columns = [
            "attacker_shift_source_dataset_name",
            "attacker_shift_source_dataset_setup",
            "attacker_shift_test_dataset_setup",
        ]
        setup_rows = dynamic[setup_columns].drop_duplicates().copy()
        identities = setup_rows.apply(
            _attacker_shift_plot_identity, axis=1
        )
        setup_rows[["task_order", "direction_order", "shift_label"]] = (
            pd.DataFrame(identities.tolist(), index=setup_rows.index)
        )
        setup_rows = setup_rows.sort_values(
            ["task_order", "direction_order"]
        ).reset_index(drop=True)
        setup_rows["plot_index"] = setup_rows.index.astype(float)
        setup_index = {
            tuple(getattr(row, column) for column in setup_columns): (
                row.plot_index
            )
            for row in setup_rows.itertuples(index=False)
        }
        for row_index, row in data.iterrows():
            setup = (
                row["attacker_shift_source_dataset_name"],
                row["attacker_shift_source_dataset_setup"],
                row["attacker_shift_test_dataset_setup"],
            )
            if setup in setup_index:
                data.at[row_index, "factor_value"] = setup_index[setup]
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
    data["phase1_objective"] = _first_available(
        data,
        "soft_mass_phase1_raw_policy_fit_mean_variance_proxy",
        "phase1_projected_mean_objective_variance_proxy",
        "phase1_oracle_optimized_objective",
    )
    data["phase2_target_weight_objective"] = _first_available(
        data,
        "phase2_mean_objective_variance_proxy",
        "soft_mass_phase2_frozen_policy_mean_variance_proxy",
    )
    if kind == "n1":
        data["factor_label"] = data["factor_value"].round().astype("Int64").astype(str)
    elif kind == "budget":
        budget_label = data["factor_value"].round().astype("Int64").astype(str)
        n1_label = pd.to_numeric(
            data["ablation_n1"], errors="coerce"
        ).round().astype("Int64").astype(str)
        data["factor_label"] = budget_label + "\n" + r"($N_1$=" + n1_label + ")"
    elif kind in {"hard_soft", "representation", "score"}:
        labels = (
            data.loc[~data["method"].eq("Static"), ["factor_value", "ablation_label"]]
            .dropna().drop_duplicates("factor_value")
            .set_index("factor_value")["ablation_label"].to_dict()
        )
        data["factor_label"] = data["factor_value"].map(labels)
        if kind == "score":
            data["factor_label"] = data["factor_label"].replace(
                SCORE_DISPLAY_LABELS
            )
    elif kind == "attacker_shift":
        dynamic_labels = data[~data["method"].eq("Static")].copy()
        dynamic_labels["label"] = dynamic_labels.apply(
            lambda row: _attacker_shift_plot_identity(row)[2], axis=1
        )
        labels = dynamic_labels.drop_duplicates("factor_value").set_index(
            "factor_value"
        )["label"].to_dict()
        data["factor_label"] = data["factor_value"].map(labels)
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


def load_metric_ablation_data(
        input_dir: Path,
        *,
        experiment_prefix: str,
        kind: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load metric score-ablation rows and pair Static at every x value."""
    if kind not in {"score_noise", "score"}:
        raise ValueError("Metric figures support score_noise and score only.")
    suffix = f"{experiment_prefix}_{kind}"
    frames: list[pd.DataFrame] = []
    inventory: list[dict] = []
    for path in _discover(input_dir, suffix):
        frame = _read_available(path)
        frame["method"] = frame["calibration_name"].map(
            _ablation_method_name
        )
        frame = frame[frame["method"].isin(METHOD_ORDER)].copy()
        dynamic_rows = frame[
            frame["method"].isin({"DAPRO", "DAPRO w/o CRC"})
            & frame["ablation_kind"].astype(str).eq(kind)
        ].copy()
        bins = pd.to_numeric(
            dynamic_rows["ablation_score_bin_count"], errors="coerce"
        )
        if bins.notna().any():
            dynamic_rows = dynamic_rows[np.isclose(
                bins, 2.0, atol=5e-7
            )].copy()
        regularization = pd.to_numeric(
            dynamic_rows["global_regularization"], errors="coerce"
        )
        if regularization.notna().any():
            dynamic_rows = dynamic_rows[np.isclose(
                regularization.loc[dynamic_rows.index], 0.001, atol=5e-10
            )].copy()
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
            copy["ablation_n1"] = pd.to_numeric(
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
            ).dropna().iloc[0]
            expanded_static.append(copy)
        selected = pd.concat(
            [dynamic_rows, *expanded_static], ignore_index=True, sort=False
        )
        selected["source_file"] = str(path)
        frames.append(selected)
        inventory.append({
            "task": "metric",
            "kind": kind,
            "source_file": str(path),
            "factor_values": " ".join(f"{value:g}" for value in values),
            "seed_count": int(selected["seed"].nunique()),
            "row_count": len(selected),
        })
    if not frames:
        raise ValueError(f"No usable metric {kind} ablation rows were found.")

    data = pd.concat(frames, ignore_index=True, sort=False)
    data["factor_value"] = pd.to_numeric(
        data["ablation_value"], errors="coerce"
    )
    data["event_rate_pct"] = pd.to_numeric(
        data["estimated_cjr"], errors="coerce"
    )
    data["event_rate_abs_error_pp"] = _first_available(
        data, "abs_diff_cjr"
    )
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
        "mean_metric_target_a_weighted_inverse_probability",
        "mean_a_weighted_inverse_probability",
    )
    data["observed_events"] = pd.to_numeric(
        data["num_events_observed"], errors="coerce"
    )
    data["event_rate_conditional_variance_pp2"] = 10000 * _first_available(
        data,
        "conditional_variance_unsafe_event_rate_estimator",
        "estimated_conditional_variance_unsafe_event_rate_estimator",
    )
    # The requested estimator variance is the empirical variance over the 50
    # random calibration/test splits.  It is stored on every row in its cell
    # solely to reuse the line-plot/statistics machinery below.
    grouping = ["source_file", "factor_value", "method"]
    data["event_rate_across_split_variance_pp2"] = data.groupby(
        grouping, observed=True
    )["event_rate_pct"].transform("var")
    data["event_rate_truth_pct"] = _first_available(
        data, "full_benchmark_cjr", "oracle_cjr"
    )
    if kind == "score":
        labels = (
            data.loc[
                ~data["method"].eq("Static"),
                ["factor_value", "ablation_label"],
            ]
            .dropna().drop_duplicates("factor_value")
            .set_index("factor_value")["ablation_label"].to_dict()
        )
        data["factor_label"] = data["factor_value"].map(labels)
        data["factor_label"] = data["factor_label"].replace(
            SCORE_DISPLAY_LABELS
        )
    else:
        data["factor_label"] = data["factor_value"].map(
            lambda value: f"{value:g}"
        )

    duplicate_key = ["source_file", "seed", "method", "factor_value"]
    duplicates = data.duplicated(duplicate_key, keep=False)
    if bool(duplicates.any()):
        raise ValueError(
            "Duplicate metric-ablation rows detected:\n"
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
    labels = (
        data[["factor_value", "factor_label"]]
        .dropna().drop_duplicates("factor_value")
        .set_index("factor_value")["factor_label"].to_dict()
    )
    axis.set_xticks(values)
    tick_labels = (
        [f"{value:g}" for value in values]
        if xlabel.startswith("Target budget")
        else [labels.get(value, f"{value:g}") for value in values]
    )
    axis.set_xticklabels(
        tick_labels,
        rotation=18 if len(values) >= 5 and not xlabel.startswith("Target budget") else 0,
        ha="right" if len(values) >= 5 and not xlabel.startswith("Target budget") else "center",
    )
    _style(axis, xlabel=xlabel, ylabel=ylabel)
    return statistics


def generate_ablation_figure(
        data: pd.DataFrame,
        *,
        kind: str,
        output_path: Path,
        quality: str,
) -> pd.DataFrame:
    if kind in {"hard_soft", "attacker_shift"}:
        return generate_categorical_box_figure(
            data,
            kind=kind,
            output_path=output_path,
            quality=quality,
        )
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


def generate_categorical_box_figure(
        data: pd.DataFrame,
        *,
        kind: str,
        output_path: Path,
        quality: str,
) -> pd.DataFrame:
    """Use boxplots for categorical coefficient and distribution shifts."""
    if kind not in {"hard_soft", "attacker_shift"}:
        raise ValueError("Categorical boxplots support hard_soft/attacker_shift.")
    spec = FACTOR_SPECS[kind]
    values = sorted(data["factor_value"].dropna().unique())
    labels = (
        data[["factor_value", "factor_label"]]
        .dropna().drop_duplicates("factor_value")
        .set_index("factor_value")["factor_label"].to_dict()
    )
    plot_data = data.copy()
    plot_data["factor_label"] = plot_data["factor_value"].map(labels)
    order = [labels.get(value, f"{value:g}") for value in values]
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.1))
    panels = (
        ("coverage_pct", "Coverage Rate", "Coverage rate (%)"),
        (
            "budget_used_per_sample", "Budget Used per Sample",
            "Budget Used per Sample",
        ),
        (
            "coverage_diff_pct", "Coverage Difference",
            "|Coverage - target| (pp)",
        ),
        (
            "mean_target_a_weight", "Mean Target-A Weight",
            r"Mean selected-target weight $A_i(q_{\hat\tau})/\pi_i$",
        ),
    )
    all_statistics = []
    for axis, (metric, panel_title, ylabel) in zip(axes.flat, panels):
        panel = plot_data.dropna(subset=["factor_label", "method", metric])
        sns.boxplot(
            data=panel,
            x="factor_label",
            y=metric,
            hue="method",
            order=order,
            hue_order=list(METHOD_ORDER),
            palette={method: METHOD_COLORS[method] for method in METHOD_ORDER},
            linewidth=0.9,
            fliersize=2.2,
            ax=axis,
        )
        _style(axis, xlabel=str(spec["xlabel"]), ylabel=ylabel)
        axis.set_title(panel_title, fontsize=11)
        axis.tick_params(axis="x", labelrotation=0)
        statistics = summarize_line_statistics(data, metric=metric)
        statistics["metric"] = metric
        all_statistics.append(statistics)
        if metric == "coverage_pct":
            axis.axhline(90.0, color="#D62728", linestyle="--", linewidth=1.2)
        elif metric == "budget_used_per_sample":
            configured_budgets = pd.to_numeric(
                plot_data.get(
                    "configured_budget_per_sample",
                    pd.Series(np.nan, index=plot_data.index),
                ),
                errors="coerce",
            ).dropna().unique()
            if len(configured_budgets) > 1:
                raise ValueError(
                    f"{kind} figure mixes configured budgets: "
                    f"{sorted(configured_budgets.tolist())}."
                )
            reference_budget = (
                float(configured_budgets[0])
                if len(configured_budgets) == 1
                else ATTACKER_SHIFT_BUDGET
                if kind == "attacker_shift"
                else 20.0
            )
            axis.axhline(
                reference_budget,
                color="#D62728",
                linestyle="--",
                linewidth=1.2,
            )
    handles, legend_labels = axes.flat[1].get_legend_handles_labels()
    if handles:
        axes.flat[1].legend(
            handles, legend_labels,
            title="Method", loc="best", frameon=True, framealpha=0.88,
            fontsize=9, title_fontsize=9,
        )
    for axis in (axes.flat[0], axes.flat[2], axes.flat[3]):
        legend = axis.get_legend()
        if legend is not None:
            legend.remove()
    budget_note = (
        f", $B={ATTACKER_SHIFT_BUDGET:g}$"
        if kind == "attacker_shift"
        else ""
    )
    figure.suptitle(
        f"{spec['title']} (50 random calibration/test splits{budget_note})",
        y=0.995,
        fontsize=13,
    )
    figure.tight_layout(rect=(0.0, 0.02, 1.0, 0.97), h_pad=2.0, w_pad=1.7)
    save_jpeg(figure, output_path, quality)
    plt.close(figure)
    return pd.concat(all_statistics, ignore_index=True)


def generate_metric_ablation_figure(
        data: pd.DataFrame,
        *,
        kind: str,
        output_path: Path,
        quality: str,
) -> pd.DataFrame:
    """Plot metric score quality, including variance over random splits."""
    spec = FACTOR_SPECS[kind]
    values = sorted(data["factor_value"].dropna().unique())
    figure, axes = plt.subplots(3, 2, figsize=(11.2, 10.0))
    panels = (
        ("event_rate_pct", "Estimated Event Rate", "Event rate (%)"),
        (
            "event_rate_abs_error_pp", "Event-Rate Error",
            "Absolute event-rate error (pp)",
        ),
        (
            "event_rate_across_split_variance_pp2",
            "Variance of Estimated Event Rate",
            r"Across-split variance (pp$^2$)",
        ),
        (
            "budget_used_per_sample", "Budget Used per Sample",
            "Budget Used per Sample",
        ),
        ("observed_events", "Observed Events", "Number of observed events"),
        (
            "mean_target_a_weight", "Mean Metric Target-A Weight",
            r"Mean $\mathbf{1}\{T_i\leq M\}/\pi_i$",
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
        if metric == "event_rate_pct":
            truth = pd.to_numeric(
                data["event_rate_truth_pct"], errors="coerce"
            ).dropna()
            if not truth.empty:
                axis.axhline(
                    float(truth.iloc[0]), color="#D62728",
                    linestyle="--", linewidth=1.2,
                )
        elif metric == "budget_used_per_sample":
            axis.axhline(
                20.0, color="#D62728", linestyle="--", linewidth=1.2
            )
    axes.flat[0].legend(
        title="Method", loc="best", frameon=True, framealpha=0.88,
        fontsize=8.5, title_fontsize=9,
    )
    for axis in axes.flat[1:]:
        legend = axis.get_legend()
        if legend is not None:
            legend.remove()
    representation_note = "all score maps use $K=2$"
    figure.suptitle(
        "Metric " + str(spec["title"]).lower()
        + f" ({representation_note}; mean $\\pm$ 1 SD across splits)",
        y=0.995,
        fontsize=13,
    )
    figure.tight_layout(rect=(0.0, 0.015, 1.0, 0.975), h_pad=2.0, w_pad=1.8)
    save_jpeg(figure, output_path, quality)
    plt.close(figure)
    return pd.concat(all_statistics, ignore_index=True)


def generate_representation_diagnostics(
        data: pd.DataFrame,
        *,
        output_path: Path,
        quality: str,
) -> pd.DataFrame:
    """Plot the extra diagnostics requested for representation capacity."""
    dynamic = data[~data["method"].eq("Static")].copy()
    values = sorted(dynamic["factor_value"].dropna().unique())
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 3.7))
    output = []
    for axis, metric, title, ylabel in (
        (
            axes[0], "phase1_objective", "Phase-I Objective",
            "Mean fitted variance proxy",
        ),
        (
            axes[1], "phase2_target_weight_objective",
            "Phase-II Target-Weight Objective",
            r"Mean $A_i(1/\pi_i-1)$",
        ),
    ):
        statistics = _draw_line(
            axis, dynamic, metric=metric, ylabel=ylabel,
            xlabel="Score-map representation", values=values,
        )
        statistics["metric"] = metric
        output.append(statistics)
        axis.set_title(title)

    variance = (
        dynamic.groupby(["factor_value", "method"], observed=True)["coverage_pct"]
        .agg(variance="var", count="count").reset_index()
    )
    variance["mean"] = variance["variance"]
    variance["std"] = np.nan
    variance["metric"] = "coverage_variance"
    labels = (
        dynamic[["factor_value", "factor_label"]].drop_duplicates("factor_value")
        .set_index("factor_value")["factor_label"].to_dict()
    )
    for method, style, marker in (
        ("DAPRO", "-", "o"), ("DAPRO w/o CRC", "-.", "s")
    ):
        rows = variance[variance["method"].eq(method)].set_index(
            "factor_value"
        ).reindex(values)
        axes[2].plot(
            values, rows["variance"], label=method,
            color=METHOD_COLORS[method], linestyle=style, marker=marker,
        )
    axes[2].set_xticks(values)
    axes[2].set_xticklabels(
        [labels.get(value, f"{value:g}") for value in values],
        rotation=18, ha="right",
    )
    _style(
        axes[2], xlabel="Score-map representation",
        ylabel=r"Variance of coverage (pp$^2$)",
    )
    axes[2].set_title("Coverage Variance")
    axes[1].legend(title="Method", loc="best", fontsize=8, framealpha=.9)
    figure.tight_layout(w_pad=1.5)
    save_jpeg(figure, output_path, quality)
    plt.close(figure)
    output.append(variance)
    return pd.concat(output, ignore_index=True, sort=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=_ROOT / "results" / "merged_calibration_dfs",
    )
    parser.add_argument(
        "--metric-input-dir",
        type=Path,
        default=_ROOT / "results" / "merged_metric_calibration_dfs",
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
        "--metric-experiment-prefix", default="dapro_metric_ablation_v1"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=("lpb", "metric"),
        default=["lpb", "metric"],
        help="Generate LPB figures, metric figures, or both (default).",
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
    generated_count = 0
    if "lpb" in args.tasks:
        for kind in args.kinds:
            data, _ = load_ablation_data(
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
            generated_count += int(path.exists())
            print(f"Generated {path}")
            if kind == "representation":
                diagnostic_path = (
                    args.output_dir / "dapro_representation_diagnostics.jpg"
                )
                generate_representation_diagnostics(
                    data,
                    output_path=diagnostic_path,
                    quality=args.quality,
                )
                generated_count += int(diagnostic_path.exists())
                print(f"Generated {diagnostic_path}")
    if "metric" in args.tasks:
        metric_kinds = [
            kind for kind in args.kinds if kind in {"score_noise", "score"}
        ]
        for kind in metric_kinds:
            data, _ = load_metric_ablation_data(
                args.metric_input_dir,
                experiment_prefix=args.metric_experiment_prefix,
                kind=kind,
            )
            counts = data.groupby(
                ["factor_value", "method"], observed=True
            )["seed"].nunique()
            if args.strict_seeds and not bool(counts.eq(50).all()):
                raise ValueError(
                    f"metric {kind} does not contain 50 splits in every "
                    f"cell:\n{counts}"
                )
            path = args.output_dir / f"dapro_metric_{kind}_ablation.jpg"
            generate_metric_ablation_figure(
                data, kind=kind, output_path=path, quality=args.quality
            )
            generated_count += int(path.exists())
            print(f"Generated {path}")
    if generated_count == 0:
        raise ValueError("No requested ablation figures were generated.")
    print(f"Generated {generated_count} ablation figures; no CSV artifacts written.")


if __name__ == "__main__":
    main()
