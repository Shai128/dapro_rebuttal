"""Paired audit for deployment-aware DAPRO experiments.

The script accepts one or more merged CSVs per dataset, validates split/source
provenance and metric identities, then reports coverage variance, all-sample
weight tails, selected-target variance proxies, and Phase-I-to-Phase-II
objective transfer.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd


METHODS = OrderedDict([
    ("calibration_optimized_allocation", "Static"),
    (
        "calibration_adaptive_optimized_allocation",
        "Local (hard pi>=0.005)",
    ),
    (
        "calibration_adaptive_optimized_crc_allocation",
        "Local (hard pi>=0.005, CRC)",
    ),
    (
        "calibration_adaptive_optimized_mixture_terminal_floor_0p005_allocation",
        "Local (trajectory mixture 0.005)",
    ),
    (
        "calibration_adaptive_optimized_no_terminal_floor_allocation",
        "Local (no terminal floor)",
    ),
    (
        "calibration_random_adaptive_optimized_allocation",
        "Random (trajectory mixture 0.005)",
    ),
    (
        "calibration_random_adaptive_optimized_hard_terminal_floor_0p005_allocation",
        "Random (hard pi>=0.005)",
    ),
    (
        "calibration_random_adaptive_optimized_no_terminal_floor_allocation",
        "Random (no floor; constant p)",
    ),
    (
        "calibration_random_adaptive_optimized_hard_terminal_floor_0p005_crc_allocation",
        "Random (hard pi>=0.005, CRC)",
    ),
    (
        "calibration_projected_optimization_platt_prob_allocation",
        "DAPRO (conditional Platt)",
    ),
    (
        "calibration_projected_optimization_platt_prob_a_target_phase1_unweighted_alpha_0p10_allocation",
        "Phase-I target DAPRO (Platt)",
    ),
    (
        "calibration_projected_optimization_cumulative_platt_prob_a_target_raw_alpha_0p10_allocation",
        "Raw-target DAPRO (cumulative Platt)",
    ),
    (
        "calibration_projected_optimization_direct_time_prob_allocation",
        "DAPRO (direct time)",
    ),
    (
        "calibration_projected_optimization_direct_time_prob_a_target_raw_alpha_0p10_allocation",
        "Direct time raw-target",
    ),
    (
        "calibration_projected_optimization_direct_bins_4_prob_a_target_raw_alpha_0p10_allocation",
        "Direct 4-bin raw-target",
    ),
    (
        "calibration_projected_optimization_direct_time_prob_a_target_phase1_unweighted_alpha_0p10_allocation",
        "Direct time Phase-I target",
    ),
    (
        "calibration_projected_optimization_direct_time_prob_a_target_phase1_robust_raw_0p10_alpha_0p10_allocation",
        "Direct Phase-I/raw blend",
    ),
    (
        "calibration_projected_optimization_direct_time_prob_a_target_raw_regularized_global_0p010_alpha_0p10_allocation",
        "Direct raw-target + global 0.01",
    ),
    (
        "calibration_projected_optimization_direct_time_prob_a_target_raw_regularized_global_0p001_alpha_0p10_allocation",
        "Direct raw-target + global 0.001",
    ),
    (
        "calibration_projected_optimization_direct_time_prob_a_target_raw_regularized_global_0p050_alpha_0p10_allocation",
        "Direct raw-target + global 0.05",
    ),
    (
        "calibration_projected_optimization_direct_time_prob_a_target_phase1_unweighted_regularized_global_0p001_alpha_0p10_allocation",
        "Direct Phase-I target + global 0.001",
    ),
    (
        "calibration_projected_optimization_direct_time_prob_a_target_phase1_unweighted_regularized_global_0p010_alpha_0p10_allocation",
        "Direct Phase-I target + global 0.01",
    ),
    (
        "calibration_projected_optimization_direct_time_prob_a_target_phase1_unweighted_regularized_global_0p050_alpha_0p10_allocation",
        "Direct Phase-I target + global 0.05",
    ),
    (
        "calibration_projected_optimization_direct_time_prob_a_band_0p07_0p13_global_0p010_allocation",
        "Direct target band + global 0.01",
    ),
])

RANDOM_NAMES = (
    "Random (trajectory mixture 0.005)",
    "Random (hard pi>=0.005)",
    "Random (no floor; constant p)",
    "Random (hard pi>=0.005, CRC)",
)


def method_label(calibration_name: str) -> str | None:
    """Map exact method names, including arbitrary registered DAPRO N1 suffixes."""
    if calibration_name in METHODS:
        return METHODS[calibration_name]
    anchored = re.fullmatch(
        r"calibration_projected_optimization_direct_time_prob"
        r"_a_target_raw_random_anchor_target_(\d+p\d+)_alpha_0p10"
        r"(_random_slack_filled)?"
        r"_budget_(crc|hoeffding)_control_(\d+)"
        r"(?:_n1_(\d+))?_allocation",
        calibration_name,
    )
    if anchored is not None:
        fraction = float(anchored.group(1).replace("p", "."))
        slack_filled = anchored.group(2) is not None
        control_size = int(anchored.group(4))
        n1 = (
            100
            if anchored.group(5) is None
            else int(anchored.group(5))
        )
        mode = anchored.group(3).upper()
        fill_label = ", Random-slack-filled" if slack_filled else ""
        return (
            f"Random-anchored target-A "
            f"(target={fraction:.2f}{fill_label}, {mode}, "
            f"control={control_size}, N1={n1})"
        )
    match = re.fullmatch(r"(.+)_n1_(\d+)_allocation", calibration_name)
    if match is None:
        return None
    base_name = f"{match.group(1)}_allocation"
    base_label = METHODS.get(base_name)
    if base_label is None:
        return None
    return f"{base_label} (N1={int(match.group(2))})"

CORE_COLUMNS = [
    "seed",
    "calibration_name",
    "target_coverage",
    "coverage",
    "size",
    "budget_used",
    "mean_weight",
    "max_weight",
    "variance_weight",
    "median_weight",
    "p90_weight",
    "p99_weight",
    "effective_sample_size_weight",
    "top_1pct_weight_share",
    "mean_inverse_probability_minus_one",
    "mean_a_weighted_inverse_probability",
    "mean_a_weighted_inverse_probability_minus_one",
    "conditional_variance_of_ht_mean",
    "all_observed_jailbreaks",
]

PHASE_COLUMNS = [
    "phase1_oracle_mean_objective_inverse_probability",
    "phase1_oracle_variance_objective_inverse_probability",
    "phase1_projected_mean_objective_inverse_probability",
    "phase1_projected_variance_objective_inverse_probability",
    "phase2_mean_objective_inverse_probability",
    "phase2_variance_objective_inverse_probability",
    "phase1_projection_mean_objective_gap",
    "phase2_minus_phase1_projected_mean_objective_gap",
    "phase2_minus_phase1_oracle_mean_objective_gap",
    "phase2_to_phase1_oracle_mean_objective_ratio",
    "phase2_minus_phase1_oracle_objective_variance_gap",
    "phase1_oracle_mean_objective_variance_proxy",
    "phase1_projected_mean_objective_variance_proxy",
    "phase2_mean_objective_variance_proxy",
    "all_mean_objective_inverse_probability",
    "all_variance_objective_inverse_probability",
    "all_mean_objective_variance_proxy",
    "phase2_expected_cost_per_sample",
    "phase2_expected_budget_gap_per_sample",
    "phase2_realized_cost_per_sample",
    "target_anchor_phase1_a_rate",
    "target_anchor_phase2_a_rate",
]


def absolute_io_path(path: str | Path) -> str:
    result = os.path.abspath(path)
    if os.name == "nt" and not result.startswith("\\\\?\\"):
        return f"\\\\?\\{result}"
    return result


def parse_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Inputs must use DATASET=PATH syntax."
        )
    dataset, path = value.split("=", 1)
    if not dataset.strip() or not path.strip():
        raise argparse.ArgumentTypeError(
            "Both dataset label and path are required."
        )
    return dataset.strip(), Path(path.strip())


def finite(frame: pd.DataFrame, columns: list[str], context: str):
    present = [column for column in columns if column in frame]
    values = frame[present].apply(pd.to_numeric, errors="coerce")
    bad = ~np.isfinite(values.to_numpy(dtype=float))
    if bad.any():
        row, column = np.argwhere(bad)[0]
        raise ValueError(
            f"Non-finite {context}: row={frame.index[row]}, "
            f"column={present[column]}."
        )


def load_inputs(
        input_specs: list[tuple[str, Path]],
        target_coverage: float,
        seed_start: int,
        seed_end: int,
) -> pd.DataFrame:
    frames = []
    for dataset, path in input_specs:
        frame = pd.read_csv(absolute_io_path(path), low_memory=False)
        missing = sorted(set(CORE_COLUMNS) - set(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        target = frame[
            np.isclose(
                pd.to_numeric(frame["target_coverage"], errors="coerce"),
                target_coverage,
                atol=1e-10,
                rtol=0,
            )
        ].copy()
        target["method"] = target["calibration_name"].map(method_label)
        target = target[target["method"].notna()].copy()
        target["dataset"] = dataset
        target["source_csv"] = str(path.resolve())
        frames.append(target)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["seed"] = pd.to_numeric(
        combined["seed"],
        errors="raise",
    ).astype(int)

    duplicate = combined.groupby(
        ["dataset", "calibration_name", "seed"],
        dropna=False,
    ).size()
    duplicate = duplicate[duplicate != 1]
    if len(duplicate):
        raise ValueError(
            "Expected one target row per dataset/method/seed: "
            f"{duplicate.to_dict()}"
        )

    expected_seeds = set(range(seed_start, seed_end))
    for (dataset, method), group in combined.groupby(
            ["dataset", "calibration_name"]
    ):
        actual = set(group["seed"])
        if actual != expected_seeds:
            raise ValueError(
                f"{dataset}/{method} has seeds {sorted(actual)}, expected "
                f"{sorted(expected_seeds)}."
            )

    finite(combined, CORE_COLUMNS[3:], "core metric")
    tolerance = 1e-8
    if not np.allclose(
        combined["mean_weight"].to_numpy(float) - 1,
        combined["mean_inverse_probability_minus_one"].to_numpy(float),
        atol=tolerance,
        rtol=1e-8,
    ):
        raise ValueError("mean_weight - 1 identity failed.")
    a_rate = (
        combined["all_observed_jailbreaks"].to_numpy(float)
        / combined["configured_cal_size"].to_numpy(float)
    )
    if not np.allclose(
        combined["mean_a_weighted_inverse_probability"].to_numpy(float),
        a_rate
        + combined[
            "mean_a_weighted_inverse_probability_minus_one"
        ].to_numpy(float),
        atol=tolerance,
        rtol=1e-8,
    ):
        raise ValueError("mean(A/pi) = mean(A) + variance-proxy identity failed.")

    provenance = [
        "execution_device",
        "torch_version",
        "predictive_bounds_source_sha256",
        "calibration_split_sha256",
        "test_split_sha256",
    ]
    missing_provenance = [
        column for column in provenance if column not in combined
    ]
    if missing_provenance:
        raise ValueError(
            f"Missing provenance columns: {missing_provenance}"
        )
    if combined[provenance].isna().any().any():
        raise ValueError("Provenance contains missing values.")
    for (dataset, seed), group in combined.groupby(["dataset", "seed"]):
        for column in ["calibration_split_sha256", "test_split_sha256"]:
            if group[column].nunique() != 1:
                raise ValueError(
                    f"{dataset}/seed={seed} does not share one {column}."
                )

    combined = combined.copy()
    return combined.sort_values(
        ["dataset", "method", "seed"]
    ).reset_index(drop=True)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, method), group in frame.groupby(
            ["dataset", "method"],
            sort=False,
    ):
        coverage_pp = group["coverage"].to_numpy(float) * 100
        target_coverage_pp = (
            group["target_coverage"].to_numpy(float) * 100
        )
        target_coverage_mean = float(np.mean(target_coverage_pp))
        coverage_standard_error = float(
            np.std(coverage_pp, ddof=1) / np.sqrt(len(coverage_pp))
        )
        coverage_ci_low = float(
            np.mean(coverage_pp) - 1.96 * coverage_standard_error
        )
        coverage_ci_high = float(
            np.mean(coverage_pp) + 1.96 * coverage_standard_error
        )
        a_rate = (
            group["all_observed_jailbreaks"].to_numpy(float)
            / group["configured_cal_size"].to_numpy(float)
        )
        mean_a_pi = group[
            "mean_a_weighted_inverse_probability"
        ].to_numpy(float)
        target_proxy = group[
            "mean_a_weighted_inverse_probability_minus_one"
        ].to_numpy(float)
        mean_weight = group["mean_weight"].to_numpy(float)
        inverse_given_target = np.divide(
            mean_a_pi,
            a_rate,
            out=np.full_like(mean_a_pi, np.nan),
            where=a_rate > 0,
        )
        inverse_given_non_target = np.divide(
            mean_weight - mean_a_pi,
            1 - a_rate,
            out=np.full_like(mean_a_pi, np.nan),
            where=a_rate < 1,
        )
        predicted_variance_pp2 = (
            group["conditional_variance_of_ht_mean"].to_numpy(float)
            * 10_000
        )
        coverage_variance = float(np.var(coverage_pp, ddof=1))
        predicted_variance_mean = float(np.mean(predicted_variance_pp2))
        expected_total_budget_per_sample = np.full(
            len(group),
            np.nan,
            dtype=np.float64,
        )
        if (
            "total_expected_budget_per_sample" in group
            and group["total_expected_budget_per_sample"].notna().all()
        ):
            expected_total_budget_per_sample = group[
                "total_expected_budget_per_sample"
            ].to_numpy(float)
        elif all(
            column in group
            for column in [
                "phase2_expected_cost_per_sample",
                "phase2_realized_cost_per_sample",
                "phase2_sample_count",
            ]
        ):
            # The stored realized total is Phase-I fully observed cost plus
            # Phase-II realized cost.  Replacing only the latter by its
            # conditional expectation recovers the full expected total.
            phase2_count = group["phase2_sample_count"].to_numpy(float)
            phase1_realized_total = (
                group["budget_used"].to_numpy(float)
                - phase2_count
                * group["phase2_realized_cost_per_sample"].to_numpy(float)
            )
            expected_total_budget_per_sample = (
                phase1_realized_total
                + phase2_count
                * group["phase2_expected_cost_per_sample"].to_numpy(float)
            ) / group["configured_cal_size"].to_numpy(float)
        configured_budget = group[
            "configured_budget_per_sample"
        ].to_numpy(float)
        expected_budget_gap = (
            expected_total_budget_per_sample - configured_budget
        )
        finite_expected_budget = np.isfinite(expected_budget_gap)
        if np.count_nonzero(finite_expected_budget) > 1:
            budget_gap_standard_error = float(
                np.std(
                    expected_budget_gap[finite_expected_budget],
                    ddof=1,
                )
                / np.sqrt(np.count_nonzero(finite_expected_budget))
            )
        else:
            budget_gap_standard_error = np.nan
        expected_budget_gap_mean = float(
            np.mean(expected_budget_gap[finite_expected_budget])
            if np.any(finite_expected_budget)
            else np.nan
        )
        expected_budget_gap_ci_low = float(
            expected_budget_gap_mean - 1.96 * budget_gap_standard_error
            if np.isfinite(budget_gap_standard_error)
            else np.nan
        )
        expected_budget_gap_ci_high = float(
            expected_budget_gap_mean + 1.96 * budget_gap_standard_error
            if np.isfinite(budget_gap_standard_error)
            else np.nan
        )
        size = group["size"].to_numpy(float)
        runtime = (
            group["method_runtime_seconds"].to_numpy(float)
            if "method_runtime_seconds" in group
            else np.full(len(group), np.nan)
        )
        rows.append({
            "dataset": dataset,
            "method": method,
            "n_seeds": len(group),
            "target_coverage_pct": target_coverage_mean,
            "coverage_mean_pct": float(np.mean(coverage_pp)),
            "coverage_bias_pp": float(
                np.mean(coverage_pp - target_coverage_pp)
            ),
            "coverage_variance_pp2": coverage_variance,
            "coverage_sd_pp": float(np.std(coverage_pp, ddof=1)),
            "coverage_mean_ci95_low_pct": coverage_ci_low,
            "coverage_mean_ci95_high_pct": coverage_ci_high,
            "coverage_mean_at_least_target": int(
                np.mean(coverage_pp) >= target_coverage_mean
            ),
            "coverage_ci95_low_at_least_target": int(
                coverage_ci_low >= target_coverage_mean
            ),
            "coverage_rmse_pp": float(np.sqrt(
                np.mean((coverage_pp - target_coverage_pp) ** 2)
            )),
            "mean_lpb": float(np.mean(size)),
            "variance_lpb": float(np.var(size, ddof=1)),
            "budget_mean_per_sample": float(np.mean(
                group["budget_used"].to_numpy(float)
                / group["configured_cal_size"].to_numpy(float)
            )),
            "expected_budget_mean_per_sample": float(
                np.mean(
                    expected_total_budget_per_sample[
                        finite_expected_budget
                    ]
                )
                if np.any(finite_expected_budget)
                else np.nan
            ),
            "expected_budget_gap_mean_per_sample": expected_budget_gap_mean,
            "expected_budget_gap_ci95_low_per_sample": (
                expected_budget_gap_ci_low
            ),
            "expected_budget_gap_ci95_high_per_sample": (
                expected_budget_gap_ci_high
            ),
            "expected_budget_violation_rate": float(
                np.mean(expected_budget_gap[finite_expected_budget] > 1e-7)
                if np.any(finite_expected_budget)
                else np.nan
            ),
            "expected_budget_mean_valid": int(
                expected_budget_gap_mean <= 1e-7
            ) if np.any(finite_expected_budget) else np.nan,
            "expected_budget_ci95_high_valid": int(
                expected_budget_gap_ci_high <= 1e-7
            ) if np.isfinite(expected_budget_gap_ci_high) else np.nan,
            "feasible_on_estimated_means": int(
                np.mean(coverage_pp) >= target_coverage_mean
                and expected_budget_gap_mean <= 1e-7
            ) if np.any(finite_expected_budget) else np.nan,
            "feasible_with_95pct_confidence": int(
                coverage_ci_low >= target_coverage_mean
                and expected_budget_gap_ci_high <= 1e-7
            ) if np.isfinite(expected_budget_gap_ci_high) else np.nan,
            "mean_weight_all_calibration": float(
                group["mean_weight"].mean()
            ),
            "variance_weight_all_calibration": float(
                group["variance_weight"].mean()
            ),
            "median_weight": float(group["median_weight"].mean()),
            "p90_weight": float(group["p90_weight"].mean()),
            "p99_weight": float(group["p99_weight"].mean()),
            "max_weight": float(group["max_weight"].max()),
            "kish_ess": float(
                group["effective_sample_size_weight"].mean()
            ),
            "top_1pct_weight_share": float(
                group["top_1pct_weight_share"].mean()
            ),
            "target_a_rate": float(np.mean(a_rate)),
            "mean_target_a_over_pi_all_calibration": float(
                np.mean(mean_a_pi)
            ),
            "mean_target_variance_proxy": float(
                np.mean(target_proxy)
            ),
            "mean_inverse_probability_given_target_a": float(
                np.nanmean(inverse_given_target)
            ),
            "mean_inverse_probability_given_non_target": float(
                np.nanmean(inverse_given_non_target)
            ),
            "predicted_conditional_variance_pp2": float(
                predicted_variance_mean
            ),
            "actual_to_predicted_variance_ratio": float(
                coverage_variance / predicted_variance_mean
                if predicted_variance_mean > 1e-12
                else np.nan
            ),
            "runtime_mean_seconds": float(
                np.nanmean(runtime)
                if np.isfinite(runtime).any()
                else np.nan
            ),
            "runtime_median_seconds": float(
                np.nanmedian(runtime)
                if np.isfinite(runtime).any()
                else np.nan
            ),
        })
    return pd.DataFrame(rows)


def summarize_rank_correlations(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, group in summary.groupby("dataset", sort=False):
        rows.append({
            "dataset": dataset,
            "n_methods": len(group),
            "spearman_coverage_variance_vs_target_proxy": float(
                group["coverage_variance_pp2"].corr(
                    group["mean_target_variance_proxy"],
                    method="spearman",
                )
            ),
            "spearman_coverage_variance_vs_mean_weight": float(
                group["coverage_variance_pp2"].corr(
                    group["mean_weight_all_calibration"],
                    method="spearman",
                )
            ),
            "spearman_coverage_variance_vs_mean_a_over_pi": float(
                group["coverage_variance_pp2"].corr(
                    group["mean_target_a_over_pi_all_calibration"],
                    method="spearman",
                )
            ),
        })
    return pd.DataFrame(rows)


def summarize_phase_transfer(frame: pd.DataFrame) -> pd.DataFrame:
    dapro = frame[frame["phase1_oracle_mean_objective_inverse_probability"].notna()]
    rows = []
    for (dataset, method), group in dapro.groupby(
            ["dataset", "method"],
            sort=False,
    ):
        row = {"dataset": dataset, "method": method, "n_seeds": len(group)}
        for column in PHASE_COLUMNS:
            if column in group and group[column].notna().all():
                values = group[column].to_numpy(float)
                row[f"{column}_mean"] = float(np.mean(values))
                row[f"{column}_across_seed_variance"] = float(
                    np.var(values, ddof=1)
                )
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_variance_comparisons(
        frame: pd.DataFrame,
        repetitions: int,
        bootstrap_seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(bootstrap_seed)
    rows = []
    for dataset, dataset_frame in frame.groupby("dataset", sort=False):
        pivot = dataset_frame.pivot(
            index="seed",
            columns="method",
            values="coverage",
        ) * 100
        n = len(pivot)
        indices = rng.integers(0, n, size=(repetitions, n))
        for random_name in RANDOM_NAMES:
            if random_name not in pivot:
                continue
            random_values = pivot[random_name].to_numpy(float)
            random_variance = np.var(
                random_values[indices],
                axis=1,
                ddof=1,
            )
            for method in pivot.columns:
                if method == random_name:
                    continue
                candidate = pivot[method].to_numpy(float)
                candidate_variance = np.var(
                    candidate[indices],
                    axis=1,
                    ddof=1,
                )
                difference = candidate_variance - random_variance
                ratio = np.divide(
                    candidate_variance,
                    random_variance,
                    out=np.full_like(candidate_variance, np.nan),
                    where=random_variance > 0,
                )
                rows.append({
                    "dataset": dataset,
                    "random_reference": random_name,
                    "method": method,
                    "variance_difference_vs_random_pp2": float(
                        np.var(candidate, ddof=1)
                        - np.var(random_values, ddof=1)
                    ),
                    "variance_ratio_vs_random": float(
                        np.var(candidate, ddof=1)
                        / max(
                            np.var(random_values, ddof=1),
                            np.finfo(float).tiny,
                        )
                    ),
                    "bootstrap_difference_ci_low": float(
                        np.quantile(difference, 0.025)
                    ),
                    "bootstrap_difference_ci_high": float(
                        np.quantile(difference, 0.975)
                    ),
                    "bootstrap_ratio_ci_low": float(
                        np.nanquantile(ratio, 0.025)
                    ),
                    "bootstrap_ratio_ci_high": float(
                        np.nanquantile(ratio, 0.975)
                    ),
                })
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows._"
    shown = frame[columns].copy()
    for column in shown.select_dtypes(include=[np.number]):
        shown[column] = shown[column].map(lambda value: f"{value:.6g}")
    header = "| " + " | ".join(shown.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(shown.columns)) + " |"
    rows = [
        "| " + " | ".join(map(str, values)) + " |"
        for values in shown.to_numpy()
    ]
    return "\n".join([header, separator, *rows])


def write_report(
        output_dir: Path,
        frame: pd.DataFrame,
        summary: pd.DataFrame,
        phase: pd.DataFrame,
        paired: pd.DataFrame,
        correlations: pd.DataFrame,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "per_seed_90pct.csv", index=False)
    summary.to_csv(output_dir / "method_summary.csv", index=False)
    phase.to_csv(output_dir / "phase_transfer_summary.csv", index=False)
    paired.to_csv(output_dir / "paired_vs_random.csv", index=False)
    correlations.to_csv(
        output_dir / "metric_rank_correlations.csv",
        index=False,
    )
    provenance_columns = [
        "dataset",
        "source_csv",
        "execution_device",
        "cuda_device_name",
        "torch_version",
        "numpy_version",
        "predictive_bounds_source_sha256",
        "calibration_split_sha256",
        "test_split_sha256",
    ]
    frame[
        [column for column in provenance_columns if column in frame]
    ].drop_duplicates().to_csv(
        output_dir / "provenance.csv",
        index=False,
    )

    best_rows = pd.concat(
        [
            (
                group[group["feasible_on_estimated_means"] == 1]
                if (group["feasible_on_estimated_means"] == 1).any()
                else group
            ).nsmallest(1, "coverage_variance_pp2")
            for _, group in summary.groupby("dataset", sort=False)
        ],
        ignore_index=True,
    )
    report = f"""# Deployment-aware DAPRO audit

All comparisons use paired seeds and the exact 90% target row. Coverage
variance is the sample variance across outer seeds in squared percentage
points. The predicted conditional variance is
`mean(A * (1/pi - 1)) / calibration_size`; it conditions on the fitted policy
and fixed candidate, whereas empirical coverage variance also includes split,
policy-fit, target-selection, acquisition, and test-set variation.

## Best observed coverage variance among mean-feasible methods

When no evaluated method is mean-feasible for a dataset, this table falls back
to its lowest-variance method and the feasibility columns in the full table
show the failure.

{markdown_table(best_rows, [
    "dataset",
    "method",
    "coverage_mean_pct",
    "coverage_mean_ci95_low_pct",
    "coverage_variance_pp2",
    "mean_weight_all_calibration",
    "mean_target_a_over_pi_all_calibration",
    "mean_target_variance_proxy",
])}

## Full method summary

{markdown_table(summary, [
    "dataset",
    "method",
    "coverage_mean_pct",
    "coverage_variance_pp2",
    "mean_lpb",
    "variance_lpb",
    "budget_mean_per_sample",
    "expected_budget_mean_per_sample",
    "expected_budget_gap_mean_per_sample",
    "expected_budget_gap_ci95_high_per_sample",
    "feasible_on_estimated_means",
    "feasible_with_95pct_confidence",
    "expected_budget_violation_rate",
    "mean_weight_all_calibration",
    "p99_weight",
    "max_weight",
    "mean_target_a_over_pi_all_calibration",
    "mean_target_variance_proxy",
    "mean_inverse_probability_given_target_a",
    "mean_inverse_probability_given_non_target",
    "predicted_conditional_variance_pp2",
    "runtime_mean_seconds",
])}

## Phase-I to Phase-II objective transfer

The raw-to-projected gap isolates deployment-map distortion. The
projected-to-Phase-II gap measures population transfer. Variances below are
within-phase variances of per-record objective contributions; the additional
`across_seed_variance` columns in the CSV quantify run-to-run instability of
each summary.

{markdown_table(phase, [
    column for column in [
        "dataset",
        "method",
        "phase1_oracle_mean_objective_inverse_probability_mean",
        "phase1_oracle_variance_objective_inverse_probability_mean",
        "phase1_projected_mean_objective_inverse_probability_mean",
        "phase2_mean_objective_inverse_probability_mean",
        "phase2_variance_objective_inverse_probability_mean",
        "phase1_oracle_mean_objective_variance_proxy_mean",
        "phase2_mean_objective_variance_proxy_mean",
        "phase1_projection_mean_objective_gap_mean",
        "phase2_minus_phase1_projected_mean_objective_gap_mean",
        "phase2_to_phase1_oracle_mean_objective_ratio_mean",
    ] if column in phase
])}

## Paired bootstrap versus each Random reference

Intervals resample the common outer seeds. They quantify uncertainty for these
specific runs, not generalization across datasets or models.

{markdown_table(paired, list(paired.columns))}

## Method-level rank correlations

These are descriptive correlations across the evaluated methods. They are
not p-values and methods are not independent experimental units.

{markdown_table(correlations, list(correlations.columns))}

## Interpretation safeguards

- `mean_weight` and `mean(A/pi)` include all calibration rows; Phase-I pilot
  rows have propensity one.
- Very large global weights can coexist with a small target variance proxy
  when the large weights occur on `A=0` rows.
- A pure sparse target objective can saturate every observed target event and
  rationally leave budget unused. Global regularization is therefore evaluated
  as a tail-robustness safeguard.
- Method-specific post-selection `A` is a diagnostic. A prospective policy
  comparison should additionally freeze one common target or cross-fit it.
"""
    (output_dir / "audit_report.md").write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        type=parse_input,
        help="Repeat DATASET=PATH for every merged CSV or supplement.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-coverage", type=float, default=0.90)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=10)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260728)
    args = parser.parse_args()

    frame = load_inputs(
        args.input,
        args.target_coverage,
        args.seed_start,
        args.seed_end,
    )
    method_summary = summarize(frame)
    phase_summary = summarize_phase_transfer(frame)
    paired = bootstrap_variance_comparisons(
        frame,
        args.bootstrap_repetitions,
        args.bootstrap_seed,
    )
    correlations = summarize_rank_correlations(method_summary)
    write_report(
        args.output_dir,
        frame,
        method_summary,
        phase_summary,
        paired,
        correlations,
    )
    print(f"Stored audit at {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
