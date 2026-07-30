"""Audit DAPRO variance objectives on one homogeneous merged experiment.

The notebook used for the original analysis is intentionally not imported:
this script validates experiment identity, seed pairing, row uniqueness, and
finite metrics before calculating any across-seed variance.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = OrderedDict([
    ("calibration_optimized_allocation", "Static"),
    ("calibration_adaptive_optimized_allocation", "Locally adaptive"),
    ("calibration_random_adaptive_optimized_allocation", "Random"),
    (
        "calibration_projected_optimization_platt_prob_allocation",
        "DAPRO",
    ),
    (
        "calibration_projected_optimization_platt_prob_a_weighted_allocation",
        "Prior-A DAPRO",
    ),
    (
        "calibration_projected_optimization_platt_prob_a_target_raw_alpha_0p10_allocation",
        "Target-A DAPRO (raw alpha)",
    ),
    (
        "calibration_projected_optimization_platt_prob_a_target_phase1_unweighted_alpha_0p10_allocation",
        "Target-A DAPRO (Phase-I)",
    ),
])

CORE_METRICS = [
    "coverage",
    "size",
    "budget_used",
    "mean_weight",
    "mean_inverse_probability_minus_one",
    "mean_a_weighted_inverse_probability_minus_one",
    "mean_estimable_a_weighted_inverse_probability_minus_one",
    "mean_prior_a_weighted_inverse_probability_minus_one",
    "all_observed_jailbreaks",
    "alpha_hat_per_tau",
]


def absolute_io_path(path: Path | str) -> str:
    """Return an absolute path that supports long Windows experiment names."""
    abs_path = os.path.abspath(path)
    if os.name == "nt" and not abs_path.startswith("\\\\?\\"):
        return f"\\\\?\\{abs_path}"
    return abs_path


def _finite_values(frame: pd.DataFrame, columns: list[str], context: str):
    values = frame[columns].apply(pd.to_numeric, errors="coerce")
    bad = ~np.isfinite(values.to_numpy(dtype=float))
    if bad.any():
        locations = np.argwhere(bad)
        examples = [
            f"row={frame.index[row]}, column={columns[col]}"
            for row, col in locations[:10]
        ]
        raise ValueError(
            f"Non-finite values in {context}: {', '.join(examples)}"
        )


def validate_experiment_identity(
        merged_csv: Path,
        cal_size: int,
        budget_per_sample: float,
        tau_prior: float,
        m_upper_bound: float,
        experiment_suffix: str,
):
    directory_name = merged_csv.parent.name
    if "__" not in directory_name:
        raise ValueError(
            "The audit requires a version-suffixed experiment directory."
        )
    base, actual_suffix = directory_name.rsplit("__", 1)
    if actual_suffix != experiment_suffix:
        raise ValueError(
            f"Experiment suffix is {actual_suffix!r}, expected "
            f"{experiment_suffix!r}."
        )
    try:
        _, budget_text, cal_size_text, tau_text, gamma_text = (
            base.rsplit("_", 4)
        )
        path_budget = float(budget_text)
        path_cal_size = int(cal_size_text)
        path_tau = float(tau_text)
        path_gamma = float(gamma_text)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Cannot parse experiment identity from {directory_name!r}."
        ) from error
    expected_gamma = m_upper_bound / budget_per_sample
    expected = {
        "calibration size": (path_cal_size, cal_size),
        "budget per sample": (path_budget, budget_per_sample),
        "tau prior": (path_tau, tau_prior),
        "gamma": (path_gamma, round(expected_gamma, 3)),
    }
    mismatches = {
        label: values
        for label, values in expected.items()
        if not np.isclose(values[0], values[1], atol=1e-12, rtol=0)
    }
    if mismatches:
        raise ValueError(
            "CLI configuration disagrees with the merged experiment path: "
            f"{mismatches}"
        )


def load_and_validate(
        merged_csv: Path,
        target_coverage: float,
        seed_start: int,
        seed_end: int,
        cal_size: int,
        budget_per_sample: float,
        tau_prior: float,
        m_upper_bound: float,
        experiment_suffix: str,
) -> pd.DataFrame:
    validate_experiment_identity(
        merged_csv,
        cal_size,
        budget_per_sample,
        tau_prior,
        m_upper_bound,
        experiment_suffix,
    )
    frame = pd.read_csv(absolute_io_path(merged_csv))
    required = {"seed", "calibration_name", "target_coverage", *CORE_METRICS}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Merged result is missing columns: {missing_columns}")
    metadata_expectations = {
        "experiment_name": merged_csv.parent.name,
        "configured_cal_size": cal_size,
        "configured_budget_per_sample": budget_per_sample,
        "configured_tau_prior": tau_prior,
        "configured_m_upper_bound": m_upper_bound,
    }
    present_metadata = set(metadata_expectations).intersection(frame.columns)
    if present_metadata and present_metadata != set(metadata_expectations):
        missing_metadata = sorted(
            set(metadata_expectations) - present_metadata
        )
        raise ValueError(
            "Merged result contains only a partial provenance schema; "
            f"missing metadata columns: {missing_metadata}."
        )
    for column, expected_value in metadata_expectations.items():
        if column not in frame:
            continue
        if frame[column].isna().any():
            raise ValueError(
                f"Metadata column {column!r} has missing values; refusing "
                "to analyze a mixed-provenance merge."
            )
        values = frame[column].unique()
        if len(values) != 1:
            raise ValueError(
                f"Metadata column {column!r} is not constant: {values}."
            )
        actual_value = values[0]
        if isinstance(expected_value, str):
            agrees = str(actual_value) == expected_value
        else:
            agrees = np.isclose(
                float(actual_value),
                float(expected_value),
                atol=1e-12,
                rtol=0,
            )
        if not agrees:
            raise ValueError(
                f"Metadata {column!r} is {actual_value!r}, expected "
                f"{expected_value!r}."
            )

    target = frame[
        np.isclose(
            pd.to_numeric(frame["target_coverage"], errors="coerce"),
            target_coverage,
            atol=1e-10,
            rtol=0,
        )
    ].copy()
    target = target[target["calibration_name"].isin(METHODS)].copy()

    missing_methods = sorted(set(METHODS) - set(target["calibration_name"]))
    if missing_methods:
        raise ValueError(
            "Expected methods are absent from the target rows: "
            f"{missing_methods}"
        )
    duplicate_counts = target.groupby(
        ["calibration_name", "seed"],
        dropna=False,
    ).size()
    duplicates = duplicate_counts[duplicate_counts != 1]
    if len(duplicates):
        raise ValueError(
            "Expected exactly one target row per method/seed; violations: "
            f"{duplicates.to_dict()}"
        )

    expected_seeds = set(range(seed_start, seed_end))
    for method in METHODS:
        actual = set(
            pd.to_numeric(
                target.loc[target["calibration_name"] == method, "seed"],
                errors="raise",
            ).astype(int)
        )
        if actual != expected_seeds:
            raise ValueError(
                f"{method} has seeds {sorted(actual)}, expected "
                f"{sorted(expected_seeds)}."
            )

    _finite_values(target, CORE_METRICS, "90%-target method comparison")
    tolerance = 1e-8
    global_proxy = target["mean_inverse_probability_minus_one"].to_numpy(
        dtype=float
    )
    if not np.allclose(
        target["mean_weight"].to_numpy(dtype=float) - 1,
        global_proxy,
        atol=tolerance,
        rtol=1e-8,
    ):
        raise ValueError(
            "`mean_weight - 1` disagrees with the global inverse-excess metric."
        )
    for proxy in [
        "mean_a_weighted_inverse_probability_minus_one",
        "mean_prior_a_weighted_inverse_probability_minus_one",
    ]:
        if np.any(target[proxy].to_numpy(dtype=float) > global_proxy + tolerance):
            raise ValueError(f"{proxy} exceeds the global inverse excess.")
    literal = target[
        "mean_a_weighted_inverse_probability_minus_one"
    ].to_numpy(dtype=float)
    estimable = target[
        "mean_estimable_a_weighted_inverse_probability_minus_one"
    ].to_numpy(dtype=float)
    if not np.allclose(literal, estimable, atol=tolerance, rtol=1e-8):
        raise ValueError(
            "Literal and q_prior-estimable target-A proxies differ; the "
            "selected bound is not uniformly inside the acquisition horizon."
        )

    target["seed"] = target["seed"].astype(int)
    target["method"] = target["calibration_name"].map(METHODS)
    method_order = list(METHODS.values())
    target["method"] = pd.Categorical(
        target["method"],
        categories=method_order,
        ordered=True,
    )
    return target.sort_values(["method", "seed"]).reset_index(drop=True)


def summarize_methods(
        target: pd.DataFrame,
        target_coverage: float,
        cal_size: int,
        budget_per_sample: float,
) -> pd.DataFrame:
    rows = []
    for method, group in target.groupby("method", observed=True, sort=False):
        coverage_pct = group["coverage"].to_numpy(dtype=float) * 100
        desired_pct = target_coverage * 100
        a_rate = (
            group["all_observed_jailbreaks"].to_numpy(dtype=float) / cal_size
        )
        target_proxy = group[
            "mean_a_weighted_inverse_probability_minus_one"
        ].to_numpy(dtype=float)
        inverse_given_a = np.divide(
            target_proxy,
            a_rate,
            out=np.full_like(target_proxy, np.nan),
            where=a_rate > 0,
        )
        budget = group["budget_used"].to_numpy(dtype=float) / cal_size
        rows.append({
            "method": str(method),
            "n_seeds": len(group),
            "coverage_mean_pct": coverage_pct.mean(),
            "coverage_bias_pp": coverage_pct.mean() - desired_pct,
            "coverage_variance_pp2": coverage_pct.var(ddof=1),
            "coverage_sd_pp": coverage_pct.std(ddof=1),
            "coverage_rmse_pp": np.sqrt(
                np.mean((coverage_pct - desired_pct) ** 2)
            ),
            "size_mean": group["size"].mean(),
            "size_variance": group["size"].var(ddof=1),
            "budget_mean_per_sample": budget.mean(),
            "budget_sd_per_sample": budget.std(ddof=1),
            "budget_min_per_sample": budget.min(),
            "budget_max_per_sample": budget.max(),
            "budget_overrun_fraction": np.mean(
                budget > budget_per_sample + 1e-12
            ),
            "mean_inverse_probability_minus_one": group[
                "mean_inverse_probability_minus_one"
            ].mean(),
            "mean_target_a_inverse_probability_minus_one": target_proxy.mean(),
            "mean_prior_a_inverse_probability_minus_one": group[
                "mean_prior_a_weighted_inverse_probability_minus_one"
            ].mean(),
            "mean_target_a_rate": a_rate.mean(),
            "mean_inverse_excess_given_target_a": np.nanmean(inverse_given_a),
            "scaled_conditional_ht_variance_pp2": (
                target_proxy.mean() / cal_size * 10_000
            ),
            "alpha_hat_mean": group["alpha_hat_per_tau"].mean(),
            "alpha_hat_variance": group["alpha_hat_per_tau"].var(ddof=1),
        })
    return pd.DataFrame(rows)


def _bootstrap_variance_comparison(
        method_values: np.ndarray,
        baseline_values: np.ndarray,
        rng: np.random.Generator,
        repetitions: int,
) -> tuple[tuple[float, float], tuple[float, float]]:
    n = len(method_values)
    indices = rng.integers(0, n, size=(repetitions, n))
    method_var = np.var(method_values[indices], axis=1, ddof=1)
    baseline_var = np.var(baseline_values[indices], axis=1, ddof=1)
    differences = method_var - baseline_var
    ratios = np.divide(
        method_var,
        baseline_var,
        out=np.full_like(method_var, np.nan),
        where=baseline_var > np.finfo(float).eps,
    )
    diff_ci = tuple(np.quantile(differences, [0.025, 0.975]))
    finite_ratios = ratios[np.isfinite(ratios)]
    ratio_ci = tuple(np.quantile(finite_ratios, [0.025, 0.975]))
    return diff_ci, ratio_ci


def _bootstrap_paired_mean_ci(
        differences: np.ndarray,
        rng: np.random.Generator,
        repetitions: int,
) -> tuple[float, float]:
    n = len(differences)
    indices = rng.integers(0, n, size=(repetitions, n))
    means = differences[indices].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def paired_comparisons(
        target: pd.DataFrame,
        cal_size: int,
        repetitions: int,
        random_seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    wide = {}
    for method, group in target.groupby("method", observed=True, sort=False):
        wide[str(method)] = group.set_index("seed").sort_index()
    baseline = wide["DAPRO"]
    baseline_coverage = baseline["coverage"].to_numpy(dtype=float) * 100
    rows = []
    for method, group in wide.items():
        if method == "DAPRO":
            continue
        coverage = group["coverage"].to_numpy(dtype=float) * 100
        diff_ci, ratio_ci = _bootstrap_variance_comparison(
            coverage,
            baseline_coverage,
            rng,
            repetitions,
        )
        row = {
            "method": method,
            "coverage_variance_difference_pp2": (
                coverage.var(ddof=1) - baseline_coverage.var(ddof=1)
            ),
            "coverage_variance_ratio": (
                coverage.var(ddof=1) / baseline_coverage.var(ddof=1)
            ),
            "bootstrap_difference_ci_low": diff_ci[0],
            "bootstrap_difference_ci_high": diff_ci[1],
            "bootstrap_ratio_ci_low": ratio_ci[0],
            "bootstrap_ratio_ci_high": ratio_ci[1],
        }
        metric_specs = {
            "global_proxy": "mean_inverse_probability_minus_one",
            "target_a_proxy": (
                "mean_a_weighted_inverse_probability_minus_one"
            ),
            "prior_a_proxy": (
                "mean_prior_a_weighted_inverse_probability_minus_one"
            ),
        }
        for label, column in metric_specs.items():
            differences = (
                group[column].to_numpy(dtype=float)
                - baseline[column].to_numpy(dtype=float)
            )
            low, high = _bootstrap_paired_mean_ci(
                differences,
                rng,
                repetitions,
            )
            row[f"paired_mean_difference_{label}"] = differences.mean()
            row[f"paired_mean_difference_{label}_ci_low"] = low
            row[f"paired_mean_difference_{label}_ci_high"] = high
        budget_differences = (
            group["budget_used"].to_numpy(dtype=float)
            - baseline["budget_used"].to_numpy(dtype=float)
        ) / cal_size
        low, high = _bootstrap_paired_mean_ci(
            budget_differences,
            rng,
            repetitions,
        )
        row["paired_mean_budget_difference_per_sample"] = (
            budget_differences.mean()
        )
        row["paired_mean_budget_difference_ci_low"] = low
        row["paired_mean_budget_difference_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_dapro_stages(target: pd.DataFrame) -> pd.DataFrame:
    dapro = target[
        target["method"].astype(str).str.contains("DAPRO")
    ].copy()
    if {
        "phase2_expected_cost_per_sample",
        "phase2_expected_budget_gap_per_sample",
    }.issubset(dapro.columns):
        dapro["derived_phase2_target_budget_per_sample"] = (
            pd.to_numeric(
                dapro["phase2_expected_cost_per_sample"],
                errors="coerce",
            )
            - pd.to_numeric(
                dapro["phase2_expected_budget_gap_per_sample"],
                errors="coerce",
            )
        )
    requested = [
        "phase1_oracle_mean_inverse_probability_minus_one",
        "phase1_projected_mean_inverse_probability_minus_one",
        "phase2_mean_inverse_probability_minus_one",
        "phase1_oracle_mean_prior_variance_proxy",
        "phase1_projected_mean_prior_variance_proxy",
        "phase2_mean_prior_variance_proxy",
        "phase1_oracle_mean_objective_variance_proxy",
        "phase1_projected_mean_objective_variance_proxy",
        "phase2_mean_objective_variance_proxy",
        "mean_inverse_probability_minus_one",
        "mean_prior_a_weighted_inverse_probability_minus_one",
        "mean_a_weighted_inverse_probability_minus_one",
        "phase2_target_budget_per_sample",
        "derived_phase2_target_budget_per_sample",
        "phase2_expected_cost_per_sample",
        "phase2_expected_budget_gap_per_sample",
        "target_anchor_index",
        "target_anchor_tau",
        "target_anchor_selection_miscoverage",
        "target_anchor_phase1_a_rate",
        "target_anchor_phase2_a_rate",
    ]
    available = [column for column in requested if column in dapro.columns]
    numeric = dapro[available].apply(pd.to_numeric, errors="coerce")
    dapro[available] = numeric
    return (
        dapro.groupby("method", observed=True, sort=False)[available]
        .mean()
        .reset_index()
    )


def make_proxy_plot(summary: pd.DataFrame, output_path: Path):
    colors = plt.cm.tab10(np.linspace(0, 1, len(summary)))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    panels = [
        (
            "mean_inverse_probability_minus_one",
            r"Mean $(1/\pi_i-1)$",
            True,
        ),
        (
            "mean_target_a_inverse_probability_minus_one",
            r"Mean $A_i(1/\pi_i-1)$ at selected 90% bound",
            False,
        ),
    ]
    for axis, (column, xlabel, log_scale) in zip(axes, panels):
        x_values = summary[column].to_numpy(dtype=float)
        for color, row in zip(colors, summary.to_dict("records")):
            axis.scatter(
                row[column],
                row["coverage_variance_pp2"],
                s=65,
                color=color,
            )
            axis.annotate(
                row["method"],
                (row[column], row["coverage_variance_pp2"]),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8,
            )
        if log_scale:
            axis.set_xscale("log")
            axis.set_xlim(x_values.min() / 1.5, x_values.max() * 2.0)
        else:
            x_span = max(x_values.max() - x_values.min(), 1e-6)
            axis.set_xlim(
                max(0, x_values.min() - 0.05 * x_span),
                x_values.max() + 0.25 * x_span,
            )
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Across-seed coverage variance (percentage points²)")
        axis.grid(alpha=0.25)
    fig.suptitle(
        "Global and target-A propensity proxies versus coverage variance"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    view = frame[columns].copy()
    headers = list(view.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in view.itertuples(index=False, name=None):
        rendered = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                rendered.append(f"{value:.6g}")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def write_report(
        output_path: Path,
        merged_csv: Path,
        summary: pd.DataFrame,
        paired: pd.DataFrame,
        target_coverage: float,
        seed_start: int,
        seed_end: int,
):
    summary_columns = [
        "method",
        "coverage_mean_pct",
        "coverage_variance_pp2",
        "coverage_rmse_pp",
        "budget_mean_per_sample",
        "mean_inverse_probability_minus_one",
        "mean_target_a_inverse_probability_minus_one",
        "mean_prior_a_inverse_probability_minus_one",
    ]
    paired_columns = [
        "method",
        "coverage_variance_difference_pp2",
        "coverage_variance_ratio",
        "bootstrap_difference_ci_low",
        "bootstrap_difference_ci_high",
    ]
    report = f"""# DAPRO variance-objective audit

Input: `{os.path.abspath(merged_csv)}`

Validated target coverage: {target_coverage:.2%}; paired seeds:
{seed_start}-{seed_end - 1}. Each method has exactly one finite row per seed.

## Method comparison

{_markdown_table(summary, summary_columns)}

## Paired bootstrap comparison with ordinary DAPRO

{_markdown_table(paired, paired_columns)}

The bootstrap resamples common seed indices. With only
{seed_end - seed_start} seeds, intervals are descriptive and should not be
treated as decisive hypothesis tests.

## Interpretation

For a fixed event indicator and independent acquisition draws,

`Var(n^-1 sum_i A_i R_i / pi_i | complete data)
 = mean_i[A_i(1/pi_i - 1)] / n`.

This identity does not equal the across-seed variance of final test coverage:
those seeds also change the split, policy fit, selected calibration candidate,
acquisition randomness, and test sample. The selected-target proxy is
target-relevant but method-dependent and post-selection; the prior-A proxy is
common across methods but targets the wider acquisition envelope.
"""
    output_path.write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Validate and analyze one merged A-weighted DAPRO run."
    )
    parser.add_argument("--merged-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-coverage", type=float, default=0.90)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=10)
    parser.add_argument("--cal-size", type=int, default=3000)
    parser.add_argument("--budget-per-sample", type=float, default=20.0)
    parser.add_argument("--tau-prior", type=float, default=0.56)
    parser.add_argument("--m-upper-bound", type=float, default=200.0)
    parser.add_argument("--experiment-suffix", type=str, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260726)
    args = parser.parse_args()

    if args.seed_end <= args.seed_start:
        parser.error("`--seed-end` must be greater than `--seed-start`.")
    if args.cal_size <= 0:
        parser.error("`--cal-size` must be positive.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    target = load_and_validate(
        args.merged_csv,
        args.target_coverage,
        args.seed_start,
        args.seed_end,
        args.cal_size,
        args.budget_per_sample,
        args.tau_prior,
        args.m_upper_bound,
        args.experiment_suffix,
    )
    summary = summarize_methods(
        target,
        args.target_coverage,
        args.cal_size,
        args.budget_per_sample,
    )
    paired = paired_comparisons(
        target,
        args.cal_size,
        args.bootstrap_repetitions,
        args.bootstrap_seed,
    )
    stages = summarize_dapro_stages(target)

    target.to_csv(args.output_dir / "per_seed_90pct.csv", index=False)
    summary.to_csv(args.output_dir / "method_summary_90pct.csv", index=False)
    paired.to_csv(args.output_dir / "paired_vs_dapro_90pct.csv", index=False)
    stages.to_csv(args.output_dir / "dapro_stage_summary_90pct.csv", index=False)
    make_proxy_plot(
        summary,
        args.output_dir / "proxy_vs_coverage_variance.png",
    )
    write_report(
        args.output_dir / "numerical_report.md",
        args.merged_csv,
        summary,
        paired,
        args.target_coverage,
        args.seed_start,
        args.seed_end,
    )
    print(f"Validated and stored analysis in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
