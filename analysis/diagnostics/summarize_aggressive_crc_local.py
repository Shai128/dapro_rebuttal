"""Summarize paired standard-versus-aggressive CRC diagnostic runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _mean_ci(values: pd.Series) -> tuple[float, float]:
    values = values.dropna().astype(float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, np.nan
    return mean, float(1.96 * values.std(ddof=1) / np.sqrt(len(values)))


def _positive_rate(values: pd.Series) -> float:
    values = values.dropna().astype(float)
    return float((values > 0).mean()) if len(values) else np.nan


def _mean_or_nan(values: pd.Series) -> float:
    values = values.dropna().astype(float)
    return float(values.mean()) if len(values) else np.nan


def summarize_lpb_metric(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    rows: list[dict] = []
    for (setup, task, method), group in frame.groupby(
        ["setup", "task", "method"], sort=True
    ):
        expected_mean, expected_half_width = _mean_ci(
            group["conditional_expected_cost"]
        )
        realized_mean, realized_half_width = _mean_ci(group["realized_cost"])
        row = {
            "setup": setup,
            "task": task,
            "method": method,
            "splits": group["seed"].nunique(),
            "expected_budget_mean": expected_mean,
            "expected_budget_95ci_half_width": expected_half_width,
            "expected_budget_split_exceed_rate": float(
                (group["conditional_expected_cost"] > group["budget_target"]).mean()
            ),
            "expected_budget_max_excess_pct": float(
                (
                    100
                    * (
                        group["conditional_expected_cost"]
                        / group["budget_target"]
                        - 1
                    )
                ).max()
            ),
            "realized_budget_mean": realized_mean,
            "realized_budget_95ci_half_width": realized_half_width,
            "realized_budget_split_exceed_rate": float(
                (group["realized_cost"] > group["budget_target"]).mean()
            ),
            "realized_budget_max_excess_pct": float(
                (100 * (group["realized_cost"] / group["budget_target"] - 1)).max()
            ),
            "selected_scale_mean": float(group["selected_crc_scale"].mean()),
            "selected_transformed_any_violation_rate": _positive_rate(
                group[
                    "aggressive_selected_transformed_envelope_violation_rate"
                ]
            ),
            "selected_transformed_mean_row_violation_rate": _mean_or_nan(
                group[
                    "aggressive_selected_transformed_envelope_violation_rate"
                ]
            ),
            "family_transformed_any_violation_rate": _positive_rate(
                group[
                    "aggressive_family_transformed_envelope_violation_rate"
                ]
            ),
            "family_transformed_mean_row_violation_rate": _mean_or_nan(
                group[
                    "aggressive_family_transformed_envelope_violation_rate"
                ]
            ),
            "selected_sufficient_cap_any_violation_rate": _positive_rate(
                group["aggressive_selected_sufficient_cap_violation_rate"]
            ),
            "selected_sufficient_cap_mean_row_violation_rate": _mean_or_nan(
                group["aggressive_selected_sufficient_cap_violation_rate"]
            ),
            "family_sufficient_cap_any_violation_rate": _positive_rate(
                group["aggressive_family_sufficient_cap_violation_rate"]
            ),
            "family_sufficient_cap_mean_row_violation_rate": _mean_or_nan(
                group["aggressive_family_sufficient_cap_violation_rate"]
            ),
            "selected_transformed_max_excess": float(
                group[
                    "aggressive_selected_transformed_envelope_max_excess"
                ].max()
            ),
            "family_transformed_max_excess": float(
                group[
                    "aggressive_family_transformed_envelope_max_excess"
                ].max()
            ),
        }
        if task == "lpb":
            row.update({
                "primary_variance": float(group["coverage_pct"].var(ddof=1)),
                "primary_mse": float(
                    np.mean((group["coverage_pct"] - 90.0) ** 2)
                ),
                "restricted_mean_variance": np.nan,
            })
        else:
            row.update({
                "primary_variance": float(group["estimate_pct"].var(ddof=1)),
                "primary_mse": float(
                    np.mean((group["estimate_pct"] - group["truth_pct"]) ** 2)
                ),
                "restricted_mean_variance": float(
                    group["estimated_restricted_mean"].var(ddof=1)
                ),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_upb(paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    # Budget and envelope diagnostics repeat for every displayed coverage
    # target, because one allocation is shared by the target grid.
    allocation = frame.sort_values("target_coverage").drop_duplicates(
        ["dataset", "seed", "method"]
    )
    allocation_rows: list[dict] = []
    for (dataset, method), group in allocation.groupby(
        ["dataset", "method"], sort=True
    ):
        expected_mean, expected_half_width = _mean_ci(
            group["expected_budget_per_sample"]
        )
        realized_mean, realized_half_width = _mean_ci(
            group["realized_budget_per_sample"]
        )
        allocation_rows.append({
            "dataset": dataset,
            "method": method,
            "splits": group["seed"].nunique(),
            "expected_budget_mean": expected_mean,
            "expected_budget_95ci_half_width": expected_half_width,
            "expected_budget_split_exceed_rate": float(
                (group["expected_budget_per_sample"] > 20).mean()
            ),
            "expected_budget_max_excess_pct": float(
                (100 * (group["expected_budget_per_sample"] / 20 - 1)).max()
            ),
            "realized_budget_mean": realized_mean,
            "realized_budget_95ci_half_width": realized_half_width,
            "realized_budget_split_exceed_rate": float(
                (group["realized_budget_per_sample"] > 20).mean()
            ),
            "realized_budget_max_excess_pct": float(
                (100 * (group["realized_budget_per_sample"] / 20 - 1)).max()
            ),
            "selected_scale_mean": float(group["selected_crc_scale"].mean()),
            "selected_transformed_any_violation_rate": _positive_rate(
                group["aggressive_selected_transformed_violation_rate"]
            ),
            "selected_transformed_mean_row_violation_rate": _mean_or_nan(
                group["aggressive_selected_transformed_violation_rate"]
            ),
            "family_transformed_any_violation_rate": _positive_rate(
                group["aggressive_family_transformed_violation_rate"]
            ),
            "family_transformed_mean_row_violation_rate": _mean_or_nan(
                group["aggressive_family_transformed_violation_rate"]
            ),
            "selected_sufficient_cap_any_violation_rate": _positive_rate(
                group["aggressive_selected_sufficient_cap_violation_rate"]
            ),
            "selected_sufficient_cap_mean_row_violation_rate": _mean_or_nan(
                group["aggressive_selected_sufficient_cap_violation_rate"]
            ),
            "family_sufficient_cap_any_violation_rate": _positive_rate(
                group["aggressive_family_sufficient_cap_violation_rate"]
            ),
            "family_sufficient_cap_mean_row_violation_rate": _mean_or_nan(
                group["aggressive_family_sufficient_cap_violation_rate"]
            ),
            "selected_transformed_max_excess": float(
                group["aggressive_selected_transformed_max_excess"].max()
            ),
            "family_transformed_max_excess": float(
                group["aggressive_family_transformed_max_excess"].max()
            ),
        })
    coverage = (
        frame.groupby(["dataset", "target_coverage", "method"], sort=True)
        .agg(
            coverage_mean_pct=("coverage_pct", "mean"),
            coverage_variance_pp2=("coverage_pct", "var"),
            mean_upb=("mean_upb", "mean"),
        )
        .reset_index()
    )
    return pd.DataFrame(allocation_rows), coverage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lpb-metric", type=Path, required=True)
    parser.add_argument("--upb", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    print("LPB and metrics")
    print(summarize_lpb_metric(args.lpb_metric).to_string(index=False))
    allocation, coverage = summarize_upb(args.upb)
    print("\nUPB allocation")
    print(allocation.to_string(index=False))
    print("\nUPB coverage")
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()
