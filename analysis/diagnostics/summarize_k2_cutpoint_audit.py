"""Summarize the isolated K2 cutpoint audit."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.concat([pd.read_csv(path) for path in args.inputs], ignore_index=True)
    numeric = [
        "cut_quantile",
        "exact_target_variance_pp2",
        "expected_cost_per_sample",
        "realized_cost_per_sample",
        "raw_projection_cost_gap",
        "fit_low_share",
        "deploy_low_share",
        "fit_soft_surrogate",
        "deploy_soft_surrogate",
        "estimate",
        "coverage_pct",
        "selected_exact_variance_pp2",
        "switched_from_oracle",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    by_setup = (
        frame.groupby(["setup_key", "task", "cut_quantile"], as_index=False)
        .agg(
            splits=("seed", "nunique"),
            exact_variance=("exact_target_variance_pp2", "mean"),
            exact_variance_sd=("exact_target_variance_pp2", "std"),
            expected_cost=("expected_cost_per_sample", "mean"),
            expected_cost_max=("expected_cost_per_sample", "max"),
            realized_cost=("realized_cost_per_sample", "mean"),
            raw_projection_gap=("raw_projection_cost_gap", "mean"),
            raw_projection_abs_gap=("raw_projection_cost_gap", lambda x: np.mean(np.abs(x))),
            fit_low_share=("fit_low_share", "mean"),
            deploy_low_share=("deploy_low_share", "mean"),
            fit_surrogate=("fit_soft_surrogate", "mean"),
            deploy_surrogate=("deploy_soft_surrogate", "mean"),
            estimator_variance=("estimate", lambda x: np.var(x, ddof=1)),
            coverage_variance_pp2=("coverage_pct", lambda x: np.var(x, ddof=1)),
            coverage_mse_pp2=("coverage_pct", lambda x: np.mean((x - 90.0) ** 2)),
            selected_exact_variance=("selected_exact_variance_pp2", "mean"),
            switch_rate=("switched_from_oracle", "mean"),
        )
    )
    median = by_setup[by_setup.cut_quantile == 0.5][
        ["setup_key", "task", "exact_variance", "expected_cost", "coverage_variance_pp2"]
    ].rename(
        columns={
            "exact_variance": "median_exact_variance",
            "expected_cost": "median_expected_cost",
            "coverage_variance_pp2": "median_coverage_variance_pp2",
        }
    )
    by_setup = by_setup.merge(median, on=["setup_key", "task"], how="left")
    by_setup["exact_variance_ratio_to_median"] = (
        by_setup.exact_variance / by_setup.median_exact_variance
    )

    pooled = (
        by_setup.groupby(["task", "cut_quantile"], as_index=False)
        .agg(
            setups=("setup_key", "nunique"),
            mean_variance_ratio=("exact_variance_ratio_to_median", "mean"),
            geometric_variance_ratio=(
                "exact_variance_ratio_to_median",
                lambda x: np.exp(np.mean(np.log(x))),
            ),
            median_variance_ratio=("exact_variance_ratio_to_median", "median"),
            wins_vs_median=("exact_variance_ratio_to_median", lambda x: int((x < 1).sum())),
            expected_cost=("expected_cost", "mean"),
            expected_cost_max=("expected_cost_max", "max"),
            mean_abs_raw_projection_gap=("raw_projection_abs_gap", "mean"),
            fit_low_share=("fit_low_share", "mean"),
            deploy_low_share=("deploy_low_share", "mean"),
            coverage_variance_pp2=("coverage_variance_pp2", "mean"),
            coverage_mse_pp2=("coverage_mse_pp2", "mean"),
        )
    )

    # Select q on Phase-I only for each split.  This is ordinary finite-grid
    # ERM, not an oracle.  Compare with the deployment-oracle lower envelope.
    selected_rows = []
    for (setup, task, seed), group in frame.groupby(["setup_key", "task", "seed"]):
        fit_pick = group.loc[group.fit_soft_surrogate.idxmin()].copy()
        fit_pick["selector"] = "phase1_erm"
        selected_rows.append(fit_pick)
        oracle_pick = group.loc[group.exact_target_variance_pp2.idxmin()].copy()
        oracle_pick["selector"] = "deployment_oracle"
        selected_rows.append(oracle_pick)
        median_pick = group.iloc[(group.cut_quantile - 0.5).abs().argmin()].copy()
        median_pick["selector"] = "fixed_median"
        selected_rows.append(median_pick)
    selected = pd.DataFrame(selected_rows)
    selected_setup = (
        selected.groupby(["setup_key", "task", "selector"], as_index=False)
        .agg(
            mean_selected_q=("cut_quantile", "mean"),
            modal_selected_q=("cut_quantile", lambda x: x.mode().iloc[0]),
            exact_variance=("exact_target_variance_pp2", "mean"),
            expected_cost=("expected_cost_per_sample", "mean"),
            realized_cost=("realized_cost_per_sample", "mean"),
            estimator_variance=("estimate", lambda x: np.var(x, ddof=1)),
            coverage_variance_pp2=("coverage_pct", lambda x: np.var(x, ddof=1)),
            coverage_mse_pp2=("coverage_pct", lambda x: np.mean((x - 90.0) ** 2)),
        )
    )
    selected_median = selected_setup[selected_setup.selector == "fixed_median"][
        ["setup_key", "task", "exact_variance"]
    ].rename(columns={"exact_variance": "fixed_median_exact_variance"})
    selected_setup = selected_setup.merge(selected_median, on=["setup_key", "task"])
    selected_setup["exact_variance_ratio_to_median"] = (
        selected_setup.exact_variance / selected_setup.fixed_median_exact_variance
    )
    selected_pooled = (
        selected_setup.groupby(["task", "selector"], as_index=False)
        .agg(
            setups=("setup_key", "nunique"),
            mean_selected_q=("mean_selected_q", "mean"),
            mean_variance_ratio=("exact_variance_ratio_to_median", "mean"),
            geometric_variance_ratio=(
                "exact_variance_ratio_to_median", lambda x: np.exp(np.mean(np.log(x)))
            ),
            wins_vs_median=("exact_variance_ratio_to_median", lambda x: int((x < 1).sum())),
            expected_cost=("expected_cost", "mean"),
            coverage_variance_pp2=("coverage_variance_pp2", "mean"),
            coverage_mse_pp2=("coverage_mse_pp2", "mean"),
        )
    )

    # Leave-one-setup-out fixed q: choose on independent setups, then report
    # the held-out setup.  This tests whether a universally better quantile is
    # supported without tuning on the target setup.
    loso_rows = []
    for task, task_frame in by_setup.groupby("task"):
        setups = sorted(task_frame.setup_key.unique())
        for heldout in setups:
            train = task_frame[task_frame.setup_key != heldout]
            choice = (
                train.groupby("cut_quantile").exact_variance_ratio_to_median.mean().idxmin()
            )
            test = task_frame[
                (task_frame.setup_key == heldout) & (task_frame.cut_quantile == choice)
            ].iloc[0]
            loso_rows.append({
                "task": task,
                "heldout_setup": heldout,
                "selected_q": choice,
                "heldout_variance_ratio_to_median": test.exact_variance_ratio_to_median,
                "heldout_expected_cost": test.expected_cost,
            })
    loso = pd.DataFrame(loso_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "k2_cutpoint_all_rows.csv", index=False)
    by_setup.to_csv(args.output_dir / "k2_cutpoint_by_setup.csv", index=False)
    pooled.to_csv(args.output_dir / "k2_cutpoint_pooled.csv", index=False)
    selected_setup.to_csv(args.output_dir / "k2_cutpoint_selectors_by_setup.csv", index=False)
    selected_pooled.to_csv(args.output_dir / "k2_cutpoint_selectors_pooled.csv", index=False)
    loso.to_csv(args.output_dir / "k2_cutpoint_loso.csv", index=False)
    print("\nFixed quantiles, pooled across setups:\n")
    print(pooled.to_string(index=False))
    print("\nCutpoint selectors, pooled across setups:\n")
    print(selected_pooled.to_string(index=False))
    print("\nLeave-one-setup-out selection:\n")
    print(loso.to_string(index=False))


if __name__ == "__main__":
    main()
