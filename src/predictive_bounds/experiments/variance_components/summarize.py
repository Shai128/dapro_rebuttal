"""Decompose coverage variance into data, policy, and acquisition components."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.predictive_bounds.experiments.variance_components.design import variance_jobs
from src.predictive_bounds.experiments.variance_components.run import parse_args


def _n1(method: str) -> str:
    match = re.search(r"_n1_(\d+)_allocation$", method)
    return f"N1={match.group(1)}" if match else "Constant"


def _merged_path(args, suffix):
    from src.predictive_bounds.utils.utils import get_calibration_experiment_name
    name = get_calibration_experiment_name(
        args.dataset_name, args.dataset_setup, args.budget_per_sample,
        args.cal_size, args.tau_prior,
        args.m_upper_bound / args.budget_per_sample, suffix,
    )
    return Path("results/merged_calibration_dfs") / name / "all_df.csv"


def load_designs(args):
    frames = []
    for job in variance_jobs(
        replicates=args.replicates, crossed_groups=args.crossed_groups,
        suffix_prefix=args.suffix_prefix,
    ):
        frame = pd.read_csv(_merged_path(args, job.suffix))
        frame = frame[np.isclose(frame["target_coverage"], 0.90)].copy()
        frame["design"] = job.design
        frame["group"] = job.suffix.rsplit("_g", 1)[-1] if "_g" in job.suffix else "one_factor"
        frame["method_display"] = frame["calibration_name"].map(_n1)
        frame["coverage_pp"] = 100 * frame["coverage"]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def decompose(frame):
    one_factor = (
        frame[frame["group"].eq("one_factor")]
        .groupby(["method_display", "design"], observed=True)["coverage_pp"]
        .var(ddof=1).rename("coverage_variance_pp2").reset_index()
    )
    rows = []
    for (method, design), group in frame[~frame["group"].eq("one_factor")].groupby(
            ["method_display", "design"], observed=True):
        matrix = group.pivot(index="group", columns="seed", values="coverage_pp").to_numpy()
        within = float(np.var(matrix, axis=1, ddof=0).mean())
        between = float(np.var(matrix.mean(axis=1), ddof=0))
        rows.extend([
            {"method_display": method, "design": design, "component": "within", "coverage_variance_pp2": within},
            {"method_display": method, "design": design, "component": "between", "coverage_variance_pp2": between},
            {"method_display": method, "design": design, "component": "total", "coverage_variance_pp2": within + between},
        ])
    return one_factor, pd.DataFrame(rows)


def latex_tables(one_factor, crossed):
    return "\n".join([
        "% Requires booktabs.",
        one_factor.to_latex(index=False, float_format="%.4f", escape=True),
        "",
        crossed.to_latex(index=False, float_format="%.4f", escape=True),
    ])


def main(argv=None):
    base = parse_args(argv)
    output_dir = base.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_designs(base)
    one_factor, crossed = decompose(frame)
    one_factor.to_csv(output_dir / "one_factor_variances.csv", index=False)
    crossed.to_csv(output_dir / "crossed_variance_components.csv", index=False)
    figure, axis = plt.subplots(figsize=(10, 5.8))
    sns.lineplot(
        data=one_factor[one_factor["method_display"].ne("Constant")],
        x="method_display", y="coverage_variance_pp2", hue="design",
        marker="o", ax=axis,
    )
    axis.set_xlabel("DAPRO Phase-I size")
    axis.set_ylabel("Coverage variance (squared pp)")
    figure.tight_layout()
    figure.savefig(output_dir / "variance_by_n1.pdf", bbox_inches="tight")
    plt.close(figure)
    one_factor_rows = frame[frame["group"].eq("one_factor")]
    figure, axis = plt.subplots(figsize=(10.5, 5.8))
    sns.boxplot(
        data=one_factor_rows, x="method_display", y="coverage_pp",
        hue="design", fliersize=2, ax=axis,
    )
    axis.set_xlabel("Method / DAPRO Phase-I size")
    axis.set_ylabel("Coverage rate (%)")
    axis.axhline(90, color="#b91c1c", linestyle="--", linewidth=1.4)
    figure.tight_layout()
    figure.savefig(output_dir / "coverage_by_randomness_boxplot.pdf", bbox_inches="tight")
    plt.close(figure)
    (output_dir / "variance_components_tables.tex").write_text(
        latex_tables(one_factor, crossed), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
