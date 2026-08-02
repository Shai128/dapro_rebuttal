"""Reproduce the crossed DAPRO coverage-variance decomposition.

The experiment design uses the outer seed for one factor at a time:

* policy x acquisition: fixed data split, ten Phase-I policy seeds, and ten
  acquisition seeds for every fitted policy;
* data x acquisition: fixed policy RNG seed, ten data splits, and ten
  acquisition seeds for every split;
* one-factor runs: 100 seeds with the other two factors fixed.

The crossed tables use population variances (``ddof=0``), so the displayed
within- and between-group terms add exactly by the law of total variance.
Coverage is converted to percentage points before taking the variance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = {
    "Original DAPRO (mean weight, N1=100)": (
        "calibration_projected_optimization_direct_time_prob_allocation"
    ),
    "Direct raw-target DAPRO (N1=200)": (
        "calibration_projected_optimization_direct_time_prob_a_target_raw_"
        "alpha_0p10_n1_200_allocation"
    ),
    "Random constant (hard floor, CRC, N1=100)": (
        "calibration_random_adaptive_optimized_hard_terminal_floor_0p005_"
        "crc_allocation"
    ),
}

ONE_FACTOR_SUFFIXES = {
    "acquisition_only": "variance_decomp_acquisition_only_100_v1",
    "policy_only": "variance_decomp_policy_only_100_v1",
    "data_only": "variance_decomp_data_only_100_v1",
}


def _glob_one(root: Path, pattern: str) -> Path:
    paths = sorted(root.glob(pattern))
    if len(paths) != 1:
        raise FileNotFoundError(
            f"Expected exactly one path for {pattern!r}; found {len(paths)}."
        )
    return paths[0]


def _target_rows(path: Path, calibration_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    selected = frame[
        frame["calibration_name"].eq(calibration_name)
        & np.isclose(frame["target_coverage"], 0.90)
    ].copy()
    if selected.empty:
        raise ValueError(
            f"No exact 90% rows for {calibration_name!r} in {path}."
        )
    return selected


def _crossed_matrix(
        merged_root: Path,
        suffix_template: str,
        calibration_name: str,
        group_seed_column: str,
        replicate_seed_column: str,
) -> np.ndarray:
    rows = []
    for group_seed in range(10):
        suffix = suffix_template.format(index=group_seed)
        path = _glob_one(merged_root, f"*__{suffix}/all_df.csv")
        selected = _target_rows(path, calibration_name)
        observed_group_seeds = selected[group_seed_column].unique()
        if (
            len(observed_group_seeds) != 1
            or int(observed_group_seeds[0]) != group_seed
        ):
            raise ValueError(
                f"{path} does not contain the expected "
                f"{group_seed_column}={group_seed}."
            )
        selected = selected.sort_values(replicate_seed_column)
        replicate_seeds = selected[replicate_seed_column].to_numpy()
        if not np.array_equal(replicate_seeds, np.arange(10)):
            raise ValueError(
                f"{path} does not contain replicate seeds 0,...,9."
            )
        rows.append(selected["coverage"].to_numpy(dtype=float) * 100)
    return np.vstack(rows)


def _format_table(frame: pd.DataFrame, digits: int = 6) -> str:
    columns = list(frame.columns)
    rendered = []
    for row in frame.itertuples(index=False, name=None):
        rendered.append([
            (
                f"{value:.{digits}f}"
                if isinstance(value, (float, np.floating))
                else str(value)
            )
            for value in row
        ])
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rendered]
    return "\n".join([header, divider, *body])


def analyze(merged_root: Path, output_dir: Path) -> None:
    one_factor_records = []
    crossed_records = []

    for label, calibration_name in METHODS.items():
        for factor, suffix in ONE_FACTOR_SUFFIXES.items():
            path = _glob_one(merged_root, f"*__{suffix}/all_df.csv")
            coverage_pp = (
                _target_rows(path, calibration_name)["coverage"].to_numpy()
                * 100
            )
            if len(coverage_pp) != 100:
                raise ValueError(
                    f"Expected 100 rows in {path}; found {len(coverage_pp)}."
                )
            one_factor_records.append({
                "method": label,
                "varied_factor": factor,
                "replicates": len(coverage_pp),
                "coverage_variance_pp2": float(
                    np.var(coverage_pp, ddof=1)
                ),
            })

        policy_matrix = _crossed_matrix(
            merged_root,
            "variance_decomp_crossed_p{index}_10_v1",
            calibration_name,
            "policy_rng_seed",
            "acquisition_rng_seed",
        )
        policy_within = float(
            np.var(policy_matrix, axis=1, ddof=0).mean()
        )
        policy_between = float(
            np.var(policy_matrix.mean(axis=1), ddof=0)
        )
        policy_total = float(np.var(policy_matrix, ddof=0))
        crossed_records.extend([
            {
                "method": label,
                "crossed_design": "fixed data: policy x acquisition",
                "component": "E_policy Var_acquisition",
                "coverage_variance_pp2": policy_within,
                "share_of_crossed_total": policy_within / policy_total,
            },
            {
                "method": label,
                "crossed_design": "fixed data: policy x acquisition",
                "component": "Var_policy E_acquisition",
                "coverage_variance_pp2": policy_between,
                "share_of_crossed_total": policy_between / policy_total,
            },
        ])

        data_matrix = _crossed_matrix(
            merged_root,
            "variance_decomp_data_cross_d{index}_10_v1",
            calibration_name,
            "data_split_seed",
            "acquisition_rng_seed",
        )
        data_within = float(np.var(data_matrix, axis=1, ddof=0).mean())
        data_between = float(np.var(data_matrix.mean(axis=1), ddof=0))
        data_total = float(np.var(data_matrix, ddof=0))
        crossed_records.extend([
            {
                "method": label,
                "crossed_design": "fixed policy RNG: data x acquisition",
                "component": "E_data Var_acquisition",
                "coverage_variance_pp2": data_within,
                "share_of_crossed_total": data_within / data_total,
            },
            {
                "method": label,
                "crossed_design": "fixed policy RNG: data x acquisition",
                "component": "Var_data E_acquisition",
                "coverage_variance_pp2": data_between,
                "share_of_crossed_total": data_between / data_total,
            },
        ])

        for design, matrix in [
            ("fixed data: policy x acquisition", policy_matrix),
            ("fixed policy RNG: data x acquisition", data_matrix),
        ]:
            total = float(np.var(matrix, ddof=0))
            crossed_records.append({
                "method": label,
                "crossed_design": design,
                "component": "total",
                "coverage_variance_pp2": total,
                "share_of_crossed_total": 1.0,
            })

    one_factor = pd.DataFrame(one_factor_records)
    crossed = pd.DataFrame(crossed_records)
    output_dir.mkdir(parents=True, exist_ok=True)
    one_factor.to_csv(output_dir / "one_factor_variances.csv", index=False)
    crossed.to_csv(output_dir / "crossed_variance_components.csv", index=False)

    policy_view = crossed[
        crossed["crossed_design"].eq(
            "fixed data: policy x acquisition"
        )
        & crossed["component"].ne("total")
    ].copy()
    policy_view["share_pct"] = (
        100 * policy_view["share_of_crossed_total"]
    )
    data_view = crossed[
        crossed["crossed_design"].eq(
            "fixed policy RNG: data x acquisition"
        )
        & crossed["component"].ne("total")
    ].copy()
    data_view["share_pct"] = 100 * data_view["share_of_crossed_total"]

    report = f"""# Toxicity coverage-variance decomposition

Coverage variances are in squared percentage points. The crossed 10 x 10
designs use population variances so their two components add exactly.

## One-factor 100-seed experiments

{_format_table(one_factor)}

These one-factor paths are diagnostic and are not additive because each is
evaluated around one fixed setting of the other factors.

## Fixed data: Phase-I policy seed x acquisition seed

{_format_table(policy_view[[
    "method",
    "component",
    "coverage_variance_pp2",
    "share_pct",
]])}

## Fixed policy RNG: data split x acquisition seed

{_format_table(data_view[[
    "method",
    "component",
    "coverage_variance_pp2",
    "share_pct",
]])}

## Interpretation

For direct raw-target DAPRO, fitted-policy variation is small on a fixed
dataset; acquisition randomness is larger, and data/calibration-test
resampling dominates the outer experiment. For original mean-weight DAPRO,
acquisition randomness is much larger than for either raw-target DAPRO or the
constant policy. Thus the poor original result is not explained primarily by
small-N1 optimizer instability: its objective produces a propensity schedule
that is intrinsically much more sensitive to which target events acquisition
reveals.
"""
    (output_dir / "variance_decomposition_report.md").write_text(
        report,
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--merged-root",
        type=Path,
        default=Path("results/merged_calibration_dfs"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/variance_decomposition_100_toxicity_v1"),
    )
    args = parser.parse_args()
    analyze(args.merged_root, args.output_dir)
    print(f"Stored variance decomposition at {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
