"""Summarize the matched CRC-DAPRO and fixed-schedule experiments.

The analysis uses exactly one row per method/seed at 90% target coverage.  It
reports the across-seed variance of the obtained coverage, expected-budget
behavior, inverse-propensity diagnostics, objective transfer, allocation
focus, and runtime.  The script also verifies the alpha non-identifiability of
the requested complement-power family numerically.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
MERGED_ROOT = ROOT / "results" / "merged_calibration_dfs"

METHODS = {
    "calibration_dapro_variance_aligned_bins_2_alpha_0p10_global_0p001_"
    "projection_margin_0p00_n1_200_allocation": "DAPRO reserve 0",
    "calibration_dapro_variance_aligned_bins_2_alpha_0p10_global_0p001_"
    "projection_margin_1p00_n1_200_allocation": "DAPRO reserve 1",
    "calibration_dapro_variance_aligned_bins_2_alpha_0p10_global_0p001_"
    "budget_crc_control_100_n1_200_allocation": "CRC-DAPRO fit100/control100",
    "calibration_dapro_variance_aligned_bins_2_alpha_0p10_global_0p001_"
    "budget_crc_control_100_n1_300_allocation": "CRC-DAPRO fit200/control100",
    "calibration_dapro_variance_aligned_bins_2_alpha_0p10_global_0p001_"
    "budget_crc_control_100_row_cap_1p00x_budget_n1_200_allocation": (
        "CRC-DAPRO fit100/control100 rowcap1"
    ),
    "calibration_dapro_variance_aligned_bins_2_alpha_0p10_global_0p001_"
    "budget_crc_control_100_row_cap_2p00x_budget_n1_200_allocation": (
        "CRC-DAPRO fit100/control100 rowcap2"
    ),
    "calibration_dapro_variance_aligned_bins_2_alpha_0p10_global_0p001_"
    "budget_crc_control_100_row_cap_1p00x_budget_n1_300_allocation": (
        "CRC-DAPRO fit200/control100 rowcap1"
    ),
    "calibration_dapro_variance_aligned_bins_2_alpha_0p10_global_0p001_"
    "budget_crc_control_100_row_cap_2p00x_budget_n1_300_allocation": (
        "CRC-DAPRO fit200/control100 rowcap2"
    ),
    "calibration_random_adaptive_optimized_allocation": "Constant empirical",
    "calibration_random_adaptive_optimized_crc_allocation": "Constant CRC",
    "calibration_random_schedule_complement_power_alpha_0p5_crc_allocation": (
        "Complement power alpha=0.5 CRC"
    ),
    "calibration_random_schedule_complement_power_alpha_1_crc_allocation": (
        "Complement power alpha=1 CRC"
    ),
    "calibration_random_schedule_complement_power_alpha_2_crc_allocation": (
        "Complement power alpha=2 CRC"
    ),
    "calibration_random_schedule_power_reach_alpha_0p5_crc_allocation": (
        "Power reach alpha=0.5 CRC"
    ),
    "calibration_random_schedule_power_reach_alpha_1_crc_allocation": (
        "Power reach alpha=1 CRC"
    ),
    "calibration_random_schedule_power_reach_alpha_2_crc_allocation": (
        "Power reach alpha=2 CRC"
    ),
}

DATASETS = {
    "dataset_toxicity_attack_toxic": ("toxicity", 20.0),
    "dataset_autoif_attack_autoif": ("autoif", 20.0),
    "dataset_hallucination3_attack_hallucination": ("hallucination", 10.0),
    "dataset_red_team_attack_default_attack_qwen25_14b_instruct_lm_target_"
    "qwen25_14b_instruct_judge_llama_guard": ("redteam_llamaguard", 10.0),
    "dataset_red_team_attack_default_attack_qwen25_14b_instruct_lm_target_"
    "qwen25_14b_instruct_judge_llm-judge": ("redteam_qwen", 20.0),
}

NUMERIC_METRICS = [
    "coverage",
    "size",
    "total_expected_budget_per_sample",
    "phase2_expected_cost_per_sample",
    "phase2_realized_cost_per_sample",
    "mean_weight",
    "max_weight",
    "p99_weight",
    "effective_sample_size_weight",
    "mean_a_weighted_inverse_probability",
    "mean_a_weighted_inverse_probability_minus_one",
    "conditional_variance_of_ht_mean",
    "phase1_projected_mean_objective_inverse_probability",
    "phase2_mean_objective_inverse_probability",
    "phase2_minus_phase1_projected_mean_objective_gap",
    "projection_transfer_cost_error_per_sample",
    "projection_transfer_assumption_satisfied",
    "risk_budget_selected_mixture_parameter",
    "risk_budget_correction_per_sample",
    "phase2_focus_expected_query_lift",
    "phase2_focus_mean_expected_queries",
    "phase2_nonfocus_mean_expected_queries",
    "phase2_focus_mean_terminal_probability",
    "phase2_nonfocus_mean_terminal_probability",
    "method_runtime_seconds",
]


def _io_path(path: Path) -> str:
    result = os.path.abspath(path)
    if os.name == "nt" and not result.startswith("\\\\?\\"):
        return f"\\\\?\\{result}"
    return result


def _dataset_for_directory(name: str) -> tuple[str, float]:
    for prefix, value in DATASETS.items():
        if name.startswith(prefix):
            return value
    raise ValueError(f"Unrecognized experiment directory: {name}")


def load_per_seed(suffix: str) -> pd.DataFrame:
    directories = sorted(
        path
        for path in MERGED_ROOT.iterdir()
        # ``Path.is_dir`` can return false for valid experiment directories
        # whose full Windows path exceeds MAX_PATH.  The merged root contains
        # only experiment directories with this suffix, and CSV loading below
        # remains the definitive existence check via the extended path.
        if path.name.endswith(f"__{suffix}")
    )
    if len(directories) != len(DATASETS):
        raise RuntimeError(
            f"Expected {len(DATASETS)} merged directories for {suffix!r}; "
            f"found {len(directories)}."
        )
    frames = []
    for directory in directories:
        dataset, target_budget = _dataset_for_directory(directory.name)
        frame = pd.read_csv(_io_path(directory / "all_df.csv"))
        frame = frame[frame["calibration_name"].isin(METHODS)].copy()
        frame = frame[np.isclose(frame["target_coverage"], 0.90)].copy()
        if frame.empty:
            raise RuntimeError(f"No 90% rows in {directory}.")
        duplicates = frame.duplicated(["seed", "calibration_name"])
        if duplicates.any():
            raise RuntimeError(
                f"Duplicate method/seed rows in {directory}: "
                f"{int(duplicates.sum())}."
            )
        frame["dataset"] = dataset
        frame["target_budget_per_sample"] = target_budget
        frame["method"] = frame["calibration_name"].map(METHODS)
        for column in NUMERIC_METRICS:
            if column not in frame:
                frame[column] = np.nan
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["expected_budget_gap_per_sample"] = (
            frame["total_expected_budget_per_sample"] - target_budget
        )
        frame["conditional_expected_budget_violation"] = (
            frame["expected_budget_gap_per_sample"] > 1e-7
        ).astype(float)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    method_sets = result.groupby("dataset")["method"].agg(
        lambda values: frozenset(values)
    )
    if method_sets.nunique() != 1:
        raise RuntimeError(
            "The merged datasets do not contain the same requested methods."
        )
    expected = len(method_sets.iloc[0])
    counts = result.groupby(["dataset", "seed"])["method"].nunique()
    if not (counts == expected).all():
        raise RuntimeError("At least one dataset/seed is missing a method.")
    return result


def verify_complement_reparameterization(per_seed: pd.DataFrame) -> pd.DataFrame:
    labels = [
        "Complement power alpha=0.5 CRC",
        "Complement power alpha=1 CRC",
        "Complement power alpha=2 CRC",
    ]
    metrics = [
        "total_expected_budget_per_sample",
        "phase2_expected_cost_per_sample",
        "mean_weight",
        "mean_a_weighted_inverse_probability",
    ]
    if not set(labels).issubset(set(per_seed["method"])):
        return pd.DataFrame()
    subset = per_seed[per_seed["method"].isin(labels)]
    rows = []
    for (dataset, seed), group in subset.groupby(["dataset", "seed"]):
        indexed = group.set_index("method")
        reference = indexed.loc[labels[1], metrics].to_numpy(dtype=float)
        for label in [labels[0], labels[2]]:
            difference = np.abs(
                indexed.loc[label, metrics].to_numpy(dtype=float) - reference
            )
            rows.append({
                "dataset": dataset,
                "seed": seed,
                "comparison": f"{label} versus alpha=1",
                **{
                    f"abs_difference_{metric}": value
                    for metric, value in zip(metrics, difference)
                },
            })
    return pd.DataFrame(rows)


def summarize(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, method), group in per_seed.groupby(["dataset", "method"]):
        budget_gap = group["expected_budget_gap_per_sample"]
        n = len(group)

        def mean(column: str) -> float:
            return float(group[column].mean())

        def sd(column: str) -> float:
            return float(group[column].std(ddof=1))

        gap_sd = float(budget_gap.std(ddof=1))
        rows.append({
            "dataset": dataset,
            "method": method,
            "n_seeds": n,
            "coverage_pct": 100 * mean("coverage"),
            "coverage_variance_pp2": float(
                np.var(100 * group["coverage"].to_numpy(), ddof=1)
            ),
            "lpb_size": mean("size"),
            "expected_budget": mean("total_expected_budget_per_sample"),
            "expected_budget_gap": float(budget_gap.mean()),
            "expected_budget_gap_sd": gap_sd,
            "expected_budget_gap_mean_95_upper": float(
                budget_gap.mean() + 1.96 * gap_sd / np.sqrt(n)
            ),
            "conditional_budget_violation_rate": mean(
                "conditional_expected_budget_violation"
            ),
            "phase2_expected_cost": mean("phase2_expected_cost_per_sample"),
            "phase2_realized_cost": mean("phase2_realized_cost_per_sample"),
            "mean_weight": mean("mean_weight"),
            "p99_weight": mean("p99_weight"),
            "max_weight": mean("max_weight"),
            "effective_sample_size": mean("effective_sample_size_weight"),
            "mean_A_over_pi": mean(
                "mean_a_weighted_inverse_probability"
            ),
            "mean_A_times_inverse_pi_minus_one": mean(
                "mean_a_weighted_inverse_probability_minus_one"
            ),
            "conditional_variance_ht_mean": mean(
                "conditional_variance_of_ht_mean"
            ),
            "phase1_projected_objective": mean(
                "phase1_projected_mean_objective_inverse_probability"
            ),
            "phase2_objective": mean(
                "phase2_mean_objective_inverse_probability"
            ),
            "objective_transfer_gap": mean(
                "phase2_minus_phase1_projected_mean_objective_gap"
            ),
            "projection_cost_transfer_error": mean(
                "projection_transfer_cost_error_per_sample"
            ),
            "projection_assumption_rate": mean(
                "projection_transfer_assumption_satisfied"
            ),
            "crc_selected_scale": mean(
                "risk_budget_selected_mixture_parameter"
            ),
            "crc_correction": mean("risk_budget_correction_per_sample"),
            "target_query_lift": mean("phase2_focus_expected_query_lift"),
            "target_expected_queries": mean(
                "phase2_focus_mean_expected_queries"
            ),
            "nontarget_expected_queries": mean(
                "phase2_nonfocus_mean_expected_queries"
            ),
            "target_terminal_pi": mean(
                "phase2_focus_mean_terminal_probability"
            ),
            "nontarget_terminal_pi": mean(
                "phase2_nonfocus_mean_terminal_probability"
            ),
            "runtime_seconds": mean("method_runtime_seconds"),
        })
    return pd.DataFrame(rows).sort_values(
        ["dataset", "coverage_variance_pp2"]
    )


def variance_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    wide = summary.pivot(
        index="dataset",
        columns="method",
        values="coverage_variance_pp2",
    )
    rows = []
    for method in wide.columns:
        ratios_empirical = wide[method] / wide["Constant empirical"]
        ratios_crc = wide[method] / wide["Constant CRC"]
        rows.append({
            "method": method,
            "geometric_variance_ratio_vs_constant_empirical": float(
                np.exp(np.log(ratios_empirical).mean())
            ),
            "geometric_variance_ratio_vs_constant_crc": float(
                np.exp(np.log(ratios_crc).mean())
            ),
            "dataset_wins_vs_constant_empirical": int(
                (ratios_empirical < 1).sum()
            ),
            "dataset_wins_vs_constant_crc": int((ratios_crc < 1).sum()),
        })
    return pd.DataFrame(rows).sort_values(
        "geometric_variance_ratio_vs_constant_crc"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", default="crc_dynamic_schedules_v1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "crc_dynamic_schedules",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_seed = load_per_seed(args.suffix)
    summary = summarize(per_seed)
    invariance = verify_complement_reparameterization(per_seed)
    comparison = variance_comparison(summary)

    per_seed.to_csv(args.output_dir / "per_seed.csv", index=False)
    summary.to_csv(args.output_dir / "method_summary.csv", index=False)
    invariance.to_csv(
        args.output_dir / "complement_alpha_invariance.csv",
        index=False,
    )
    comparison.to_csv(
        args.output_dir / "geometric_variance_comparison.csv",
        index=False,
    )
    print(summary.to_string(index=False))
    print("\nCross-dataset variance comparison")
    print(comparison.to_string(index=False))
    print("\nMaximum complement-alpha differences")
    if invariance.empty:
        print("not part of this experiment")
    else:
        difference_columns = [
            column
            for column in invariance
            if column.startswith("abs_difference_")
        ]
        print(invariance[difference_columns].max().to_string())


if __name__ == "__main__":
    main()
