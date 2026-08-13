"""Summarize the metric/LPB score-value experiments from raw diagnostic CSVs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analysis.diagnostics.dapro_binning_audit import SETUPS
from src.dataset_utils.data_utils import get_data


ROOT = Path("outputs/dapro_binning_audit")


def event_times(setup_key: str) -> np.ndarray:
    dataset, setup = SETUPS[setup_key]
    loaded = get_data(True, torch.device("cpu"), dataset, setup, load_x=False)
    return torch.cat([loaded[10], loaded[11]]).numpy().astype(np.int64)


def metric_outer_summary() -> pd.DataFrame:
    files = {
        ("toxicity_qwen", "hazard_k2"): ROOT / "ht_toxicity_hazard_k2_50.csv",
        ("toxicity_qwen", "future_risk_smooth_k4"): (
            ROOT / "ht_toxicity_future_smooth_k4_50.csv"
        ),
        ("red_qwen", "hazard_k2"): ROOT / "ht_red_hazard_k2_50.csv",
        ("red_qwen", "future_risk_smooth_k4"): (
            ROOT / "ht_red_future_smooth_k4_50.csv"
        ),
    }
    truth_by_setup = {}
    for setup_key in {key[0] for key in files}:
        times = event_times(setup_key)
        truths = []
        for seed in range(50):
            np.random.seed(seed)
            rows = np.random.permutation(len(times))[:3000]
            truths.append(100 * float(np.mean(times[rows] <= 200)))
        truth_by_setup[setup_key] = np.asarray(truths)

    rows = []
    for (setup_key, policy), path in files.items():
        data = pd.read_csv(path).sort_values("seed")
        truth = truth_by_setup[setup_key]
        residual = data["estimated_cjr"].to_numpy() - truth
        covariance = float(np.cov(truth, residual, ddof=1)[0, 1])
        rows.append({
            "setup": setup_key,
            "policy": policy,
            "n_outer_splits": len(data),
            "mean_estimated_metric_pct": data["estimated_cjr"].mean(),
            "observed_outer_sample_variance_pp2": data["estimated_cjr"].var(ddof=1),
            "mean_split_truth_pct": truth.mean(),
            "split_truth_variance_pp2": truth.var(ddof=1),
            "one_draw_residual_variance_pp2": residual.var(ddof=1),
            "truth_residual_sample_covariance_pp2": covariance,
            "mean_exact_conditional_variance_pp2": data["exact_variance_pp2"].mean(),
            "law_total_variance_prediction_pp2": (
                truth.var(ddof=1) + data["exact_variance_pp2"].mean()
            ),
            "mean_expected_cost_per_sample": data["expected_cost_per_sample"].mean(),
            "mean_realized_cost_per_sample": data["realized_cost_per_sample"].mean(),
        })
    return pd.DataFrame(rows)


def metric_acquisition_summary() -> pd.DataFrame:
    files = {
        ("toxicity_qwen", "hazard_k2"): (
            ROOT / "metric_acq500_toxicity_hazard_k2_seed0.csv"
        ),
        ("toxicity_qwen", "future_risk_smooth_k4"): (
            ROOT / "metric_acq500_toxicity_future_smooth_k4_seed0.csv"
        ),
        ("red_qwen", "hazard_k2"): (
            ROOT / "metric_acq500_red_hazard_k2_seed0.csv"
        ),
        ("red_qwen", "future_risk_smooth_k4"): (
            ROOT / "metric_acq500_red_future_smooth_k4_seed0.csv"
        ),
    }
    rows = []
    for (setup_key, policy), path in files.items():
        data = pd.read_csv(path)
        times = event_times(setup_key)
        np.random.seed(0)
        outer = np.random.permutation(len(times))[:3000]
        fixed_truth = 100 * float(np.mean(times[outer] <= 200))
        empirical_variance = data["estimated_cjr"].var(ddof=1)
        exact_variance = float(data["exact_variance_pp2"].iloc[0])
        rows.append({
            "setup": setup_key,
            "policy": policy,
            "fixed_data_seed": 0,
            "fixed_policy_seed": 0,
            "n_acquisition_replicates": len(data),
            "fixed_split_truth_pct": fixed_truth,
            "mean_estimated_metric_pct": data["estimated_cjr"].mean(),
            "empirical_acquisition_variance_pp2": empirical_variance,
            "exact_conditional_variance_pp2": exact_variance,
            "empirical_to_exact_ratio": empirical_variance / exact_variance,
            "expected_cost_per_sample": data["expected_cost_per_sample"].iloc[0],
            "mean_realized_cost_per_sample": data["realized_cost_per_sample"].mean(),
        })
    return pd.DataFrame(rows)


def lpb_outer_summary() -> pd.DataFrame:
    rows = []
    for setup in ["toxicity", "red"]:
        screen = pd.read_csv(ROOT / f"lpb_screen10_{setup}.csv")
        extension = pd.read_csv(ROOT / f"lpb_extend40_{setup}.csv")
        data = pd.concat([
            screen[screen["variant"].isin(["hazard_k2", "target_value_k4"])],
            extension,
        ], ignore_index=True)
        data.to_csv(ROOT / f"lpb_outer50_{setup}_current_vs_value.csv", index=False)
        for variant, group in data.groupby("variant"):
            rows.append({
                "setup": f"{setup}_qwen",
                "policy": variant,
                "n_outer_splits": len(group),
                "mean_coverage_pct": group["coverage_pct"].mean(),
                "coverage_sample_variance_pp2": group["coverage_pct"].var(ddof=1),
                "mean_selected_size": group["selected_size"].mean(),
                "mean_selected_fixed_candidate_conditional_variance_pp2": (
                    group["selected_exact_conditional_variance_pp2"].mean()
                ),
                "mean_full_cal_fixed_candidate_conditional_variance_pp2": (
                    group["fixed_candidate_exact_conditional_variance_pp2"].mean()
                ),
                "mean_target_anchor_conditional_variance_pp2": (
                    group["anchor_exact_conditional_variance_pp2"].mean()
                ),
                "switch_from_full_cal_oracle_rate": (
                    group["switched_from_oracle_fixed"].mean()
                ),
                "mean_absolute_candidate_index_displacement": (
                    group["selected_index_minus_oracle"].abs().mean()
                ),
                "candidate_switch_coverage_effect_variance_pp2": (
                    group["candidate_switch_coverage_effect_pp"].var(ddof=1)
                ),
                "mean_expected_cost_per_sample": group["expected_cost_per_sample"].mean(),
                "mean_realized_cost_per_sample": group["realized_cost_per_sample"].mean(),
            })
    return pd.DataFrame(rows)


def lpb_acquisition_summary() -> pd.DataFrame:
    rows = []
    for setup in ["toxicity", "red"]:
        data = pd.concat([
            pd.read_csv(ROOT / f"lpb_acq50_{setup}_seed{seed}.csv")
            for seed in range(5)
        ], ignore_index=True)
        data.to_csv(ROOT / f"lpb_acq250_{setup}_five_fixed_splits.csv", index=False)
        strata = data.groupby(["data_seed", "variant"]).agg(
            coverage_variance_pp2=("coverage_pct", lambda values: values.var(ddof=1)),
            fixed_candidate_ht_variance_pp2=(
                "fixed_candidate_ht_alpha_pct", lambda values: values.var(ddof=1)
            ),
            mean_exact_fixed_candidate_variance_pp2=(
                "fixed_candidate_exact_conditional_variance_pp2", "mean"
            ),
            mean_exact_selected_candidate_variance_pp2=(
                "selected_exact_conditional_variance_pp2", "mean"
            ),
            switch_rate=("switched_from_oracle_fixed", "mean"),
            switch_effect_variance_pp2=(
                "candidate_switch_coverage_effect_pp", lambda values: values.var(ddof=1)
            ),
            expected_cost_per_sample=("expected_cost_per_sample", "mean"),
            realized_cost_per_sample=("realized_cost_per_sample", "mean"),
        ).reset_index()
        for variant, group in strata.groupby("variant"):
            rows.append({
                "setup": f"{setup}_qwen",
                "policy": variant,
                "n_fixed_outer_splits": len(group),
                "acquisition_replicates_per_split": 50,
                "mean_within_split_coverage_variance_pp2": (
                    group["coverage_variance_pp2"].mean()
                ),
                "median_within_split_coverage_variance_pp2": (
                    group["coverage_variance_pp2"].median()
                ),
                "mean_fixed_candidate_ht_variance_pp2": (
                    group["fixed_candidate_ht_variance_pp2"].mean()
                ),
                "mean_exact_fixed_candidate_variance_pp2": (
                    group["mean_exact_fixed_candidate_variance_pp2"].mean()
                ),
                "mean_exact_selected_candidate_variance_pp2": (
                    group["mean_exact_selected_candidate_variance_pp2"].mean()
                ),
                "mean_switch_rate": group["switch_rate"].mean(),
                "mean_switch_effect_variance_pp2": (
                    group["switch_effect_variance_pp2"].mean()
                ),
                "mean_expected_cost_per_sample": (
                    group["expected_cost_per_sample"].mean()
                ),
                "mean_realized_cost_per_sample": (
                    group["realized_cost_per_sample"].mean()
                ),
            })
    return pd.DataFrame(rows)


def main() -> None:
    metric_outer_summary().to_csv(
        ROOT / "score_value_metric_outer50_summary.csv", index=False
    )
    metric_acquisition_summary().to_csv(
        ROOT / "score_value_metric_acq500_summary.csv", index=False
    )
    lpb_outer_summary().to_csv(
        ROOT / "score_value_lpb_outer50_summary.csv", index=False
    )
    lpb_acquisition_summary().to_csv(
        ROOT / "score_value_lpb_acquisition_summary.csv", index=False
    )


if __name__ == "__main__":
    main()
