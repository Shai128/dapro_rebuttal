"""Reproduce the LPB score-noise and alternative-score mechanism audit."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analysis.diagnostics.dapro_binning_audit import load_setup, rank_auc
from analysis.diagnostics.lpb_dapro_binning_audit import (
    lpb_quantiles,
    target_value_scores,
)
from src.predictive_bounds.experiments.full_bounds.config import (
    method_display_name,
)
from src.predictive_bounds.utils.utils import make_lpb_tau_grid


DEFAULT_RESULT = Path(
    "results/merged_calibration_dfs/"
    "dataset_toxicity_attack_toxic_attack_qwen25_14b_instruct_lm_target_"
    "qwen25_14b_instruct_judge_detoxify_20_calibration_"
    "dapro_lpb_ablation_v1_score_noise/all_df.csv"
)
DEFAULT_OUTPUT = Path("outputs/dapro_binning_audit")


def summarize_noise_result(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(path)
    frame = frame[np.isclose(frame["target_coverage"], 0.90)].copy()
    frame["method"] = frame["calibration_name"].map(method_display_name)
    frame = frame[frame["method"].isin({"DAPRO", "DAPRO w/o CRC"})]
    frame["score_noise_lambda"] = pd.to_numeric(
        frame["ablation_value"], errors="coerce"
    )
    frame["coverage_pct"] = 100 * frame["coverage"]
    frame["coverage_difference_pp"] = (
        frame["coverage_pct"] - 90.0
    ).abs()
    frame["budget_used_per_sample"] = (
        frame["actual_event_stopped_budget_per_sample"]
    )
    frame["selected_candidate_conditional_variance_pp2"] = (
        10_000 * frame["conditional_variance_of_ht_mean"]
    )
    frame["target_anchor_conditional_variance_pp2"] = (
        10_000
        * frame["phase2_mean_objective_variance_proxy"]
        * frame["phase2_sample_count"]
        / frame["configured_cal_size"].pow(2)
    )
    metric_columns = [
        "coverage_pct",
        "coverage_difference_pp",
        "size",
        "budget_used_per_sample",
        "mean_calibrated_a_weighted_inverse_probability",
        "selected_candidate_conditional_variance_pp2",
        "target_anchor_conditional_variance_pp2",
        "phase2_mean_objective_variance_proxy",
        "phase2_mean_objective_inverse_probability",
        "target_anchor_phase2_a_rate",
        "ablation_score_mean_timewise_pearson_correlation",
        "ablation_score_original_k2_bin_agreement",
        "risk_budget_selected_mixture_parameter",
    ]
    summary = (
        frame.groupby(["score_noise_lambda", "method"], observed=True)[
            metric_columns
        ]
        .agg(["mean", "var", "std", "count"])
    )
    summary.columns = ["__".join(column) for column in summary.columns]
    summary = summary.reset_index()

    paired_rows = []
    for method in ("DAPRO", "DAPRO w/o CRC"):
        method_frame = frame[frame["method"].eq(method)]
        for metric in metric_columns:
            wide = method_frame.pivot(
                index="seed", columns="score_noise_lambda", values=metric
            )
            if 0.0 not in wide or 1.0 not in wide:
                continue
            difference = (wide[1.0] - wide[0.0]).dropna()
            standard_error = difference.std(ddof=1) / np.sqrt(len(difference))
            paired_rows.append({
                "method": method,
                "metric": metric,
                "n_paired_splits": len(difference),
                "lambda_0_mean": wide.loc[difference.index, 0.0].mean(),
                "lambda_1_mean": wide.loc[difference.index, 1.0].mean(),
                "paired_mean_difference_lambda1_minus_lambda0": difference.mean(),
                "paired_standard_error": standard_error,
                "normal_95ci_low": difference.mean() - 1.96 * standard_error,
                "normal_95ci_high": difference.mean() + 1.96 * standard_error,
            })
    return summary, pd.DataFrame(paired_rows)


def _binary_median_score(values: np.ndarray) -> np.ndarray:
    return (values >= np.median(values)).astype(np.int8)


def compute_lpb_target_score_auc() -> pd.DataFrame:
    """Compare current hazard with remaining named-target probability."""
    grid, times, _, _ = load_setup("toxicity_qwen")
    taus = make_lpb_tau_grid(device="cpu")
    quantiles = lpb_quantiles(grid, taus, 200)
    anchor_index = int(torch.argmin(torch.abs(taus - 0.10)).item())
    horizons = quantiles[:, anchor_index]
    target_value = target_value_scores(grid, horizons, 200)
    event = (times < horizons).numpy()
    times_np = times.numpy()
    horizons_np = horizons.numpy()
    rows = []
    for step in (0, 1, 4, 9, 19, 49, 99, 149):
        hazard = grid[:, step, step].to(torch.float64).numpy()
        scores = {"instantaneous_hazard": hazard, "target_value": target_value[:, step]}
        active = times_np > step
        eligible = active & (horizons_np > step + 1)
        for name, values in scores.items():
            row = {
                "turn": step + 1,
                "score": name,
                "active_count": int(active.sum()),
                "eligible_count": int(eligible.sum()),
                "active_target_rate": float(event[active].mean()),
                "eligible_target_rate": float(event[eligible].mean()),
                "active_raw_auc": rank_auc(values[active], event[active]),
                "active_k2_auc": rank_auc(
                    _binary_median_score(values[active]), event[active]
                ),
                "eligible_raw_auc": rank_auc(values[eligible], event[eligible]),
                "eligible_k2_auc": rank_auc(
                    _binary_median_score(values[eligible]), event[eligible]
                ),
                "active_zero_score_share": float((values[active] == 0).mean()),
            }
            rows.append(row)
    result = pd.DataFrame(rows)
    result["anchor_tau"] = float(taus[anchor_index])
    result["full_target_rate"] = float(event.mean())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-noise-result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary, paired = summarize_noise_result(args.score_noise_result)
    auc = compute_lpb_target_score_auc()
    summary.to_csv(args.output_dir / "score_noise_lpb_deep_summary.csv", index=False)
    paired.to_csv(args.output_dir / "score_noise_lpb_endpoint_paired.csv", index=False)
    auc.to_csv(args.output_dir / "score_noise_lpb_target_auc.csv", index=False)
    print(summary.to_string(index=False))
    print("\nPaired lambda endpoint changes:\n", paired.to_string(index=False))
    print("\nTarget-score AUC audit:\n", auc.to_string(index=False))


if __name__ == "__main__":
    main()
