"""Summarize the paired LPB score/map audit artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path("outputs/dapro_binning_audit")


def load_outer(setup: str) -> pd.DataFrame:
    first = pd.read_csv(ROOT / f"lpb_screen10_{setup}.csv")
    rest = pd.read_csv(ROOT / f"lpb_extend40_{setup}.csv")
    return pd.concat([first, rest], ignore_index=True)


def summarize_outer(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in frame.groupby("variant"):
        rows.append({
            "variant": variant,
            "n_splits": len(group),
            "coverage_mean_pct": group["coverage_pct"].mean(),
            "coverage_variance_pp2": group["coverage_pct"].var(ddof=1),
            "expected_cost_per_sample": group[
                "expected_cost_per_sample"
            ].mean(),
            "realized_cost_per_sample": group[
                "realized_cost_per_sample"
            ].mean(),
            "selected_conditional_variance_pp2": group[
                "selected_exact_conditional_variance_pp2"
            ].mean(),
            "full_cal_oracle_candidate_conditional_variance_pp2": group[
                "fixed_candidate_exact_conditional_variance_pp2"
            ].mean(),
            "anchor_conditional_variance_pp2": group[
                "anchor_exact_conditional_variance_pp2"
            ].mean(),
            "candidate_switch_rate": group[
                "switched_from_oracle_fixed"
            ].mean(),
            "mean_abs_candidate_index_displacement": group[
                "selected_index_minus_oracle"
            ].abs().mean(),
        })
    return pd.DataFrame(rows)


def summarize_acquisition(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in frame.groupby("variant"):
        rows.append({
            "variant": variant,
            "n_acquisition_draws": len(group),
            "fixed_split_coverage_variance_pp2": group[
                "coverage_pct"
            ].var(ddof=1),
            "fixed_split_alpha_variance_rate2": (
                group["selected_alpha_hat"].var(ddof=1)
            ),
            "candidate_switch_rate": group[
                "switched_from_oracle_fixed"
            ].mean(),
        })
    return pd.DataFrame(rows)


def main() -> None:
    outer_rows = []
    acquisition_rows = []
    for setup in ("toxicity", "red"):
        outer = summarize_outer(load_outer(setup))
        outer.insert(0, "setup", setup)
        outer_rows.append(outer)
        acquisition = summarize_acquisition(
            pd.read_csv(ROOT / f"lpb_acq50_{setup}_seed0.csv")
        )
        acquisition.insert(0, "setup", setup)
        acquisition_rows.append(acquisition)
    outer_summary = pd.concat(outer_rows, ignore_index=True)
    acquisition_summary = pd.concat(acquisition_rows, ignore_index=True)
    outer_summary.to_csv(ROOT / "lpb_score_map_50split_summary.csv", index=False)
    acquisition_summary.to_csv(
        ROOT / "lpb_score_map_fixed_split_acquisition_summary.csv", index=False
    )
    print("OUTER SPLITS")
    print(outer_summary.to_string(index=False))
    print("\nFIXED-SPLIT ACQUISITION")
    print(acquisition_summary.to_string(index=False))


if __name__ == "__main__":
    main()
