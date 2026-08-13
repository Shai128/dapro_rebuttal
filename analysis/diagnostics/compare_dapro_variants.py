"""Summarize the matched DAPRO coefficient/controller comparison.

The script consumes checked-in or locally generated experiment outputs.  It
does not run an allocator and does not modify production registries.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs" / "dapro_universal_comparison"


def _method(name: str) -> str:
    controller = "crc" if "budget_crc" in name else "no_crc"
    if "soft_prefix" in name:
        family = "generalized_soft"
    elif "variance_aligned" in name:
        family = "definitive"
    elif "a_target" in name:
        family = "target_a"
    else:
        raise ValueError(f"Unrecognized DAPRO name: {name}")
    return f"{family}_{controller}"


def _setup(experiment: str) -> str:
    if "dataset_autoif" in experiment:
        return "autoif_qwen"
    if "dataset_toxicity" in experiment:
        return "toxicity_phi" if "mini_phi" in experiment else "toxicity_qwen"
    if "dataset_red_team" in experiment:
        return "red_llamaguard" if "llama_guard" in experiment else "red_qwen"
    if "dataset_hallucination3" in experiment:
        if "mini_phi" in experiment:
            return "hallucination_phi"
        if "llama_31_8B" in experiment:
            return "hallucination_llama"
        return "hallucination_qwen"
    return experiment


def _split_method(method: str) -> tuple[str, str]:
    if method.endswith("_no_crc"):
        return method[:-7], "no_crc"
    if method.endswith("_crc"):
        return method[:-4], "crc"
    raise ValueError(f"Method lacks a controller suffix: {method!r}")


def _rank_summary(
    frame: pd.DataFrame,
    value: str,
    *,
    group: tuple[str, ...] = ("controller", "setup"),
) -> pd.DataFrame:
    work = frame.copy()
    best = work.groupby(list(group))[value].transform("min")
    work["regret"] = np.divide(
        work[value], best,
        out=np.ones(len(work), dtype=float),
        where=best.to_numpy() > 1e-12,
    )
    work["winner"] = np.isclose(work[value], best, rtol=1e-8, atol=1e-10)
    return (
        work.groupby(["controller", "family"])
        .agg(
            cells=(value, "size"),
            wins=("winner", "sum"),
            geometric_mean_regret=(
                "regret", lambda x: float(np.exp(np.log(x).mean()))
            ),
            worst_regret=("regret", "max"),
            mean_value=(value, "mean"),
            mean_cost=("cost", "mean"),
            mean_split_budget_valid_rate=("budget_valid_rate", "mean"),
        )
        .reset_index()
    )


def metric_summary(suffix: str, expected_setups: int) -> pd.DataFrame:
    files = list(
        (ROOT / "results" / "tmp_metric_calibration_results").glob(
            f"*{suffix}/*/seed=*.csv"
        )
    )
    if not files:
        raise FileNotFoundError(f"No metric files found for {suffix!r}")
    rows = []
    for file in files:
        frame = pd.read_csv(file)
        frame["experiment"] = file.parents[1].name
        rows.append(frame)
    data = pd.concat(rows, ignore_index=True)
    data["method"] = data["allocator_name"].map(_method)
    data["setup"] = data["experiment"].map(_setup)
    data[["family", "controller"]] = data["method"].apply(
        lambda value: pd.Series(_split_method(value))
    )
    grouped = []
    for keys, group in data.groupby(
        ["setup", "family", "controller"], sort=True
    ):
        grouped.append({
            "setup": keys[0],
            "family": keys[1],
            "controller": keys[2],
            "n_splits": len(group),
            "metric_variance_pp2": group["estimated_cjr"].var(ddof=1),
            "mean_exact_acquisition_variance_pp2": (
                10_000
                * group["conditional_variance_unsafe_event_rate_estimator"].mean()
            ),
            "mean_estimate_pct": group["estimated_cjr"].mean(),
            "truth_pct": group["full_benchmark_cjr"].iloc[0],
            "bias_pp": (
                group["estimated_cjr"].mean()
                - group["full_benchmark_cjr"].iloc[0]
            ),
            "cost": group["total_expected_budget_per_sample"].mean(),
            "budget_valid_rate": group["total_expected_budget_valid"].mean(),
            "crc_selector_valid_rate": (
                group["risk_budget_selector_valid"].mean()
                if "risk_budget_selector_valid" in group
                else np.nan
            ),
        })
    result = pd.DataFrame(grouped)
    if result["setup"].nunique() != expected_setups:
        raise RuntimeError(
            f"Expected {expected_setups} metric setups, got "
            f"{result['setup'].nunique()} for {suffix}"
        )
    return result


def lpb_corrected_summary() -> pd.DataFrame:
    files = list(
        (ROOT / "results" / "tmp_calibration_results").glob(
            "*lpb_universal_compare_v1/*/seed=*.csv"
        )
    )
    if not files:
        raise FileNotFoundError("No corrected LPB comparison files found")
    rows = []
    for file in files:
        frame = pd.read_csv(file)
        frame = frame[np.isclose(frame["target_coverage"], 0.90)].copy()
        frame["experiment"] = file.parents[1].name
        rows.append(frame)
    data = pd.concat(rows, ignore_index=True)
    data["method"] = data["calibration_name"].map(_method)
    data["setup"] = data["experiment"].map(_setup)
    data[["family", "controller"]] = data["method"].apply(
        lambda value: pd.Series(_split_method(value))
    )
    grouped = []
    for keys, group in data.groupby(
        ["setup", "family", "controller"], sort=True
    ):
        coverage_pct = 100 * group["coverage"]
        grouped.append({
            "setup": keys[0],
            "family": keys[1],
            "controller": keys[2],
            "n_splits": len(group),
            "mean_coverage_pct": coverage_pct.mean(),
            "coverage_variance_pp2": coverage_pct.var(ddof=1),
            "coverage_mse_to_90_pp2": np.mean((coverage_pct - 90) ** 2),
            "mean_selected_acquisition_variance_pp2": (
                10_000 * group["conditional_variance_of_ht_mean"].mean()
            ),
            "mean_size": group["size"].mean(),
            "cost": group["total_expected_budget_per_sample"].mean(),
            "budget_valid_rate": group["total_expected_budget_valid"].mean(),
            "crc_selector_valid_rate": (
                group["risk_budget_selector_valid"].mean()
                if "risk_budget_selector_valid" in group
                else np.nan
            ),
        })
    result = pd.DataFrame(grouped)
    if result["setup"].nunique() != 8:
        raise RuntimeError(
            f"Expected 8 corrected LPB setups, got {result['setup'].nunique()}"
        )
    return result


def broad_lpb_summary() -> pd.DataFrame:
    path = OUTPUT / "lpb_32setup_50split_historical_summary.csv"
    broad = pd.read_csv(path)
    return broad


def _markdown_table(frame: pd.DataFrame, digits: int = 3) -> str:
    printable = frame.copy()
    for column in printable.select_dtypes(include=[np.number]).columns:
        printable[column] = printable[column].map(
            lambda x: f"{x:.{digits}f}" if np.isfinite(x) else "NA"
        )
    headers = list(printable.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    lines.extend(
        "| " + " | ".join(map(str, row)) + " |"
        for row in printable.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metric50 = metric_summary("dapro_universal_compare_v1", 8)
    metric200 = metric_summary("dapro_universal_compare_n200_v1", 4)
    lpb = lpb_corrected_summary()
    broad = broad_lpb_summary()

    metric50.to_csv(OUTPUT / "metric_n50_matched_summary.csv", index=False)
    metric200.to_csv(OUTPUT / "metric_n200_matched_summary.csv", index=False)
    lpb.to_csv(OUTPUT / "lpb_n50_corrected_matched_summary.csv", index=False)

    metric_rank = _rank_summary(
        metric50.rename(columns={
            "mean_exact_acquisition_variance_pp2": "score"
        }),
        "score",
    )
    lpb_rank = _rank_summary(
        lpb.rename(columns={"coverage_mse_to_90_pp2": "score"}),
        "score",
    )
    metric_rank.to_csv(OUTPUT / "metric_n50_method_ranks.csv", index=False)
    lpb_rank.to_csv(OUTPUT / "lpb_n50_method_ranks.csv", index=False)

    broad_agg = (
        broad.groupby(["controller", "n1", "method"])
        .agg(
            cells=("experiment", "size"),
            mean_coverage_variance_pp2=("coverage_var_pp2", "mean"),
            mean_coverage_mse90_pp2=("coverage_mse_to_90_pp2", "mean"),
            mean_acquisition_variance_pp2=("cond_var_pp2", "mean"),
            mean_cost=("cost", "mean"),
            budget_valid_rate=("split_budget_valid_rate", "mean"),
        )
        .reset_index()
    )
    broad_agg.to_csv(OUTPUT / "lpb_32setup_aggregate.csv", index=False)

    report = f"""# Matched DAPRO variant comparison

## Scope

The corrected experiment uses eight locally cached real setups, budget 20,
horizon 200, N1=50, identical outer splits and acquisition CRNs, and the
task-appropriate target A.  No-CRC uses the one-turn projection reserve; CRC
uses the corrected causal shared-PAV row envelope and an independent control
fold of 25 rows.  Metric results use 10 splits.  LPB results use the same 10
splits and evaluate target coverage 90%.  A second metric screen uses N1=200
on four setups and five paired splits.  The historical LPB breadth check uses
32 setup/budget cells, 50 splits, and N1=50/100/200; its CRC rows predate the
causal cap correction and are used only as a ranking-sensitivity check.

## Metric N1=50: exact acquisition-variance ranking

{_markdown_table(metric_rank)}

## LPB N1=50: MSE-to-90 ranking

{_markdown_table(lpb_rank)}

## Broad LPB sensitivity check

{_markdown_table(broad_agg)}

## Interpretation

At N1=50, soft-prefix Generalized DAPRO is the only coefficient estimator
that wins the exact metric acquisition comparison on every setup under both
controllers.  It also has the best aggregate LPB MSE, coverage variance, and
selected-candidate acquisition variance.  Some LPB setups are exact ties
because acquisition noise is already zero.

Hard Target-A is theoretically exact for a frozen target when its coefficient
is known, but estimating a sparse binary endpoint coefficient from 25--50
policy-fit rows is unstable.  Definitive's 0.001 global mass improves it
substantially, but it remains noisier than the Rao--Blackwellized soft mass at
N1=50.  CRC controls cost, not coefficient estimation; for sparse Target-A it
can select a severely contracted policy and amplify the efficiency loss.

There is no theorem that the soft method is universally best.  Its advantage
depends on the conditional PMF being informative enough that reduced
coefficient variance outweighs model misspecification.  In the broad LPB
archive, total 50-split coverage rankings occasionally change at N1=200
because calibration/test split variance dominates tiny acquisition
differences.  Nevertheless, the matched metric N1=200 screen still favors
soft Generalized on exact acquisition variance in all four tested setups,
with and without CRC.
"""
    (OUTPUT / "README.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
