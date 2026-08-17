"""Audit current LPB, UPB, and metric result families on matched cells.

The script intentionally reads every merged result directory, including the
hallucination runs whose directory names predate the unified-suite suffixes.
It writes one row per exact method configuration and task cell.  Shared
baselines are not duplicated across DAPRO N1 values.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOTS = {
    "lpb": ROOT / "results" / "merged_calibration_dfs",
    "upb": ROOT / "results" / "merged_upb_calibration_dfs",
    "metric": ROOT / "results" / "merged_metric_calibration_dfs",
}
OUTPUT = ROOT / "analysis" / "diagnostics" / "all_results_method_selection"


def _metadata(path: Path) -> dict[str, object]:
    name = path.parent.name
    dataset_match = re.search(r"^dataset_(.*?)_attack_", name)
    model_match = re.search(r"_lm_target_(.*?)_judge_", name)
    budget_match = re.search(r"_(\d+(?:\.\d+)?)_(?:calibration|m)(?:_|$)", name)
    if not (dataset_match and model_match and budget_match):
        raise ValueError(f"Cannot parse result metadata from {name}")
    return {
        "dataset": dataset_match.group(1),
        "target_model": model_match.group(1),
        "budget": float(budget_match.group(1)),
        "source": name,
    }


def _method(name: str) -> tuple[str, str, bool, float | None]:
    lower = name.lower()
    crc = "crc_control_" in lower or "budget_crc_control_" in lower
    control_match = re.search(r"(?:budget_)?crc_control_(\d+)", lower)
    control = float(control_match.group(1)) if control_match else None
    n1_match = re.search(r"_n1_(\d+)", lower)
    n1 = float(n1_match.group(1)) if n1_match else None
    if n1 is None and control is not None:
        n1 = 2.0 * control

    if "oracle" in lower:
        family = "Oracle"
        crc = False
        n1 = None
    elif lower in {"optimized", "calibration_optimized_allocation"}:
        family = "Static"
        crc = False
        n1 = None
    elif "endpoint_block_terminal_residual_aht" in lower or (
        "upb_residual_aht" in lower and "model_anchor" in lower
    ):
        family = "Endpoint/block"
    elif "information_gain" in lower:
        family = "Information-gain"
    elif "residual_sequential" in lower:
        family = "Residual"
    elif "soft_prefix" in lower:
        family = "Soft-prefix"
    else:
        family = f"Other: {name}"
    label = family + (" + CRC" if crc else "")
    return family, label, crc, n1


COMMON_COLUMNS = {
    "seed",
    "calibration_name",
    "allocator_name",
    "coverage",
    "size",
    "infinite_bound_rate",
    "target_coverage",
    "policy_target_coverage",
    "budget_used",
    "budget_per_sample",
    "configured_budget_per_sample",
    "total_expected_budget_per_sample",
    "total_expected_budget_valid",
    "risk_budget_selector_valid",
    "estimated_conditional_variance_upb_coverage_estimator",
    "conditional_variance_of_ht_mean",
    "estimated_cjr",
    "full_benchmark_cjr",
    "estimated_rmttu",
    "full_benchmark_rmttu",
    "unsafe_event_rate_estimator_kind",
    "aht_estimator_kind",
}


def _read_task(task: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(RESULT_ROOTS[task].rglob("all_df.csv")):
        header = pd.read_csv(path, nrows=0).columns
        usecols = sorted(COMMON_COLUMNS.intersection(header))
        frame = pd.read_csv(path, usecols=usecols, low_memory=False)
        metadata = _metadata(path)
        for key, value in metadata.items():
            frame[key] = value
        name_col = "allocator_name" if task == "metric" else "calibration_name"
        mapped = frame[name_col].astype(str).map(_method)
        frame["family"] = mapped.map(lambda value: value[0])
        frame["method"] = mapped.map(lambda value: value[1])
        frame["crc"] = mapped.map(lambda value: value[2])
        frame["n1"] = mapped.map(lambda value: value[3])

        if task == "lpb":
            frame = frame[np.isclose(frame["target_coverage"], 0.90)]
            frame["analysis_target"] = 0.90
        elif task == "upb":
            target = frame["target_coverage"]
            policy = frame["policy_target_coverage"]
            keep_target = np.isclose(target, 0.70) | np.isclose(target, 0.80) | np.isclose(target, 0.90)
            keep_policy = policy.isna() | np.isclose(policy, target)
            frame = frame[keep_target & keep_policy]
            frame["analysis_target"] = frame["target_coverage"]
        else:
            frame["analysis_target"] = np.nan
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No {task} result files found")
    result = pd.concat(frames, ignore_index=True)
    duplicates = result.duplicated(
        ["source", "seed", "method", "n1", "analysis_target"], keep=False
    )
    if duplicates.any():
        sample = result.loc[
            duplicates,
            ["source", "seed", "method", "n1", "analysis_target"],
        ].head()
        raise ValueError(f"Duplicate selected rows for {task}:\n{sample}")
    return result


def _safe_var(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    return float(finite.var(ddof=1)) if len(finite) >= 2 else np.nan


def _safe_mean(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    return float(finite.mean()) if len(finite) else np.nan


def _bound_summary(frame: pd.DataFrame, task: str) -> pd.DataFrame:
    keys = [
        "dataset", "target_model", "budget", "source", "analysis_target",
        "family", "method", "crc", "n1",
    ]
    rows = []
    for key, group in frame.groupby(keys, dropna=False, sort=False):
        row = dict(zip(keys, key))
        coverage = pd.to_numeric(group["coverage"], errors="coerce")
        target = float(row["analysis_target"])
        row.update({
            "seeds": int(group["seed"].nunique()),
            "coverage_mean": float(coverage.mean()),
            "coverage_variance": _safe_var(coverage),
            "coverage_mse_to_target": float(((coverage - target) ** 2).mean()),
            "coverage_bias": float(coverage.mean() - target),
            "size_mean": _safe_mean(group["size"]),
            "infinite_rate_mean": _safe_mean(group.get("infinite_bound_rate", pd.Series(dtype=float))),
            "expected_cost_mean": _safe_mean(group.get("total_expected_budget_per_sample", pd.Series(dtype=float))),
            "realized_cost_mean": (
                _safe_mean(group["budget_used"])
                / _safe_mean(group.get("configured_cal_size", pd.Series(dtype=float)))
                if "configured_cal_size" in group and _safe_mean(group.get("configured_cal_size", pd.Series(dtype=float))) > 0
                else np.nan
            ),
            "conditional_variance_mean": _safe_mean(
                group[
                    "estimated_conditional_variance_upb_coverage_estimator"
                    if task == "upb"
                    else "conditional_variance_of_ht_mean"
                ]
            ),
            "split_budget_valid_rate": _safe_mean(group.get("total_expected_budget_valid", pd.Series(dtype=float))),
            "crc_selector_valid_rate": _safe_mean(group.get("risk_budget_selector_valid", pd.Series(dtype=float))),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def _metric_summary(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "dataset", "target_model", "budget", "source", "family", "method",
        "crc", "n1",
    ]
    rows = []
    for key, group in frame.groupby(keys, dropna=False, sort=False):
        row = dict(zip(keys, key))
        cjr = pd.to_numeric(group["estimated_cjr"], errors="coerce")
        cjr_truth = pd.to_numeric(group["full_benchmark_cjr"], errors="coerce")
        rmttu = pd.to_numeric(group["estimated_rmttu"], errors="coerce")
        rmttu_truth = pd.to_numeric(group["full_benchmark_rmttu"], errors="coerce")
        row.update({
            "seeds": int(group["seed"].nunique()),
            "cjr_mean": float(cjr.mean()),
            "cjr_variance": _safe_var(cjr),
            "cjr_bias": float((cjr - cjr_truth).mean()),
            "cjr_mae": float((cjr - cjr_truth).abs().mean()),
            "cjr_mse": float(((cjr - cjr_truth) ** 2).mean()),
            "rmttu_mean": float(rmttu.mean()),
            "rmttu_variance": _safe_var(rmttu),
            "rmttu_bias": float((rmttu - rmttu_truth).mean()),
            "rmttu_mae": float((rmttu - rmttu_truth).abs().mean()),
            "rmttu_mse": float(((rmttu - rmttu_truth) ** 2).mean()),
            "expected_cost_mean": _safe_mean(group.get("total_expected_budget_per_sample", pd.Series(dtype=float))),
            "realized_cost_mean": _safe_mean(group.get("budget_per_sample", pd.Series(dtype=float))),
            "split_budget_valid_rate": _safe_mean(group.get("total_expected_budget_valid", pd.Series(dtype=float))),
            "crc_selector_valid_rate": _safe_mean(group.get("risk_budget_selector_valid", pd.Series(dtype=float))),
            "estimator_kind": ";".join(sorted(set(
                group.get("unsafe_event_rate_estimator_kind", pd.Series(dtype=str)).dropna().astype(str)
            ))),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def _add_static_ratios(frame: pd.DataFrame, metrics: list[str], task: str) -> pd.DataFrame:
    cell = ["source", "analysis_target"] if task != "metric" else ["source"]
    static = frame[frame["family"] == "Static"][cell + metrics].copy()
    static = static.rename(columns={metric: f"static_{metric}" for metric in metrics})
    if static.duplicated(cell).any():
        raise ValueError(f"Static baseline duplicated for {task}")
    result = frame.merge(static, on=cell, how="left", validate="many_to_one")
    for metric in metrics:
        denominator = result[f"static_{metric}"]
        result[f"{metric}_ratio_to_static"] = result[metric] / denominator
    return result


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lpb = _add_static_ratios(
        _bound_summary(_read_task("lpb"), "lpb"),
        ["coverage_variance", "coverage_mse_to_target"],
        "lpb",
    )
    upb = _add_static_ratios(
        _bound_summary(_read_task("upb"), "upb"),
        ["coverage_variance", "coverage_mse_to_target", "conditional_variance_mean"],
        "upb",
    )
    metric = _add_static_ratios(
        _metric_summary(_read_task("metric")),
        ["cjr_variance", "cjr_mse", "cjr_mae", "rmttu_variance", "rmttu_mse", "rmttu_mae"],
        "metric",
    )
    metric["joint_geometric_mse_ratio"] = np.sqrt(
        metric["cjr_mse_ratio_to_static"]
        * metric["rmttu_mse_ratio_to_static"]
    )
    metric["joint_worst_mse_ratio"] = metric[[
        "cjr_mse_ratio_to_static", "rmttu_mse_ratio_to_static"
    ]].max(axis=1)

    for task, frame in (("lpb", lpb), ("upb", upb), ("metric", metric)):
        sort_columns = ["dataset", "target_model", "budget"]
        if "analysis_target" in frame:
            sort_columns.append("analysis_target")
        sort_columns.extend(["method", "n1"])
        frame.sort_values(
            sort_columns,
            na_position="first",
        ).to_csv(OUTPUT / f"{task}_config_summary.csv", index=False)
        print(
            task,
            f"files={frame['source'].nunique()}",
            f"cells={len(frame)}",
            f"datasets={sorted(frame['dataset'].unique())}",
        )


if __name__ == "__main__":
    main()
