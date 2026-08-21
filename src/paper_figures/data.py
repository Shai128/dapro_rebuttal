"""Load exactly the recommended paper cells from merged result directories."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

from src.evaluation.result_matrix import (
    MatrixResult,
    method_display_name as metric_method_display_name,
    parse_lpb_result,
    parse_metric_result,
    parse_upb_result,
)
from src.paper_figures.config import (
    DATASET_DISPLAY,
    METHOD_ORDER,
    RECOMMENDED_CONFIGURATIONS,
)
from src.predictive_bounds.experiments.full_bounds.config import (
    method_display_name as bound_method_display_name,
)


_N1_RE = re.compile(r"(?:^|_)n1_(?P<n1>\d+)(?:_|$)")

BOUND_SOURCE_COLUMNS = {
    "seed",
    "calibration_name",
    "target_coverage",
    "policy_target_coverage",
    "coverage",
    "size",
    "budget_used",
    "reported_assigned_budget_per_sample",
    "actual_event_stopped_budget_total",
    "actual_event_stopped_budget_per_sample",
    "reported_budget_semantics",
    "configured_cal_size",
    "n_observed_events",
    "mean_weight",
    "mean_a_weighted_inverse_probability",
    "mean_calibrated_a_weighted_inverse_probability",
    "mean_prior_a_weighted_inverse_probability",
    "mean_tau_0p10_a_weighted_inverse_probability",
}

METRIC_SOURCE_COLUMNS = {
    "seed",
    "allocator_name",
    "calibration_name",
    "configured_dapro_n1",
    "configured_crc_control_size",
    "evaluation_sample_size",
    "estimated_cjr",
    "estimated_rmttu",
    "estimated_restricted_mean_time_to_event",
    "full_benchmark_cjr",
    "full_benchmark_rmttu",
    "full_benchmark_restricted_mean_time_to_event",
    "unsafe_event_rate_estimator_kind",
    "budget_per_sample",
    "reported_assigned_budget_per_sample",
    "actual_event_stopped_budget_total",
    "actual_event_stopped_budget_per_sample",
    "reported_budget_semantics",
    "num_events_observed",
    "observed_jailbreaks",
    "mean_metric_target_a_weighted_inverse_probability",
    "mean_metric_a_weighted_inverse_probability",
    "mean_a_weighted_weight",
}


def _source_rank(task: str, path: Path) -> int:
    name = path.parent.name
    if "schema_metrics_validation" in name:
        return -100
    columns = set(pd.read_csv(path, nrows=0).columns)
    current_schema_columns = {
        "lpb": {
            "reported_budget_semantics",
            "actual_event_stopped_budget_per_sample",
            "mean_calibrated_a_weighted_inverse_probability",
            "mean_prior_a_weighted_inverse_probability",
            "mean_tau_0p10_a_weighted_inverse_probability",
        },
        "upb": {
            "reported_budget_semantics",
            "actual_event_stopped_budget_per_sample",
            "mean_calibrated_a_weighted_inverse_probability",
            "mean_prior_a_weighted_inverse_probability",
            "mean_tau_0p10_a_weighted_inverse_probability",
        },
        "metrics": {
            "reported_budget_semantics",
            "actual_event_stopped_budget_per_sample",
            "estimated_restricted_mean_time_to_event",
            "mean_metric_target_a_weighted_inverse_probability",
        },
    }[task]
    # Schema completeness outranks historical experiment suffixes.  This is
    # essential after a fresh run: a new ``full_bounds_v*`` directory must not
    # be shadowed by an older ``*_unified_aht_v*`` download simply because the
    # older filename happens to contain a preferred legacy token.
    schema_score = 100 * len(columns & current_schema_columns)
    if current_schema_columns.issubset(columns):
        schema_score += 10_000
    preferences = {
        "lpb": ("lpb_unified_aht_v1", "_calibration"),
        "upb": ("upb_unified_aht_v2", "_calibration_upb"),
        "metrics": ("metric_unified_aht_v1", "_m"),
    }[task]
    for index, token in enumerate(preferences):
        if token in name:
            return schema_score + 10 - index
    return schema_score


def _discover_sources(input_dir: Path, task: str) -> list[MatrixResult]:
    parser = {
        "lpb": parse_lpb_result,
        "upb": parse_upb_result,
        "metrics": parse_metric_result,
    }[task]
    recommendations = RECOMMENDED_CONFIGURATIONS[task]
    grouped: dict[tuple[str, str], list[tuple[int, MatrixResult]]] = {}
    for path in sorted(input_dir.rglob("all_df.csv")):
        metadata = parser(path)
        if metadata is None or metadata.dataset_key not in recommendations:
            continue
        recommendation = recommendations[metadata.dataset_key]
        if not np.isclose(
            metadata.budget_per_sample,
            recommendation.budget_per_sample,
            atol=1e-9,
        ):
            continue
        rank = _source_rank(task, path)
        if rank < 0:
            continue
        grouped.setdefault(
            (metadata.dataset_key, metadata.target_model), []
        ).append((rank, metadata))

    selected = []
    for key, candidates in sorted(grouped.items()):
        best_rank = max(rank for rank, _ in candidates)
        best = [metadata for rank, metadata in candidates if rank == best_rank]
        if len(best) != 1:
            paths = "\n".join(str(metadata.path) for metadata in best)
            raise ValueError(
                f"Ambiguous {task} result sources for {key}:\n{paths}"
            )
        selected.append(best[0])
    if not selected:
        raise FileNotFoundError(
            f"No recommended {task} result files found below {input_dir}."
        )
    return selected


def _read_available(path: Path, requested: set[str]) -> pd.DataFrame:
    available = set(pd.read_csv(path, nrows=0).columns)
    return pd.read_csv(path, usecols=sorted(requested & available))


def _method_n1(identifier: str) -> int | None:
    match = _N1_RE.search(str(identifier))
    return int(match.group("n1")) if match is not None else None


def _first_available(
    frame: pd.DataFrame, candidates: tuple[str, ...]
) -> pd.Series:
    for candidate in candidates:
        if candidate in frame:
            values = pd.to_numeric(frame[candidate], errors="coerce")
            if bool(values.notna().any()):
                return values
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _paper_budget_used_per_sample(
    frame: pd.DataFrame,
    *,
    assigned: pd.Series,
    realized: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Return the paper's method-aware budget metric and its provenance.

    Static is charged its assigned censoring horizon ``sum(C_i) / n`` even
    when an event terminates generation earlier.  DAPRO is charged the actual
    event-stopped turns generated across policy fitting, CRC control, and
    deployment.  Uncalibrated and Oracle are deliberately left undefined
    because they are excluded from allocation-budget figures.
    """
    budget = pd.Series(np.nan, index=frame.index, dtype=float)
    semantics = pd.Series("not_applicable", index=frame.index, dtype=object)
    static = frame["method"].eq("Static")
    dapro = frame["method"].isin({"DAPRO", "DAPRO w/o CRC"})
    budget.loc[static] = assigned.loc[static]
    semantics.loc[static] = "assigned_sum_C_i_per_sample"
    budget.loc[dapro] = realized.loc[dapro]
    semantics.loc[dapro] = "actual_event_stopped_turns_per_sample"
    return budget, semantics


def _missing_method_rows(
    frame: pd.DataFrame,
    *,
    task: str,
    dataset_key: str,
    target_model: str,
    source_file: Path,
) -> list[dict]:
    present = set(frame["method"].dropna())
    return [
        {
            "task": task,
            "dataset_key": dataset_key,
            "target_model": target_model,
            "issue": "missing_method",
            "item": method,
            "source_file": str(source_file),
        }
        for method in METHOD_ORDER
        if method not in present
    ]


def load_bound_paper_data(
    input_dir: Path, task: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load LPB or UPB rows at the one recommended cell per dataset."""
    if task not in {"lpb", "upb"}:
        raise ValueError(f"Expected lpb or upb, got {task!r}.")
    recommendations = RECOMMENDED_CONFIGURATIONS[task]
    frames: list[pd.DataFrame] = []
    inventory: list[dict] = []
    gaps: list[dict] = []
    for metadata in _discover_sources(input_dir, task):
        recommendation = recommendations[metadata.dataset_key]
        frame = _read_available(metadata.path, BOUND_SOURCE_COLUMNS)
        source_columns = set(frame.columns)
        for column in BOUND_SOURCE_COLUMNS:
            if column not in frame:
                frame[column] = np.nan
        for column in BOUND_SOURCE_COLUMNS - {
            "calibration_name",
            "reported_budget_semantics",
        }:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame[np.isclose(
            frame["target_coverage"],
            recommendation.target_coverage,
            atol=5e-7,
        )].copy()
        policy_target = frame["policy_target_coverage"]
        frame = frame[
            policy_target.isna()
            | np.isclose(
                policy_target,
                recommendation.target_coverage,
                atol=5e-7,
            )
        ].copy()
        frame["method"] = frame["calibration_name"].map(
            bound_method_display_name
        )
        frame = frame[frame["method"].isin(METHOD_ORDER)].copy()
        n1 = frame["calibration_name"].map(_method_n1)
        dynamic = frame["method"].isin({"DAPRO", "DAPRO w/o CRC"})
        frame = frame[~dynamic | n1.eq(recommendation.n1)].copy()
        frame["dataset_key"] = metadata.dataset_key
        frame["dataset_display"] = DATASET_DISPLAY[metadata.dataset_key]
        frame["target_model"] = metadata.target_model_display
        frame["target_model_key"] = metadata.target_model
        frame["bound_type"] = task.upper()
        frame["source_file"] = str(metadata.path)
        frame["target_coverage_pct"] = 100 * recommendation.target_coverage
        frame["target_budget"] = recommendation.budget_per_sample
        frame["dapro_n1"] = recommendation.n1
        frame["crc_control_size"] = recommendation.crc_control_size
        frame["coverage_pct"] = 100 * frame["coverage"]
        frame["coverage_diff_pct"] = (
            frame["coverage_pct"] - frame["target_coverage_pct"]
        ).abs()
        calibration_size = frame["configured_cal_size"].fillna(3000)
        assigned_budget = _first_available(
            frame, ("reported_assigned_budget_per_sample",)
        ).where(
            lambda values: values.notna(),
            pd.to_numeric(frame["budget_used"], errors="coerce")
            / calibration_size,
        )
        verified_assigned = frame["reported_budget_semantics"].astype(str).eq(
            "sum_assigned_C_i"
        )
        assigned_budget = assigned_budget.where(verified_assigned)
        realized_budget = _first_available(
            frame, ("actual_event_stopped_budget_per_sample",)
        ).where(
            lambda values: values.notna(),
            pd.to_numeric(
                frame["actual_event_stopped_budget_total"], errors="coerce"
            ) / calibration_size,
        )
        (
            frame["budget_used_per_sample"],
            frame["budget_used_semantics"],
        ) = _paper_budget_used_per_sample(
            frame,
            assigned=assigned_budget,
            realized=realized_budget,
        )
        frame["mean_selected_a_weight"] = _first_available(
            frame,
            (
                "mean_calibrated_a_weighted_inverse_probability",
                "mean_a_weighted_inverse_probability",
            ),
        )
        frame["mean_prior_a_weight"] = _first_available(
            frame, ("mean_prior_a_weighted_inverse_probability",)
        )
        frame["mean_tau_0p10_a_weight"] = _first_available(
            frame, ("mean_tau_0p10_a_weighted_inverse_probability",)
        )
        duplicated = frame.duplicated(["seed", "method"], keep=False)
        if bool(duplicated.any()):
            duplicate_rows = frame.loc[
                duplicated, ["seed", "method", "calibration_name"]
            ]
            raise ValueError(
                f"Duplicate selected rows in {metadata.path}:\n"
                f"{duplicate_rows.to_string(index=False)}"
            )
        gaps.extend(_missing_method_rows(
            frame,
            task=task,
            dataset_key=metadata.dataset_key,
            target_model=metadata.target_model_display,
            source_file=metadata.path,
        ))
        if not bool(verified_assigned.any()):
            gaps.append({
                "task": task,
                "dataset_key": metadata.dataset_key,
                "target_model": metadata.target_model_display,
                "issue": "unverified_metric_semantics",
                "item": "assigned_budget_per_sample",
                "source_file": str(metadata.path),
            })
        dynamic_budget_missing = frame["method"].isin(
            {"DAPRO", "DAPRO w/o CRC"}
        ) & frame["budget_used_per_sample"].isna()
        if bool(dynamic_budget_missing.any()):
            gaps.append({
                "task": task,
                "dataset_key": metadata.dataset_key,
                "target_model": metadata.target_model_display,
                "issue": "missing_metric_value",
                "item": "actual_event_stopped_budget_per_sample",
                "source_file": str(metadata.path),
            })
        for requested, output in (
            ("mean_prior_a_weighted_inverse_probability", "mean_prior_a_weight"),
            ("mean_tau_0p10_a_weighted_inverse_probability", "mean_tau_0p10_a_weight"),
        ):
            if requested not in source_columns:
                gaps.append({
                    "task": task,
                    "dataset_key": metadata.dataset_key,
                    "target_model": metadata.target_model_display,
                    "issue": "missing_metric_column",
                    "item": output,
                    "source_file": str(metadata.path),
                })
        inventory.append({
            "task": task,
            "dataset_key": metadata.dataset_key,
            "target_model": metadata.target_model_display,
            "budget_per_sample": recommendation.budget_per_sample,
            "target_coverage": recommendation.target_coverage,
            "n1": recommendation.n1,
            "crc_control_size": recommendation.crc_control_size,
            "seed_count": frame["seed"].nunique(),
            "method_count": frame["method"].nunique(),
            "source_file": str(metadata.path),
        })
        frames.append(frame)
    return (
        pd.concat(frames, ignore_index=True),
        pd.DataFrame(inventory),
        pd.DataFrame(gaps),
    )


def _derive_restricted_mean(frame: pd.DataFrame, *, horizon: float) -> None:
    """Fill exact ordinary-HT RMST identities in legacy aggregate files."""
    direct = _first_available(
        frame, ("estimated_restricted_mean_time_to_event",)
    )
    legacy = (
        horizon
        - pd.to_numeric(frame["estimated_cjr"], errors="coerce") / 100
        * (
            horizon
            - pd.to_numeric(frame["estimated_rmttu"], errors="coerce")
        )
    )
    estimator_kind = frame["unsafe_event_rate_estimator_kind"].astype(str)
    ordinary = estimator_kind.eq("ordinary_ht") | estimator_kind.eq("nan")
    frame["estimated_restricted_mean"] = direct.where(
        direct.notna(), legacy.where(ordinary)
    )
    direct_truth = _first_available(
        frame, ("full_benchmark_restricted_mean_time_to_event",)
    )
    legacy_truth = (
        horizon
        - pd.to_numeric(frame["full_benchmark_cjr"], errors="coerce") / 100
        * (
            horizon
            - pd.to_numeric(frame["full_benchmark_rmttu"], errors="coerce")
        )
    )
    frame["full_benchmark_restricted_mean"] = direct_truth.where(
        direct_truth.notna(), legacy_truth
    )
    frame["restricted_mean_source"] = np.where(
        direct.notna(),
        "stored_standard_rmst",
        np.where(
            ordinary & legacy.notna(),
            "exact_ordinary_ht_identity_from_event_rate_and_event_time",
            "unavailable",
        ),
    )


def load_metric_paper_data(
    input_dir: Path, *, horizon: float = 200.0
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load metric-estimation rows at the recommended budget and N1."""
    recommendations = RECOMMENDED_CONFIGURATIONS["metrics"]
    frames: list[pd.DataFrame] = []
    inventory: list[dict] = []
    gaps: list[dict] = []
    for metadata in _discover_sources(input_dir, "metrics"):
        recommendation = recommendations[metadata.dataset_key]
        frame = _read_available(metadata.path, METRIC_SOURCE_COLUMNS)
        for column in METRIC_SOURCE_COLUMNS:
            if column not in frame:
                frame[column] = np.nan
        frame["method"] = frame["calibration_name"].map(
            metric_method_display_name
        )
        frame = frame[frame["method"].isin(METHOD_ORDER)].copy()
        configured_n1 = pd.to_numeric(
            frame["configured_dapro_n1"], errors="coerce"
        )
        identifier_n1 = frame["allocator_name"].map(_method_n1)
        effective_n1 = configured_n1.where(configured_n1.notna(), identifier_n1)
        dynamic = frame["method"].isin({"DAPRO", "DAPRO w/o CRC"})
        frame = frame[~dynamic | effective_n1.eq(recommendation.n1)].copy()
        frame["dataset_key"] = metadata.dataset_key
        frame["dataset_display"] = DATASET_DISPLAY[metadata.dataset_key]
        frame["target_model"] = metadata.target_model_display
        frame["target_model_key"] = metadata.target_model
        frame["source_file"] = str(metadata.path)
        frame["target_budget"] = recommendation.budget_per_sample
        frame["dapro_n1"] = recommendation.n1
        frame["crc_control_size"] = recommendation.crc_control_size
        frame["observed_events"] = _first_available(
            frame, ("num_events_observed", "observed_jailbreaks")
        )
        frame["mean_metric_a_weight"] = _first_available(
            frame,
            (
                "mean_metric_target_a_weighted_inverse_probability",
                "mean_metric_a_weighted_inverse_probability",
                "mean_a_weighted_weight",
            ),
        )
        verified_assigned = frame["reported_budget_semantics"].astype(str).eq(
            "sum_assigned_C_i"
        )
        assigned_budget = _first_available(
            frame,
            ("reported_assigned_budget_per_sample", "budget_per_sample"),
        ).where(verified_assigned)
        evaluation_size = pd.to_numeric(
            frame["evaluation_sample_size"], errors="coerce"
        ).fillna(3000.0)
        realized_budget = _first_available(
            frame, ("actual_event_stopped_budget_per_sample",)
        ).where(
            lambda values: values.notna(),
            pd.to_numeric(
                frame["actual_event_stopped_budget_total"], errors="coerce"
            ) / evaluation_size,
        )
        frame["assigned_budget_per_sample"] = assigned_budget
        (
            frame["budget_per_sample"],
            frame["budget_used_semantics"],
        ) = _paper_budget_used_per_sample(
            frame,
            assigned=assigned_budget,
            realized=realized_budget,
        )
        _derive_restricted_mean(frame, horizon=horizon)
        duplicated = frame.duplicated(["seed", "method"], keep=False)
        if bool(duplicated.any()):
            duplicate_rows = frame.loc[
                duplicated, ["seed", "method", "allocator_name"]
            ]
            raise ValueError(
                f"Duplicate selected rows in {metadata.path}:\n"
                f"{duplicate_rows.to_string(index=False)}"
            )
        gaps.extend(_missing_method_rows(
            frame,
            task="metrics",
            dataset_key=metadata.dataset_key,
            target_model=metadata.target_model_display,
            source_file=metadata.path,
        ))
        if not bool(verified_assigned.any()):
            gaps.append({
                "task": "metrics",
                "dataset_key": metadata.dataset_key,
                "target_model": metadata.target_model_display,
                "issue": "unverified_metric_semantics",
                "item": "assigned_budget_per_sample",
                "source_file": str(metadata.path),
            })
        dynamic_budget_missing = frame["method"].isin(
            {"DAPRO", "DAPRO w/o CRC"}
        ) & frame["budget_per_sample"].isna()
        if bool(dynamic_budget_missing.any()):
            gaps.append({
                "task": "metrics",
                "dataset_key": metadata.dataset_key,
                "target_model": metadata.target_model_display,
                "issue": "missing_metric_value",
                "item": "actual_event_stopped_budget_per_sample",
                "source_file": str(metadata.path),
            })
        inventory.append({
            "task": "metrics",
            "dataset_key": metadata.dataset_key,
            "target_model": metadata.target_model_display,
            "budget_per_sample": recommendation.budget_per_sample,
            "n1": recommendation.n1,
            "crc_control_size": recommendation.crc_control_size,
            "seed_count": frame["seed"].nunique(),
            "method_count": frame["method"].nunique(),
            "source_file": str(metadata.path),
        })
        frames.append(frame)
    return (
        pd.concat(frames, ignore_index=True),
        pd.DataFrame(inventory),
        pd.DataFrame(gaps),
    )
