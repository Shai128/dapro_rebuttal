"""Audit the proposed realized-residual CRC accounting on saved DAPRO runs.

The production CRC candidate family is affine in a scale ``s``:

    rho_s = epsilon + s (rho_base - epsilon),  s in {1, ..., 0}.

Consequently, the control-fold and Phase-II expected costs at every candidate
can be reconstructed exactly from a saved selected scale, its expected cost,
and the full-budget trajectory lengths.  This script changes only the
selector to the proposed rule

    (sum_i c_i(s) + C_max) / (n + 1)
        <= (B_remaining - sum_i b_i) / m,

where ``B_remaining`` is measured after the policy-fit fold.  It does not
refit the DAPRO shape using the realized control-fold costs.

The output is a diagnostic, not a validity certificate: the proposed rule is
not standard conformal risk control because its right-hand side depends on the
same control outcomes used to select ``s``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


TASK_SPECS = {
    "lpb": {
        "directory": "merged_calibration_dfs",
        "name_column": "calibration_name",
        "oracle_name": "oracle_survival_calibration",
    },
    "upb": {
        "directory": "merged_upb_calibration_dfs",
        "name_column": "calibration_name",
        "oracle_name": "oracle_survival_upb_calibration",
    },
    "metric": {
        "directory": "merged_metric_calibration_dfs",
        "name_column": "allocator_name",
        "oracle_name": "oracle_split_full_budget",
    },
}


def _number(row: pd.Series, name: str) -> float:
    value = pd.to_numeric(pd.Series([row.get(name, np.nan)]), errors="coerce").iloc[0]
    return float(value)


def _full_cost(oracle: pd.Series) -> float:
    for name in (
        "total_expected_budget",
        "actual_event_stopped_budget_total",
        "budget_used",
        "total_budget_utilized",
    ):
        value = _number(oracle, name)
        if math.isfinite(value) and value > 0:
            return value
    per_sample = _number(oracle, "total_expected_budget_per_sample")
    sample_count = _number(oracle, "evaluation_sample_size")
    if math.isfinite(per_sample) and math.isfinite(sample_count):
        return per_sample * sample_count
    raise ValueError("Could not recover the split full-budget trajectory cost.")


def _configured_budget(row: pd.Series, sample_count: float) -> float:
    configured_total = _number(row, "configured_total_budget")
    if math.isfinite(configured_total) and configured_total > 0:
        return configured_total
    configured_per_sample = _number(row, "configured_budget_per_sample")
    if not math.isfinite(configured_per_sample):
        raise ValueError("Missing configured budget.")
    return configured_per_sample * sample_count


def _grid_floor(value: float, candidate_count: int) -> float:
    """Largest member of linspace(1, 0, K) no greater than ``value``."""
    if value >= 1:
        return 1.0
    if value <= 0:
        return 0.0
    steps = candidate_count - 1
    return math.floor(value * steps + 1e-12) / steps


def reconstruct_row(
    row: pd.Series,
    oracle: pd.Series,
    *,
    task: str,
    source_file: Path,
) -> dict[str, object]:
    fit_count = _number(row, "risk_budget_policy_fit_size")
    control_count = _number(row, "risk_budget_control_size")
    deployment_count = _number(row, "phase2_sample_count")
    sample_count = fit_count + control_count + deployment_count
    fit_cost = _number(row, "risk_budget_policy_fit_realized_cost_total")
    pilot_cost = _number(row, "risk_budget_control_pilot_cost_total")
    current_phase2_cost = _number(row, "phase2_expected_cost_total")
    current_control_mean = _number(
        row, "risk_budget_empirical_control_cost_per_sample"
    )
    current_scale = _number(row, "risk_budget_selected_mixture_parameter")
    epsilon = _number(row, "terminal_pi_min")
    candidate_bound = _number(
        row, "risk_budget_maximum_candidate_cost_per_sample"
    )
    pilot_bound = _number(row, "risk_budget_maximum_pilot_cost_per_sample")
    candidate_count = int(round(_number(row, "risk_budget_candidate_count")))
    full_cost = _full_cost(oracle)
    budget_total = _configured_budget(row, sample_count)

    required = {
        "fit_count": fit_count,
        "control_count": control_count,
        "deployment_count": deployment_count,
        "fit_cost": fit_cost,
        "pilot_cost": pilot_cost,
        "current_phase2_cost": current_phase2_cost,
        "current_control_mean": current_control_mean,
        "current_scale": current_scale,
        "epsilon": epsilon,
        "candidate_bound": candidate_bound,
        "pilot_bound": pilot_bound,
        "full_cost": full_cost,
        "budget_total": budget_total,
    }
    missing = [key for key, value in required.items() if not math.isfinite(value)]
    if missing:
        raise ValueError(f"Missing values: {missing}")
    if control_count <= 0 or deployment_count <= 0 or candidate_count < 2:
        raise ValueError("Invalid fold or candidate count.")
    if current_scale <= 0:
        raise ValueError("Cannot reconstruct a more aggressive base from scale zero.")

    pilot_mean = pilot_cost / control_count
    floor_control_mean = epsilon * pilot_mean
    base_control_mean = floor_control_mean + (
        current_control_mean - floor_control_mean
    ) / current_scale

    budget_after_fit = budget_total - fit_cost
    control_to_deployment = control_count / deployment_count
    current_target = budget_after_fit / deployment_count
    current_envelope = candidate_bound + control_to_deployment * pilot_bound
    current_allowed_control_mean = (
        (control_count + 1) * current_target
        - control_to_deployment * pilot_cost
        - current_envelope
    ) / control_count
    if base_control_mean <= floor_control_mean + 1e-15:
        reconstructed_current_continuous_scale = (
            1.0 if current_allowed_control_mean >= base_control_mean else 0.0
        )
    else:
        reconstructed_current_continuous_scale = (
            current_allowed_control_mean - floor_control_mean
        ) / (base_control_mean - floor_control_mean)
    reconstructed_current_scale = _grid_floor(
        reconstructed_current_continuous_scale, candidate_count
    )
    realized_residual_target = (budget_after_fit - pilot_cost) / deployment_count
    allowed_control_mean = (
        (control_count + 1) * realized_residual_target - candidate_bound
    ) / control_count
    if base_control_mean <= floor_control_mean + 1e-15:
        continuous_scale = 1.0 if allowed_control_mean >= base_control_mean else 0.0
    else:
        continuous_scale = (
            allowed_control_mean - floor_control_mean
        ) / (base_control_mean - floor_control_mean)
    proposed_scale = _grid_floor(continuous_scale, candidate_count)
    proposed_control_mean = floor_control_mean + proposed_scale * (
        base_control_mean - floor_control_mean
    )
    proposed_selector_lhs = (
        control_count * proposed_control_mean + candidate_bound
    ) / (control_count + 1)
    selector_feasible = proposed_selector_lhs <= realized_residual_target + 1e-10

    latent_phase2_cost = full_cost - fit_cost - pilot_cost
    floor_phase2_cost = epsilon * latent_phase2_cost
    base_phase2_cost = floor_phase2_cost + (
        current_phase2_cost - floor_phase2_cost
    ) / current_scale
    proposed_phase2_cost = floor_phase2_cost + proposed_scale * (
        base_phase2_cost - floor_phase2_cost
    )
    proposed_total_cost = fit_cost + pilot_cost + proposed_phase2_cost
    current_total_cost = fit_cost + pilot_cost + current_phase2_cost

    def residual_selection(correction: float) -> tuple[float, float, bool, float]:
        allowed = (
            (control_count + 1) * realized_residual_target - correction
        ) / control_count
        if base_control_mean <= floor_control_mean + 1e-15:
            continuous = 1.0 if allowed >= base_control_mean else 0.0
        else:
            continuous = (
                allowed - floor_control_mean
            ) / (base_control_mean - floor_control_mean)
        scale = _grid_floor(continuous, candidate_count)
        control_mean = floor_control_mean + scale * (
            base_control_mean - floor_control_mean
        )
        lhs = (
            control_count * control_mean + correction
        ) / (control_count + 1)
        feasible = lhs <= realized_residual_target + 1e-10
        phase2 = floor_phase2_cost + scale * (
            base_phase2_cost - floor_phase2_cost
        )
        total = fit_cost + pilot_cost + phase2
        return scale, lhs, feasible, total

    # This is the exact selector in the attached proof: the transformed loss
    # is assumed bounded by the full horizon M.
    (
        proof_scale,
        proof_selector_lhs,
        proof_selector_feasible,
        proof_total_cost,
    ) = residual_selection(pilot_bound)

    # A generic deterministic transformed-loss envelope follows from
    # c<=C_max, b<=M and mu_b>=0.  It proves validity without separately
    # assuming K<=M, but is usually still more conservative than production.
    transformed_rho = (control_count + 1) / deployment_count
    transformed_bound = candidate_bound + transformed_rho * pilot_bound
    (
        transformed_scale,
        transformed_selector_lhs,
        transformed_selector_feasible,
        transformed_total_cost,
    ) = residual_selection(transformed_bound)

    name_column = TASK_SPECS[task]["name_column"]
    return {
        "task": task,
        "source_file": str(source_file),
        "experiment": source_file.parent.name,
        "seed": int(round(_number(row, "seed"))),
        "method": str(row[name_column]),
        "configured_budget_per_sample": budget_total / sample_count,
        "sample_count": sample_count,
        "policy_fit_count": fit_count,
        "control_count": control_count,
        "deployment_count": deployment_count,
        "pilot_mean_cost": pilot_mean,
        "candidate_bound": candidate_bound,
        "current_scale": current_scale,
        "reconstructed_current_scale": reconstructed_current_scale,
        "current_scale_reconstruction_error": (
            reconstructed_current_scale - current_scale
        ),
        "proposed_scale": proposed_scale,
        "scale_increase": proposed_scale - current_scale,
        "realized_residual_target_per_deployment": realized_residual_target,
        "proposed_selector_lhs": proposed_selector_lhs,
        "proposed_selector_feasible": int(selector_feasible),
        "current_expected_budget_per_sample": current_total_cost / sample_count,
        "proposed_expected_budget_per_sample": proposed_total_cost / sample_count,
        "proposed_expected_budget_gap_per_sample": (
            proposed_total_cost - budget_total
        ) / sample_count,
        "proposed_expected_budget_valid_on_split": int(
            proposed_total_cost <= budget_total + 1e-8
        ),
        "proof_envelope_automatic_from_declared_bounds": int(
            transformed_bound <= pilot_bound + 1e-12
        ),
        "proof_scale": proof_scale,
        "proof_selector_lhs": proof_selector_lhs,
        "proof_selector_feasible": int(proof_selector_feasible),
        "proof_expected_budget_per_sample": proof_total_cost / sample_count,
        "proof_expected_budget_gap_per_sample": (
            proof_total_cost - budget_total
        ) / sample_count,
        "proof_expected_budget_valid_on_split": int(
            proof_total_cost <= budget_total + 1e-8
        ),
        "transformed_bound": transformed_bound,
        "transformed_scale": transformed_scale,
        "transformed_selector_lhs": transformed_selector_lhs,
        "transformed_selector_feasible": int(transformed_selector_feasible),
        "transformed_expected_budget_per_sample": (
            transformed_total_cost / sample_count
        ),
        "transformed_expected_budget_gap_per_sample": (
            transformed_total_cost - budget_total
        ) / sample_count,
        "transformed_expected_budget_valid_on_split": int(
            transformed_total_cost <= budget_total + 1e-8
        ),
    }


def audit_task(results_root: Path, task: str) -> tuple[list[dict], list[dict]]:
    spec = TASK_SPECS[task]
    directory = results_root / spec["directory"]
    records: list[dict] = []
    errors: list[dict] = []
    for source_file in sorted(directory.rglob("all_df.csv")):
        try:
            frame = pd.read_csv(source_file, low_memory=False)
        except Exception as exc:  # pragma: no cover - diagnostic robustness
            errors.append({"task": task, "source_file": str(source_file), "error": str(exc)})
            continue
        name_column = spec["name_column"]
        if name_column not in frame or "seed" not in frame:
            continue
        names = frame[name_column].fillna("").astype(str)
        mask = (
            names.str.contains("dapro_soft_prefix_bins_2", regex=False)
            & names.str.contains("budget_crc", regex=False)
        )
        # LPB/metric production runs use the corrected causal shared-envelope
        # policy.  The current UPB pipeline intentionally uses its uncapped
        # endpoint-dynamic family, whose candidate support is the full horizon.
        if task != "upb":
            mask &= names.str.contains("causal_shared_pav", regex=False)
        dapro = frame.loc[mask].drop_duplicates(["seed", name_column])
        oracle = frame.loc[names.eq(spec["oracle_name"])].drop_duplicates("seed")
        oracle_by_seed = {int(row.seed): row for _, row in oracle.iterrows()}
        for _, row in dapro.iterrows():
            seed = int(round(_number(row, "seed")))
            if seed not in oracle_by_seed:
                errors.append(
                    {
                        "task": task,
                        "source_file": str(source_file),
                        "seed": seed,
                        "error": "Missing split full-budget oracle row.",
                    }
                )
                continue
            try:
                records.append(
                    reconstruct_row(
                        row,
                        oracle_by_seed[seed],
                        task=task,
                        source_file=source_file,
                    )
                )
            except Exception as exc:
                errors.append(
                    {
                        "task": task,
                        "source_file": str(source_file),
                        "seed": seed,
                        "method": row[name_column],
                        "error": str(exc),
                    }
                )
    return records, errors


def summarize(records: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame()
    keys = [
        "task",
        "experiment",
        "method",
        "configured_budget_per_sample",
        "policy_fit_count",
        "control_count",
    ]
    return (
        records.groupby(keys, dropna=False)
        .agg(
            split_count=("seed", "size"),
            mean_current_budget=("current_expected_budget_per_sample", "mean"),
            mean_proposed_budget=("proposed_expected_budget_per_sample", "mean"),
            maximum_proposed_budget=("proposed_expected_budget_per_sample", "max"),
            split_valid_rate=("proposed_expected_budget_valid_on_split", "mean"),
            mean_scale_increase=("scale_increase", "mean"),
            maximum_scale_increase=("scale_increase", "max"),
            selector_feasible_rate=("proposed_selector_feasible", "mean"),
            proof_selector_feasible_rate=("proof_selector_feasible", "mean"),
            proof_envelope_rate=(
                "proof_envelope_automatic_from_declared_bounds", "mean"
            ),
            mean_proof_budget=("proof_expected_budget_per_sample", "mean"),
            maximum_proof_budget=("proof_expected_budget_per_sample", "max"),
            proof_split_valid_rate=(
                "proof_expected_budget_valid_on_split", "mean"
            ),
            transformed_selector_feasible_rate=(
                "transformed_selector_feasible", "mean"
            ),
            mean_transformed_budget=(
                "transformed_expected_budget_per_sample", "mean"
            ),
            transformed_split_valid_rate=(
                "transformed_expected_budget_valid_on_split", "mean"
            ),
        )
        .reset_index()
        .assign(
            marginal_mean_valid=lambda data: (
                data["mean_proposed_budget"]
                <= data["configured_budget_per_sample"] + 1e-10
            ).astype(int),
            proof_marginal_mean_valid=lambda data: (
                data["mean_proof_budget"]
                <= data["configured_budget_per_sample"] + 1e-10
            ).astype(int),
            transformed_marginal_mean_valid=lambda data: (
                data["mean_transformed_budget"]
                <= data["configured_budget_per_sample"] + 1e-10
            ).astype(int),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/realized_residual_crc_audit"),
    )
    args = parser.parse_args()

    all_records: list[dict] = []
    all_errors: list[dict] = []
    for task in TASK_SPECS:
        records, errors = audit_task(args.results_root, task)
        all_records.extend(records)
        all_errors.extend(errors)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    records_frame = pd.DataFrame(all_records)
    summary_frame = summarize(records_frame)
    records_frame.to_csv(output_dir / "split_level.csv", index=False)
    summary_frame.to_csv(output_dir / "configuration_summary.csv", index=False)
    pd.DataFrame(all_errors).to_csv(output_dir / "errors.csv", index=False)

    if records_frame.empty:
        print("No eligible saved DAPRO CRC rows were found.")
        return
    print(
        records_frame.groupby("task")
        .agg(
            splits=("seed", "size"),
            split_valid_rate=("proposed_expected_budget_valid_on_split", "mean"),
            mean_scale_increase=("scale_increase", "mean"),
            mean_budget=("proposed_expected_budget_per_sample", "mean"),
            mean_target=("configured_budget_per_sample", "mean"),
            proof_feasible=("proof_selector_feasible", "mean"),
            mean_proof_budget=("proof_expected_budget_per_sample", "mean"),
            transformed_feasible=("transformed_selector_feasible", "mean"),
            mean_transformed_budget=(
                "transformed_expected_budget_per_sample", "mean"
            ),
        )
        .to_string()
    )
    print("\nConfiguration-level marginal failures:")
    failures = summary_frame.loc[summary_frame["marginal_mean_valid"].eq(0)]
    print(failures.to_string(index=False) if len(failures) else "none")
    print(f"\nWrote {output_dir}")


if __name__ == "__main__":
    main()
