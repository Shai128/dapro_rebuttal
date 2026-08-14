"""Paired UPB comparison of block and genuinely history-adaptive DAPRO.

This diagnostic deliberately stays outside the production registry until the
comparison identifies a winner.  It uses identical calibration/test splits and
common acquisition uniforms for every adaptive allocator.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.predictive_bounds.budget_allocators.DAPRO import (
    InformationGainCRCUPBDAPRO,
    InformationGainUPBDAPRO,
    SoftPrefixEndpointCRCUPBDAPRO,
    SoftPrefixEndpointUPBDAPRO,
    SoftTargetCRCUPBDAPRO,
    SoftTargetUPBDAPRO,
)
from src.predictive_bounds.budget_allocators.optimized_allocators import (
    OptimizedBudgetAllocator,
)
from src.predictive_bounds.calibration.oracle_survival_calibration import (
    OracleSurvivalUPBCalibration,
)
from src.predictive_bounds.calibration.survival_upb_calibration_with_known_weights import (
    SurvivalUPBCalibrationWithKnownWeights,
)
from src.predictive_bounds.construct_calibrated_bound import (
    _make_common_acquisition_uniforms,
    compute_metrics_bound,
)
from src.predictive_bounds.utils.utils import (
    make_upb_tau_grid,
    setup_experiment_data,
    split_data,
)
from src.train_model.models.utils import SurvivalModelPrediction


DATASETS = {
    "toxicity": (
        "dataset_toxicity",
        "attack_toxic_attack_qwen25_14b_instruct_lm_target_"
        "qwen25_14b_instruct_judge_detoxify",
    ),
    "red_qwen": (
        "dataset_red_team",
        "attack_default_attack_qwen25_14b_instruct_lm_target_"
        "qwen25_14b_instruct_judge_llm-judge_qwen25_14b_instruct",
    ),
}

TARGETS = (0.60, 0.70, 0.80, 0.90)


def _methods(grid, taus, *, budget: float, n1: int, target: float):
    common = (grid, budget, taus, 0.97, 200)
    return [
        (
            "Static",
            SurvivalUPBCalibrationWithKnownWeights(
                OptimizedBudgetAllocator(budget, taus, 0.97, 200),
                taus,
                0.97,
            ),
        ),
        ("Full-budget oracle", OracleSurvivalUPBCalibration(taus, 0.97)),
        (
            "Endpoint block",
            SurvivalUPBCalibrationWithKnownWeights(
                SoftTargetUPBDAPRO(
                    *common,
                    n1=n1,
                    target_coverage=target,
                    projection_budget_margin=1.0,
                ),
                taus,
                0.97,
            ),
        ),
        (
            "Endpoint block + CRC",
            SurvivalUPBCalibrationWithKnownWeights(
                SoftTargetCRCUPBDAPRO(
                    *common,
                    n1=n1,
                    budget_control_size=n1 // 2,
                    target_coverage=target,
                ),
                taus,
                0.97,
            ),
        ),
        (
            "Dynamic endpoint mass",
            SurvivalUPBCalibrationWithKnownWeights(
                SoftPrefixEndpointUPBDAPRO(
                    *common,
                    n1=n1,
                    target_coverage=target,
                    projection_budget_margin=1.0,
                ),
                taus,
                0.97,
            ),
        ),
        (
            "Dynamic endpoint mass + CRC",
            SurvivalUPBCalibrationWithKnownWeights(
                SoftPrefixEndpointCRCUPBDAPRO(
                    *common,
                    n1=n1,
                    budget_control_size=n1 // 2,
                    target_coverage=target,
                ),
                taus,
                0.97,
            ),
        ),
        (
            "Dynamic information gain",
            SurvivalUPBCalibrationWithKnownWeights(
                InformationGainUPBDAPRO(
                    *common,
                    n1=n1,
                    target_coverage=target,
                    projection_budget_margin=1.0,
                ),
                taus,
                0.97,
            ),
        ),
        (
            "Dynamic information gain + CRC",
            SurvivalUPBCalibrationWithKnownWeights(
                InformationGainCRCUPBDAPRO(
                    *common,
                    n1=n1,
                    budget_control_size=n1 // 2,
                    target_coverage=target,
                ),
                taus,
                0.97,
            ),
        ),
    ]


def _metric(metrics: dict, name: str, index: int):
    return metrics.get(f"{name}_{index}", metrics.get(name, np.nan))


def run_dataset(
        dataset_key: str,
        *,
        seed_start: int,
        seed_end: int,
        n1: int,
        output: Path,
        selected_methods: set[str] | None = None,
):
    dataset_name, setup = DATASETS[dataset_key]
    taus = make_upb_tau_grid(device="cpu")
    _, all_times, all_quantiles, all_probabilities, all_grid, test_size = (
        setup_experiment_data(
            3000,
            True,
            "cpu",
            dataset_name,
            setup,
            taus,
            200,
            bound_type="upb",
        )
    )
    rows = []
    if output.exists():
        rows = pd.read_csv(output).to_dict("records")
    completed = {
        (str(row["dataset"]), int(row["seed"]), str(row["method"]))
        for row in rows
    }
    target_tensor = torch.tensor(TARGETS, dtype=torch.float32)
    for seed in range(seed_start, seed_end):
        (
            _,
            _,
            cal_times,
            cal_probability,
            cal_quantiles,
            test_times,
            test_quantiles,
            test_probability,
            cal_indices,
            _,
        ) = split_data(
            seed,
            3000,
            test_size,
            None,
            all_times,
            all_probabilities,
            all_quantiles,
        )
        cal_grid = all_grid[cal_indices]
        cal_prediction = SurvivalModelPrediction(cal_quantiles, cal_probability)
        test_prediction = SurvivalModelPrediction(test_quantiles, test_probability)
        common_uniforms = torch.as_tensor(
            _make_common_acquisition_uniforms(
                seed,
                len(all_grid),
                all_grid.shape[1],
                selected_indices=cal_indices,
            ).copy(),
            dtype=torch.float64,
        )
        for label, calibration in _methods(
                cal_grid, taus, budget=20.0, n1=n1, target=0.70
        ):
            if selected_methods is not None and label not in selected_methods:
                continue
            identity = (dataset_key, seed, label)
            if identity in completed:
                continue
            np.random.seed(seed)
            torch.manual_seed(seed)
            allocator = getattr(calibration, "budget_allocator", None)
            if allocator is not None:
                allocator.set_acquisition_randomness(
                    seed=seed, uniforms=common_uniforms
                )
            started = time.perf_counter()
            calibration.calibrate(None, cal_times, cal_prediction)
            bounds = calibration.get_calibrated_upb(
                target_tensor, None, test_prediction
            )
            coverage, size = compute_metrics_bound(bounds, test_times, "upb")
            metrics = calibration.compute_metrics(cal_prediction, target_tensor)
            elapsed = time.perf_counter() - started
            allocation = calibration.allocation_result
            extra = allocation.additional_metrics or {}
            for index, target in enumerate(TARGETS):
                rows.append({
                    "dataset": dataset_key,
                    "seed": seed,
                    "method": label,
                    "allocator_name": calibration.name,
                    "n1": n1,
                    "target_coverage": target,
                    "coverage_pct": 100.0 * float(coverage[index]),
                    "mean_upb": float(size[index]),
                    "infinite_upb_rate": float(
                        (bounds[:, index] == 201).to(torch.float64).mean()
                    ),
                    "estimated_conditional_variance_pp2": float(_metric(
                        metrics,
                        "estimated_conditional_variance_upb_coverage_estimator",
                        index,
                    )),
                    "mean_exact_path_variance": float(_metric(
                        metrics,
                        "mean_upb_exact_sequential_aht_path_variance",
                        index,
                    )),
                    "expected_budget_per_sample": float(
                        extra.get("total_expected_budget_per_sample", np.nan)
                    ),
                    "realized_budget_per_sample": float(
                        allocation.total_budget_used / len(cal_times)
                    ),
                    "expected_budget_valid": float(
                        extra.get("total_expected_budget_valid", np.nan)
                    ),
                    "crc_selector_valid": float(
                        extra.get("risk_budget_selector_valid", np.nan)
                    ),
                    "min_terminal_pi": float(allocation.C_probs.min()),
                    "max_weight": float(allocation.C_probs.reciprocal().max()),
                    "runtime_seconds": elapsed,
                    "sequential_prefix_updates": int(
                        extra.get("upb_sequential_prefix_updates", 0)
                    ),
                })
            completed.add(identity)
            output.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(output, index=False)
            print(
                f"{dataset_key} seed={seed} {label}: "
                f"coverage70={100*float(coverage[1]):.3f}, "
                f"condvar70={rows[-3]['estimated_conditional_variance_pp2']:.3f}, "
                f"budget={extra.get('total_expected_budget_per_sample', np.nan):.3f}, "
                f"seconds={elapsed:.2f}",
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=10)
    parser.add_argument("--n1", type=int, default=50)
    parser.add_argument(
        "--method",
        action="append",
        choices=[
            "Static",
            "Full-budget oracle",
            "Endpoint block",
            "Endpoint block + CRC",
            "Dynamic endpoint mass",
            "Dynamic endpoint mass + CRC",
            "Dynamic information gain",
            "Dynamic information gain + CRC",
        ],
        help="Repeat to run only selected methods.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/upb_dynamic_comparison.csv")
    )
    args = parser.parse_args()
    run_dataset(
        args.dataset,
        seed_start=args.seed_start,
        seed_end=args.seed_end,
        n1=args.n1,
        output=args.output,
        selected_methods=None if not args.method else set(args.method),
    )


if __name__ == "__main__":
    main()
