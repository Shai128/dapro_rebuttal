"""Small paired validation of soft-prefix versus terminal UPB DAPRO.

This is an analysis-only experiment.  It deliberately instantiates the two
existing CRC-controlled UPB implementations on identical calibration/test
splits and identical acquisition uniforms; it does not change the paper
method registry.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.predictive_bounds.budget_allocators.DAPRO import (
    DefinitiveCRCUPBDAPRO,
    SoftPrefixEndpointCRCUPBDAPRO,
)
from src.predictive_bounds.calibration.survival_upb_calibration_with_known_weights import (
    SurvivalUPBCalibrationWithKnownWeights,
)
from src.predictive_bounds.construct_calibrated_bound import (
    _make_common_acquisition_uniforms,
)
from src.predictive_bounds.utils.utils import (
    make_upb_tau_grid,
    setup_experiment_data,
    split_data,
)
from src.train_model.models.utils import SurvivalModelPrediction
from src.utils.utils import set_seeds


DATASET = "dataset_toxicity"
SETUP = (
    "attack_toxic_attack_qwen25_14b_instruct_lm_target_"
    "qwen25_14b_instruct_judge_detoxify"
)
CAL_SIZE = 3000
BUDGET = 20.0
MAX_TIME = 200
TAU_PRIOR = 0.98
TARGET_COVERAGE = 0.80
N1 = 100
CRC_SIZE = 50
SEEDS = range(5)


def main() -> None:
    device = "cpu"
    taus = make_upb_tau_grid(device=device)
    (
        _max_time,
        all_times,
        all_quantiles,
        all_probabilities,
        all_conditional,
        test_size,
    ) = setup_experiment_data(
        CAL_SIZE,
        True,
        device,
        DATASET,
        SETUP,
        taus,
        MAX_TIME,
        bound_type="upb",
    )
    rows: list[dict] = []
    target = torch.tensor([TARGET_COVERAGE], dtype=torch.float32)

    for seed in SEEDS:
        (
            x_cal,
            x_test,
            t_cal,
            p_cal,
            q_cal,
            t_test,
            q_test,
            p_test,
            cal_idx,
            _test_idx,
        ) = split_data(
            seed,
            CAL_SIZE,
            test_size,
            None,
            all_times,
            all_probabilities,
            all_quantiles,
        )
        conditional = all_conditional[cal_idx]
        cal_prediction = SurvivalModelPrediction(q_cal, p_cal)
        test_prediction = SurvivalModelPrediction(q_test, p_test)
        uniforms = _make_common_acquisition_uniforms(
            seed,
            len(all_conditional),
            int(all_conditional.shape[1]),
            selected_indices=cal_idx,
        )

        allocators = [
            DefinitiveCRCUPBDAPRO(
                conditional,
                BUDGET,
                taus,
                TAU_PRIOR,
                MAX_TIME,
                n1=N1,
                budget_control_size=CRC_SIZE,
                row_cost_cap_multiplier=2.0,
                target_coverage=TARGET_COVERAGE,
                score_bin_count=2,
                global_regularization=0.001,
            ),
            SoftPrefixEndpointCRCUPBDAPRO(
                conditional,
                BUDGET,
                taus,
                TAU_PRIOR,
                MAX_TIME,
                n1=N1,
                budget_control_size=CRC_SIZE,
                row_cost_cap_multiplier=2.0,
                target_coverage=TARGET_COVERAGE,
                score_bin_count=2,
                global_regularization=0.001,
            ),
        ]
        labels = ("Hard terminal + CRC", "Soft prefix + CRC")
        for label, allocator in zip(labels, allocators):
            set_seeds(seed)
            allocator.set_acquisition_randomness(seed=seed, uniforms=uniforms)
            calibration = SurvivalUPBCalibrationWithKnownWeights(
                allocator, taus, TAU_PRIOR
            )
            calibration.calibrate(x_cal, t_cal, cal_prediction)
            with torch.no_grad():
                bound = calibration.get_calibrated_upb(
                    target, x_test, test_prediction
                ).reshape(-1)
                metrics = calibration.compute_metrics(cal_prediction, target)
            coverage = (t_test.reshape(-1) <= bound).float().mean().item()
            row = {
                "seed": seed,
                "method": label,
                "coverage": coverage,
                "coverage_difference": coverage - TARGET_COVERAGE,
                "size": bound.float().mean().item(),
            }
            for key in (
                "budget_used",
                "actual_event_stopped_budget_per_sample",
                "mean_calibrated_a_weighted_inverse_probability",
                "estimated_conditional_variance_upb_coverage_estimator",
                "mean_upb_exact_sequential_aht_path_variance",
                "n_observed_events",
            ):
                if key in metrics:
                    row[key] = metrics[key]
                elif f"{key}_0" in metrics:
                    row[key] = metrics[f"{key}_0"]
            rows.append(row)

    output = Path("analysis/results/soft_prefix_terminal_upb_validation.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    summary = frame.groupby("method").agg(
        coverage_mean=("coverage", "mean"),
        coverage_across_split_variance=("coverage", "var"),
        size_mean=("size", "mean"),
        size_across_split_variance=("size", "var"),
        budget_mean=("actual_event_stopped_budget_per_sample", "mean"),
        mean_target_weight=(
            "mean_calibrated_a_weighted_inverse_probability", "mean"
        ),
    )
    print(summary.to_string())
    print(f"\nWrote {output.resolve()}")


if __name__ == "__main__":
    main()
