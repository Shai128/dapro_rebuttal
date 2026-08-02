"""Record which calibration samples receive budget from each method.

Rows contain realized cost, terminal inclusion probability, target-event
membership, and several difficulty measures.  Quartiles are computed within a
split before methods are expanded, so every method is compared on identical
sample strata.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from src.predictive_bounds.calibration.calibration_utils import get_prior
from src.predictive_bounds.construct_calibrated_bound import (
    _make_common_acquisition_uniforms,
)
from src.predictive_bounds.experiments.common.bounds import (
    make_bound_grid,
    select_paper_calibrations,
)
from src.predictive_bounds.experiments.common.results import (
    stable_experiment_name,
    write_table_shard,
)
from src.predictive_bounds.experiments.full_bounds.config import (
    UNCALIBRATED,
    calibration_names,
)
from src.predictive_bounds.survival_utils.quantiles import (
    compute_conditional_quantiles_single_step,
)
from src.predictive_bounds.utils.utils import setup_experiment_data, split_data
from src.train_model.models.utils import SurvivalModelPrediction
from src.utils.utils import set_seeds


STRATIFIERS = (
    "estimated_bound",
    "estimated_median_t0",
    "true_time_to_event",
    "normalized_time_to_event",
    "estimated_median_t3",
)


def rank_quartiles(values: np.ndarray) -> np.ndarray:
    """Assign deterministic equal-count quartiles even when values are tied."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 4:
        raise ValueError("At least four one-dimensional values are required.")
    order = np.argsort(values, kind="stable")
    labels = np.empty(len(values), dtype=object)
    for quartile, indices in enumerate(np.array_split(order, 4), start=1):
        labels[indices] = f"Q{quartile}"
    return labels


def metadata(args) -> dict:
    return {
        "experiment_type": "allocation_focus",
        "dataset_name": args.dataset_name,
        "dataset_setup": args.dataset_setup,
        "bound_type": args.bound_type,
        "cal_size": args.cal_size,
        "budget_per_sample": args.budget_per_sample,
        "tau_prior": args.tau_prior,
        "target_coverage": args.target_coverage,
        "m_upper_bound": args.m_upper_bound,
        "future_time": args.future_time,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default="dataset_toxicity")
    parser.add_argument(
        "--dataset-setup",
        default=("attack_toxic_attack_qwen25_14b_instruct_lm_target_"
                 "qwen25_14b_instruct_judge_detoxify"),
    )
    parser.add_argument("--bound-type", choices=["lpb", "upb"], default="lpb")
    parser.add_argument("--cal-size", type=int, default=3000)
    parser.add_argument("--budget-per-sample", type=float, default=20.0)
    parser.add_argument("--tau-prior", type=float, default=0.56)
    parser.add_argument("--target-coverage", type=float, default=0.90)
    parser.add_argument("--m-upper-bound", type=float, default=200.0)
    parser.add_argument("--future-time", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def run_seed(args, seed, grid, data) -> pd.DataFrame:
    max_time, times, quantiles, probabilities, conditional_grid, test_size = data
    split = split_data(
        seed, args.cal_size, test_size, None, times, probabilities, quantiles
    )
    (_, _, t_cal, p_cal, q_cal, _, _, _, cal_idx, _) = split
    conditional_cal = conditional_grid[cal_idx]
    prediction = SurvivalModelPrediction(q_cal, p_cal)
    names = tuple(
        name for name in calibration_names(args.bound_type)
        if name != UNCALIBRATED
    )
    methods = select_paper_calibrations(
        conditional_cal, t_cal, q_cal, p_cal, grid,
        budget_per_sample=args.budget_per_sample,
        tau_prior=args.tau_prior,
        m_upper_bound=args.m_upper_bound,
        device=q_cal.device,
        calibration_names=names,
    )
    uniforms = _make_common_acquisition_uniforms(
        seed, len(t_cal), int(conditional_cal.shape[1])
    )
    tau_prior_quantile = get_prior(q_cal, grid.taus, args.tau_prior)
    median_index = int(torch.argmin(torch.abs(grid.taus - 0.5)).item())
    median_t0 = q_cal[:, median_index]
    raw_bound_tau = (
        1.0 - grid.target_coverage
        if args.bound_type == "lpb"
        else grid.target_coverage
    )
    raw_bound_index = int(torch.argmin(
        torch.abs(grid.taus - raw_bound_tau)
    ).item())
    estimated_bound = q_cal[:, raw_bound_index]
    future_index = min(max(args.future_time - 1, 0), conditional_cal.shape[1] - 1)
    median_t3 = compute_conditional_quantiles_single_step(
        conditional_cal[:, future_index], 0.5
    ) + 1
    base = {
        "sample_id": np.asarray(cal_idx, dtype=int),
        "true_time_to_event": t_cal.detach().cpu().numpy(),
        "estimated_median_t0": median_t0.detach().cpu().numpy(),
        "estimated_median_t3": median_t3.detach().cpu().numpy(),
        "normalized_time_to_event": (
            t_cal.reshape(-1).to(torch.float64)
            / tau_prior_quantile.reshape(-1).to(torch.float64).clamp_min(1)
        ).detach().cpu().numpy(),
        "estimated_prior_lpb": tau_prior_quantile.detach().cpu().numpy(),
        "estimated_bound": estimated_bound.detach().cpu().numpy(),
    }
    rows = []
    target_tensor = torch.tensor([grid.target_coverage], device=q_cal.device)
    # LPB APIs take miscoverage alpha, while UPB APIs take target coverage.
    api_target = 1.0 - target_tensor if args.bound_type == "lpb" else target_tensor
    for method in methods:
        set_seeds(seed)
        if hasattr(method, "budget_allocator"):
            method.budget_allocator.set_acquisition_randomness(
                seed=seed, uniforms=uniforms
            )
        method.calibrate(None, t_cal, prediction)
        with torch.no_grad():
            if args.bound_type == "lpb":
                selected_bound = method.get_calibrated_lpb(
                    api_target, None, prediction
                ).reshape(-1)
            else:
                selected_bound = method.get_calibrated_upb(
                    api_target, None, prediction
                ).reshape(-1)
        allocation = method.allocation_result
        C = allocation.C.reshape(-1).to(t_cal.device)
        pi = allocation.C_probs.reshape(-1).to(torch.float64)
        realized_cost = torch.minimum(
            t_cal.reshape(-1).to(torch.float64), C.to(torch.float64)
        ).clamp(max=max_time)
        if args.bound_type == "lpb":
            target_event = t_cal.reshape(-1) < selected_bound
        else:
            target_event = (
                (t_cal.reshape(-1) > selected_bound)
                & (selected_bound != args.m_upper_bound)
            )
        method_frame = pd.DataFrame({
            **base,
            "seed": seed,
            "method": method.name,
            "calibrated_bound": selected_bound.detach().cpu().numpy(),
            "realized_budget": realized_cost.detach().cpu().numpy(),
            "terminal_inclusion_probability": pi.detach().cpu().numpy(),
            "inverse_probability": (1 / pi).detach().cpu().numpy(),
            "target_event": target_event.detach().cpu().numpy().astype(int),
            "target_event_observed": (
                target_event & (selected_bound <= C)
            ).detach().cpu().numpy().astype(int),
        })
        for stratifier in STRATIFIERS:
            method_frame[f"{stratifier}_quartile"] = rank_quartiles(
                method_frame[stratifier].to_numpy()
            )
        rows.append(method_frame)
    return pd.concat(rows, ignore_index=True)


def main(argv=None):
    args = parse_args(argv)
    device = args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu"
    grid = make_bound_grid(args.bound_type, device, target_coverage=args.target_coverage)
    data = setup_experiment_data(
        args.cal_size, True, device, args.dataset_name, args.dataset_setup,
        grid.taus, args.m_upper_bound,
    )
    current_metadata = metadata(args)
    name = stable_experiment_name("allocation_focus", current_metadata)
    for seed in range(args.seed_start, args.seed_end):
        frame = run_seed(args, seed, grid, data)
        write_table_shard(
            "allocation_focus", name, seed, frame, current_metadata
        )
    print(name)


if __name__ == "__main__":
    main()
