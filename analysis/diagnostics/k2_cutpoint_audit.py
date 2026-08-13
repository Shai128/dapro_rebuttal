"""Offline study of the quantile cutpoint used by hazard/K2 DAPRO.

This file deliberately does not alter the production allocator.  It reproduces
the soft-prefix Generalized-DAPRO objective and cumulative budget correction,
but replaces the fixed median edge by an arbitrary empirical quantile.  Both
metric estimation and LPB construction are evaluated on common outer splits.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analysis.diagnostics.dapro_binning_audit import SETUPS, load_setup
from analysis.diagnostics.lpb_dapro_binning_audit import (
    lpb_quantiles,
    strict_select,
)
from src.predictive_bounds.budget_allocators.optimization_solver_utils import (
    solve_exact_fast,
)
from src.predictive_bounds.budget_allocators.projected_optimization_utils import (
    correct_projected_cumulative_probabilities_to_budget,
)
from src.predictive_bounds.calibration.calibration_utils import get_prior
from src.predictive_bounds.utils.utils import make_lpb_tau_grid


@dataclass
class Policy:
    fit_q: np.ndarray
    deploy_q: np.ndarray
    fit_rho: np.ndarray
    deploy_rho: np.ndarray
    cutpoints: list[float | None]
    fit_low_share: float
    deploy_low_share: float
    empty_low_cells: int
    empty_high_cells: int


def solve_quantile_k2(
    *,
    fit_scores: np.ndarray,
    deploy_scores: np.ndarray,
    fit_lengths: np.ndarray,
    deploy_lengths: np.ndarray,
    fit_prior: np.ndarray,
    deploy_prior: np.ndarray,
    target_budget: float,
    objective_masses: np.ndarray,
    cut_quantile: float,
    terminal_pi_min: float = 0.005,
) -> tuple[Policy, dict[str, float | int | str | None]]:
    """Fit the production direct-bin policy with a non-median K2 edge."""
    if not 0 < cut_quantile < 1:
        raise ValueError("The cut quantile must lie strictly between 0 and 1.")
    n_fit, width = fit_scores.shape
    fit_bins = np.zeros((n_fit, width), dtype=np.int64)
    deploy_bins = np.zeros((len(deploy_scores), width), dtype=np.int64)
    cutpoints: list[float | None] = []
    active_fit = np.arange(width)[None, :] < fit_lengths[:, None]
    active_deploy = np.arange(width)[None, :] < deploy_lengths[:, None]
    for step in range(width):
        active = active_fit[:, step]
        if not np.any(active):
            cutpoints.append(None)
            continue
        edge = float(np.quantile(fit_scores[active, step], cut_quantile))
        cutpoints.append(edge)
        # Match production: observations equal to the cutpoint enter the high
        # bin.  This matters when the hazard score has an atom at zero.
        fit_bins[:, step] = np.searchsorted([edge], fit_scores[:, step], side="right")
        deploy_bins[:, step] = np.searchsorted(
            [edge], deploy_scores[:, step], side="right"
        )

    optimal = solve_exact_fast(
        torch.as_tensor(fit_bins, dtype=torch.float64),
        torch.as_tensor(fit_lengths),
        target_budget,
        objective_masses=torch.as_tensor(objective_masses, dtype=torch.float64),
        terminal_pi_min=None,
        verbose=False,
    )
    optimal[~active_fit] = 1.0
    table = np.ones((2, width), dtype=np.float64)
    empty_low = 0
    empty_high = 0
    for step in range(width):
        active = active_fit[:, step]
        if not np.any(active):
            continue
        observed = []
        for bin_index in range(2):
            rows = active & (fit_bins[:, step] == bin_index)
            if np.any(rows):
                table[bin_index, step] = float(optimal[rows, step].mean())
                observed.append(bin_index)
            elif bin_index == 0:
                empty_low += 1
            else:
                empty_high += 1
        if len(observed) == 1:
            table[1 - observed[0], step] = table[observed[0], step]
        table[:, step] = np.maximum.accumulate(table[:, step])

    fit_raw_q = table[fit_bins, np.arange(width)[None, :]]
    deploy_raw_q = table[deploy_bins, np.arange(width)[None, :]]
    fit_raw_rho = np.cumprod(fit_raw_q, axis=1)
    deploy_raw_rho = np.cumprod(deploy_raw_q, axis=1)
    raw_fit_cost = float((fit_raw_rho * active_fit).sum() / len(fit_lengths))
    raw_deploy_cost = float(
        (deploy_raw_rho * active_deploy).sum() / len(deploy_lengths)
    )
    fit_q_t, deploy_q_t, correction = (
        correct_projected_cumulative_probabilities_to_budget(
            torch.as_tensor(fit_raw_rho),
            torch.as_tensor(deploy_raw_rho),
            torch.as_tensor(fit_lengths),
            torch.as_tensor(fit_prior),
            torch.as_tensor(deploy_prior),
            target_budget,
            terminal_pi_min=terminal_pi_min,
        )
    )
    fit_q = fit_q_t.numpy()
    deploy_q = deploy_q_t.numpy()
    fit_rho = np.cumprod(fit_q, axis=1)
    deploy_rho = np.cumprod(deploy_q, axis=1)
    fit_cell_count = int(active_fit.sum())
    deploy_cell_count = int(active_deploy.sum())
    policy = Policy(
        fit_q=fit_q,
        deploy_q=deploy_q,
        fit_rho=fit_rho,
        deploy_rho=deploy_rho,
        cutpoints=cutpoints,
        fit_low_share=float(((fit_bins == 0) & active_fit).sum() / fit_cell_count),
        deploy_low_share=float(
            ((deploy_bins == 0) & active_deploy).sum() / deploy_cell_count
        ),
        empty_low_cells=empty_low,
        empty_high_cells=empty_high,
    )
    return policy, {
        "raw_fit_phase2_cost": raw_fit_cost,
        "raw_deploy_phase2_cost": raw_deploy_cost,
        "raw_projection_cost_gap": raw_deploy_cost - raw_fit_cost,
        **correction,
    }


def policy_metrics(
    *,
    task: str,
    policy: Policy,
    fit_times: np.ndarray,
    deploy_times: np.ndarray,
    fit_lengths: np.ndarray,
    deploy_lengths: np.ndarray,
    fit_objective_masses: np.ndarray,
    deploy_objective_masses: np.ndarray,
    phase1_cost: float,
    total_n: int,
    endpoint_target: np.ndarray,
) -> dict[str, float]:
    width = policy.deploy_rho.shape[1]
    fit_active = np.arange(width)[None, :] < fit_lengths[:, None]
    deploy_active = np.arange(width)[None, :] < deploy_lengths[:, None]
    fit_end = policy.fit_rho[np.arange(len(fit_lengths)), fit_lengths - 1]
    deploy_end = policy.deploy_rho[
        np.arange(len(deploy_lengths)), deploy_lengths - 1
    ]
    expected_phase2 = float((policy.deploy_rho * deploy_active).sum())
    exact_variance = float(
        (endpoint_target * (1.0 / deploy_end - 1.0)).sum()
        / total_n**2
        * 10_000
    )
    fit_soft = float(
        np.mean(np.sum(fit_objective_masses * (1.0 / policy.fit_rho - 1.0), axis=1))
    )
    deploy_soft = float(
        np.mean(
            np.sum(
                deploy_objective_masses * (1.0 / policy.deploy_rho - 1.0),
                axis=1,
            )
        )
    )
    return {
        "expected_cost_per_sample": (phase1_cost + expected_phase2) / total_n,
        "exact_target_variance_pp2": exact_variance,
        "fit_soft_surrogate": fit_soft,
        "deploy_soft_surrogate": deploy_soft,
        "fit_mean_endpoint_pi": float(fit_end.mean()),
        "deploy_mean_endpoint_pi": float(deploy_end.mean()),
        "deploy_min_endpoint_pi": float(deploy_end.min()),
    }


def realized_metric(
    *,
    policy: Policy,
    fit_times: np.ndarray,
    deploy_times: np.ndarray,
    fit_lengths: np.ndarray,
    deploy_lengths: np.ndarray,
    phase1_cost: float,
    total_n: int,
    uniforms: np.ndarray,
    width: int,
) -> dict[str, float | int]:
    active = np.arange(width)[None, :] < deploy_lengths[:, None]
    keep = np.logical_and.accumulate(uniforms < policy.deploy_q, axis=1)
    endpoint = keep[np.arange(len(deploy_lengths)), deploy_lengths - 1]
    pi = policy.deploy_rho[np.arange(len(deploy_lengths)), deploy_lengths - 1]
    target = deploy_times <= width
    estimate = ((fit_times <= width).sum() + (target & endpoint).dot(1.0 / pi)) / total_n
    return {
        "estimate": float(estimate),
        "realized_cost_per_sample": float(
            (phase1_cost + (keep & active).sum()) / total_n
        ),
        "observed_phase2_target_events": int((target & endpoint).sum()),
    }


def realized_lpb(
    *,
    policy: Policy,
    calibration_idx: np.ndarray,
    test_idx: np.ndarray,
    fit_local: np.ndarray,
    deploy_local: np.ndarray,
    times: np.ndarray,
    quantiles: np.ndarray,
    prior: np.ndarray,
    phase1_cost: float,
    uniforms: np.ndarray,
    taus: np.ndarray,
    width: int,
) -> dict[str, float | int]:
    deploy_idx = calibration_idx[deploy_local]
    fit_idx = calibration_idx[fit_local]
    deploy_lengths = np.minimum(times[deploy_idx], prior[deploy_idx]).astype(np.int64)
    active = np.arange(width)[None, :] < deploy_lengths[:, None]
    keep = np.logical_and.accumulate(uniforms < policy.deploy_q, axis=1)
    endpoint = keep[np.arange(len(deploy_lengths)), deploy_lengths - 1]
    pi_deploy = policy.deploy_rho[
        np.arange(len(deploy_lengths)), deploy_lengths - 1
    ]
    acquired_counts = (keep & active).sum(axis=1).astype(np.int64)
    c_deploy = np.where(endpoint, prior[deploy_idx], acquired_counts)

    c = np.empty(len(calibration_idx), dtype=np.int64)
    pi = np.empty(len(calibration_idx), dtype=np.float64)
    c[fit_local] = prior[fit_idx]
    pi[fit_local] = 1.0
    c[deploy_local] = c_deploy
    pi[deploy_local] = pi_deploy
    cal_times = times[calibration_idx]
    cal_quantiles = quantiles[calibration_idx]
    latent = cal_times[:, None] < cal_quantiles
    observed = latent & (cal_quantiles <= c[:, None])
    miscoverage = (observed / pi[:, None]).mean(axis=0)
    selected = strict_select(miscoverage)
    oracle_curve = latent.mean(axis=0)
    oracle_selected = strict_select(oracle_curve)
    test_coverage = (times[test_idx, None] >= quantiles[test_idx]).mean(axis=0)
    selected_a = latent[:, selected]
    exact_selected = float(
        (selected_a * (1.0 / pi - 1.0)).sum()
        / len(calibration_idx) ** 2
        * 10_000
    )
    return {
        "estimate": float(miscoverage[selected]),
        "selected_index": selected,
        "selected_tau": float(taus[selected]),
        "coverage_pct": float(100 * test_coverage[selected]),
        "oracle_selected_index": oracle_selected,
        "oracle_selected_tau": float(taus[oracle_selected]),
        "oracle_fixed_coverage_pct": float(100 * test_coverage[oracle_selected]),
        "selected_exact_variance_pp2": exact_selected,
        "switched_from_oracle": int(selected != oracle_selected),
        "realized_cost_per_sample": float(
            (phase1_cost + (keep & active).sum()) / len(calibration_idx)
        ),
        "observed_phase2_target_events": int(endpoint.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", choices=SETUPS, required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--tasks", default="metric,lpb")
    parser.add_argument("--cut-quantiles", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--n1", type=int, default=50)
    parser.add_argument("--budget", type=float, default=20.0)
    parser.add_argument("--projection-margin", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    width = 200
    grid, times_tensor, dataset, setup = load_setup(args.setup)
    times = times_tensor.numpy().astype(np.int64)
    step = torch.arange(width)
    hazard = grid[:, step, step].to(torch.float64).numpy()
    taus_t = make_lpb_tau_grid(device="cpu")
    lpb_q_t = lpb_quantiles(grid, taus_t, width)
    lpb_q = lpb_q_t.numpy().astype(np.int64)
    prior = get_prior(lpb_q_t, taus_t, 0.56).numpy().astype(np.int64)
    anchor = strict_select(taus_t.numpy(), target=0.10)
    anchor_horizon = lpb_q[:, anchor]
    del grid

    tasks = [value.strip() for value in args.tasks.split(",")]
    cut_quantiles = [float(value) for value in args.cut_quantiles.split(",")]
    seeds = [int(value) for value in args.seeds.split(",")]
    rows: list[dict] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        np.random.seed(seed)
        outer = np.random.permutation(len(times))
        calibration_idx = outer[:3000]
        test_idx = outer[3000:]
        inner = np.random.RandomState(seed).permutation(len(calibration_idx))
        fit_local = inner[: args.n1]
        deploy_local = inner[args.n1 :]
        fit_idx = calibration_idx[fit_local]
        deploy_idx = calibration_idx[deploy_local]
        common_uniforms = np.random.default_rng(seed).random((len(calibration_idx), width))[
            deploy_local
        ]
        for task in tasks:
            if task == "metric":
                fit_prior = np.full(len(fit_idx), width, dtype=np.int64)
                deploy_prior = np.full(len(deploy_idx), width, dtype=np.int64)
                fit_lengths = np.minimum(times[fit_idx], width).astype(np.int64)
                deploy_lengths = np.minimum(times[deploy_idx], width).astype(np.int64)
                fit_active = np.arange(width)[None, :] < fit_lengths[:, None]
                deploy_active = np.arange(width)[None, :] < deploy_lengths[:, None]
                fit_masses = hazard[fit_idx] * fit_active
                deploy_masses = hazard[deploy_idx] * deploy_active
                endpoint_target = times[deploy_idx] <= width
            elif task == "lpb":
                fit_prior = prior[fit_idx]
                deploy_prior = prior[deploy_idx]
                fit_lengths = np.minimum(times[fit_idx], fit_prior).astype(np.int64)
                deploy_lengths = np.minimum(times[deploy_idx], deploy_prior).astype(np.int64)
                fit_active = np.arange(width)[None, :] < fit_lengths[:, None]
                deploy_active = np.arange(width)[None, :] < deploy_lengths[:, None]
                fit_target = np.arange(1, width + 1)[None, :] < anchor_horizon[fit_idx, None]
                deploy_target = np.arange(1, width + 1)[None, :] < anchor_horizon[deploy_idx, None]
                fit_masses = (hazard[fit_idx] * fit_active * (fit_target + 0.001)) / 1.001
                deploy_masses = (
                    hazard[deploy_idx] * deploy_active * (deploy_target + 0.001)
                ) / 1.001
                endpoint_target = times[deploy_idx] < anchor_horizon[deploy_idx]
            else:
                raise ValueError(f"Unknown task: {task!r}")

            phase1_cost = float(fit_lengths.sum())
            target_budget = (
                (args.budget * len(calibration_idx) - phase1_cost) / len(deploy_idx)
                - args.projection_margin
            )
            for cut_quantile in cut_quantiles:
                started = time.perf_counter()
                policy, correction = solve_quantile_k2(
                    fit_scores=hazard[fit_idx],
                    deploy_scores=hazard[deploy_idx],
                    fit_lengths=fit_lengths,
                    deploy_lengths=deploy_lengths,
                    fit_prior=fit_prior,
                    deploy_prior=deploy_prior,
                    target_budget=target_budget,
                    objective_masses=fit_masses,
                    cut_quantile=cut_quantile,
                )
                deterministic = policy_metrics(
                    task=task,
                    policy=policy,
                    fit_times=times[fit_idx],
                    deploy_times=times[deploy_idx],
                    fit_lengths=fit_lengths,
                    deploy_lengths=deploy_lengths,
                    fit_objective_masses=fit_masses,
                    deploy_objective_masses=deploy_masses,
                    phase1_cost=phase1_cost,
                    total_n=len(calibration_idx),
                    endpoint_target=endpoint_target,
                )
                if task == "metric":
                    realized = realized_metric(
                        policy=policy,
                        fit_times=times[fit_idx],
                        deploy_times=times[deploy_idx],
                        fit_lengths=fit_lengths,
                        deploy_lengths=deploy_lengths,
                        phase1_cost=phase1_cost,
                        total_n=len(calibration_idx),
                        uniforms=common_uniforms,
                        width=width,
                    )
                else:
                    realized = realized_lpb(
                        policy=policy,
                        calibration_idx=calibration_idx,
                        test_idx=test_idx,
                        fit_local=fit_local,
                        deploy_local=deploy_local,
                        times=times,
                        quantiles=lpb_q,
                        prior=prior,
                        phase1_cost=phase1_cost,
                        uniforms=common_uniforms,
                        taus=taus_t.numpy(),
                        width=width,
                    )
                rows.append({
                    "setup_key": args.setup,
                    "dataset": dataset,
                    "data_setup": setup,
                    "task": task,
                    "seed": seed,
                    "n1": args.n1,
                    "budget": args.budget,
                    "projection_margin": args.projection_margin,
                    "cut_quantile": cut_quantile,
                    "fit_low_share": policy.fit_low_share,
                    "deploy_low_share": policy.deploy_low_share,
                    "empty_low_cells": policy.empty_low_cells,
                    "empty_high_cells": policy.empty_high_cells,
                    "runtime_seconds": time.perf_counter() - started,
                    **correction,
                    **deterministic,
                    **realized,
                })
                print(json.dumps({
                    "setup": args.setup,
                    "task": task,
                    "seed": seed,
                    "q": cut_quantile,
                    "variance": deterministic["exact_target_variance_pp2"],
                    "cost": deterministic["expected_cost_per_sample"],
                }), flush=True)
                pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
