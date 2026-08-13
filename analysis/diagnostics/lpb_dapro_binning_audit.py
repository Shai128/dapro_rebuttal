"""Offline LPB DAPRO bin/score/candidate-switching audit.

This is intentionally a diagnostic, not an allocator implementation.  It
reproduces the production no-CRC Generalized-DAPRO split, soft LPB objective,
cumulative budget correction, acquisition CRNs, Horvitz--Thompson calibration,
and strict-prefix LPB selector.  Alternative score maps are confined here.
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

from analysis.diagnostics.dapro_binning_audit import (
    SETUPS,
    bin_stats,
    load_setup,
    smooth_rank_policy,
)
from src.predictive_bounds.budget_allocators.optimization_solver_utils import (
    solve_binned_deployable_policy,
)
from src.predictive_bounds.budget_allocators.projected_optimization_utils import (
    correct_projected_cumulative_probabilities_to_budget,
)
from src.predictive_bounds.calibration.calibration_utils import (
    get_prior,
    quantiles_to_interaction_counts,
    select_calibration_positions,
)
from src.predictive_bounds.survival_utils.compute_mean_time_given_pmf import (
    compute_quantiles_survival_time,
)
from src.predictive_bounds.utils.utils import make_lpb_tau_grid


VARIANTS = {
    "hazard_k2": ("hazard", 2, "hard_bin"),
    "hazard_k4": ("hazard", 4, "hard_bin"),
    "target_value_k4": ("target_value", 4, "hard_bin"),
    "target_value_smooth_k4": ("target_value", 4, "smooth_rank"),
}


def lpb_quantiles(grid: torch.Tensor, taus: torch.Tensor, width: int) -> torch.Tensor:
    raw = compute_quantiles_survival_time(
        grid[:, 0].unsqueeze(1),
        taus,
        tail_distribution="geometric",
    ).squeeze(1)
    return quantiles_to_interaction_counts(
        raw,
        width=width,
        upper_bound=width,
    )


def target_value_scores(
    grid: torch.Tensor,
    target_horizons: torch.Tensor,
    width: int,
) -> np.ndarray:
    """Causal probability of the named strict LPB event from each prefix."""
    horizons = target_horizons.reshape(-1).to(torch.long)
    output = np.zeros((len(grid), width), dtype=np.float64)
    for step in range(width):
        pmf = grid[:, step, :]
        cumulative = pmf.cumsum(dim=1)
        upper = (horizons - 2).clamp(min=0, max=pmf.shape[1] - 1)
        mass = cumulative.gather(1, upper.unsqueeze(1)).squeeze(1)
        if step:
            mass = mass - cumulative[:, step - 1]
        eligible = horizons > step + 1
        valid = pmf[:, step:].sum(dim=1).clamp_min(
            torch.finfo(pmf.dtype).tiny
        )
        value = torch.where(eligible, mass.clamp_min(0) / valid, 0.0)
        output[:, step] = value.to(torch.float64).numpy()
    return output


def strict_select(values: np.ndarray, target: float = 0.10) -> int:
    return int(select_calibration_positions(
        torch.as_tensor(values, dtype=torch.float64),
        torch.tensor([target], dtype=torch.float64),
    ).item())


@dataclass
class FittedPolicy:
    variant: str
    objective: str
    outer_idx: np.ndarray
    calibration_idx: np.ndarray
    test_idx: np.ndarray
    fit_local: np.ndarray
    deploy_local: np.ndarray
    fit_lengths: np.ndarray
    deploy_lengths: np.ndarray
    deploy_q: np.ndarray
    deploy_pi: np.ndarray
    deploy_active: np.ndarray
    phase1_cost: float
    expected_total_cost: float
    exact_anchor_variance_pp2: float
    diagnostics: dict


def fit_policy(
    *,
    variant: str,
    objective: str,
    data_seed: int,
    policy_seed: int,
    times: np.ndarray,
    quantiles: torch.Tensor,
    prior: np.ndarray,
    scores: dict[str, np.ndarray],
    hazard: np.ndarray,
    anchor_horizons: np.ndarray,
    budget: float,
    n1: int,
    projection_margin: float,
    width: int,
) -> FittedPolicy:
    score_name, n_bins, projection_kind = VARIANTS[variant]
    np.random.seed(data_seed)
    outer_idx = np.random.permutation(len(times))
    calibration_idx = outer_idx[:3000]
    test_idx = outer_idx[3000:]

    inner = np.random.RandomState(policy_seed).permutation(len(calibration_idx))
    fit_local = inner[:n1]
    deploy_local = inner[n1:]
    fit_idx = calibration_idx[fit_local]
    deploy_idx = calibration_idx[deploy_local]
    fit_times = times[fit_idx]
    deploy_times = times[deploy_idx]
    fit_prior = prior[fit_idx]
    deploy_prior = prior[deploy_idx]
    fit_lengths = np.minimum(fit_times, fit_prior).astype(np.int64)
    deploy_lengths = np.minimum(deploy_times, deploy_prior).astype(np.int64)
    phase1_cost = float(fit_lengths.sum())
    target_budget = (
        (budget * len(calibration_idx) - phase1_cost) / len(deploy_idx)
        - projection_margin
    )
    if target_budget <= 0:
        raise ValueError("Phase-I rows exhaust the LPB budget.")

    active_fit = np.arange(width)[None, :] < fit_lengths[:, None]
    target_mask = (
        np.arange(1, width + 1)[None, :]
        < anchor_horizons[fit_idx, None]
    )
    named_mass = hazard[fit_idx] * active_fit * target_mask
    all_event_mass = hazard[fit_idx] * active_fit
    objective_masses = None
    objective_weights = None
    hard_target = (fit_times < anchor_horizons[fit_idx]).astype(np.float64)
    if objective == "soft":
        objective_masses = (named_mass + 0.001 * all_event_mass) / 1.001
    elif objective == "hard":
        objective_weights = hard_target
    elif objective == "hard_reg":
        objective_weights = (hard_target + 0.001) / 1.001
    else:
        raise ValueError(f"Unknown objective: {objective!r}")

    fit_scores = torch.as_tensor(scores[score_name][fit_idx], dtype=torch.float64)
    deploy_scores = torch.as_tensor(
        scores[score_name][deploy_idx], dtype=torch.float64
    )
    _, fit_raw_q, deploy_raw_q, solver_diagnostics = (
        solve_binned_deployable_policy(
            fit_scores,
            deploy_scores,
            torch.as_tensor(fit_lengths),
            target_budget,
            (
                None
                if objective_weights is None
                else torch.as_tensor(objective_weights)
            ),
            n_bins,
            objective_masses=(
                None
                if objective_masses is None
                else torch.as_tensor(objective_masses)
            ),
        )
    )
    if projection_kind == "smooth_rank":
        fit_raw_q, deploy_raw_q = smooth_rank_policy(
            scores[score_name][fit_idx],
            scores[score_name][deploy_idx],
            fit_lengths,
            fit_raw_q,
            n_bins,
        )
    fit_q, deploy_q_tensor, correction = (
        correct_projected_cumulative_probabilities_to_budget(
            fit_raw_q.cumprod(dim=1),
            deploy_raw_q.cumprod(dim=1),
            torch.as_tensor(fit_lengths),
            torch.as_tensor(fit_prior),
            torch.as_tensor(deploy_prior),
            target_budget,
            terminal_pi_min=0.005,
        )
    )
    deploy_q = deploy_q_tensor.detach().cpu().numpy()
    deploy_rho = np.cumprod(deploy_q, axis=1)
    deploy_active = np.arange(width)[None, :] < deploy_lengths[:, None]
    deploy_pi = deploy_rho[np.arange(len(deploy_idx)), deploy_lengths - 1]
    expected_deploy_cost = float((deploy_rho * deploy_active).sum())
    expected_total_cost = phase1_cost + expected_deploy_cost

    anchor_a = deploy_times < anchor_horizons[deploy_idx]
    exact_anchor_variance_pp2 = float(
        (anchor_a * (1.0 / deploy_pi - 1.0)).sum()
        / len(calibration_idx) ** 2
        * 10_000
    )
    diagnostics = {
        **solver_diagnostics,
        **correction,
        **bin_stats(
            scores[score_name][fit_idx],
            fit_lengths,
            fit_raw_q,
            n_bins,
        ),
    }
    return FittedPolicy(
        variant=variant,
        objective=objective,
        outer_idx=outer_idx,
        calibration_idx=calibration_idx,
        test_idx=test_idx,
        fit_local=fit_local,
        deploy_local=deploy_local,
        fit_lengths=fit_lengths,
        deploy_lengths=deploy_lengths,
        deploy_q=deploy_q,
        deploy_pi=deploy_pi,
        deploy_active=deploy_active,
        phase1_cost=phase1_cost,
        expected_total_cost=expected_total_cost,
        exact_anchor_variance_pp2=exact_anchor_variance_pp2,
        diagnostics=diagnostics,
    )


def acquire_and_calibrate(
    *,
    fitted: FittedPolicy,
    acquisition_seed: int,
    times: np.ndarray,
    quantiles: np.ndarray,
    prior: np.ndarray,
    taus: np.ndarray,
    width: int,
) -> dict[str, float | int]:
    cal_idx = fitted.calibration_idx
    deploy_idx = cal_idx[fitted.deploy_local]
    uniforms = np.random.default_rng(acquisition_seed).random(
        (len(times), width)
    )[deploy_idx]
    sequential_keep = np.logical_and.accumulate(
        uniforms < fitted.deploy_q,
        axis=1,
    )
    acquired = sequential_keep & fitted.deploy_active
    endpoint_observed = sequential_keep[
        np.arange(len(deploy_idx)), fitted.deploy_lengths - 1
    ]
    acquired_counts = acquired.sum(axis=1).astype(np.int64)
    deploy_prior = prior[deploy_idx].astype(np.int64)
    deploy_times = times[deploy_idx].astype(np.int64)
    succeeded = endpoint_observed
    deploy_c = np.where(succeeded, deploy_prior, acquired_counts)

    c = np.empty(len(cal_idx), dtype=np.int64)
    pi = np.empty(len(cal_idx), dtype=np.float64)
    fit_idx = cal_idx[fitted.fit_local]
    c[fitted.fit_local] = prior[fit_idx].astype(np.int64)
    pi[fitted.fit_local] = 1.0
    c[fitted.deploy_local] = deploy_c
    pi[fitted.deploy_local] = fitted.deploy_pi

    cal_times = times[cal_idx]
    cal_q = quantiles[cal_idx]
    latent = cal_times[:, None] < cal_q
    observed = latent & (cal_q <= c[:, None])
    miscoverage = (observed / pi[:, None]).mean(axis=0)
    selected = strict_select(miscoverage)

    oracle_miscoverage = latent.mean(axis=0)
    oracle_selected = strict_select(oracle_miscoverage)
    test_q = quantiles[fitted.test_idx]
    test_times = times[fitted.test_idx]
    coverage_curve = (test_times[:, None] >= test_q).mean(axis=0)
    size_curve = test_q.mean(axis=0)

    selected_a = latent[:, selected]
    fixed_a = latent[:, oracle_selected]
    selected_exact_variance = float(
        (selected_a * (1.0 / pi - 1.0)).sum()
        / len(cal_idx) ** 2
        * 10_000
    )
    fixed_exact_variance = float(
        (fixed_a * (1.0 / pi - 1.0)).sum()
        / len(cal_idx) ** 2
        * 10_000
    )
    return {
        "acquisition_seed": acquisition_seed,
        "selected_index": selected,
        "selected_tau": float(taus[selected]),
        "selected_alpha_hat": float(miscoverage[selected]),
        "coverage": float(coverage_curve[selected]),
        "coverage_pct": float(100 * coverage_curve[selected]),
        "selected_size": float(size_curve[selected]),
        "selected_exact_conditional_variance_pp2": selected_exact_variance,
        "oracle_fixed_index": oracle_selected,
        "oracle_fixed_tau": float(taus[oracle_selected]),
        "oracle_fixed_alpha": float(oracle_miscoverage[oracle_selected]),
        "oracle_fixed_coverage": float(coverage_curve[oracle_selected]),
        "oracle_fixed_coverage_pct": float(100 * coverage_curve[oracle_selected]),
        "fixed_candidate_ht_alpha": float(miscoverage[oracle_selected]),
        "fixed_candidate_ht_alpha_pct": float(100 * miscoverage[oracle_selected]),
        "fixed_candidate_exact_conditional_variance_pp2": fixed_exact_variance,
        "switched_from_oracle_fixed": int(selected != oracle_selected),
        "selected_index_minus_oracle": int(selected - oracle_selected),
        "candidate_switch_coverage_effect_pp": float(
            100 * (coverage_curve[selected] - coverage_curve[oracle_selected])
        ),
        "realized_cost_per_sample": float(
            (fitted.phase1_cost + acquired.sum()) / len(cal_idx)
        ),
        "observed_phase2_anchor_events": int(
            (
                (deploy_times < quantiles[deploy_idx, strict_select(taus)])
                & endpoint_observed
            ).sum()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", choices=SETUPS, required=True)
    parser.add_argument("--mode", choices=["outer", "acquisition"], default="outer")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--fixed-data-seed", type=int, default=0)
    parser.add_argument("--fixed-policy-seed", type=int, default=0)
    parser.add_argument("--n1", type=int, default=50)
    parser.add_argument("--budget", type=float, default=20.0)
    parser.add_argument("--projection-margin", type=float, default=1.0)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument(
        "--objectives",
        default="soft",
        help="Comma-separated coefficient estimators: hard, hard_reg, soft.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    width = 200
    grid, times_tensor, dataset, setup = load_setup(args.setup)
    times = times_tensor.numpy().astype(np.int64)
    taus_tensor = make_lpb_tau_grid(device="cpu")
    quantiles_tensor = lpb_quantiles(grid, taus_tensor, width)
    quantiles = quantiles_tensor.numpy().astype(np.int64)
    prior = get_prior(quantiles_tensor, taus_tensor, 0.56).numpy().astype(np.int64)
    anchor_index = strict_select(taus_tensor.numpy(), target=0.10)
    anchor_horizons = quantiles[:, anchor_index]
    step = torch.arange(width)
    hazard = grid[:, step, step].to(torch.float64).numpy()
    scores = {
        "hazard": hazard,
        "target_value": target_value_scores(grid, quantiles_tensor[:, anchor_index], width),
    }
    del grid

    args.output.parent.mkdir(parents=True, exist_ok=True)
    seeds = [int(value) for value in args.seeds.split(",")]
    variants = args.variants.split(",")
    objectives = args.objectives.split(",")
    unknown_objectives = sorted(
        set(objectives) - {"hard", "hard_reg", "soft"}
    )
    if unknown_objectives:
        raise ValueError(f"Unknown objectives: {unknown_objectives}")
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")

    rows: list[dict] = []
    if args.mode == "outer":
        jobs = [(seed, seed, seed) for seed in seeds]
    else:
        jobs = [
            (args.fixed_data_seed, args.fixed_policy_seed, acquisition_seed)
            for acquisition_seed in seeds
        ]

    fitted_cache: dict[tuple[str, str, int, int], FittedPolicy] = {}
    for data_seed, policy_seed, acquisition_seed in jobs:
        for variant in variants:
            for objective in objectives:
                key = (variant, objective, data_seed, policy_seed)
                if key not in fitted_cache:
                    started = time.perf_counter()
                    fitted_cache[key] = fit_policy(
                        variant=variant,
                        objective=objective,
                        data_seed=data_seed,
                        policy_seed=policy_seed,
                        times=times,
                        quantiles=quantiles_tensor,
                        prior=prior,
                        scores=scores,
                        hazard=hazard,
                        anchor_horizons=anchor_horizons,
                        budget=args.budget,
                        n1=args.n1,
                        projection_margin=args.projection_margin,
                        width=width,
                    )
                    fit_seconds = time.perf_counter() - started
                else:
                    fit_seconds = 0.0
                fitted = fitted_cache[key]
                result = acquire_and_calibrate(
                    fitted=fitted,
                    acquisition_seed=acquisition_seed,
                    times=times,
                    quantiles=quantiles,
                    prior=prior,
                    taus=taus_tensor.numpy(),
                    width=width,
                )
                row = {
                    "setup_key": args.setup,
                    "dataset": dataset,
                    "data_setup": setup,
                    "mode": args.mode,
                    "data_seed": data_seed,
                    "policy_seed": policy_seed,
                    "acquisition_seed": acquisition_seed,
                    "variant": variant,
                    "objective": objective,
                    "score": VARIANTS[variant][0],
                    "bins": VARIANTS[variant][1],
                    "projection_kind": VARIANTS[variant][2],
                    "n1": args.n1,
                    "budget": args.budget,
                    "projection_margin": args.projection_margin,
                    "target_anchor_index": anchor_index,
                    "target_anchor_tau": float(taus_tensor[anchor_index]),
                    "expected_cost_per_sample": (
                        fitted.expected_total_cost / len(fitted.calibration_idx)
                    ),
                    "anchor_exact_conditional_variance_pp2": (
                        fitted.exact_anchor_variance_pp2
                    ),
                    "policy_fit_runtime_seconds": fit_seconds,
                    **result,
                    **fitted.diagnostics,
                }
                rows.append(row)
                print(json.dumps({
                    "setup": args.setup,
                    "mode": args.mode,
                    "data_seed": data_seed,
                    "policy_seed": policy_seed,
                    "acquisition_seed": acquisition_seed,
                    "variant": variant,
                    "objective": objective,
                    "coverage_pct": result["coverage_pct"],
                    "selected_index": result["selected_index"],
                }), flush=True)
                pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
