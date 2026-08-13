"""Isolated diagnostics for distribution-free, CRC-free K=2 DAPRO control.

This module intentionally does not register a production allocator.  It
temporarily replaces only the projection-budget correction and (for the
online compensator experiment) the acquisition simulator used by
``SoftTargetDAPRO``.  The learned hazard score, median K=2 cutpoints, soft
target masses, and convex DAPRO optimizer are therefore identical to the
production implementation.

Two controllers are evaluated:

``all_high``
    Bound every possible K=2 score path by the path that selects the larger
    continuation probability at every time.  A single cumulative-logit
    intercept is selected using only the known deployment prior horizons so
    the aggregate upper-bound cost is at most the remaining budget.  This is
    a static causal policy and gives a conditional expected-budget guarantee.

``online_compensator``
    Keep a Phase-I-frozen time-spend profile.  At each deployment time, after
    observing the current active set and its causal K=2 scores, select one
    common logit intercept so the sum of current continuation probabilities
    equals at most the current compensator tranche.  The sum of all tranches
    is deterministically at most the remaining budget.  This gives expected
    budget control without a projection-error assumption or a CRC split.

The script reports metric-estimation variance and LPB coverage/size across
random calibration--test splits, along with realized and conditional expected
cost.  Run from the repository root, for example::

    python -m analysis.diagnostics.k2_budget_safe_experiment \
        --setups toxicity_qwen red_qwen --seeds 0:20 --tasks metric lpb
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import pandas as pd
import torch

import src.predictive_bounds.budget_allocators.DAPRO as dapro_module
from src.predictive_bounds.budget_allocators.DAPRO import (
    SoftTargetCRCDAPRO,
    SoftTargetDAPRO,
    construct_final_result,
)
from src.predictive_bounds.budget_allocators.budget_allocator import (
    BudgetAllocationResult,
)
from src.predictive_bounds.budget_allocators.projected_optimization_utils import (
    _continuation_from_cumulative,
    expected_acquisition_cost,
)
from src.predictive_bounds.calibration.calibration_utils import (
    get_prior,
    select_calibration_positions,
)
from src.predictive_bounds.utils.utils import setup_experiment_data, split_data
from src.train_model.models.utils import SurvivalModelPrediction
from src.utils.utils import set_seeds


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "outputs" / "k2_budget_safe"


@dataclass(frozen=True)
class Setup:
    dataset_name: str
    data_setup: str
    budget: float


SETUPS = {
    "toxicity_qwen": Setup(
        "dataset_toxicity",
        "attack_toxic_attack_qwen25_14b_instruct_lm_target_"
        "qwen25_14b_instruct_judge_detoxify",
        20.0,
    ),
    "toxicity_phi": Setup(
        "dataset_toxicity",
        "attack_toxic_attack_qwen25_14b_instruct_lm_target_"
        "mini_phi_4_instruct_judge_detoxify",
        20.0,
    ),
    "red_qwen": Setup(
        "dataset_red_team",
        "attack_default_attack_qwen25_14b_instruct_lm_target_"
        "qwen25_14b_instruct_judge_llm-judge_qwen25_14b_instruct",
        20.0,
    ),
    "red_guard": Setup(
        "dataset_red_team",
        "attack_default_attack_qwen25_14b_instruct_lm_target_"
        "qwen25_14b_instruct_judge_llama_guard",
        10.0,
    ),
    "hallucination_qwen": Setup(
        "dataset_hallucination3",
        "attack_hallucination_attack_qwen25_14b_instruct_lm_target_"
        "qwen25_14b_instruct_judge_llm-judge_qwen25_14b_instruct",
        10.0,
    ),
    "autoif_qwen": Setup(
        "dataset_autoif",
        "attack_autoif_helper_qwen25_14b_instruct_lm_target_"
        "qwen25_14b_instruct_judge_autoif",
        20.0,
    ),
}


ORIGINAL_CORRECTION = (
    dapro_module.correct_projected_cumulative_probabilities_to_budget
)
ORIGINAL_ADAPTIVE = dapro_module.adaptive_budget_allocation


def _active(q: torch.Tensor, width: int, device: torch.device) -> torch.Tensor:
    time = torch.arange(width, device=device).unsqueeze(0)
    return time < q.to(device=device, dtype=torch.long).reshape(-1, 1)


def _direct_cost(cumulative: torch.Tensor, q: torch.Tensor) -> float:
    active = _active(q, cumulative.shape[1], cumulative.device)
    return float((cumulative * active.to(torch.float64)).sum(1).mean().item())


def _conditional_from_unmasked_cumulative(cumulative: torch.Tensor) -> torch.Tensor:
    cumulative = cumulative.to(torch.float64).clamp(
        min=torch.finfo(torch.float64).tiny,
        max=1.0,
    )
    previous = torch.cat(
        [torch.ones((len(cumulative), 1), dtype=torch.float64,
                     device=cumulative.device), cumulative[:, :-1]],
        dim=1,
    )
    return (cumulative / previous.clamp_min(
        torch.finfo(torch.float64).tiny
    )).clamp(max=1.0)


def _path_envelope_correction(
    validation_cumulative: torch.Tensor,
    deployment_cumulative: torch.Tensor,
    validation_active_lengths: torch.Tensor,
    validation_prior_q: torch.Tensor,
    deployment_prior_q: torch.Tensor,
    target_budget_per_sample: float,
    terminal_pi_min: float | None = None,
    tolerance: float = 1e-7,
    *,
    condition_on_x0: bool,
):
    """Distribution-free K-bin path-envelope controller.

    The maximum table entry at each time is recoverable from the validation
    lookup paths because the deployable K-bin table is constant within a bin.
    A future deployment score can only select one of those entries, so the
    product of the per-time maxima dominates every possible cumulative path.
    The intercept is selected against deployment ``q_prior``, which is an
    initial-prompt quantity and hence is known before sequential acquisition.
    """
    val = validation_cumulative.to(torch.float64)
    dep = deployment_cumulative.to(torch.float64)
    width = val.shape[1]
    val_conditionals = _conditional_from_unmasked_cumulative(val)
    high_conditionals = val_conditionals.max(dim=0).values.clamp(max=1.0)
    high_raw = high_conditionals.cumprod(0).reshape(1, -1)
    if condition_on_x0:
        # The first hazard score is a function of X_{i0}, so its selected bin
        # is known before any paid trajectory step.  Keep that personalized
        # first continuation and dominate only the genuinely future bins.
        dep_first = dep[:, :1].clamp(max=1.0)
        if width == 1:
            envelope_raw = dep_first
        else:
            future_high = high_conditionals[1:].cumprod(0).reshape(1, -1)
            envelope_raw = torch.cat(
                [dep_first, dep_first * future_high], dim=1
            )
        controller_name = "x0_then_all_high_path"
    else:
        envelope_raw = high_raw.repeat(len(deployment_prior_q), 1)
        controller_name = "all_high_path"

    epsilon = 0.0 if terminal_pi_min is None else float(terminal_pi_min)
    if not 0 <= epsilon < 1:
        raise ValueError("terminal floor must lie in [0,1).")

    def transform(raw: torch.Tensor, shift: float, q: torch.Tensor) -> torch.Tensor:
        clipped = raw.clamp(1e-12, 1 - 1e-12)
        shifted = torch.sigmoid(torch.logit(clipped) + shift)
        shifted = torch.cummin(shifted, dim=1).values
        if epsilon:
            shifted = epsilon + (1 - epsilon) * shifted
        return torch.where(
            _active(q, width, shifted.device),
            shifted,
            torch.ones((), dtype=torch.float64, device=shifted.device),
        )

    def high_cost(shift: float) -> float:
        high = transform(
            envelope_raw,
            shift,
            deployment_prior_q,
        )
        return _direct_cost(high, deployment_prior_q)

    low, high = -80.0, 80.0
    minimum_cost = high_cost(low)
    if minimum_cost > target_budget_per_sample + tolerance:
        raise ValueError(
            "All-high K=2 controller is infeasible under the requested "
            f"terminal floor: minimum={minimum_cost:.8g}, "
            f"target={target_budget_per_sample:.8g}."
        )
    if high_cost(high) <= target_budget_per_sample + tolerance:
        shift = high
        boundary = "maximum"
    else:
        for _ in range(100):
            mid = (low + high) / 2
            if high_cost(mid) <= target_budget_per_sample:
                low = mid
            else:
                high = mid
        shift = low
        boundary = "all_high_path"

    val_final_cum = transform(val, shift, validation_prior_q)
    dep_final_cum = transform(dep, shift, deployment_prior_q)
    dep_bound = high_cost(shift)
    val_final = _continuation_from_cumulative(
        val_final_cum, validation_prior_q
    )
    dep_final = _continuation_from_cumulative(
        dep_final_cum, deployment_prior_q
    )
    dep_raw_monotone = torch.cummin(dep, dim=1).values
    dep_raw_cost = _direct_cost(dep_raw_monotone, deployment_prior_q)
    dep_selected_cost = _direct_cost(dep_final_cum, deployment_prior_q)
    dep_active = _active(deployment_prior_q, width, dep.device)
    log_distortion = torch.abs(
        torch.log(dep_final_cum.clamp_min(torch.finfo(torch.float64).tiny))
        - torch.log(dep_raw_monotone.clamp_min(torch.finfo(torch.float64).tiny))
    )[dep_active]
    changed = (
        torch.abs(dep_final_cum - dep_raw_monotone) > 1e-10
    )[dep_active]
    return val_final, dep_final, {
        "projection_space": f"k2_{controller_name}_cumulative_envelope",
        "projection_raw_base_phase1_expected_cost": _direct_cost(
            torch.cummin(val, dim=1).values, validation_active_lengths
        ),
        "projection_pre_intercept_mixed_phase1_expected_cost": _direct_cost(
            val_final_cum, validation_active_lengths
        ),
        "projection_raw_phase1_expected_cost": _direct_cost(
            torch.cummin(val, dim=1).values, validation_active_lengths
        ),
        "projection_budget_logit_shift": shift,
        "projection_corrected_phase1_expected_cost": _direct_cost(
            val_final_cum, validation_active_lengths
        ),
        "projection_budget_boundary": boundary,
        "k2_safe_controller": controller_name,
        "k2_safe_deployment_upper_bound_cost_per_sample": dep_bound,
        "k2_safe_deployment_selected_cost_per_sample": dep_selected_cost,
        "k2_safe_deployment_raw_cost_per_sample": dep_raw_cost,
        "k2_safe_envelope_utilization": dep_selected_cost / max(
            dep_bound, np.finfo(np.float64).tiny
        ),
        "k2_safe_fraction_cumulative_reach_changed": float(
            changed.to(torch.float64).mean().item()
        ),
        "k2_safe_mean_abs_log_cumulative_distortion": float(
            log_distortion.mean().item()
        ),
        "k2_safe_target_cost_per_sample": target_budget_per_sample,
        "k2_safe_valid": int(
            dep_bound <= target_budget_per_sample + tolerance
        ),
    }


def all_high_correction(*args, **kwargs):
    return _path_envelope_correction(
        *args, **kwargs, condition_on_x0=False
    )


def x0_high_correction(*args, **kwargs):
    return _path_envelope_correction(
        *args, **kwargs, condition_on_x0=True
    )


class OnlineState:
    profile: torch.Tensor | None = None
    target_per_sample: float = np.nan
    phase2_compensator: float = np.nan
    phase2_budget: float = np.nan
    phase2_realized_cost: float = np.nan
    mean_abs_shift: float = np.nan
    max_abs_shift: float = np.nan
    fraction_conditionals_changed: float = np.nan
    mean_abs_log_conditional_distortion: float = np.nan
    deployment_probabilities: torch.Tensor | None = None
    deployment_prior_q: torch.Tensor | None = None
    deployment_event_times: torch.Tensor | None = None
    maximum_conditionals: torch.Tensor | None = None


ONLINE_STATE = OnlineState()


def online_identity_correction(
    validation_cumulative: torch.Tensor,
    deployment_cumulative: torch.Tensor,
    validation_active_lengths: torch.Tensor,
    validation_prior_q: torch.Tensor,
    deployment_prior_q: torch.Tensor,
    target_budget_per_sample: float,
    terminal_pi_min: float | None = None,
    tolerance: float = 1e-7,
):
    """Freeze a Phase-I spend profile; defer control to deployment time."""
    del terminal_pi_min, tolerance
    val = torch.cummin(validation_cumulative.to(torch.float64), dim=1).values
    dep = torch.cummin(deployment_cumulative.to(torch.float64), dim=1).values
    val_active = _active(
        validation_active_lengths, val.shape[1], val.device
    )
    profile = (val * val_active.to(torch.float64)).sum(0)
    # Give every potentially deployable time positive mass.  This prevents a
    # zero tranche before the final possible acquisition time while changing
    # the learned Phase-I profile only below floating-point reporting scale.
    possible = _active(
        deployment_prior_q, dep.shape[1], dep.device
    ).any(0)
    scale = max(float(profile.sum().item()), 1.0)
    profile = profile + possible.to(torch.float64) * scale * 1e-12
    ONLINE_STATE.profile = profile.detach().clone()
    ONLINE_STATE.target_per_sample = float(target_budget_per_sample)
    ONLINE_STATE.phase2_compensator = np.nan
    ONLINE_STATE.phase2_budget = np.nan
    ONLINE_STATE.phase2_realized_cost = np.nan

    val_p = _continuation_from_cumulative(val, validation_prior_q)
    dep_p = _continuation_from_cumulative(dep, deployment_prior_q)
    ONLINE_STATE.maximum_conditionals = val_p.max(dim=0).values.detach().clone()
    raw_cost = _direct_cost(val, validation_active_lengths)
    return val_p, dep_p, {
        "projection_space": "raw_k2_online_compensator",
        "projection_raw_base_phase1_expected_cost": raw_cost,
        "projection_pre_intercept_mixed_phase1_expected_cost": raw_cost,
        "projection_raw_phase1_expected_cost": raw_cost,
        "projection_budget_logit_shift": 0.0,
        "projection_corrected_phase1_expected_cost": raw_cost,
        "projection_budget_boundary": "online_predictable_compensator",
        "k2_safe_controller": "online_predictable_compensator",
        "k2_safe_target_cost_per_sample": target_budget_per_sample,
        "k2_safe_valid": 1,
    }


def _odds_shift_to_sum(
    raw: torch.Tensor,
    target: float,
) -> tuple[torch.Tensor, float]:
    """Largest finite common logit shift with sum at most ``target``."""
    raw64 = raw.to(torch.float64).clamp(1e-12, 1 - 1e-12)
    n = len(raw64)
    if n == 0:
        return raw64, 0.0
    if target >= n:
        return torch.ones_like(raw64), 80.0
    target = max(float(target), np.finfo(np.float64).tiny)
    logits = torch.logit(raw64)
    lo, hi = -80.0, 80.0
    for _ in range(100):
        mid = (lo + hi) / 2
        total = float(torch.sigmoid(logits + mid).sum().item())
        if total <= target:
            lo = mid
        else:
            hi = mid
    return torch.sigmoid(logits + lo), lo


def online_compensator_allocation(
    continuation_probabilities,
    prior_q,
    event_times,
    T_max_curr,
    device,
    reach_t_max_is_success=False,
    uniforms=None,
):
    """Simulate a causal K=2 policy with a bounded compensator."""
    profile = ONLINE_STATE.profile
    if profile is None:
        raise RuntimeError("online correction must run before acquisition")
    profile = profile.to(device=device, dtype=torch.float64)
    ONLINE_STATE.deployment_probabilities = (
        continuation_probabilities.detach().clone()
    )
    ONLINE_STATE.deployment_prior_q = prior_q.detach().clone()
    ONLINE_STATE.deployment_event_times = event_times.detach().clone()
    sim_n = len(prior_q)
    budget = float(ONLINE_STATE.target_per_sample * sim_n)
    remaining = budget
    compensator = 0.0
    realized = 0.0
    sim_c = torch.zeros(sim_n, dtype=torch.long, device=device)
    sim_active = torch.ones(sim_n, dtype=torch.bool, device=device)
    observed_path_propensity = torch.ones(
        sim_n, dtype=torch.float64, device=device
    )
    shifts: list[float] = []
    changed_count = 0
    active_count = 0
    log_distortion_sum = 0.0

    for step in range(T_max_curr):
        sim_active = (
            sim_active
            & (event_times > step)
            & (prior_q > step)
        )
        if not bool(sim_active.any()):
            break
        future_profile = float(profile[step:].sum().item())
        if future_profile <= 0:
            tranche = remaining
        else:
            tranche = remaining * float(profile[step].item()) / future_profile
        # When later times remain possible, tranche < remaining.  The tiny
        # subtraction also protects that strict inequality numerically.
        if bool((profile[step + 1:] > 0).any()):
            tranche = min(tranche, remaining * (1 - 1e-14))
        active_rows = torch.where(sim_active)[0]
        final_active, shift = _odds_shift_to_sum(
            continuation_probabilities[active_rows, step], tranche
        )
        raw_active = continuation_probabilities[
            active_rows, step
        ].to(torch.float64).clamp_min(torch.finfo(torch.float64).tiny)
        active_count += len(active_rows)
        changed_count += int(
            (torch.abs(final_active - raw_active) > 1e-10).sum().item()
        )
        log_distortion_sum += float(torch.abs(
            torch.log(final_active.clamp_min(torch.finfo(torch.float64).tiny))
            - torch.log(raw_active)
        ).sum().item())
        current_spend = float(final_active.sum().item())
        # Bisection returns the feasible endpoint; tolerate only roundoff.
        if current_spend > remaining + 1e-8:
            raise RuntimeError("online compensator exceeded remaining credit")
        compensator += current_spend
        remaining = max(0.0, remaining - current_spend)
        shifts.append(shift)

        rand = (
            torch.rand(len(active_rows), device=device, dtype=torch.float64)
            if uniforms is None
            else uniforms[active_rows, step].to(torch.float64)
        )
        kept_local = rand < final_active
        kept_rows = active_rows[kept_local]
        realized += float(len(kept_rows))
        sim_c[kept_rows] += 1
        observed_path_propensity[kept_rows] *= final_active[kept_local]
        next_active = torch.zeros_like(sim_active)
        next_active[kept_rows] = True
        sim_active = next_active

    succeeded = (sim_c >= prior_q) | (sim_c >= event_times)
    if reach_t_max_is_success:
        succeeded = succeeded | (sim_c == T_max_curr)
    sim_c = torch.where(succeeded, prior_q.to(torch.long), sim_c)
    ONLINE_STATE.phase2_compensator = compensator
    ONLINE_STATE.phase2_budget = budget
    ONLINE_STATE.phase2_realized_cost = realized
    ONLINE_STATE.mean_abs_shift = float(np.mean(np.abs(shifts))) if shifts else 0.0
    ONLINE_STATE.max_abs_shift = float(np.max(np.abs(shifts))) if shifts else 0.0
    ONLINE_STATE.fraction_conditionals_changed = (
        changed_count / active_count if active_count else 0.0
    )
    ONLINE_STATE.mean_abs_log_conditional_distortion = (
        log_distortion_sum / active_count if active_count else 0.0
    )
    return sim_c, realized, observed_path_propensity


def row_account_allocation(
    continuation_probabilities,
    prior_q,
    event_times,
    T_max_curr,
    device,
    reach_t_max_is_success=False,
    uniforms=None,
):
    """Independent per-row predictable accounts with score-aware tranches.

    Every row starts with the same credit ``b``.  At an active time, its
    current K2 probability is divided by the Phase-I maximum table value,
    yielding a score multiplier in ``(0,1]``.  The current tranche uses that
    multiplier, while all unknown future multipliers are pessimistically set
    to one.  Debiting the selected continuation probability itself makes the
    sum of predictable costs on every possible realized row path at most
    ``b``.  Rows depend only on their own histories and coins.
    """
    profile = ONLINE_STATE.profile
    high = ONLINE_STATE.maximum_conditionals
    if profile is None or high is None:
        raise RuntimeError("online correction must run before acquisition")
    profile = profile.to(device=device, dtype=torch.float64)
    high = high.to(device=device, dtype=torch.float64).clamp_min(1e-12)
    sim_n = len(prior_q)
    per_row_budget = float(ONLINE_STATE.target_per_sample)
    credit = torch.full(
        (sim_n,), per_row_budget, dtype=torch.float64, device=device
    )
    compensator = 0.0
    realized = 0.0
    sim_c = torch.zeros(sim_n, dtype=torch.long, device=device)
    sim_active = torch.ones(sim_n, dtype=torch.bool, device=device)
    observed_path_propensity = torch.ones(
        sim_n, dtype=torch.float64, device=device
    )
    changed_count = 0
    active_count = 0
    log_distortion_sum = 0.0

    future_profile = torch.flip(
        torch.cumsum(torch.flip(profile, dims=[0]), dim=0), dims=[0]
    ) - profile
    for step in range(T_max_curr):
        sim_active = (
            sim_active & (event_times > step) & (prior_q > step)
        )
        if not bool(sim_active.any()):
            break
        rows = torch.where(sim_active)[0]
        raw = continuation_probabilities[rows, step].to(torch.float64).clamp(
            1e-12, 1 - 1e-12
        )
        score_ratio = (raw / high[step]).clamp(max=1.0)
        current_weight = profile[step] * score_ratio
        denominator = current_weight + future_profile[step]
        tranche = torch.where(
            denominator > 0,
            credit[rows] * current_weight / denominator,
            credit[rows],
        ).clamp(max=1.0)
        # Strict positivity in real arithmetic; the replacement is far below
        # any displayed precision and its debit is still exact.
        final = tranche.clamp_min(torch.finfo(torch.float64).tiny)
        credit[rows] = (credit[rows] - final).clamp_min(0.0)
        compensator += float(final.sum().item())
        active_count += len(rows)
        changed_count += int((torch.abs(final - raw) > 1e-10).sum().item())
        log_distortion_sum += float(torch.abs(
            torch.log(final) - torch.log(raw)
        ).sum().item())

        rand = (
            torch.rand(len(rows), device=device, dtype=torch.float64)
            if uniforms is None
            else uniforms[rows, step].to(torch.float64)
        )
        kept_local = rand < final
        kept_rows = rows[kept_local]
        realized += float(len(kept_rows))
        sim_c[kept_rows] += 1
        observed_path_propensity[kept_rows] *= final[kept_local]
        next_active = torch.zeros_like(sim_active)
        next_active[kept_rows] = True
        sim_active = next_active

    succeeded = (sim_c >= prior_q) | (sim_c >= event_times)
    if reach_t_max_is_success:
        succeeded = succeeded | (sim_c == T_max_curr)
    sim_c = torch.where(succeeded, prior_q.to(torch.long), sim_c)
    ONLINE_STATE.phase2_compensator = compensator
    ONLINE_STATE.phase2_budget = per_row_budget * sim_n
    ONLINE_STATE.phase2_realized_cost = realized
    ONLINE_STATE.mean_abs_shift = np.nan
    ONLINE_STATE.max_abs_shift = np.nan
    ONLINE_STATE.fraction_conditionals_changed = (
        changed_count / active_count if active_count else 0.0
    )
    ONLINE_STATE.mean_abs_log_conditional_distortion = (
        log_distortion_sum / active_count if active_count else 0.0
    )
    return sim_c, realized, observed_path_propensity


@contextmanager
def patched(
    correction: Callable | None = None,
    adaptive: Callable | None = None,
) -> Iterator[None]:
    old_correction = dapro_module.correct_projected_cumulative_probabilities_to_budget
    old_adaptive = dapro_module.adaptive_budget_allocation
    if correction is not None:
        dapro_module.correct_projected_cumulative_probabilities_to_budget = correction
    if adaptive is not None:
        dapro_module.adaptive_budget_allocation = adaptive
    try:
        yield
    finally:
        dapro_module.correct_projected_cumulative_probabilities_to_budget = old_correction
        dapro_module.adaptive_budget_allocation = old_adaptive


def _allocate(
    method: str,
    conditional_grid: torch.Tensor,
    budget: float,
    taus: torch.Tensor,
    tau_prior: float,
    max_time: int,
    n1: int,
    task: str,
    t_cal: torch.Tensor,
    prediction: SurvivalModelPrediction,
    seed: int,
    acquisition_seed: int | None = None,
):
    common = dict(
        conditional_grid=conditional_grid,
        budget_per_sample=budget,
        taus_range=taus,
        tau_prior=tau_prior,
        m_upper_bound=max_time,
        n1=n1,
    )
    target = dict(
        metric_estimation_horizon=max_time if task == "metric" else None
    )
    if method == "soft_no_crc":
        allocator = SoftTargetDAPRO(
            **common, projection_budget_margin=1.0, **target
        )
        correction = None
        adaptive = None
    elif method == "soft_crc":
        allocator = SoftTargetCRCDAPRO(
            **common,
            budget_control_size=n1 // 2,
            row_cost_cap_multiplier=2.0,
            **target,
        )
        correction = None
        adaptive = None
    elif method in {"k2_all_high", "k2_x0_high"}:
        allocator = SoftTargetDAPRO(
            **common, projection_budget_margin=0.0, **target
        )
        correction = (
            all_high_correction
            if method == "k2_all_high"
            else x0_high_correction
        )
        adaptive = None
    elif method in {"k2_online_compensator", "k2_row_accounts"}:
        allocator = SoftTargetDAPRO(
            **common,
            projection_budget_margin=0.0,
            terminal_pi_min=None,
            **target,
        )
        correction = online_identity_correction
        adaptive = (
            online_compensator_allocation
            if method == "k2_online_compensator"
            else row_account_allocation
        )
    else:
        raise ValueError(method)

    set_seeds(seed)
    uniforms = np.random.default_rng(
        seed if acquisition_seed is None else acquisition_seed
    ).random(
        (len(t_cal), conditional_grid.shape[1])
    )
    allocator.set_acquisition_randomness(seed=seed, uniforms=uniforms)
    with patched(correction, adaptive):
        result = allocator.allocate_budget(
            prediction.probability_est, None, t_cal, prediction.quantile_est
        )
    return result


def _result_row(
    *,
    setup_name: str,
    task: str,
    method: str,
    seed: int,
    result,
    t_cal: torch.Tensor,
    quantile_cal: torch.Tensor,
    t_test: torch.Tensor,
    quantile_test: torch.Tensor,
    max_time: int,
    budget: float,
    target_tau: float,
    acquisition_rep: int,
) -> dict:
    n = len(t_cal)
    propensity = result.C_probs.reshape(-1).to(torch.float64)
    c = result.C.reshape(-1).to(torch.long)
    observed_event = (t_cal <= max_time) & (c >= t_cal)
    target_a = (t_cal <= max_time).to(torch.float64)
    row = {
        "setup": setup_name,
        "task": task,
        "method": method,
        "seed": seed,
        "acquisition_rep": acquisition_rep,
        "budget_target": budget,
        "realized_cost": float(result.total_budget_used) / n,
        "minimum_propensity": float(propensity.min().item()),
        "maximum_weight": float(propensity.reciprocal().max().item()),
    }
    if method in {"k2_online_compensator", "k2_row_accounts"}:
        phase1_cost = float(result.total_budget_used) - ONLINE_STATE.phase2_realized_cost
        expected_total = phase1_cost + ONLINE_STATE.phase2_compensator
        row.update({
            "conditional_expected_cost": expected_total / n,
            "guaranteed_upper_cost": (
                phase1_cost + ONLINE_STATE.phase2_budget
            ) / n,
            "controller_valid": int(
                ONLINE_STATE.phase2_compensator
                <= ONLINE_STATE.phase2_budget + 1e-7
            ),
            "controller_mean_abs_shift": ONLINE_STATE.mean_abs_shift,
            "controller_max_abs_shift": ONLINE_STATE.max_abs_shift,
            "controller_fraction_changed": (
                ONLINE_STATE.fraction_conditionals_changed
            ),
            "controller_mean_abs_log_distortion": (
                ONLINE_STATE.mean_abs_log_conditional_distortion
            ),
        })
    else:
        metrics = result.additional_metrics or {}
        row.update({
            "conditional_expected_cost": float(
                metrics.get("total_expected_budget_per_sample", np.nan)
            ),
            "guaranteed_upper_cost": float(
                metrics.get(
                    "k2_safe_deployment_upper_bound_cost_per_sample",
                    np.nan,
                )
            ),
            "controller_valid": int(
                metrics.get(
                    "k2_safe_valid",
                    metrics.get("total_expected_budget_valid", 0),
                )
            ),
            "controller_mean_abs_shift": np.nan,
            "controller_max_abs_shift": np.nan,
            "controller_fraction_changed": float(
                metrics.get(
                    "k2_safe_fraction_cumulative_reach_changed", np.nan
                )
            ),
            "controller_mean_abs_log_distortion": float(
                metrics.get(
                    "k2_safe_mean_abs_log_cumulative_distortion", np.nan
                )
            ),
        })

    if task == "metric":
        contribution = (
            observed_event.to(torch.float64) / propensity
        )
        row["estimate_pct"] = 100 * float(contribution.mean().item())
        row["truth_pct"] = 100 * float(target_a.mean().item())
        # Exact for policies frozen before Phase II.  For the cross-row online
        # policy this is only a logged-path plug-in diagnostic; its across-run
        # estimator variance remains the primary comparison.
        row["acquisition_variance_pp2"] = (
            10_000
            * float((target_a * (propensity.reciprocal() - 1)).sum().item())
            / n**2
        )
        row["estimated_acquisition_variance_pp2"] = (
            10_000
            * float((
                observed_event.to(torch.float64)
                * (1 - propensity)
                / propensity.square()
            ).sum().item())
            / n**2
        )
    else:
        f = quantile_cal
        estimable = (
            (t_cal.reshape(-1, 1) < f)
            & (f <= c.reshape(-1, 1))
        )
        miscoverage = (
            estimable.to(torch.float64)
            / propensity.reshape(-1, 1)
        ).mean(0)
        position = select_calibration_positions(
            miscoverage,
            torch.tensor(
                [target_tau], dtype=torch.float64, device=f.device
            ),
        )
        selected = quantile_test[:, position].squeeze()
        coverage = (t_test >= selected).to(torch.float64).mean()
        row.update({
            "coverage_pct": 100 * float(coverage.item()),
            "bound_size": float(selected.to(torch.float64).mean().item()),
            "selected_position": int(position.reshape(-1)[0].item()),
        })
    return row


def run_setup(
    setup_name: str,
    setup: Setup,
    seeds: range,
    tasks: tuple[str, ...],
    methods: tuple[str, ...],
    *,
    cal_size: int,
    n1: int,
    device: str,
    output: Path,
    online_reps: int,
) -> pd.DataFrame:
    torch_device = torch.device(device)
    taus = torch.tensor(
        np.arange(0.01, 1.0, 0.01),
        dtype=torch.float32,
        device=torch_device,
    )
    max_time, t_all, q_all, p_all, grid_all, test_size = setup_experiment_data(
        cal_size,
        True,
        torch_device,
        setup.dataset_name,
        setup.data_setup,
        taus,
        200,
    )
    rows = []
    for seed in seeds:
        (
            _, _, t_cal, p_cal, q_cal, t_test, q_test, _, cal_idx, _,
        ) = split_data(
            seed, cal_size, test_size, None, t_all, p_all, q_all
        )
        grid = grid_all[cal_idx]
        q_cal = q_cal.clip(max=max_time)
        q_test = q_test.clip(max=max_time)
        prediction = SurvivalModelPrediction(q_cal, p_cal)
        for task in tasks:
            task_q_cal = q_cal.clone()
            task_q_test = q_test.clone()
            if task == "metric":
                task_q_cal[:] = max_time
                task_q_test[:] = max_time
                prediction = SurvivalModelPrediction(task_q_cal, p_cal)
            else:
                prediction = SurvivalModelPrediction(task_q_cal, p_cal)
            for method in methods:
                repetitions = (
                    online_reps
                    if method == "k2_online_compensator"
                    else 1
                )
                for acquisition_rep in range(repetitions):
                    if acquisition_rep == 0:
                        result = _allocate(
                            method,
                            grid,
                            setup.budget,
                            taus,
                            0.56,
                            max_time,
                            n1,
                            task,
                            t_cal,
                            prediction,
                            seed,
                            acquisition_seed=(1_000_003 * seed),
                        )
                    else:
                        # Reuse the exactly same learned K2 table and frozen
                        # Phase-I spend profile.  Only acquisition randomness
                        # changes, making within-split design variance cheap
                        # to estimate and avoiding accidental policy refits.
                        set_seeds(seed)
                        permutation = np.random.permutation(len(t_cal))
                        val_idxs = permutation[:n1]
                        test_idxs = permutation[n1:]
                        prior_q = get_prior(
                            prediction.quantile_est, taus, 0.56
                        )
                        val_prior = prior_q[val_idxs]
                        raw_p = ONLINE_STATE.deployment_probabilities
                        internal_q = ONLINE_STATE.deployment_prior_q
                        internal_t = ONLINE_STATE.deployment_event_times
                        if raw_p is None or internal_q is None or internal_t is None:
                            raise RuntimeError("missing cached online policy")
                        uniforms = torch.as_tensor(
                            np.random.default_rng(
                                1_000_003 * seed + acquisition_rep
                            ).random(raw_p.shape),
                            dtype=raw_p.dtype,
                            device=raw_p.device,
                        )
                        test_c, phase2_used, test_propensity = (
                            online_compensator_allocation(
                                raw_p,
                                internal_q,
                                internal_t,
                                max_time,
                                raw_p.device,
                                uniforms=uniforms,
                            )
                        )
                        final_c, final_propensity = construct_final_result(
                            len(t_cal),
                            val_idxs,
                            val_prior,
                            test_idxs,
                            internal_q,
                            test_c,
                            test_propensity,
                            raw_p.device,
                        )
                        phase1_used = torch.minimum(
                            t_cal[val_idxs], val_prior
                        ).sum().item()
                        result = BudgetAllocationResult(
                            prediction.quantile_est,
                            final_c,
                            final_propensity,
                            phase1_used + phase2_used,
                            additional_metrics={},
                        )
                    rows.append(_result_row(
                        setup_name=setup_name,
                        task=task,
                        method=method,
                        seed=seed,
                        result=result,
                        t_cal=t_cal,
                        quantile_cal=task_q_cal,
                        t_test=t_test,
                        quantile_test=task_q_test,
                        max_time=max_time,
                        budget=setup.budget,
                        target_tau=0.10,
                        acquisition_rep=acquisition_rep,
                    ))
                    output.parent.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(rows).to_csv(output, index=False)
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    # Acquisition replicate zero is the matched across-split comparison.
    primary = frame[frame["acquisition_rep"] == 0].copy()
    rows = []
    for keys, group in primary.groupby(["setup", "task", "method"]):
        all_reps = frame[
            (frame["setup"] == keys[0])
            & (frame["task"] == keys[1])
            & (frame["method"] == keys[2])
        ]
        row = {
            "setup": keys[0],
            "task": keys[1],
            "method": keys[2],
            "splits": len(group),
            "mean_realized_cost": group["realized_cost"].mean(),
            "mean_conditional_expected_cost": group[
                "conditional_expected_cost"
            ].mean(),
            "expected_budget_valid_rate": (
                group["conditional_expected_cost"]
                <= group["budget_target"] + 1e-7
            ).mean(),
            "controller_valid_rate": group["controller_valid"].mean(),
            "median_minimum_propensity": group[
                "minimum_propensity"
            ].median(),
            "median_maximum_weight": group["maximum_weight"].median(),
            "mean_fraction_policy_values_changed": group[
                "controller_fraction_changed"
            ].mean(),
            "mean_abs_log_policy_distortion": group[
                "controller_mean_abs_log_distortion"
            ].mean(),
        }
        if keys[1] == "metric":
            row.update({
                "estimate_variance_pp2": group["estimate_pct"].var(ddof=1),
                "estimate_mse_pp2": np.mean(
                    (group["estimate_pct"] - group["truth_pct"]) ** 2
                ),
                "mean_logged_acquisition_variance_pp2": group[
                    "acquisition_variance_pp2"
                ].mean(),
                "mean_design_unbiased_acquisition_variance_pp2": all_reps[
                    "estimated_acquisition_variance_pp2"
                ].mean(),
                "mean_within_split_acquisition_variance_pp2": (
                    all_reps.groupby("seed")["estimate_pct"]
                    .var(ddof=1).mean()
                    if all_reps["acquisition_rep"].nunique() > 1
                    else np.nan
                ),
            })
        else:
            row.update({
                "coverage_mean_pct": group["coverage_pct"].mean(),
                "coverage_variance_pp2": group["coverage_pct"].var(ddof=1),
                "coverage_mse90_pp2": np.mean(
                    (group["coverage_pct"] - 90) ** 2
                ),
                "mean_bound_size": group["bound_size"].mean(),
                "mean_within_split_coverage_variance_pp2": (
                    all_reps.groupby("seed")["coverage_pct"]
                    .var(ddof=1).mean()
                    if all_reps["acquisition_rep"].nunique() > 1
                    else np.nan
                ),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def _parse_range(value: str) -> range:
    start, stop = map(int, value.split(":"))
    return range(start, stop)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setups", nargs="+", choices=SETUPS, default=[
        "toxicity_qwen", "red_qwen", "hallucination_qwen", "autoif_qwen"
    ])
    parser.add_argument("--tasks", nargs="+", choices=["metric", "lpb"],
                        default=["metric", "lpb"])
    parser.add_argument("--methods", nargs="+", choices=[
        "soft_no_crc", "soft_crc", "k2_all_high",
        "k2_x0_high", "k2_online_compensator", "k2_row_accounts",
    ], default=[
        "soft_no_crc", "soft_crc", "k2_all_high",
        "k2_x0_high", "k2_online_compensator",
    ])
    parser.add_argument("--seeds", type=_parse_range, default=range(0, 20))
    parser.add_argument("--cal-size", type=int, default=3000)
    parser.add_argument("--n1", type=int, default=50)
    parser.add_argument(
        "--online-reps", type=int, default=1,
        help="Acquisition-randomness replicates per split for the online method.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    all_frames = []
    for name in args.setups:
        path = args.output_dir / f"{name}_rows.csv"
        frame = run_setup(
            name,
            SETUPS[name],
            args.seeds,
            tuple(args.tasks),
            tuple(args.methods),
            cal_size=args.cal_size,
            n1=args.n1,
            device=args.device,
            output=path,
            online_reps=args.online_reps,
        )
        all_frames.append(frame)
    combined = pd.concat(all_frames, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output_dir / "all_rows.csv", index=False)
    summary = summarize(combined)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
