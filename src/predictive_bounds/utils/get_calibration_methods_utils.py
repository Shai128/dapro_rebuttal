import numpy as np
import torch

from src.predictive_bounds.bandit_algorithms.discounted_ucb import DiscountedUCB
from src.predictive_bounds.bandit_algorithms.new_bandits_algorithm import NewBanditsAlgorithm
from src.predictive_bounds.bandit_algorithms.rexp3 import Rexp3
from src.predictive_bounds.budget_allocators.adaptive_optimized_allocator import AdaptiveOptimizedBudgetAllocator
from src.predictive_bounds.budget_allocators.bandits_allocator import BanditsBudgetAllocator
from src.predictive_bounds.budget_allocators.basic_allocator import BasicBudgetAllocator
from src.predictive_bounds.budget_allocators.budget_allocator import BudgetAllocator
from src.predictive_bounds.budget_allocators.naive_allocator import NaiveBudgetAllocator
from src.predictive_bounds.budget_allocators.optimized_allocators import OptimizedBudgetAllocator
from src.predictive_bounds.budget_allocators.oracle_dapro_allocator import (
    CRCOracleTargetADAPRO,
    GlobalOracleTargetADAPRO,
    SplitOracleTargetADAPRO,
)
from src.predictive_bounds.budget_allocators.DAPRO import (
    AWeightedDAPRO,
    BandRegularizedTargetAWeightedDAPRO,
    DAPRO,
    DefinitiveCRCDAPRO,
    DefinitiveCRCUPBDAPRO,
    DefinitiveDAPRO,
    LegacyMeanWeightDAPRO,
    RandomAnchoredTargetAWeightedDAPRO,
    RegularizedTargetAWeightedDAPRO,
    RobustTargetAWeightedDAPRO,
    TargetAWeightedDAPRO,
)
from src.predictive_bounds.budget_allocators.random_adaptive_optimized_allocator import (
    ConstantCRCBudgetAllocator,
    RandomAdaptiveOptimizedBudgetAllocator,
)
from src.predictive_bounds.budget_allocators.trimmed_allocator import TrimmedBudgetAllocator
from src.predictive_bounds.calibration.abstract_calibration import SurvivalLPBCalibration
from src.predictive_bounds.calibration.dummy_calibration import UncalibratedLPBSurvivalCalibration
from src.predictive_bounds.calibration.survival_calibration_with_known_weights import SurvivalCalibrationWithKnownWeights
from src.predictive_bounds.reward_functions.reward_by_probability import RewardByProbability
from src.predictive_bounds.reward_functions.reward_by_probability_diff import RewardByProbabilityDiff
from src.predictive_bounds.utils.get_best_params_utils import get_best_rexp3_params, get_best_discounted_ucb_params, \
    new_alg_best_params
from src.predictive_bounds.budget_allocators.uniform_allocator import UniformBudgetAllocator, UnweightedUniformBudgetAllocator
from src.predictive_bounds.budget_allocators.full_budget_oracle_allocator import FullBudgetOracleAllocator
from typing import List


def is_budget_sufficient_for_split(N, n1, total_budget, censored_event_time, prior_q):
    if n1 > N:
        return False
    perm = np.random.permutation(N)
    val_idxs = perm[:n1]

    t_val = censored_event_time[val_idxs]
    val_prior_q = prior_q[val_idxs]
    val_budget_used = torch.minimum(t_val, val_prior_q).sum().item()

    return total_budget > val_budget_used


def get_baseline_calibrations(
        conditional_grid,
        budget_per_sample,
        taus_range,
        tau_prior,
        m_upper_bound,
        include_a_weighted=True,
        evaluate_dapro_projection=False,
        dapro_n1_values=(200,100),
        definitive_dapro_margins=(1.0,),
):
    basic_allocation = BasicBudgetAllocator(budget_per_sample, taus_range, tau_prior)
    trimmed_allocation = TrimmedBudgetAllocator(budget_per_sample, taus_range, tau_prior, m_upper_bound)
    optimized_allocation = OptimizedBudgetAllocator(budget_per_sample, taus_range, tau_prior, m_upper_bound)
    adaptive_allocation = AdaptiveOptimizedBudgetAllocator(conditional_grid, budget_per_sample, taus_range, tau_prior, m_upper_bound)
    adaptive_mixture_floor_allocation = AdaptiveOptimizedBudgetAllocator(
        conditional_grid,
        budget_per_sample,
        taus_range,
        tau_prior,
        m_upper_bound,
        terminal_pi_min=1.0 / float(m_upper_bound),
        terminal_floor_mode="mixture",
    )
    adaptive_no_floor_allocation = AdaptiveOptimizedBudgetAllocator(
        conditional_grid,
        budget_per_sample,
        taus_range,
        tau_prior,
        m_upper_bound,
        terminal_pi_min=None,
        terminal_floor_mode="none",
    )
    adaptive_crc_allocation = AdaptiveOptimizedBudgetAllocator(
        conditional_grid,
        budget_per_sample,
        taus_range,
        tau_prior,
        m_upper_bound,
        budget_control_mode="crc",
    )

    all_allocations: List[BudgetAllocator] = [
        basic_allocation,
        trimmed_allocation,
        optimized_allocation,
        adaptive_allocation,
        adaptive_mixture_floor_allocation,
        adaptive_no_floor_allocation,
        adaptive_crc_allocation,
    ]
    for n1 in dapro_n1_values:
        if include_a_weighted:
            for margin in definitive_dapro_margins:
                all_allocations.append(DefinitiveDAPRO(
                    conditional_grid,
                    budget_per_sample,
                    taus_range,
                    tau_prior,
                    m_upper_bound,
                    n1=n1,
                    projection_budget_margin=margin,
                ))
            if n1 >= 50:
                all_allocations.append(DefinitiveCRCDAPRO(
                    conditional_grid,
                    budget_per_sample,
                    taus_range,
                    tau_prior,
                    m_upper_bound,
                    n1=n1,
                    budget_control_size=min(100, n1 // 2),
                    row_cost_cap_multiplier=None,
                ))
                for row_cost_cap_multiplier in [1.0, 2.0]:
                    all_allocations.append(DefinitiveCRCDAPRO(
                        conditional_grid,
                        budget_per_sample,
                        taus_range,
                        tau_prior,
                        m_upper_bound,
                        n1=n1,
                        budget_control_size=min(100, n1 // 2),
                        row_cost_cap_multiplier=row_cost_cap_multiplier,
                    ))
        elif n1 >= 100:
            all_allocations.append(DefinitiveCRCUPBDAPRO(
                conditional_grid,
                budget_per_sample,
                taus_range,
                tau_prior,
                m_upper_bound,
                n1=n1,
                budget_control_size=min(100, n1 // 2),
                row_cost_cap_multiplier=2.0,
            ))
    if include_a_weighted:
        for n1 in dapro_n1_values:
            all_allocations.append(SplitOracleTargetADAPRO(
                conditional_grid,
                budget_per_sample,
                taus_range,
                tau_prior,
                m_upper_bound,
                n1=n1,
            ))
            if n1 >= 2:
                all_allocations.append(CRCOracleTargetADAPRO(
                    conditional_grid,
                    budget_per_sample,
                    taus_range,
                    tau_prior,
                    m_upper_bound,
                    n1=n1,
                    budget_control_size=min(100, n1 // 2),
                ))
        all_allocations.append(GlobalOracleTargetADAPRO(
            conditional_grid,
            budget_per_sample,
            taus_range,
            tau_prior,
            m_upper_bound,
        ))
    all_allocations.extend([
        RandomAdaptiveOptimizedBudgetAllocator(
            conditional_grid,
            budget_per_sample,
            taus_range,
            tau_prior,
            m_upper_bound,
        ),
        RandomAdaptiveOptimizedBudgetAllocator(
            conditional_grid,
            budget_per_sample,
            taus_range,
            tau_prior,
            m_upper_bound,
            terminal_pi_min=1.0 / float(m_upper_bound),
            terminal_floor_mode="hard",
        ),
        RandomAdaptiveOptimizedBudgetAllocator(
            conditional_grid,
            budget_per_sample,
            taus_range,
            tau_prior,
            m_upper_bound,
            terminal_pi_min=None,
            terminal_floor_mode="none",
        ),
        ConstantCRCBudgetAllocator(
            conditional_grid,
            budget_per_sample,
            taus_range,
            tau_prior,
            m_upper_bound,
        ),
        RandomAdaptiveOptimizedBudgetAllocator(
            conditional_grid,
            budget_per_sample,
            taus_range,
            tau_prior,
            m_upper_bound,
            terminal_pi_min=1.0 / float(m_upper_bound),
            terminal_floor_mode="hard",
            budget_control_mode="crc",
        ),
        RandomAdaptiveOptimizedBudgetAllocator(
            conditional_grid,
            budget_per_sample,
            taus_range,
            tau_prior,
            m_upper_bound,
            budget_control_mode="crc",
        ),
    ])
    for schedule_family in ["complement_power", "power_reach"]:
        for schedule_alpha in [0.5, 1.0, 2.0]:
            all_allocations.append(
                RandomAdaptiveOptimizedBudgetAllocator(
                    conditional_grid,
                    budget_per_sample,
                    taus_range,
                    tau_prior,
                    m_upper_bound,
                    schedule_family=schedule_family,
                    schedule_alpha=schedule_alpha,
                    budget_control_mode="crc",
                )
            )
    for projection in [
        'platt',
        'cumulative_platt',
        'direct_time',
        'direct_bins_2',
        'direct_bins_4',
    ]:
        for n1 in dapro_n1_values:
            all_allocations.append(
                LegacyMeanWeightDAPRO(
                    conditional_grid,
                    budget_per_sample,
                    taus_range,
                    tau_prior,
                    m_upper_bound,
                    projection=projection,
                    score='prob',
                    n1=n1,
                    evaluate_projection=evaluate_dapro_projection,
                )
            )
            if not include_a_weighted:
                continue
            if projection == "direct_bins_2" and n1 >= 50:
                control_size = min(100, n1 // 2)
                crc_kwargs = {
                    "budget_control_mode": "crc",
                    "budget_control_size": control_size,
                    "risk_candidate_row_cost_cap": 2.0 * budget_per_sample,
                }
                all_allocations.extend([
                    LegacyMeanWeightDAPRO(
                        conditional_grid,
                        budget_per_sample,
                        taus_range,
                        tau_prior,
                        m_upper_bound,
                        projection=projection,
                        score="prob",
                        n1=n1,
                        evaluate_projection=evaluate_dapro_projection,
                        **crc_kwargs,
                    ),
                    TargetAWeightedDAPRO(
                        conditional_grid,
                        budget_per_sample,
                        taus_range,
                        tau_prior,
                        m_upper_bound,
                        projection=projection,
                        score="prob",
                        n1=n1,
                        anchor_kind="raw_alpha",
                        target_alpha=0.10,
                        **crc_kwargs,
                    ),
                ])
            all_allocations.append(
                AWeightedDAPRO(
                    conditional_grid,
                    budget_per_sample,
                    taus_range,
                    tau_prior,
                    m_upper_bound,
                    projection=projection,
                    score='prob',
                    n1=n1,
                )
            )
            for anchor_kind in ["raw_alpha", "phase1_unweighted"]:
                all_allocations.append(
                    TargetAWeightedDAPRO(
                        conditional_grid,
                        budget_per_sample,
                        taus_range,
                        tau_prior,
                        m_upper_bound,
                        projection=projection,
                        score='prob',
                        n1=n1,
                        anchor_kind=anchor_kind,
                        target_alpha=0.10,
                    )
                )
            if projection == "direct_time":
                if n1 >= 100:
                    for target_policy_fraction in [
                        0.00,
                        0.25,
                        0.50,
                        0.75,
                        1.00,
                    ]:
                        all_allocations.append(
                            RandomAnchoredTargetAWeightedDAPRO(
                                conditional_grid,
                                budget_per_sample,
                                taus_range,
                                tau_prior,
                                m_upper_bound,
                                projection=projection,
                                score='prob',
                                n1=n1,
                                target_alpha=0.10,
                                target_policy_fraction=(
                                    target_policy_fraction
                                ),
                                budget_control_mode="crc",
                                budget_control_size=n1 // 2,
                            )
                        )
                        if target_policy_fraction in {0.50, 0.75}:
                            all_allocations.append(
                                RandomAnchoredTargetAWeightedDAPRO(
                                    conditional_grid,
                                    budget_per_sample,
                                    taus_range,
                                    tau_prior,
                                    m_upper_bound,
                                    projection=projection,
                                    score='prob',
                                    n1=n1,
                                    target_alpha=0.10,
                                    target_policy_fraction=(
                                        target_policy_fraction
                                    ),
                                    fill_random_slack=True,
                                    budget_control_mode="crc",
                                    budget_control_size=n1 // 2,
                                )
                            )
                for robustness_weight in [0.10, 0.50]:
                    all_allocations.append(
                        RobustTargetAWeightedDAPRO(
                            conditional_grid,
                            budget_per_sample,
                            taus_range,
                            tau_prior,
                            m_upper_bound,
                            projection=projection,
                            score='prob',
                            n1=n1,
                            target_alpha=0.10,
                            robustness_weight=robustness_weight,
                        )
                    )
                for global_regularization in [0.001, 0.01, 0.05]:
                    all_allocations.append(
                        RegularizedTargetAWeightedDAPRO(
                            conditional_grid,
                            budget_per_sample,
                            taus_range,
                            tau_prior,
                            m_upper_bound,
                            projection=projection,
                            score='prob',
                            n1=n1,
                            target_alpha=0.10,
                            global_regularization=global_regularization,
                        )
                    )
                    all_allocations.append(
                        RegularizedTargetAWeightedDAPRO(
                            conditional_grid,
                            budget_per_sample,
                            taus_range,
                            tau_prior,
                            m_upper_bound,
                            projection=projection,
                            score='prob',
                            n1=n1,
                            anchor_kind="phase1_unweighted",
                            target_alpha=0.10,
                            global_regularization=global_regularization,
                        )
                    )
                all_allocations.append(
                    BandRegularizedTargetAWeightedDAPRO(
                        conditional_grid,
                        budget_per_sample,
                        taus_range,
                        tau_prior,
                        m_upper_bound,
                        projection=projection,
                        score='prob',
                        n1=n1,
                        target_alphas=tuple(
                            0.07 + 0.01 * offset
                            for offset in range(7)
                        ),
                        global_regularization=0.01,
                    )
                )
    # for projection in ['platt', 'beta']:
    #     for score in ['prob', 'quantile']:
    #         all_allocations.append(DAPRO(conditional_grid,
    #                                      budget_per_sample, taus_range,
    #                                      tau_prior, m_upper_bound,
    #                                      projection=projection, score=score))
    #         for n1 in [25, 50, 100, 150, 200, 250, 300, 400, 500, 750, 1000, 1250, 1500]:
    #             all_allocations.append(DAPRO(conditional_grid,
    #                                                    budget_per_sample, taus_range,
    #                                                    tau_prior, m_upper_bound,
    #                                                    projection=projection, score=score,
    #                                                    n1=n1))
    #         for score_error_lambda in list(np.arange(0, 1, 0.1)) + [0.95, 0.99]:
    #             score_error_lambda = np.round(score_error_lambda, 2)
    #             all_allocations.append(ProjectedOptimizationBudgetAllocatorScoreError(conditional_grid,
    #                                                    budget_per_sample, taus_range,
    #                                                    tau_prior, m_upper_bound,
    #                                                    projection=projection, score=score,
    #                                                    score_error_lambda=score_error_lambda))

    dummy_calibration = UncalibratedLPBSurvivalCalibration(taus_range)
    all_calibrations: List[SurvivalLPBCalibration] = [dummy_calibration]
    all_calibrations.extend([SurvivalCalibrationWithKnownWeights(allocation, taus_range, tau_prior) for
                             allocation in all_allocations])

    return all_calibrations


def get_new_allocation_algorithms(conditional_grid, budget_per_sample, taus_range, tau_prior, m_upper_bound, allocations, device):
    all_allocations = []
    do_diffs = [False, True]
    if allocations == 'none':
        return []
    elif allocations == 'all':
        k_values = [1,2, 3,4, 5, 10, 50, 5000]
        first_step_budget_values = [0.01, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.75, 0.95]
    elif allocations == 'one':
        k,b = new_alg_best_params()
        k_values = [k]
        first_step_budget_values = [b, 0.95]
        # first_step_budget_values = [b]
    else:
        raise Exception("not valid allocations")
    for do_diff in do_diffs:
        for k in k_values:
            for b in first_step_budget_values:
                reward_function = RewardByProbability(conditional_grid)
                bandits_algorithm = NewBanditsAlgorithm(reward_function, device=device, k=k, n_monte_carlo=1)
                new_allocation = BanditsBudgetAllocator(budget_per_sample, taus_range, tau_prior, m_upper_bound,
                 first_step_budget_ratio=b, bandits_algorithm=bandits_algorithm, do_diff=do_diff)
                # new_allocation = NewBudgetAllocator(budget_per_sample, taus_range, tau_prior, m_upper_bound,
                #                                     k=k, first_step_budget_ratio=b, do_diff=do_diff)
                all_allocations.append(new_allocation)
    return all_allocations


def get_discounted_ucb_allocation_algorithms(budget_per_sample, taus_range, tau_prior, m_upper_bound, allocations, device):
    all_allocations = []
    do_diffs = [False, True]
    if allocations == 'none':
        return []
    elif allocations == 'all':
        xi_values = [0.01, 0.05, 0.1, 0.2, 0.5, 0.75, 0.9, 1.]
        gamma_values = [0.01, 0.05, 0.1, 0.2, 0.5, 0.75, 0.9, 1.]
        first_step_budget_values = [ 0.1, 0.2, ]
    elif allocations == 'one':
        xi, gamma, b = get_best_discounted_ucb_params()
        xi_values = [xi]
        gamma_values = [gamma]
        first_step_budget_values = [b]
    else:
        raise Exception("not valid allocations")
    for do_diff in do_diffs:
        for gamma in gamma_values:
            for xi in xi_values:
                for b in first_step_budget_values:
                    reward_function = RewardByProbabilityDiff(None)
                    bandits_algorithm = DiscountedUCB(reward_function, gamma=gamma, xi=xi, device=device)
                    new_allocation = BanditsBudgetAllocator(budget_per_sample, taus_range, tau_prior, m_upper_bound,
                                                            first_step_budget_ratio=b,
                                                            bandits_algorithm=bandits_algorithm, do_diff=do_diff)
                    all_allocations.append(new_allocation)
    return all_allocations


def get_bandits_allocation_algorithms(budget_per_sample, taus_range, tau_prior, m_upper_bound, allocations, device):
    all_allocations = []
    do_diffs = [False, True]
    # return []
    if allocations == 'none':
        return []
    elif allocations == 'all':
        first_step_budget_values = [0.1, 0.2]
        gamma_values = [0., 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.75, 0.8, 0.95, 1.]
        delta_values = [0., 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    elif allocations == 'one':
        g,d, rb = get_best_rexp3_params()
        first_step_budget_values = [rb]
        gamma_values = [g]
        delta_values = [d]
    else:
        raise Exception("not valid allocations")
    for do_diff in do_diffs:
        for gamma in gamma_values:
            for delta in delta_values:
                for b in first_step_budget_values:
                    reward_function = RewardByProbabilityDiff(None)
                    bandits_algorithm = Rexp3(reward_function, gamma=gamma, delta_t=delta, device=device)
                    new_allocation = BanditsBudgetAllocator(budget_per_sample, taus_range, tau_prior, m_upper_bound,
                                                            first_step_budget_ratio=b,
                                                            bandits_algorithm=bandits_algorithm, do_diff=do_diff)
                    all_allocations.append(new_allocation)

    return all_allocations

def get_metric_allocators(
        conditional_grid,
        budget_per_sample,
        m_upper_bound,
        taus_range,
        tau_prior,
        device,
        t_tilde_cal=None,
        cal_model_prediction=None,
        *,
        dapro_n1=200,
        crc_control_size=100,
        include_legacy_dapro=True,
        include_locally_adaptive=True,
) -> List[BudgetAllocator]:
    """Return the fixed-benchmark metric-estimation comparison.

    Target-A methods use ``A_i = 1{T_i <= m_upper_bound}``, the binary event
    whose population mean is estimated by the unsafe-event-rate estimator.
    This differs intentionally from the LPB construction target anchored at a
    model quantile.  ``device`` and the optional observed-data arguments are
    retained for compatibility with older callers.
    """
    del device, t_tilde_cal, cal_model_prediction
    if dapro_n1 <= 0:
        raise ValueError("`dapro_n1` must be positive.")
    if not 0 < crc_control_size < dapro_n1:
        raise ValueError("CRC requires 0 < crc_control_size < dapro_n1.")

    common = dict(
        conditional_grid=conditional_grid,
        budget_per_sample=budget_per_sample,
        taus_range=taus_range,
        tau_prior=tau_prior,
        m_upper_bound=m_upper_bound,
    )
    crc = dict(
        budget_control_mode="crc",
        budget_control_size=crc_control_size,
        risk_candidate_row_cost_cap=2.0 * budget_per_sample,
    )
    target = dict(metric_estimation_horizon=m_upper_bound)

    allocators = [
        # Uniform sampling, with and without inverse-probability correction.
        UniformBudgetAllocator(
            budget_per_sample, taus_range, tau_prior, m_upper_bound
        ),
        UnweightedUniformBudgetAllocator(
            budget_per_sample, taus_range, tau_prior, m_upper_bound
        ),
        # The repeated, deterministic reference is evaluated on the complete
        # fixed benchmark rather than on each calibration subsample.
        FullBudgetOracleAllocator(taus_range, tau_prior, m_upper_bound),
        OptimizedBudgetAllocator(
            budget_per_sample, taus_range, tau_prior, m_upper_bound
        ),
        ConstantCRCBudgetAllocator(**common, phase1_size=100),
        # DAPRO: legacy, metric-aligned Target-A, and definitive objective.
        LegacyMeanWeightDAPRO(
            **common,
            projection="direct_bins_2",
            score="prob",
            n1=dapro_n1,
        ),
        TargetAWeightedDAPRO(
            **common,
            projection="direct_bins_2",
            score="prob",
            n1=dapro_n1,
            anchor_kind="raw_alpha",
            target_alpha=0.10,
            **target,
        ),
        DefinitiveDAPRO(
            **common,
            n1=dapro_n1,
            projection_budget_margin=1.0,
            **target,
        ),
        LegacyMeanWeightDAPRO(
            **common,
            projection="direct_bins_2",
            score="prob",
            n1=dapro_n1,
            **crc,
        ),
        TargetAWeightedDAPRO(
            **common,
            projection="direct_bins_2",
            score="prob",
            n1=dapro_n1,
            anchor_kind="raw_alpha",
            target_alpha=0.10,
            **target,
            **crc,
        ),
        DefinitiveCRCDAPRO(
            **common,
            n1=dapro_n1,
            budget_control_size=crc_control_size,
            row_cost_cap_multiplier=2.0,
            **target,
        ),
        SplitOracleTargetADAPRO(
            **common,
            n1=dapro_n1,
            **target,
        ),
        CRCOracleTargetADAPRO(
            **common,
            n1=dapro_n1,
            budget_control_size=crc_control_size,
            **target,
        ),
        GlobalOracleTargetADAPRO(
            **common,
            **target,
        ),
    ]
    if not include_legacy_dapro:
        allocators = [
            allocator
            for allocator in allocators
            if type(allocator) is not LegacyMeanWeightDAPRO
        ]
    if not include_locally_adaptive:
        allocators = [
            allocator
            for allocator in allocators
            if type(allocator) is not AdaptiveOptimizedBudgetAllocator
        ]
    return allocators
