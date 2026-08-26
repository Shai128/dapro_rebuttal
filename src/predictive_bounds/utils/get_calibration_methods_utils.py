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
    SoftTargetCRCDAPRO,
    SoftTargetCRCUPBDAPRO,
    SoftTargetDAPRO,
    SoftTargetUPBDAPRO,
    SoftPrefixEndpointUPBDAPRO,
    SoftPrefixEndpointCRCUPBDAPRO,
    InformationGainDAPRO,
    InformationGainCRCDAPRO,
    ResidualDAPRO,
    ResidualCRCDAPRO,
    InformationGainUPBDAPRO,
    InformationGainCRCUPBDAPRO,
    ResidualUPBDAPRO,
    ResidualCRCUPBDAPRO,
    TargetAWeightedDAPRO,
)
from src.predictive_bounds.budget_allocators.endpoint_residual_allocator import (
    EndpointResidualAHTAllocator,
)
from src.predictive_bounds.budget_allocators.random_adaptive_optimized_allocator import (
    ConstantCRCBudgetAllocator,
    RandomAdaptiveOptimizedBudgetAllocator,
)
from src.predictive_bounds.budget_allocators.trimmed_allocator import TrimmedBudgetAllocator
from src.predictive_bounds.calibration.abstract_calibration import (
    SurvivalLPBCalibration,
    SurvivalUPBCalibration,
)
from src.predictive_bounds.calibration.dummy_calibration import (
    UncalibratedLPBSurvivalCalibration,
    UncalibratedUPBSurvivalCalibration,
)
from src.predictive_bounds.calibration.oracle_survival_calibration import (
    OracleSurvivalCalibration,
    OracleSurvivalUPBCalibration,
)
from src.predictive_bounds.calibration.survival_calibration_with_known_weights import SurvivalCalibrationWithKnownWeights
from src.predictive_bounds.calibration.survival_upb_calibration_with_known_weights import (
    SurvivalUPBCalibrationWithKnownWeights,
)
from src.predictive_bounds.reward_functions.reward_by_probability import RewardByProbability
from src.predictive_bounds.reward_functions.reward_by_probability_diff import RewardByProbabilityDiff
from src.predictive_bounds.utils.get_best_params_utils import get_best_rexp3_params, get_best_discounted_ucb_params, \
    new_alg_best_params
from src.predictive_bounds.budget_allocators.uniform_allocator import UniformBudgetAllocator, UnweightedUniformBudgetAllocator
from src.predictive_bounds.budget_allocators.full_budget_oracle_allocator import (
    FullBudgetOracleAllocator,
    SplitFullBudgetOracleAllocator,
)
from src.predictive_bounds.budget_allocators.metric_optimal_allocator import (
    MetricOptimalPMFAllocator,
    MetricPrefixNeymanCRCAllocator,
    MetricOptimalPooledTimeAllocator,
)
from src.predictive_bounds.budget_allocators.dapro_ablation import (
    AblationHardTargetCRCDAPRO,
    AblationHardTargetDAPRO,
    AblationSoftTargetCRCDAPRO,
    AblationSoftTargetDAPRO,
)
from src.predictive_bounds.budget_allocators.uncalibrated_metric_allocator import (
    UncalibratedMetricAllocator,
)
from typing import List


# One paper-wide Generalized-DAPRO specification.  Registries should vary
# these values only in an ablation whose x-axis explicitly names that value.
# Keeping the constants here (rather than relying on constructor defaults)
# makes LPB, UPB, metric estimation, construction, and merge registries agree.
CANONICAL_DAPRO_SCORE_BIN_COUNT = 2
CANONICAL_DAPRO_GLOBAL_REGULARIZATION = 0.001
CANONICAL_DAPRO_PROJECTION_BUDGET_MARGIN = 0.0
CANONICAL_DAPRO_CRC_ROW_COST_CAP_MULTIPLIER = 2.0
CANONICAL_DAPRO_SCORE_KIND = "hazard"


def get_dapro_ablation_calibrations(
        conditional_grid,
        budget_per_sample,
        taus_range,
        tau_prior,
        m_upper_bound,
        *,
        ablation_kind: str,
        dapro_n1_values=(50, 100, 200, 300, 400),
        score_noise_lambdas=(0.0, 0.1, 0.25, 0.5, 0.75, 1.0),
        score_noise_seed: int = 314159,
) -> List[SurvivalLPBCalibration]:
    """Return Static plus paired raw/CRC DAPRO LPB ablations.

    The caller controls which configurations coexist in an experiment:
    ``n1`` instantiates every requested Phase-I size, ``score_noise`` every
    requested lambda at the single supplied N1, and ``budget`` one paired
    DAPRO configuration at the single supplied N1.  Static is evaluated once
    and is replicated across x-axis values only by the summarizer.
    """
    kind = str(ablation_kind).lower()
    supported = {
        "n1", "score_noise", "budget", "hard_soft", "representation",
        "score", "attacker_shift",
    }
    if kind not in supported:
        raise ValueError(f"DAPRO ablations support {sorted(supported)}.")
    n1_values = tuple(dict.fromkeys(int(value) for value in dapro_n1_values))
    if not n1_values or any(value < 2 for value in n1_values):
        raise ValueError("DAPRO N1 values must be integers of at least two.")
    lambdas = tuple(dict.fromkeys(float(value) for value in score_noise_lambdas))
    if not lambdas or any(not 0.0 <= value <= 1.0 for value in lambdas):
        raise ValueError("Score-noise lambdas must be distinct values in [0, 1].")
    if kind != "n1" and len(n1_values) != 1:
        raise ValueError(f"The {kind} ablation requires exactly one N1 value.")

    static = OptimizedBudgetAllocator(
        budget_per_sample, taus_range, tau_prior, m_upper_bound
    )
    calibrations: list[SurvivalLPBCalibration] = [
        SurvivalCalibrationWithKnownWeights(static, taus_range, tau_prior),
    ]
    common = dict(
        conditional_grid=conditional_grid,
        budget_per_sample=budget_per_sample,
        taus_range=taus_range,
        tau_prior=tau_prior,
        m_upper_bound=m_upper_bound,
        target_alpha=0.10,
        score_bin_count=CANONICAL_DAPRO_SCORE_BIN_COUNT,
    )
    configurations: list[dict]
    if kind == "n1":
        configurations = [
            dict(n1=n1, value=float(n1), label=f"N1={n1}")
            for n1 in n1_values
        ]
    elif kind == "score_noise":
        configurations = [
            dict(
                n1=n1_values[0], value=lam, label=f"lambda={lam:g}",
                noise=lam,
            )
            for lam in lambdas
        ]
    elif kind == "budget":
        configurations = [
            dict(
                n1=n1_values[0], value=float(budget_per_sample),
                label=f"B={budget_per_sample:g}",
            )
        ]
    elif kind == "hard_soft":
        configurations = [
            dict(n1=n1_values[0], value=0.0, label="Hard", hard=True),
            dict(n1=n1_values[0], value=1.0, label="Soft", hard=False),
        ]
    elif kind == "representation":
        configurations = [
            dict(n1=n1_values[0], value=float(k), label=f"K={k}", bins=k)
            for k in (1, 2, 4, 8)
        ] + [
            dict(
                n1=n1_values[0], value=9.0, label="Continuous",
                bins=4, smooth=True,
            )
        ]
    elif kind == "score":
        configurations = [
            dict(n1=n1_values[0], value=float(index), label=label,
                 score_kind=score_kind,
                 bins=CANONICAL_DAPRO_SCORE_BIN_COUNT)
            for index, (score_kind, label) in enumerate([
                ("hazard", "Current hazard"),
                ("remaining_quantile", "Remaining-time quantile"),
                ("target_value", "Causal target value"),
                ("random", "Random"),
                ("oracle_remaining_time", "Oracle remaining time"),
            ])
        ]
    else:  # attacker_shift
        configurations = [
            dict(
                n1=n1_values[0], value=0.0,
                label="Source calibration -> shifted test",
            )
        ]

    for configuration in configurations:
        n1 = int(configuration["n1"])
        allocator_common = {
            **common,
            "n1": n1,
            "score_bin_count": int(configuration.get(
                "bins", CANONICAL_DAPRO_SCORE_BIN_COUNT
            )),
            "smooth_score_rank_map": bool(configuration.get("smooth", False)),
            "global_regularization": float(
                configuration.get(
                    "global_regularization",
                    CANONICAL_DAPRO_GLOBAL_REGULARIZATION,
                )
            ),
            "ablation_kind": kind,
            "ablation_value": float(configuration["value"]),
            "ablation_label": str(configuration["label"]),
            "score_kind": str(configuration.get(
                "score_kind", CANONICAL_DAPRO_SCORE_KIND
            )),
            "score_noise_lambda": float(configuration.get("noise", 0.0)),
            "score_noise_seed": score_noise_seed,
        }
        hard = bool(configuration.get("hard", False))
        raw_class = AblationHardTargetDAPRO if hard else AblationSoftTargetDAPRO
        crc_class = (
            AblationHardTargetCRCDAPRO
            if hard else AblationSoftTargetCRCDAPRO
        )
        raw = raw_class(
            **allocator_common,
            projection_budget_margin=(
                CANONICAL_DAPRO_PROJECTION_BUDGET_MARGIN
            ),
        )
        crc_kwargs = dict(allocator_common)
        # Hard and soft CRC wrappers expose the same public control size.
        crc = crc_class(
            **crc_kwargs,
            budget_control_size=n1 // 2,
        )
        calibrations.extend([
            SurvivalCalibrationWithKnownWeights(raw, taus_range, tau_prior),
            SurvivalCalibrationWithKnownWeights(crc, taus_range, tau_prior),
        ])
    return calibrations


def get_metric_dapro_ablation_allocators(
        conditional_grid,
        budget_per_sample,
        taus_range,
        tau_prior,
        m_upper_bound,
        *,
        ablation_kind: str,
        dapro_n1: int = 50,
        crc_control_size: int = 25,
        score_noise_lambdas=(0.0, 0.1, 0.25, 0.5, 0.75, 1.0),
        score_noise_seed: int = 314159,
) -> List[BudgetAllocator]:
    """Return Static plus paired raw/CRC metric score ablations.

    These allocators optimize the event-rate target
    ``A_i = 1{T_i <= m_upper_bound}``.  Both named-score and score-noise
    studies retain the paper-wide two-bin representation, so the score is the
    only DAPRO component changed along either x-axis.
    """
    kind = str(ablation_kind).lower()
    if kind not in {"score_noise", "score", "hard_soft"}:
        raise ValueError(
            "Metric DAPRO ablations support 'score_noise', 'score', and "
            "'hard_soft'."
        )
    dapro_n1 = int(dapro_n1)
    crc_control_size = int(crc_control_size)
    if dapro_n1 < 2:
        raise ValueError("`dapro_n1` must be at least two.")
    if not 0 < crc_control_size < dapro_n1:
        raise ValueError("CRC requires 0 < crc_control_size < dapro_n1.")
    lambdas = tuple(dict.fromkeys(float(value) for value in score_noise_lambdas))
    if not lambdas or any(not 0.0 <= value <= 1.0 for value in lambdas):
        raise ValueError("Score-noise lambdas must lie in [0, 1].")

    allocators: list[BudgetAllocator] = [
        OptimizedBudgetAllocator(
            budget_per_sample, taus_range, tau_prior, m_upper_bound
        )
    ]
    if kind == "score_noise":
        configurations = [
            dict(
                value=lam,
                label=f"lambda={lam:g}",
                score_kind="hazard",
                noise=lam,
                bins=CANONICAL_DAPRO_SCORE_BIN_COUNT,
            )
            for lam in lambdas
        ]
    elif kind == "score":
        configurations = [
            dict(value=float(index), label=label, score_kind=score_kind,
                 noise=0.0, bins=CANONICAL_DAPRO_SCORE_BIN_COUNT)
            for index, (score_kind, label) in enumerate([
                ("hazard", "Current hazard"),
                ("remaining_quantile", "Remaining-time quantile"),
                ("target_value", "Causal event-rate target value"),
                ("random", "Random"),
                ("oracle_remaining_time", "Oracle remaining time"),
            ])
        ]
    else:
        configurations = [
            dict(
                value=0.0, label="Hard terminal", score_kind="hazard",
                noise=0.0, bins=CANONICAL_DAPRO_SCORE_BIN_COUNT, hard=True,
            ),
            dict(
                value=1.0, label="Soft prefix", score_kind="hazard",
                noise=0.0, bins=CANONICAL_DAPRO_SCORE_BIN_COUNT, hard=False,
            ),
        ]

    common = dict(
        conditional_grid=conditional_grid,
        budget_per_sample=budget_per_sample,
        taus_range=taus_range,
        tau_prior=tau_prior,
        m_upper_bound=m_upper_bound,
        n1=dapro_n1,
        target_alpha=0.10,
        metric_estimation_horizon=m_upper_bound,
        global_regularization=CANONICAL_DAPRO_GLOBAL_REGULARIZATION,
    )
    for configuration in configurations:
        ablation = dict(
            ablation_kind=kind,
            ablation_value=float(configuration["value"]),
            ablation_label=str(configuration["label"]),
            score_kind=str(configuration["score_kind"]),
            score_noise_lambda=float(configuration["noise"]),
            score_noise_seed=score_noise_seed,
            score_bin_count=int(configuration["bins"]),
        )
        hard = bool(configuration.get("hard", False))
        raw_class = AblationHardTargetDAPRO if hard else AblationSoftTargetDAPRO
        crc_class = (
            AblationHardTargetCRCDAPRO
            if hard else AblationSoftTargetCRCDAPRO
        )
        allocators.extend([
            raw_class(
                **common,
                **ablation,
                projection_budget_margin=(
                    CANONICAL_DAPRO_PROJECTION_BUDGET_MARGIN
                ),
            ),
            crc_class(
                **common,
                **ablation,
                budget_control_size=crc_control_size,
            ),
        ])
    return allocators


def get_unified_bound_calibrations(
        conditional_grid,
        budget_per_sample,
        taus_range,
        tau_prior,
        m_upper_bound,
        *,
        bound_type: str,
        dapro_n1_values=(200, 100, 50),
        target_coverages=(0.90,),
) -> List[SurvivalLPBCalibration | SurvivalUPBCalibration]:
    """Registry for the paper's unified LPB/UPB AHT comparison.

    Every learned schedule is included both as a raw, zero-margin projection
    and with an independent CRC controller.  The raw variants do not reserve
    or assume a worst-case projection-error ``eta``.
    """
    if bound_type not in {"lpb", "upb"}:
        raise ValueError("`bound_type` must be 'lpb' or 'upb'.")
    n1_values = tuple(dict.fromkeys(int(value) for value in dapro_n1_values))
    if not n1_values or any(value < 2 for value in n1_values):
        raise ValueError("DAPRO N1 values must be integers of at least two.")

    static = OptimizedBudgetAllocator(
        budget_per_sample, taus_range, tau_prior, m_upper_bound
    )
    if bound_type == "lpb":
        calibrations: list = [
            UncalibratedLPBSurvivalCalibration(taus_range, tau_prior),
            OracleSurvivalCalibration(taus_range, tau_prior),
            SurvivalCalibrationWithKnownWeights(static, taus_range, tau_prior),
        ]
        target_alpha = 0.10
        common = dict(
            conditional_grid=conditional_grid,
            budget_per_sample=budget_per_sample,
            taus_range=taus_range,
            tau_prior=tau_prior,
            m_upper_bound=m_upper_bound,
            target_alpha=target_alpha,
            score_bin_count=CANONICAL_DAPRO_SCORE_BIN_COUNT,
            global_regularization=CANONICAL_DAPRO_GLOBAL_REGULARIZATION,
        )
        for n1 in n1_values:
            control = n1 // 2
            allocations = [
                SoftTargetDAPRO(
                    **common,
                    n1=n1,
                    projection_budget_margin=(
                        CANONICAL_DAPRO_PROJECTION_BUDGET_MARGIN
                    ),
                ),
                SoftTargetCRCDAPRO(
                    **common,
                    n1=n1,
                    budget_control_size=control,
                    row_cost_cap_multiplier=(
                        CANONICAL_DAPRO_CRC_ROW_COST_CAP_MULTIPLIER
                    ),
                ),
                # InformationGainDAPRO(
                #     **common, n1=n1, projection_budget_margin=0.0
                # ),
                # InformationGainCRCDAPRO(
                #     **common, n1=n1, budget_control_size=control
                # ),
                # ResidualDAPRO(
                #     **common, n1=n1, projection_budget_margin=0.0
                # ),
                # ResidualCRCDAPRO(
                #     **common, n1=n1, budget_control_size=control
                # ),
            ]
            calibrations.extend(
                SurvivalCalibrationWithKnownWeights(
                    allocation, taus_range, tau_prior
                ) for allocation in allocations
            )
        # endpoint_allocations = [EndpointResidualAHTAllocator(
        #     conditional_grid,
        #     budget_per_sample,
        #     taus_range,
        #     tau_prior,
        #     m_upper_bound,
        #     target_kind="lpb",
        #     target_alpha=target_alpha,
        # )]
        # endpoint_allocations.extend(
        #     EndpointResidualAHTAllocator(
        #         conditional_grid,
        #         budget_per_sample,
        #         taus_range,
        #         tau_prior,
        #         m_upper_bound,
        #         target_kind="lpb",
        #         target_alpha=target_alpha,
        #         crc_control_size=n1 // 2,
        #     ) for n1 in n1_values
        # )
        # calibrations.extend(
        #     SurvivalCalibrationWithKnownWeights(
        #         allocation, taus_range, tau_prior
        #     ) for allocation in endpoint_allocations
        # )
        return calibrations

    calibrations = [
        UncalibratedUPBSurvivalCalibration(taus_range, tau_prior),
        OracleSurvivalUPBCalibration(taus_range, tau_prior),
        SurvivalUPBCalibrationWithKnownWeights(static, taus_range, tau_prior),
    ]
    for coverage in target_coverages:
        for n1 in n1_values:
            control = n1 // 2
            common = dict(
                conditional_grid=conditional_grid,
                budget_per_sample=budget_per_sample,
                taus_range=taus_range,
                tau_prior=tau_prior,
                m_upper_bound=m_upper_bound,
                target_coverage=float(coverage),
                n1=n1,
                score_bin_count=CANONICAL_DAPRO_SCORE_BIN_COUNT,
                global_regularization=CANONICAL_DAPRO_GLOBAL_REGULARIZATION,
            )
            allocations = [
                SoftPrefixEndpointUPBDAPRO(
                    **common,
                    projection_budget_margin=(
                        CANONICAL_DAPRO_PROJECTION_BUDGET_MARGIN
                    ),
                ),
                SoftPrefixEndpointCRCUPBDAPRO(
                    **common,
                    budget_control_size=control,
                    row_cost_cap_multiplier=(
                        CANONICAL_DAPRO_CRC_ROW_COST_CAP_MULTIPLIER
                    ),
                ),
                # InformationGainUPBDAPRO(
                #     **common, projection_budget_margin=0.0
                # ),
                # InformationGainCRCUPBDAPRO(
                #     **common, budget_control_size=control
                # ),
                # ResidualUPBDAPRO(
                #     **common, projection_budget_margin=0.0
                # ),
                # ResidualCRCUPBDAPRO(
                #     **common, budget_control_size=control
                # ),
            ]
            calibrations.extend(
                SurvivalUPBCalibrationWithKnownWeights(
                    allocation, taus_range, tau_prior
                ) for allocation in allocations
            )
        # endpoint = [SoftTargetUPBDAPRO(
        #     conditional_grid,
        #     budget_per_sample,
        #     taus_range,
        #     tau_prior,
        #     m_upper_bound,
        #     n1=max(n1_values),
        #     target_coverage=float(coverage),
        #     projection_budget_margin=0.0,
        # )]
        # endpoint.extend(
        #     SoftTargetCRCUPBDAPRO(
        #         conditional_grid,
        #         budget_per_sample,
        #         taus_range,
        #         tau_prior,
        #         m_upper_bound,
        #         n1=n1,
        #         budget_control_size=n1 // 2,
        #         target_coverage=float(coverage),
        #     ) for n1 in n1_values
        # )
        # calibrations.extend(
        #     SurvivalUPBCalibrationWithKnownWeights(
        #         allocation, taus_range, tau_prior
        #     ) for allocation in endpoint
        # )
    return calibrations


def get_upb_calibrations(
        conditional_grid,
        budget_per_sample,
        taus_range,
        tau_prior,
        m_upper_bound,
        *,
        dapro_n1_values=(200, 100, 50),
        projection_budget_margin: float = (
            CANONICAL_DAPRO_PROJECTION_BUDGET_MARGIN
        ),
        target_coverage: float = 0.70,
) -> List[SurvivalUPBCalibration]:
    """Return the shared UPB construction/merge registry.

    The paper comparison retains Static, Constant+CRC, and the power-schedule
    baseline.  Each exposes candidate-specific reach propensities required by
    UPB calibration.  The only DAPRO family instantiated is canonical
    history-adaptive soft-prefix Generalized DAPRO, with a raw zero-margin
    controller or the same independent capped-CRC controller used elsewhere.
    """
    reach = {"reach_t_max_is_success": True}
    allocations: List[BudgetAllocator] = [
        OptimizedBudgetAllocator(
            budget_per_sample, taus_range, tau_prior, m_upper_bound
        ),
        ConstantCRCBudgetAllocator(
            conditional_grid,
            budget_per_sample,
            taus_range,
            tau_prior,
            m_upper_bound,
            **reach,
        ),
        RandomAdaptiveOptimizedBudgetAllocator(
            conditional_grid,
            budget_per_sample,
            taus_range,
            tau_prior,
            m_upper_bound,
            schedule_family="power_reach",
            schedule_alpha=2.0,
            budget_control_mode="crc",
            **reach,
        ),
    ]
    # Keep the legacy registry aligned with the paper suite: UPB changes the
    # target event, but retains the same history-adaptive hazard/K2 policy,
    # regularization, raw zero-margin controller, and capped CRC controller.
    for requested_n1 in tuple(dict.fromkeys(dapro_n1_values)):
        common = dict(
            conditional_grid=conditional_grid,
            budget_per_sample=budget_per_sample,
            taus_range=taus_range,
            tau_prior=tau_prior,
            m_upper_bound=m_upper_bound,
            n1=int(requested_n1),
            target_coverage=target_coverage,
            score_bin_count=CANONICAL_DAPRO_SCORE_BIN_COUNT,
            global_regularization=CANONICAL_DAPRO_GLOBAL_REGULARIZATION,
        )
        allocations.append(SoftPrefixEndpointUPBDAPRO(
            **common,
            projection_budget_margin=projection_budget_margin,
        ))
        if requested_n1 >= 2:
            allocations.append(SoftPrefixEndpointCRCUPBDAPRO(
                **common,
                budget_control_size=int(requested_n1) // 2,
                row_cost_cap_multiplier=(
                    CANONICAL_DAPRO_CRC_ROW_COST_CAP_MULTIPLIER
                ),
            ))
    return [
        UncalibratedUPBSurvivalCalibration(taus_range, tau_prior),
        OracleSurvivalUPBCalibration(taus_range, tau_prior),
        *[
            SurvivalUPBCalibrationWithKnownWeights(
                allocation, taus_range, tau_prior
            )
            for allocation in allocations
        ],
    ]


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
        definitive_dapro_margins=(0.0,),
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
                all_allocations.append(SoftTargetDAPRO(
                    conditional_grid,
                    budget_per_sample,
                    taus_range,
                    tau_prior,
                    m_upper_bound,
                    n1=n1,
                    projection_budget_margin=margin,
                ))
            if n1 >= 50:
                all_allocations.append(SoftTargetCRCDAPRO(
                    conditional_grid,
                    budget_per_sample,
                    taus_range,
                    tau_prior,
                    m_upper_bound,
                    n1=n1,
                    budget_control_size=min(100, n1 // 2),
                    row_cost_cap_multiplier=2.0,
                ))
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

    dummy_calibration = UncalibratedLPBSurvivalCalibration(
        taus_range, tau_prior
    )
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
        include_dapro_comparison=False,
        method_suite="legacy",
        dapro_ablation_kind="score_noise",
        score_noise_lambdas=(0.0, 0.1, 0.25, 0.5, 0.75, 1.0),
        score_noise_seed=314159,
) -> List[BudgetAllocator]:
    """Return the exact server/local metric-estimation comparison.

    The ``unified_aht`` paper suite contains the initial-PMF Uncalibrated
    plug-in, Static, soft-prefix Generalized DAPRO with raw zero-margin
    probabilities, the same DAPRO schedule with an independent CRC
    controller, and the split full-budget Oracle.  The legacy registry remains
    available for historical experiments.
    """
    del (
        device,
        t_tilde_cal,
        cal_model_prediction,
        include_legacy_dapro,
        include_locally_adaptive,
    )
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
    target = dict(metric_estimation_horizon=m_upper_bound)

    if method_suite == "dapro_ablation":
        return get_metric_dapro_ablation_allocators(
            conditional_grid,
            budget_per_sample,
            taus_range,
            tau_prior,
            m_upper_bound,
            ablation_kind=dapro_ablation_kind,
            dapro_n1=dapro_n1,
            crc_control_size=crc_control_size,
            score_noise_lambdas=score_noise_lambdas,
            score_noise_seed=score_noise_seed,
        )

    if method_suite == "unified_aht":
        allocations = [
            UncalibratedMetricAllocator(
                conditional_grid,
                taus_range,
                tau_prior,
                m_upper_bound,
            ),
            OptimizedBudgetAllocator(
                budget_per_sample, taus_range, tau_prior, m_upper_bound
            ),
            SoftTargetDAPRO(
                **common,
                n1=dapro_n1,
                score_bin_count=CANONICAL_DAPRO_SCORE_BIN_COUNT,
                global_regularization=CANONICAL_DAPRO_GLOBAL_REGULARIZATION,
                projection_budget_margin=(
                    CANONICAL_DAPRO_PROJECTION_BUDGET_MARGIN
                ),
                **target,
            ),
            SoftTargetCRCDAPRO(
                **common,
                n1=dapro_n1,
                budget_control_size=crc_control_size,
                score_bin_count=CANONICAL_DAPRO_SCORE_BIN_COUNT,
                global_regularization=CANONICAL_DAPRO_GLOBAL_REGULARIZATION,
                row_cost_cap_multiplier=(
                    CANONICAL_DAPRO_CRC_ROW_COST_CAP_MULTIPLIER
                ),
                **target,
            ),
            # InformationGainDAPRO(
            #     **common,
            #     n1=dapro_n1,
            #     projection_budget_margin=0.0,
            #     **target,
            # ),
            # InformationGainCRCDAPRO(
            #     **common,
            #     n1=dapro_n1,
            #     budget_control_size=crc_control_size,
            #     **target,
            # ),
            # ResidualDAPRO(
            #     **common,
            #     n1=dapro_n1,
            #     projection_budget_margin=0.0,
            #     **target,
            # ),
            # ResidualCRCDAPRO(
            #     **common,
            #     n1=dapro_n1,
            #     budget_control_size=crc_control_size,
            #     **target,
            # ),
            # EndpointResidualAHTAllocator(
            #     conditional_grid,
            #     budget_per_sample,
            #     taus_range,
            #     tau_prior,
            #     m_upper_bound,
            #     target_kind="metric",
            # ),
            # EndpointResidualAHTAllocator(
            #     conditional_grid,
            #     budget_per_sample,
            #     taus_range,
            #     tau_prior,
            #     m_upper_bound,
            #     target_kind="metric",
            #     crc_control_size=crc_control_size,
            # ),
            SplitFullBudgetOracleAllocator(
                taus_range, tau_prior, m_upper_bound
            ),
        ]
        return allocations
    if method_suite != "legacy":
        raise ValueError(
            "`method_suite` must be 'legacy', 'unified_aht', or "
            "'dapro_ablation'."
        )

    allocators = [
        # Intentional duplicate allocation with and without IPCW correction.
        UniformBudgetAllocator(
            budget_per_sample, taus_range, tau_prior, m_upper_bound
        ),
        UnweightedUniformBudgetAllocator(
            budget_per_sample, taus_range, tau_prior, m_upper_bound
        ),
        # Keep the original naive Static allocation and its event-stopped
        # budget underuse exactly as defined by the baseline.
        OptimizedBudgetAllocator(
            budget_per_sample, taus_range, tau_prior, m_upper_bound
        ),
        # Constant continuation with CRC. It has no policy-fit fold, so the
        # requested N1//2 CRC size is its fully observed control size.
        # ConstantCRCBudgetAllocator(
        #     **common,
        #     phase1_size=crc_control_size,
        # ),
        # No-split, model-budget, initial-PMF cumulative-reach optimum.
        # MetricOptimalPMFAllocator(
        #     budget_per_sample,
        #     taus_range,
        #     tau_prior,
        #     m_upper_bound,
        # ),
        # # The same variance objective restricted to one shared, time-varying
        # # cumulative-reach schedule.  This is a simple O(NM), no-split,
        # # model-budget alternative to DAPRO.
        # MetricOptimalPooledTimeAllocator(
        #     budget_per_sample,
        #     taus_range,
        #     tau_prior,
        #     m_upper_bound,
        # ),
        # A closed-form current-prefix value/cost index.  CRC selects only one
        # global scale; no score bins or DAPRO coordinate optimization are fit.
        # MetricPrefixNeymanCRCAllocator(
        #     conditional_grid,
        #     budget_per_sample,
        #     taus_range,
        #     tau_prior,
        #     m_upper_bound,
        #     control_size=crc_control_size,
        #     row_cost_cap_multiplier=2.0,
        # ),
        # Generalized DAPRO: identical soft-prefix metric objective with an
        # assumption-based projection controller or an independent CRC fold.
        SoftTargetDAPRO(
            **common,
            n1=dapro_n1,
            score_bin_count=CANONICAL_DAPRO_SCORE_BIN_COUNT,
            global_regularization=CANONICAL_DAPRO_GLOBAL_REGULARIZATION,
            projection_budget_margin=(
                CANONICAL_DAPRO_PROJECTION_BUDGET_MARGIN
            ),
            **target,
        ),
        SoftTargetCRCDAPRO(
            **common,
            n1=dapro_n1,
            budget_control_size=crc_control_size,
            score_bin_count=CANONICAL_DAPRO_SCORE_BIN_COUNT,
            global_regularization=CANONICAL_DAPRO_GLOBAL_REGULARIZATION,
            row_cost_cap_multiplier=(
                CANONICAL_DAPRO_CRC_ROW_COST_CAP_MULTIPLIER
            ),
            **target,
        ),
        # Infinite budget on the random calibration side only.
        SplitFullBudgetOracleAllocator(
            taus_range, tau_prior, m_upper_bound
        ),
        # Fixed truth over the complete calibration+test union.
        FullBudgetOracleAllocator(taus_range, tau_prior, m_upper_bound),
    ]
    if include_dapro_comparison:
        comparison = [
            TargetAWeightedDAPRO(
                **common,
                n1=dapro_n1,
                projection="direct_bins_2",
                score="prob",
                anchor_kind="raw_alpha",
                target_alpha=0.10,
                metric_estimation_horizon=m_upper_bound,
            ),
            DefinitiveDAPRO(
                **common,
                n1=dapro_n1,
                metric_estimation_horizon=m_upper_bound,
                projection_budget_margin=1.0,
            ),
            TargetAWeightedDAPRO(
                **common,
                n1=dapro_n1,
                projection="direct_bins_2",
                score="prob",
                anchor_kind="raw_alpha",
                target_alpha=0.10,
                metric_estimation_horizon=m_upper_bound,
                projection_budget_margin=0.0,
                budget_control_mode="crc",
                budget_control_size=crc_control_size,
                risk_candidate_row_cost_cap=(
                    2.0 * budget_per_sample
                ),
            ),
            DefinitiveCRCDAPRO(
                **common,
                n1=dapro_n1,
                metric_estimation_horizon=m_upper_bound,
                budget_control_size=crc_control_size,
                row_cost_cap_multiplier=2.0,
            ),
        ]
        # Soft projection/CRC variants are already in the main registry.
        allocators[7:7] = comparison[:2]
        allocators[-2:-2] = comparison[2:]
    return allocators
