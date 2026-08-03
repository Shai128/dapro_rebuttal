import numpy as np

from src.predictive_bounds.budget_allocators.budget_allocator import (
    BudgetAllocationResult,
    BudgetAllocator,
    summarize_expected_budget,
)
from src.predictive_bounds.budget_allocators.dapro_projection_metrics import (
    compute_dapro_projection_metrics,
)
from src.predictive_bounds.budget_allocators.optimization_solver_utils import (
    solve_binned_deployable_policy,
    solve_exact_fast,
    solve_time_only_cumulative_policy,
)
from src.predictive_bounds.budget_allocators.projected_optimization_utils import adaptive_budget_allocation, \
    construct_final_result, split_to_two_sets, project_to_test_platt, project_to_test_ir, project_to_test_beta, \
    correct_projected_probabilities_to_budget, expected_acquisition_cost, \
    project_cumulative_probabilities_to_test_platt, \
    correct_projected_cumulative_probabilities_to_budget
from src.predictive_bounds.budget_allocators.risk_controlled_budget import (
    affine_cumulative_policy_family,
    cumulative_policy_costs,
    select_crc_budget_candidate,
    select_hoeffding_budget_candidate,
    solve_constant_continuation_policy,
)
from src.predictive_bounds.calibration.calibration_utils import (
    get_prior,
    quantiles_to_interaction_counts,
    select_calibration_positions,
)

import torch

from src.predictive_bounds.survival_utils.compute_mean_time_given_pmf import compute_quantile_survival_time


class LegacyMeanWeightDAPRO(BudgetAllocator):
    """Original mean-inverse-propensity DAPRO retained for ablations only."""

    def __init__(self, conditional_grid, budget_per_sample, taus_range, tau_prior, m_upper_bound, projection: str, score: str,
                 reach_t_max_is_success:bool=False, n1: int = 100,
                 evaluate_projection: bool = False,
                 terminal_pi_min: float | None = 0.005,
                 budget_control_mode: str | None = None,
                 budget_control_size: int = 0,
                 budget_control_delta: float = 0.05,
                 budget_candidate_count: int = 401,
                 risk_candidate_row_cost_cap: float | None = None,
                 random_anchor_target_fraction: float | None = None,
                 random_anchor_fill_slack: bool = False,
                 projection_budget_margin: float = 0.0):
        super().__init__(budget_per_sample, taus_range, tau_prior)
        self.conditional_grid = conditional_grid
        self.min_pi = 0.005
        self.projection = projection
        self.score = score
        self.n1 = n1
        self.reach_t_max_is_success = reach_t_max_is_success
        self.evaluate_projection = evaluate_projection
        self.terminal_pi_min = terminal_pi_min
        if budget_control_mode not in {None, "crc", "hoeffding"}:
            raise ValueError(
                "`budget_control_mode` must be one of: None, crc, hoeffding."
            )
        if budget_control_mode is None:
            if budget_control_size != 0:
                raise ValueError(
                    "`budget_control_size` must be zero when budget control "
                    "is disabled."
                )
        elif not 0 < budget_control_size < n1:
            raise ValueError(
                "Risk-controlled DAPRO requires an independent budget-control "
                "fold with 0 < budget_control_size < n1."
            )
        if not 0 < budget_control_delta < 1:
            raise ValueError("`budget_control_delta` must lie in (0, 1).")
        if budget_candidate_count < 2:
            raise ValueError("`budget_candidate_count` must be at least two.")
        if (
                risk_candidate_row_cost_cap is not None
                and (
                    not np.isfinite(risk_candidate_row_cost_cap)
                    or risk_candidate_row_cost_cap <= 0
                )
        ):
            raise ValueError(
                "`risk_candidate_row_cost_cap` must be finite and positive."
            )
        if (
                risk_candidate_row_cost_cap is not None
                and budget_control_mode is None
        ):
            raise ValueError(
                "A risk-candidate row cap is meaningful only with independent "
                "budget control."
            )
        if (
                not np.isfinite(projection_budget_margin)
                or projection_budget_margin < 0
        ):
            raise ValueError(
                "`projection_budget_margin` must be finite and nonnegative."
            )
        if budget_control_mode is not None and projection_budget_margin != 0:
            raise ValueError(
                "Use either independent risk budget control or a projection "
                "budget margin, not both."
            )
        if (
                random_anchor_target_fraction is not None
                and not 0 <= random_anchor_target_fraction <= 1
        ):
            raise ValueError(
                "`random_anchor_target_fraction` must lie in [0, 1]."
            )
        if random_anchor_fill_slack and random_anchor_target_fraction is None:
            raise ValueError(
                "Random slack filling requires a Random-anchored policy."
            )
        risk_compatible_projections = {
            "direct_time",
            "direct_bins_2",
            "direct_bins_4",
        }
        if (
                (budget_control_mode is not None
                 or random_anchor_target_fraction is not None)
                and projection not in risk_compatible_projections
        ):
            raise ValueError(
                "Risk control and Random anchoring require a direct-time or "
                "direct-score-bin cumulative policy."
            )
        if (
                random_anchor_target_fraction is not None
                and projection != "direct_time"
        ):
            raise ValueError(
                "Random anchoring currently requires the direct-time "
                "cumulative policy."
            )
        if (
                (budget_control_mode is not None
                 or random_anchor_target_fraction is not None)
                and terminal_pi_min is None
        ):
            raise ValueError(
                "Risk control and Random anchoring require a positive "
                "terminal propensity floor."
            )
        self.budget_control_mode = budget_control_mode
        self.budget_control_size = int(budget_control_size)
        self.budget_control_delta = float(budget_control_delta)
        self.budget_candidate_count = int(budget_candidate_count)
        self.risk_candidate_row_cost_cap = (
            None
            if risk_candidate_row_cost_cap is None
            else float(risk_candidate_row_cost_cap)
        )
        self.projection_budget_margin = float(projection_budget_margin)
        self.random_anchor_target_fraction = (
            None
            if random_anchor_target_fraction is None
            else float(random_anchor_target_fraction)
        )
        self.random_anchor_fill_slack = bool(random_anchor_fill_slack)
        self.use_a_weighted_objective = False
        if projection not in [
            'ir',
            'platt',
            'beta',
            'cumulative_platt',
            'direct_time',
            'direct_bins_2',
            'direct_bins_4',
        ]:
            raise Exception(f"unknown projection {projection}, must be ir or platt")
        if score not in ['prob', 'quantile']:
            raise Exception(f"unknown projection {score}, must be 'prob', 'quantile'")

    @property
    def name(self) -> str:
        base = f"projected_optimization_{self.projection}_{self.score}"
        base += self.budget_control_name_suffix
        if self.n1 != 100:
            base += f"_n1_{self.n1}"
        return base

    @property
    def budget_control_name_suffix(self) -> str:
        """Distinguish risk-controlled variants from empirical allocators."""
        if self.budget_control_mode is None:
            return ""
        suffix = (
            f"_budget_{self.budget_control_mode}"
            f"_control_{self.budget_control_size}"
        )
        if self.risk_candidate_row_cost_cap is not None:
            multiplier = (
                self.risk_candidate_row_cost_cap / self.budget_per_sample
            )
            formatted = f"{multiplier:.2f}".replace(".", "p")
            suffix += f"_row_cap_{formatted}x_budget"
        return suffix

    @property
    def objective_kind(self) -> str:
        return "mean_inverse_probability"

    def phase1_objective_weights(
            self,
            event_times: torch.Tensor,
            prior_q: torch.Tensor,
            quantile_est: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        return None

    def phase2_objective_weights(
            self,
            event_times: torch.Tensor,
            prior_q: torch.Tensor,
            quantile_est: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Audit the frozen Phase-I objective on the Phase-II population."""
        return self.phase1_objective_weights(
            event_times,
            prior_q,
            quantile_est,
        )

    def phase2_target_indicator(
            self,
            event_times: torch.Tensor,
            prior_q: torch.Tensor,
            quantile_est: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the binary event whose acquisition variance is audited."""
        return (
            event_times.reshape(-1) < prior_q.reshape(-1)
        ).to(torch.float64)

    def objective_metadata(self) -> dict:
        return {}

    def allocate_budget(self, probability_est: torch.Tensor, x: torch.Tensor, t: torch.Tensor,
                        quantile_est: torch.Tensor) -> BudgetAllocationResult:
        device = self.conditional_grid.device
        N, T_max_curr, T_max_future = self.conditional_grid.shape
        prior_q = get_prior(quantile_est, self.taus_range, self.tau_prior)
        if self.score == 'prob':
            scores = self.conditional_grid[:, torch.arange(0, T_max_curr), torch.arange(0, T_max_curr)]
        elif self.score == 'quantile':
            quantile_counts = quantiles_to_interaction_counts(
                compute_quantile_survival_time(
                    self.conditional_grid,
                    quantile=0.9,
                    tail_distribution='geometric',
                ),
                width=T_max_curr,
            )
            scores = quantile_counts.squeeze().reciprocal()
        else:
            assert False
        val_idxs, test_idxs, val_grid, val_prior_q, t_val, val_scores, test_grid, test_prior_q, t_test, test_scores, val_max_steps, \
            target_budget_avg, val_budget_used = split_to_two_sets(self.conditional_grid, prior_q, t, scores,
                                                                   self.budget_per_sample, val_size=self.n1 ,
                                                                   reach_t_max_is_success=self.reach_t_max_is_success)

        val_quantile_est = quantile_est[val_idxs]
        test_quantile_est = quantile_est[test_idxs]
        if self.budget_control_mode is None:
            policy_fit_count = len(t_val)
            phase1_weights = self.phase1_objective_weights(
                t_val,
                val_prior_q,
                val_quantile_est,
            )
            policy_fit_weights = phase1_weights
        else:
            policy_fit_count = len(t_val) - self.budget_control_size
            policy_fit_weights = self.phase1_objective_weights(
                t_val[:policy_fit_count],
                val_prior_q[:policy_fit_count],
                val_quantile_est[:policy_fit_count],
            )
            # The full-fold objective is computed after the policy has been
            # frozen.  This keeps the independent control labels out of both
            # the target-anchor choice and the learned time-policy shape.
            phase1_weights = None
        policy_shape_budget_per_sample = (
            target_budget_avg - self.projection_budget_margin
        )
        if policy_shape_budget_per_sample < 0:
            raise ValueError(
                "The projection-error reserve exceeds the Phase-II budget: "
                f"target={target_budget_avg:.6g}, "
                f"reserve={self.projection_budget_margin:.6g}."
            )
        policy_fit_realized_cost_total = float(
            val_max_steps[:policy_fit_count].sum().item()
        )
        if self.budget_control_mode is not None:
            # The candidate family supplied to CRC/UCB must be fixed before the
            # independent budget-control fold is observed.  In particular, do
            # not use ``target_budget_avg`` here: split_to_two_sets computes it
            # after subtracting the realized cost of *all* Phase-I rows,
            # including the control fold.  Fit the family against the budget
            # remaining after the policy-fit rows only; the selector below then
            # accounts for control-fold costs exactly once.
            remaining_count_after_fit = N - policy_fit_count
            policy_shape_budget_per_sample = (
                self.budget_per_sample * N
                - policy_fit_realized_cost_total
            ) / remaining_count_after_fit
        direct_time_projection = self.projection == 'direct_time'
        direct_bin_count = {
            'direct_bins_2': 2,
            'direct_bins_4': 4,
        }.get(self.projection)
        cumulative_projection = self.projection == 'cumulative_platt'
        if direct_time_projection:
            direct_weights = (
                torch.ones(
                    policy_fit_count,
                    dtype=torch.float64,
                    device=device,
                )
                if policy_fit_weights is None
                else policy_fit_weights
            )
            (
                direct_conditionals,
                direct_cumulative,
                direct_diagnostics,
            ) = solve_time_only_cumulative_policy(
                val_max_steps[:policy_fit_count],
                policy_shape_budget_per_sample,
                direct_weights,
                T_max_curr,
                self.terminal_pi_min,
            )
            base_direct_cumulative = direct_cumulative.copy()
            base_direct_cost = direct_diagnostics[
                "direct_time_expected_cost"
            ]
            random_anchor_metrics = {}
            if self.random_anchor_target_fraction is not None:
                (
                    random_cumulative,
                    random_probability,
                    random_cost,
                ) = solve_constant_continuation_policy(
                    val_max_steps[:policy_fit_count].detach().cpu().numpy(),
                    budget_per_sample=policy_shape_budget_per_sample,
                    time_width=T_max_curr,
                    terminal_pi_min=self.terminal_pi_min,
                )
                target_fraction = self.random_anchor_target_fraction
                direct_cumulative = (
                    target_fraction * direct_cumulative
                    + (1 - target_fraction) * random_cumulative
                )
                fit_lengths_numpy = val_max_steps[
                    :policy_fit_count
                ].detach().cpu().numpy()
                anchor_cost = float(cumulative_policy_costs(
                    direct_cumulative[None, :],
                    fit_lengths_numpy,
                )[:, 0].mean())
                pre_fill_cost = anchor_cost
                upper_envelope_cost = anchor_cost
                slack_fill_coefficient = 0.0
                if self.random_anchor_fill_slack:
                    # Sparse target-A objectives can saturate every weighted
                    # endpoint before exhausting the fit budget.  Spend that
                    # otherwise useless slack only where the stable Random
                    # schedule has greater reach.  The pointwise maximum of
                    # two non-increasing cumulative schedules is itself
                    # non-increasing, and this operation never lowers a
                    # target-focused reach probability.
                    upper_envelope = np.maximum(
                        direct_cumulative,
                        random_cumulative,
                    )
                    upper_envelope_cost = float(cumulative_policy_costs(
                        upper_envelope[None, :],
                        fit_lengths_numpy,
                    )[:, 0].mean())
                    fill_target = min(
                        policy_shape_budget_per_sample,
                        upper_envelope_cost,
                    )
                    fill_denominator = upper_envelope_cost - anchor_cost
                    if (
                            fill_denominator > 1e-12
                            and anchor_cost < fill_target - 1e-12
                    ):
                        slack_fill_coefficient = float(np.clip(
                            (fill_target - anchor_cost) / fill_denominator,
                            0.0,
                            1.0,
                        ))
                        direct_cumulative = (
                            direct_cumulative
                            + slack_fill_coefficient
                            * (upper_envelope - direct_cumulative)
                        )
                        anchor_cost = float(cumulative_policy_costs(
                            direct_cumulative[None, :],
                            fit_lengths_numpy,
                        )[:, 0].mean())
                random_anchor_metrics = {
                    "random_anchor_target_fraction": target_fraction,
                    "random_anchor_constant_probability": random_probability,
                    "random_anchor_reference_expected_cost": random_cost,
                    "random_anchor_target_expected_cost": base_direct_cost,
                    "random_anchor_blended_expected_cost": anchor_cost,
                    "random_anchor_pre_fill_expected_cost": pre_fill_cost,
                    "random_anchor_upper_envelope_expected_cost": (
                        upper_envelope_cost
                    ),
                    "random_anchor_slack_fill_coefficient": (
                        slack_fill_coefficient
                    ),
                    "random_anchor_post_fill_expected_cost": anchor_cost,
                    "random_anchor_fit_budget_target": (
                        policy_shape_budget_per_sample
                    ),
                    "random_anchor_slack_fill_enabled": int(
                        self.random_anchor_fill_slack
                    ),
                }

            risk_control_metrics = {}
            if self.budget_control_mode is not None:
                parameters = np.linspace(
                    0.0,
                    -1.0,
                    self.budget_candidate_count,
                )
                policy_family = affine_cumulative_policy_family(
                    direct_cumulative,
                    parameters,
                    terminal_pi_min=self.terminal_pi_min,
                )
                control_lengths = val_max_steps[
                    policy_fit_count:
                ].detach().cpu().numpy().astype(np.int64)
                candidate_costs = cumulative_policy_costs(
                    policy_family,
                    control_lengths,
                )
                policy_fit_lengths = val_max_steps[
                    :policy_fit_count
                ].detach().cpu().numpy().astype(np.int64)
                total_budget_after_policy_fit = (
                    self.budget_per_sample * N
                    - policy_fit_realized_cost_total
                )
                selector_kwargs = {
                    "expected_costs": candidate_costs,
                    "pilot_costs": control_lengths.astype(np.float64),
                    "total_budget_after_policy_fit": (
                        total_budget_after_policy_fit
                    ),
                    "deployment_sample_count": len(test_idxs),
                    "maximum_cost_per_sample": T_max_curr,
                    "maximum_candidate_cost_per_sample": float(
                        policy_family[0].sum()
                    ),
                    "maximum_pilot_cost_per_sample": T_max_curr,
                }
                if self.budget_control_mode == "crc":
                    selection = select_crc_budget_candidate(
                        **selector_kwargs,
                    )
                else:
                    selection = select_hoeffding_budget_candidate(
                        **selector_kwargs,
                        delta=self.budget_control_delta,
                    )
                direct_cumulative = policy_family[
                    selection.selected_index
                ]
                selected_parameter = float(
                    parameters[selection.selected_index]
                )
                crc_rho = len(control_lengths) / len(test_idxs)
                selected_control_costs = candidate_costs[
                    :,
                    selection.selected_index,
                ]
                crc_empirical_combined_loss = float(
                    selected_control_costs.mean()
                    + crc_rho * control_lengths.mean()
                )
                crc_metrics = {}
                if self.budget_control_mode == "crc":
                    safest_candidate_cost_bound = float(
                        policy_family[-1].sum()
                    )
                    crc_worst_case_selector_left_side = (
                        len(control_lengths)
                        * safest_candidate_cost_bound
                        + crc_rho
                        * len(control_lengths)
                        * T_max_curr
                        + float(policy_family[0].sum())
                        + crc_rho * T_max_curr
                    ) / (len(control_lengths) + 1)
                    crc_metrics = {
                        "risk_budget_crc_rho": crc_rho,
                        (
                            "risk_budget_crc_empirical_combined_loss_"
                            "per_sample"
                        ): crc_empirical_combined_loss,
                        (
                            "risk_budget_crc_composite_target_per_"
                            "deployment_sample"
                        ): selection.deployment_budget_per_sample,
                        "risk_budget_crc_target_semantics": (
                            "remaining_total_budget_after_policy_fit_divided_"
                            "by_deployment_count_for_composite_c_plus_rho_b_"
                            "loss"
                        ),
                        (
                            "risk_budget_crc_worst_case_control_selector_"
                            "left_side"
                        ): crc_worst_case_selector_left_side,
                        (
                            "risk_budget_crc_worst_case_control_feasible_"
                            "conditional_on_fit"
                        ): int(
                            crc_worst_case_selector_left_side
                            <= selection.deployment_budget_per_sample + 1e-12
                        ),
                    }
                risk_control_metrics = {
                    "risk_budget_control_enabled": 1,
                    "risk_budget_control_mode": self.budget_control_mode,
                    "risk_budget_control_size": self.budget_control_size,
                    "risk_budget_policy_fit_size": policy_fit_count,
                    "risk_budget_policy_fit_realized_cost_total": (
                        policy_fit_realized_cost_total
                    ),
                    "risk_budget_policy_shape_target_per_sample": (
                        policy_shape_budget_per_sample
                    ),
                    "risk_budget_candidate_count": (
                        self.budget_candidate_count
                    ),
                    "risk_budget_maximum_candidate_cost_per_sample": (
                        float(policy_family[0].sum())
                    ),
                    "risk_budget_maximum_pilot_cost_per_sample": (
                        T_max_curr
                    ),
                    "risk_budget_selected_index": (
                        selection.selected_index
                    ),
                    "risk_budget_selected_mixture_parameter": (
                        selected_parameter
                    ),
                    "risk_budget_empirical_control_cost_per_sample": (
                        selection.empirical_expected_cost_per_sample
                    ),
                    "risk_budget_control_pilot_cost_total": float(
                        control_lengths.sum()
                    ),
                    "risk_budget_control_pilot_cost_per_sample": float(
                        control_lengths.mean()
                    ),
                    "risk_budget_control_to_deployment_ratio": crc_rho,
                    "risk_budget_deployment_target_per_sample": (
                        selection.deployment_budget_per_sample
                    ),
                    "risk_budget_selector_left_side_per_sample": (
                        selection.selector_left_side_per_sample
                    ),
                    "risk_budget_correction_per_sample": (
                        selection.correction_per_sample
                    ),
                    "risk_budget_guarantee_kind": (
                        selection.guarantee_kind
                    ),
                    "risk_budget_selector_valid": int(
                        selection.selector_left_side_per_sample
                        <= selection.deployment_budget_per_sample + 1e-12
                    ),
                    **crc_metrics,
                }

            previous_cumulative = np.concatenate(
                [[1.0], direct_cumulative[:-1]]
            )
            direct_conditionals = np.clip(
                direct_cumulative
                / np.maximum(
                    previous_cumulative,
                    np.finfo(np.float64).tiny,
                ),
                0.0,
                1.0,
            )
            direct_cost = float(cumulative_policy_costs(
                direct_cumulative[None, :],
                val_max_steps.detach().cpu().numpy(),
            )[:, 0].mean())
            direct_diagnostics = {
                **direct_diagnostics,
                "direct_time_base_policy_fit_expected_cost": (
                    base_direct_cost
                ),
                "direct_time_selected_phase1_expected_cost": direct_cost,
                **random_anchor_metrics,
                **risk_control_metrics,
            }
            optimal_P = np.broadcast_to(
                direct_conditionals,
                (len(val_scores), T_max_curr),
            ).copy()
            direct_tensor = torch.as_tensor(
                direct_conditionals,
                dtype=torch.float64,
                device=device,
            )
            p_val = direct_tensor.unsqueeze(0).repeat(
                len(val_scores),
                1,
            )
            p_test = direct_tensor.unsqueeze(0).repeat(
                len(test_scores),
                1,
            )
            if self.budget_control_mode is not None:
                # The target anchor and policy shape are already frozen; this
                # full-fold value is diagnostic only.
                phase1_weights = self.phase1_objective_weights(
                    t_val,
                    val_prior_q,
                    val_quantile_est,
                )
            budget_correction_metrics = {
                "projection_space": "direct_time_cumulative_probability",
                "projection_raw_base_phase1_expected_cost": (
                    base_direct_cost
                ),
                "projection_pre_intercept_mixed_phase1_expected_cost": (
                    direct_cost
                ),
                "projection_raw_phase1_expected_cost": direct_cost,
                "projection_budget_logit_shift": 0.0,
                "projection_corrected_phase1_expected_cost": direct_cost,
                "projection_budget_boundary": direct_diagnostics[
                    "direct_time_budget_boundary"
                ],
                **direct_diagnostics,
            }
        else:
            bin_diagnostics = {}
            if (
                    direct_bin_count is not None
                    and self.budget_control_mode is not None
            ):
                # Learn the score cutpoints, bin table, and Phase-I budget
                # correction exclusively on the policy-fit fold.  Scores and
                # q_prior are label-free and may then be evaluated on the
                # control/deployment rows; their event-derived active lengths
                # enter only the independent CRC selector below.
                control_count = self.budget_control_size
                remaining_scores = torch.cat(
                    [
                        val_scores[policy_fit_count:],
                        test_scores,
                    ],
                    dim=0,
                )
                remaining_prior_q = torch.cat(
                    [
                        val_prior_q[policy_fit_count:],
                        test_prior_q,
                    ],
                    dim=0,
                )
                (
                    optimal_fit,
                    p_fit_binned,
                    p_remaining_binned,
                    bin_diagnostics,
                ) = solve_binned_deployable_policy(
                    val_scores[:policy_fit_count],
                    remaining_scores,
                    val_max_steps[:policy_fit_count],
                    policy_shape_budget_per_sample,
                    policy_fit_weights,
                    direct_bin_count,
                )
                fit_raw_cumulative = p_fit_binned.cumprod(dim=1)
                remaining_raw_cumulative = (
                    p_remaining_binned.cumprod(dim=1)
                )
                (
                    p_fit_base,
                    p_remaining_base,
                    budget_correction_metrics,
                ) = correct_projected_cumulative_probabilities_to_budget(
                    fit_raw_cumulative,
                    remaining_raw_cumulative,
                    val_max_steps[:policy_fit_count],
                    val_prior_q[:policy_fit_count],
                    remaining_prior_q,
                    policy_shape_budget_per_sample,
                    terminal_pi_min=self.terminal_pi_min,
                )

                row_cap_metrics = {
                    "risk_budget_row_cost_cap_enabled": 0,
                    "risk_budget_row_cost_cap_per_sample": np.nan,
                    "risk_budget_row_cost_cap_fit_changed_fraction": 0.0,
                    "risk_budget_row_cost_cap_remaining_changed_fraction": (
                        0.0
                    ),
                }
                if self.risk_candidate_row_cost_cap is not None:
                    row_cost_cap = self.risk_candidate_row_cost_cap

                    def apply_row_cost_cap(
                            base_conditionals: torch.Tensor,
                            row_prior_q: torch.Tensor,
                    ) -> tuple[torch.Tensor, torch.Tensor]:
                        base_cumulative = base_conditionals.cumprod(dim=1)
                        row_q = row_prior_q.to(
                            device=base_cumulative.device,
                            dtype=torch.long,
                        ).reshape(-1).clamp(min=0, max=T_max_curr)
                        active = (
                            torch.arange(
                                T_max_curr,
                                device=base_cumulative.device,
                            ).unsqueeze(0)
                            < row_q.unsqueeze(1)
                        )
                        raw_cost = (
                            base_cumulative * active.to(torch.float64)
                        ).sum(dim=1)
                        floor_cost = self.terminal_pi_min * row_q.to(
                            torch.float64
                        )
                        if torch.any(floor_cost > row_cost_cap + 1e-10):
                            raise ValueError(
                                "The terminal floor is incompatible with the "
                                "risk-candidate row cost cap."
                            )
                        denominator = (raw_cost - floor_cost).clamp_min(
                            torch.finfo(torch.float64).tiny
                        )
                        coefficient = torch.where(
                            raw_cost > row_cost_cap,
                            (
                                (row_cost_cap - floor_cost) / denominator
                            ).clamp(0.0, 1.0),
                            torch.ones_like(raw_cost),
                        )
                        capped_cumulative = (
                            self.terminal_pi_min
                            + coefficient.unsqueeze(1)
                            * (base_cumulative - self.terminal_pi_min)
                        )
                        capped_cumulative = torch.where(
                            active,
                            capped_cumulative,
                            torch.ones(
                                (),
                                dtype=torch.float64,
                                device=base_cumulative.device,
                            ),
                        )
                        previous = torch.cat(
                            [
                                torch.ones(
                                    (len(base_cumulative), 1),
                                    dtype=torch.float64,
                                    device=base_cumulative.device,
                                ),
                                capped_cumulative[:, :-1],
                            ],
                            dim=1,
                        )
                        capped_conditionals = (
                            capped_cumulative
                            / previous.clamp_min(
                                torch.finfo(torch.float64).tiny
                            )
                        ).clamp(max=1.0)
                        capped_conditionals = torch.where(
                            active,
                            capped_conditionals,
                            torch.ones(
                                (),
                                dtype=torch.float64,
                                device=base_cumulative.device,
                            ),
                        )
                        return capped_conditionals, coefficient

                    p_fit_base, fit_cap_coefficient = apply_row_cost_cap(
                        p_fit_base,
                        val_prior_q[:policy_fit_count],
                    )
                    (
                        p_remaining_base,
                        remaining_cap_coefficient,
                    ) = apply_row_cost_cap(
                        p_remaining_base,
                        remaining_prior_q,
                    )
                    row_cap_metrics = {
                        "risk_budget_row_cost_cap_enabled": 1,
                        "risk_budget_row_cost_cap_per_sample": row_cost_cap,
                        (
                            "risk_budget_row_cost_cap_fit_changed_fraction"
                        ): float((fit_cap_coefficient < 1).to(
                            torch.float64
                        ).mean().item()),
                        (
                            "risk_budget_row_cost_cap_remaining_changed_"
                            "fraction"
                        ): float((remaining_cap_coefficient < 1).to(
                            torch.float64
                        ).mean().item()),
                    }

                base_remaining_cumulative = p_remaining_base.cumprod(dim=1)
                control_lengths = val_max_steps[
                    policy_fit_count:
                ].to(torch.long)
                control_time = torch.arange(
                    T_max_curr,
                    device=device,
                ).unsqueeze(0)
                control_active = (
                    control_time < control_lengths.unsqueeze(1)
                )
                base_control_costs = (
                    base_remaining_cumulative[:control_count]
                    * control_active.to(torch.float64)
                ).sum(dim=1).detach().cpu().numpy()
                floor_control_costs = (
                    self.terminal_pi_min
                    * control_lengths.detach().cpu().numpy().astype(
                        np.float64
                    )
                )
                mixture_scales = np.linspace(
                    1.0,
                    0.0,
                    self.budget_candidate_count,
                )
                candidate_costs = (
                    floor_control_costs[:, None]
                    + (
                        base_control_costs - floor_control_costs
                    )[:, None]
                    * mixture_scales[None, :]
                )
                total_budget_after_policy_fit = (
                    self.budget_per_sample * N
                    - policy_fit_realized_cost_total
                )
                selector_kwargs = {
                    "expected_costs": candidate_costs,
                    "pilot_costs": (
                        control_lengths.detach().cpu().numpy().astype(
                            np.float64
                        )
                    ),
                    "total_budget_after_policy_fit": (
                        total_budget_after_policy_fit
                    ),
                    "deployment_sample_count": len(test_idxs),
                    "maximum_cost_per_sample": T_max_curr,
                    # The score-bin map is data adaptive.  Without an
                    # additional row-local cap, its distribution-free support
                    # bound is the full horizon even when observed rows are
                    # much cheaper.
                    "maximum_candidate_cost_per_sample": (
                        T_max_curr
                        if self.risk_candidate_row_cost_cap is None
                        else min(
                            T_max_curr,
                            self.risk_candidate_row_cost_cap,
                        )
                    ),
                    "maximum_pilot_cost_per_sample": T_max_curr,
                }
                if self.budget_control_mode == "crc":
                    selection = select_crc_budget_candidate(
                        **selector_kwargs,
                    )
                else:
                    selection = select_hoeffding_budget_candidate(
                        **selector_kwargs,
                        delta=self.budget_control_delta,
                    )
                selected_scale = float(
                    mixture_scales[selection.selected_index]
                )

                def apply_selected_scale(
                        base_conditionals: torch.Tensor,
                        row_prior_q: torch.Tensor,
                ) -> torch.Tensor:
                    base_cumulative = base_conditionals.cumprod(dim=1)
                    row_q = row_prior_q.to(
                        device=base_cumulative.device,
                        dtype=torch.long,
                    ).reshape(-1).clamp(min=0, max=T_max_curr)
                    active = (
                        torch.arange(
                            T_max_curr,
                            device=base_cumulative.device,
                        ).unsqueeze(0)
                        < row_q.unsqueeze(1)
                    )
                    selected_cumulative = (
                        self.terminal_pi_min
                        + selected_scale
                        * (base_cumulative - self.terminal_pi_min)
                    )
                    selected_cumulative = torch.where(
                        active,
                        selected_cumulative,
                        torch.ones(
                            (),
                            dtype=torch.float64,
                            device=base_cumulative.device,
                        ),
                    )
                    previous = torch.cat(
                        [
                            torch.ones(
                                (len(base_cumulative), 1),
                                dtype=torch.float64,
                                device=base_cumulative.device,
                            ),
                            selected_cumulative[:, :-1],
                        ],
                        dim=1,
                    )
                    conditionals = (
                        selected_cumulative
                        / previous.clamp_min(
                            torch.finfo(torch.float64).tiny
                        )
                    ).clamp(max=1.0)
                    return torch.where(
                        active,
                        conditionals,
                        torch.ones(
                            (),
                            dtype=torch.float64,
                            device=base_cumulative.device,
                        ),
                    )

                p_fit = apply_selected_scale(
                    p_fit_base,
                    val_prior_q[:policy_fit_count],
                )
                p_remaining = apply_selected_scale(
                    p_remaining_base,
                    remaining_prior_q,
                )
                p_val = torch.cat(
                    [p_fit, p_remaining[:control_count]],
                    dim=0,
                )
                p_test = p_remaining[control_count:]
                projected_probabilities = torch.cat(
                    [p_val, p_test],
                    dim=0,
                )
                # Downstream diagnostics expect a full Phase-I conditional
                # matrix.  In the CRC path this is the selected deployable
                # policy, not the label-dependent oracle solution.
                optimal_P = p_val.detach().cpu().numpy()
                phase1_weights = self.phase1_objective_weights(
                    t_val,
                    val_prior_q,
                    val_quantile_est,
                )
                selected_fit_cost = expected_acquisition_cost(
                    p_fit,
                    val_max_steps[:policy_fit_count],
                )
                selected_full_phase1_cost = expected_acquisition_cost(
                    p_val,
                    val_max_steps,
                )
                selected_control_costs = candidate_costs[
                    :, selection.selected_index
                ]
                control_lengths_numpy = (
                    control_lengths.detach().cpu().numpy().astype(np.float64)
                )
                crc_rho = control_count / len(test_idxs)
                risk_control_metrics = {
                    "risk_budget_control_enabled": 1,
                    "risk_budget_control_mode": self.budget_control_mode,
                    "risk_budget_control_size": control_count,
                    "risk_budget_policy_fit_size": policy_fit_count,
                    "risk_budget_policy_fit_realized_cost_total": (
                        policy_fit_realized_cost_total
                    ),
                    "risk_budget_policy_shape_target_per_sample": (
                        policy_shape_budget_per_sample
                    ),
                    "risk_budget_candidate_count": (
                        self.budget_candidate_count
                    ),
                    "risk_budget_maximum_candidate_cost_per_sample": (
                        T_max_curr
                        if self.risk_candidate_row_cost_cap is None
                        else min(
                            T_max_curr,
                            self.risk_candidate_row_cost_cap,
                        )
                    ),
                    "risk_budget_maximum_pilot_cost_per_sample": (
                        T_max_curr
                    ),
                    "risk_budget_selected_index": selection.selected_index,
                    "risk_budget_selected_mixture_parameter": (
                        selected_scale
                    ),
                    "risk_budget_empirical_control_cost_per_sample": (
                        selection.empirical_expected_cost_per_sample
                    ),
                    "risk_budget_control_pilot_cost_total": float(
                        control_lengths_numpy.sum()
                    ),
                    "risk_budget_control_pilot_cost_per_sample": float(
                        control_lengths_numpy.mean()
                    ),
                    "risk_budget_control_to_deployment_ratio": crc_rho,
                    "risk_budget_deployment_target_per_sample": (
                        selection.deployment_budget_per_sample
                    ),
                    "risk_budget_selector_left_side_per_sample": (
                        selection.selector_left_side_per_sample
                    ),
                    "risk_budget_correction_per_sample": (
                        selection.correction_per_sample
                    ),
                    "risk_budget_guarantee_kind": selection.guarantee_kind,
                    "risk_budget_selector_valid": int(
                        selection.selector_left_side_per_sample
                        <= selection.deployment_budget_per_sample + 1e-12
                    ),
                    "risk_budget_selected_policy_fit_expected_cost": (
                        selected_fit_cost
                    ),
                    "risk_budget_selected_full_phase1_expected_cost": (
                        selected_full_phase1_cost
                    ),
                    "risk_budget_crc_empirical_combined_loss_per_sample": (
                        float(
                            selected_control_costs.mean()
                            + crc_rho * control_lengths_numpy.mean()
                        )
                    ),
                    **row_cap_metrics,
                }
                budget_correction_metrics.update({
                    "projection_corrected_phase1_expected_cost": (
                        selected_full_phase1_cost
                    ),
                    "projection_budget_boundary": "independent_crc_scale",
                    **bin_diagnostics,
                    **risk_control_metrics,
                })
            elif direct_bin_count is not None:
                (
                    optimal_P,
                    p_val_binned,
                    p_test_binned,
                    bin_diagnostics,
                ) = solve_binned_deployable_policy(
                    val_scores,
                    test_scores,
                    val_max_steps,
                    policy_shape_budget_per_sample,
                    phase1_weights,
                    direct_bin_count,
                )
                projected_probabilities = torch.cat(
                    [
                        p_val_binned.cumprod(dim=1),
                        p_test_binned.cumprod(dim=1),
                    ],
                    dim=0,
                )
                cumulative_projection = True
            else:
                optimal_P = solve_exact_fast(
                    val_scores,
                    val_max_steps,
                    policy_shape_budget_per_sample,
                    objective_weights=phase1_weights,
                    terminal_pi_min=None,
                    verbose=False,
                )
                optimal_P[optimal_P == 0] = 1
                projection_targets = torch.cat(
                    [val_scores, test_scores],
                    dim=0,
                )
                if self.projection == 'ir':
                    projected_probabilities = project_to_test_ir(
                        optimal_P, val_scores, projection_targets,
                        val_prior_q,
                        t_val,
                        T_max_curr,
                        device
                    )
                elif self.projection == 'platt':
                    projected_probabilities = project_to_test_platt(
                        optimal_P, val_scores, projection_targets,
                        val_prior_q, t_val, T_max_curr, device
                    )
                elif self.projection == 'beta':
                    projected_probabilities = project_to_test_beta(
                        optimal_P, val_scores, projection_targets,
                        val_prior_q, t_val, T_max_curr, device
                    )
                elif cumulative_projection:
                    projected_probabilities = (
                        project_cumulative_probabilities_to_test_platt(
                            optimal_P,
                            val_scores,
                            projection_targets,
                            val_prior_q,
                            t_val,
                            T_max_curr,
                            device,
                        )
                    )
                else:
                    assert False
            if not (
                    direct_bin_count is not None
                    and self.budget_control_mode is not None
            ):
                p_val_raw = projected_probabilities[:len(val_scores)]
                p_test_raw = projected_probabilities[len(val_scores):]
                correction = (
                    correct_projected_cumulative_probabilities_to_budget
                    if cumulative_projection
                    else correct_projected_probabilities_to_budget
                )
                p_val, p_test, budget_correction_metrics = correction(
                    p_val_raw,
                    p_test_raw,
                    val_max_steps,
                    val_prior_q,
                    test_prior_q,
                    policy_shape_budget_per_sample,
                    terminal_pi_min=self.terminal_pi_min,
                )
                budget_correction_metrics.update(bin_diagnostics)
        acquisition_uniforms = self.get_acquisition_uniforms(
            N,
            T_max_curr,
            device=device,
            dtype=p_test.dtype,
        )
        test_uniforms = (
            None
            if acquisition_uniforms is None
            else acquisition_uniforms[test_idxs]
        )
        if acquisition_uniforms is None:
            self.reset_acquisition_rng()
        test_C, test_total_used, test_C_probs = adaptive_budget_allocation(
            p_test,
            test_prior_q,
            t_test,
            T_max_curr,
            device,
            reach_t_max_is_success=self.reach_t_max_is_success,
            uniforms=test_uniforms,
        )
        total_budget_used = test_total_used + val_budget_used

        final_C, final_C_probs = construct_final_result(N, val_idxs, val_prior_q, test_idxs, test_prior_q, test_C,
                                                        test_C_probs, device)
        # p_val = torch.Tensor(optimal_P)
        # store(p_val, p_test, test_idxs, val_idxs, t, prior_q, final_C)
        # test_weights = (1 / test_C_probs).mean().item()
        # all_weights = (1 / final_C_probs).mean().item()
        # print(
        #     f"projected optimized new test weights {test_weights} | overall weights {all_weights} | total_budget_used {total_budget_used} "
        #     f"| total_budget {total_budget} | # observed: {(final_C > t).float().sum().item()}"
        #     f"| achieved prior: {(final_C.squeeze() >= prior_q).float().sum().item()}")
        t_range = np.arange(optimal_P.shape[1])
        val_mask = (
            t_range[None, :]
            < val_max_steps.cpu().detach().numpy()[:, None]
        )
        val_log_probabilities = np.where(
            val_mask,
            np.log(np.clip(optimal_P, np.exp(-700), 1.0)),
            0.0,
        )
        val_terminal_probability = np.exp(np.clip(
            np.sum(val_log_probabilities, axis=1),
            -700,
            0,
        ))
        val_inverse_excess = 1 / val_terminal_probability - 1
        val_inverse_probability = val_inverse_excess + 1
        phase1_a = (
            t_val.reshape(-1) < val_prior_q.reshape(-1)
        ).detach().cpu().numpy().astype(np.float64)

        p_val_np = p_val.detach().cpu().numpy()
        projected_val_terminal_probability = np.exp(
            np.sum(
                np.where(
                    val_mask,
                    np.log(np.clip(p_val_np, np.exp(-700), 1.0)),
                    0.0,
                ),
                axis=1,
            )
        )
        projected_val_inverse_excess = (
            1 / projected_val_terminal_probability - 1
        )
        projected_val_inverse_probability = (
            projected_val_inverse_excess + 1
        )
        test_inverse_excess = (
            1 / test_C_probs.detach().cpu().numpy() - 1
        )
        test_inverse_probability = test_inverse_excess + 1
        phase2_a = (
            t_test.reshape(-1) < test_prior_q.reshape(-1)
        ).detach().cpu().numpy().astype(np.float64)
        phase1_objective_weights = (
            np.ones(len(t_val), dtype=np.float64)
            if phase1_weights is None
            else phase1_weights.detach().cpu().numpy().astype(np.float64)
        )
        phase2_weights = self.phase2_objective_weights(
            t_test,
            test_prior_q,
            test_quantile_est,
        )
        phase2_objective_weights = (
            np.ones(len(t_test), dtype=np.float64)
            if phase2_weights is None
            else phase2_weights.detach().cpu().numpy().astype(np.float64)
        )

        val_obj = float(np.mean(1 / val_terminal_probability))
        val_obj2 = val_obj
        val_budget = float(np.mean(np.sum(
            np.exp(np.cumsum(np.where(val_mask, np.log(optimal_P), 0.0), axis=1))
            * val_mask,
            axis=1,
        )))
        corrected_phase1_budget = budget_correction_metrics[
            "projection_corrected_phase1_expected_cost"
        ]
        valid_budget = int(
            budget_correction_metrics.get(
                "risk_budget_selector_valid",
                corrected_phase1_budget <= target_budget_avg + 1e-7,
            )
        )
        phase2_active_lengths = torch.minimum(t_test, test_prior_q)
        phase2_expected_cost = expected_acquisition_cost(
            p_test,
            phase2_active_lengths,
        )
        phase2_time = torch.arange(
            p_test.shape[1],
            device=p_test.device,
        ).unsqueeze(0)
        phase2_active = (
            phase2_time < phase2_active_lengths.long().unsqueeze(1)
        )
        phase2_path = torch.where(
            phase2_active,
            p_test.to(torch.float64),
            torch.ones((), dtype=torch.float64, device=p_test.device),
        ).cumprod(dim=1)
        phase2_row_expected_cost = (
            phase2_path * phase2_active.to(torch.float64)
        ).sum(dim=1).detach().cpu().numpy()
        phase2_terminal_probability = torch.exp(torch.where(
            phase2_active,
            torch.log(p_test.to(torch.float64).clamp_min(
                torch.finfo(torch.float64).tiny
            )),
            torch.zeros((), dtype=torch.float64, device=p_test.device),
        ).sum(dim=1)).detach().cpu().numpy()
        phase2_focus_a = self.phase2_target_indicator(
            t_test,
            test_prior_q,
            test_quantile_est,
        ).detach().cpu().numpy().astype(bool)

        def focus_mean(values, mask):
            return float(np.mean(values[mask])) if np.any(mask) else np.nan

        phase2_focus_share = float(np.mean(phase2_focus_a))
        phase2_focus_query_share = (
            float(np.sum(phase2_row_expected_cost[phase2_focus_a]))
            / max(
                float(np.sum(phase2_row_expected_cost)),
                np.finfo(np.float64).tiny,
            )
        )
        phase2_focus_query_lift = (
            phase2_focus_query_share / phase2_focus_share
            if phase2_focus_share > 0
            else np.nan
        )
        if (
                np.std(phase2_focus_a.astype(np.float64)) > 0
                and np.std(phase2_terminal_probability) > 0
        ):
            phase2_focus_pi_correlation = float(np.corrcoef(
                phase2_focus_a.astype(np.float64),
                phase2_terminal_probability,
            )[0, 1])
        else:
            phase2_focus_pi_correlation = np.nan
        phase2_expected_budget_gap = (
            phase2_expected_cost - target_budget_avg
        )
        phase2_expected_cost_total = (
            phase2_expected_cost * len(test_idxs)
        )
        total_expected_budget = (
            float(val_budget_used) + phase2_expected_cost_total
        )
        projection_assumption_budget_upper_bound = (
            float(val_budget_used)
            + len(test_idxs)
            * (
                corrected_phase1_budget
                + self.projection_budget_margin
            )
        )
        expected_budget_metrics = summarize_expected_budget(
            total_expected_budget,
            N,
            self.budget_per_sample,
            cost_semantics=(
                "phase1_fully_observed_plus_phase2_expected_interactions"
            ),
        )
        phase1_oracle_optimized_objective = float(np.mean(
            phase1_objective_weights * val_inverse_excess
        ))
        phase1_oracle_objective_contributions = (
            phase1_objective_weights * val_inverse_probability
        )
        phase1_projected_objective_contributions = (
            phase1_objective_weights * projected_val_inverse_probability
        )
        phase2_objective_contributions = (
            phase2_objective_weights * test_inverse_probability
        )
        phase1_oracle_objective_mean = float(np.mean(
            phase1_oracle_objective_contributions
        ))
        phase1_projected_objective_mean = float(np.mean(
            phase1_projected_objective_contributions
        ))
        phase2_objective_mean = float(np.mean(
            phase2_objective_contributions
        ))
        phase1_oracle_objective_variance = float(np.var(
            phase1_oracle_objective_contributions
        ))
        phase1_projected_objective_variance = float(np.var(
            phase1_projected_objective_contributions
        ))
        phase2_objective_variance = float(np.var(
            phase2_objective_contributions
        ))
        all_objective_contributions = np.concatenate(
            [
                phase1_objective_weights,
                phase2_objective_contributions,
            ]
        )
        all_objective_variance_proxy = np.concatenate(
            [
                np.zeros(len(phase1_objective_weights), dtype=np.float64),
                phase2_objective_weights * test_inverse_excess,
            ]
        )

        additional_metrics = {
            'val_obj': val_obj,
            'val_obj2': val_obj2,
            'val_icw': val_obj,
            'test_icw': (1/test_C_probs).mean().item(),
            'test_icw2': (1/test_C_probs).mean().item(),
            'all_icw': (1/final_C_probs).mean().item(),
            'valid_budget': valid_budget,
            'phase1_projected_budget_valid': valid_budget,
            'val_budget': val_budget,
            'objective_kind': self.objective_kind,
            'val_obj_semantics': 'mean_inverse_probability',
            'phase1_oracle_optimized_objective': (
                phase1_oracle_optimized_objective
            ),
            'phase1_oracle_mean_objective_inverse_probability': (
                phase1_oracle_objective_mean
            ),
            'phase1_oracle_variance_objective_inverse_probability': (
                phase1_oracle_objective_variance
            ),
            'phase1_projected_mean_objective_inverse_probability': (
                phase1_projected_objective_mean
            ),
            'phase1_projected_variance_objective_inverse_probability': (
                phase1_projected_objective_variance
            ),
            'phase2_mean_objective_inverse_probability': (
                phase2_objective_mean
            ),
            'phase2_variance_objective_inverse_probability': (
                phase2_objective_variance
            ),
            'phase1_projection_mean_objective_gap': (
                phase1_projected_objective_mean
                - phase1_oracle_objective_mean
            ),
            'phase2_minus_phase1_oracle_mean_objective_gap': (
                phase2_objective_mean - phase1_oracle_objective_mean
            ),
            'phase2_minus_phase1_projected_mean_objective_gap': (
                phase2_objective_mean - phase1_projected_objective_mean
            ),
            'phase2_to_phase1_oracle_mean_objective_ratio': (
                phase2_objective_mean
                / max(
                    phase1_oracle_objective_mean,
                    np.finfo(np.float64).tiny,
                )
            ),
            'phase2_minus_phase1_oracle_objective_variance_gap': (
                phase2_objective_variance
                - phase1_oracle_objective_variance
            ),
            'all_mean_objective_inverse_probability': float(
                np.mean(all_objective_contributions)
            ),
            'all_variance_objective_inverse_probability': float(
                np.var(all_objective_contributions)
            ),
            'all_mean_objective_variance_proxy': float(
                np.mean(all_objective_variance_proxy)
            ),
            'phase1_a_rate': float(np.mean(phase1_a)),
            'phase1_sample_count': len(val_idxs),
            'phase2_sample_count': len(test_idxs),
            'phase1_expected_cost_total': float(val_budget_used),
            'phase1_expected_cost_per_sample': (
                float(val_budget_used) / len(val_idxs)
            ),
            'phase1_realized_cost_total': float(val_budget_used),
            'phase1_realized_cost_per_sample': (
                float(val_budget_used) / len(val_idxs)
            ),
            'phase1_objective_weight_mean': float(
                np.mean(phase1_objective_weights)
            ),
            'phase2_objective_weight_mean': float(
                np.mean(phase2_objective_weights)
            ),
            'terminal_pi_min': (
                self.terminal_pi_min
                if self.terminal_pi_min is not None
                else np.nan
            ),
            'phase1_oracle_mean_inverse_probability_minus_one': float(
                np.mean(val_inverse_excess)
            ),
            'phase1_oracle_mean_prior_variance_proxy': float(
                np.mean(np.where(phase1_a > 0, val_inverse_excess, 0.0))
            ),
            'phase1_projected_mean_inverse_probability_minus_one': float(
                np.mean(projected_val_inverse_excess)
            ),
            'phase1_projected_mean_prior_variance_proxy': float(
                np.mean(np.where(
                    phase1_a > 0,
                    projected_val_inverse_excess,
                    0.0,
                ))
            ),
            'phase1_oracle_mean_objective_variance_proxy': float(
                np.mean(
                    phase1_objective_weights * val_inverse_excess
                )
            ),
            'phase1_projected_mean_objective_variance_proxy': float(
                np.mean(
                    phase1_objective_weights
                    * projected_val_inverse_excess
                )
            ),
            'phase2_mean_inverse_probability_minus_one': float(
                np.mean(test_inverse_excess)
            ),
            'phase2_mean_inverse_probability': float(
                np.mean(test_inverse_probability)
            ),
            'phase2_variance_inverse_probability': float(
                np.var(test_inverse_probability)
            ),
            'phase2_mean_prior_variance_proxy': float(
                np.mean(np.where(phase2_a > 0, test_inverse_excess, 0.0))
            ),
            'phase2_mean_objective_variance_proxy': float(
                np.mean(
                    phase2_objective_weights * test_inverse_excess
                )
            ),
            'phase2_target_budget_per_sample': target_budget_avg,
            'phase2_expected_cost_total': phase2_expected_cost_total,
            'phase2_expected_cost_per_sample': phase2_expected_cost,
            'phase2_focus_a_rate': phase2_focus_share,
            'phase2_focus_expected_query_share': (
                phase2_focus_query_share
            ),
            'phase2_focus_expected_query_lift': phase2_focus_query_lift,
            'phase2_focus_mean_expected_queries': focus_mean(
                phase2_row_expected_cost,
                phase2_focus_a,
            ),
            'phase2_nonfocus_mean_expected_queries': focus_mean(
                phase2_row_expected_cost,
                ~phase2_focus_a,
            ),
            'phase2_focus_mean_terminal_probability': focus_mean(
                phase2_terminal_probability,
                phase2_focus_a,
            ),
            'phase2_nonfocus_mean_terminal_probability': focus_mean(
                phase2_terminal_probability,
                ~phase2_focus_a,
            ),
            'phase2_focus_terminal_probability_correlation': (
                phase2_focus_pi_correlation
            ),
            'phase2_expected_budget_gap_total': (
                phase2_expected_budget_gap * len(test_idxs)
            ),
            'projection_budget_margin_per_sample': (
                self.projection_budget_margin
            ),
            'projection_reserved_budget_per_sample': (
                policy_shape_budget_per_sample
            ),
            'projection_transfer_cost_error_per_sample': (
                phase2_expected_cost - corrected_phase1_budget
            ),
            'projection_transfer_assumption_satisfied': int(
                phase2_expected_cost - corrected_phase1_budget
                <= self.projection_budget_margin + 1e-7
            ),
            'projection_expected_total_budget_upper_bound_under_assumption': (
                projection_assumption_budget_upper_bound
            ),
            'projection_expected_budget_guarantee_valid_under_assumption': int(
                projection_assumption_budget_upper_bound
                <= self.budget_per_sample * N + 1e-7 * N
            ),
            'expected_budget_guarantee_kind': (
                'projection_transfer_assumption'
                if self.budget_control_mode is None
                else budget_correction_metrics.get(
                    'risk_budget_guarantee_kind',
                    self.budget_control_mode,
                )
            ),
            'expected_budget_guarantee_requires_projection_accuracy': int(
                self.budget_control_mode is None
            ),
            'expected_budget_guarantee_is_marginal_finite_sample': int(
                self.budget_control_mode == 'crc'
            ),
            'phase2_expected_budget_gap_per_sample': (
                phase2_expected_budget_gap
            ),
            'phase2_expected_budget_valid': int(
                phase2_expected_budget_gap <= 1e-7
            ),
            'phase2_realized_cost_per_sample': (
                test_total_used / len(test_idxs)
            ),
            **expected_budget_metrics,
            **self.objective_metadata(),
            **budget_correction_metrics,
        }
        if self.evaluate_projection:
            max_steps_all = torch.minimum(prior_q, t)
            all_objective_weights = self.phase1_objective_weights(
                t,
                prior_q,
                quantile_est,
            )
            optimal_P_all = solve_exact_fast(
                scores,
                max_steps_all,
                target_budget_avg,
                objective_weights=all_objective_weights,
                terminal_pi_min=None,
                verbose=False,
            )
            optimal_P_all[optimal_P_all == 0] = 1
            oracle_p_test = torch.as_tensor(
                optimal_P_all[test_idxs],
                dtype=p_test.dtype,
                device=device,
            )
            oracle_p_val = torch.as_tensor(
                optimal_P_all[val_idxs],
                dtype=p_test.dtype,
                device=device,
            )
            learned_p_val = torch.as_tensor(
                optimal_P,
                dtype=p_test.dtype,
                device=device,
            )
            val_time = torch.arange(T_max_curr, device=device).unsqueeze(0)
            val_active_mask = val_time < val_max_steps.long().unsqueeze(1)
            val_probability_error = learned_p_val - oracle_p_val
            active_val_error = val_probability_error[val_active_mask]
            val_mae_over_time = torch.where(
                val_active_mask,
                val_probability_error.abs(),
                torch.zeros_like(val_probability_error),
            ).sum(dim=0) / val_active_mask.sum(dim=0).clamp_min(1)
            projection_metrics = compute_dapro_projection_metrics(
                projected_probabilities=p_test,
                oracle_probabilities=oracle_p_test,
                prior_q=test_prior_q,
                event_times=t_test,
                val_budget_used=val_budget_used,
                target_budget_avg=target_budget_avg,
                realized_test_budget=test_total_used,
                total_sample_count=N,
                budget_per_sample=self.budget_per_sample,
            )
            projection_metrics.update(
                {
                    "projection_val_probability_mae": (
                        active_val_error.abs().mean().item()
                    ),
                    "projection_val_probability_rmse": (
                        active_val_error.square().mean().sqrt().item()
                    ),
                    "projection_val_probability_bias": (
                        active_val_error.mean().item()
                    ),
                    "projection_val_mae_over_time_max": (
                        val_mae_over_time.max().item()
                    ),
                    # Backward-compatible names from the original snippet.
                    "epsilon_val_mae": active_val_error.abs().mean().item(),
                    "epsilon_val_mae_max_time": (
                        val_mae_over_time.max().item()
                    ),
                }
            )
            additional_metrics.update(
                {
                    "projection_evaluation_enabled": 1,
                    "projection_validation_size": len(val_idxs),
                    "projection_test_size": len(test_idxs),
                    **projection_metrics,
                }
            )
        return BudgetAllocationResult(quantile_est, final_C, final_C_probs, total_budget_used, additional_metrics=additional_metrics)


class AWeightedDAPRO(LegacyMeanWeightDAPRO):
    """DAPRO variant minimizing the conditional acquisition-variance proxy.

    Phase I uses ``A_i = I(T_i < f_prior(X_i))`` and minimizes
    ``mean(A_i * (1 / pi_i - 1))``.  The constant ``-A_i`` is omitted inside
    the solver. After projection, the same explicit always-continue
    exploration mixture is applied to both this method and ordinary DAPRO,
    preserving finite deployment weights without changing the Phase-I
    optimization target.
    """

    def __init__(self, *args, terminal_pi_min: float = 0.005, **kwargs):
        super().__init__(
            *args,
            terminal_pi_min=terminal_pi_min,
            **kwargs,
        )
        self.use_a_weighted_objective = True

    @property
    def name(self) -> str:
        base = (
            f"projected_optimization_{self.projection}_{self.score}"
            "_a_weighted"
        )
        base += self.budget_control_name_suffix
        if self.n1 != 100:
            base += f"_n1_{self.n1}"
        return base

    @property
    def objective_kind(self) -> str:
        return "mean_prior_a_weighted_inverse_probability_minus_one"

    def phase1_objective_weights(
            self,
            event_times: torch.Tensor,
            prior_q: torch.Tensor,
            quantile_est: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return (
            event_times.reshape(-1) < prior_q.reshape(-1)
        ).to(torch.float64)


class TargetAWeightedDAPRO(AWeightedDAPRO):
    """A-weighted DAPRO anchored to a fixed 90%-coverage candidate.

    ``raw_alpha`` freezes the largest model tau strictly below
    ``target_alpha``. ``phase1_unweighted`` instead selects the candidate from
    the fully observed Phase-I rows, using the exact same strict-prefix rule as
    the final LPB calibrator. In both cases the anchor is fixed before Phase II
    is sampled, avoiding a policy/target feedback loop.
    """

    _VALID_ANCHORS = {"raw_alpha", "phase1_unweighted"}

    def __init__(
            self,
            *args,
            anchor_kind: str,
            target_alpha: float = 0.10,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if anchor_kind not in self._VALID_ANCHORS:
            raise ValueError(
                f"`anchor_kind` must be one of {sorted(self._VALID_ANCHORS)}; "
                f"got {anchor_kind!r}."
            )
        if not 0 < target_alpha < 1:
            raise ValueError(
                f"`target_alpha` must lie in (0, 1); got {target_alpha}."
            )
        if target_alpha >= self.tau_prior:
            raise ValueError(
                "`target_alpha` must be strictly smaller than `tau_prior` "
                "so its named target is inside the acquisition envelope; "
                f"got target_alpha={target_alpha}, tau_prior={self.tau_prior}."
            )
        self.anchor_kind = anchor_kind
        self.target_alpha = float(target_alpha)
        self._target_anchor_index = None
        self._target_anchor_selection_miscoverage = np.nan
        self._target_anchor_phase1_rate = np.nan
        self._target_anchor_phase2_rate = np.nan
        self._target_anchor_phase1_within_prior = 0
        self._target_anchor_phase2_within_prior = 0

    @property
    def name(self) -> str:
        alpha = f"{self.target_alpha:.2f}".replace(".", "p")
        anchor = (
            "raw"
            if self.anchor_kind == "raw_alpha"
            else "phase1_unweighted"
        )
        base = (
            f"projected_optimization_{self.projection}_{self.score}"
            f"_a_target_{anchor}_alpha_{alpha}"
        )
        base += self.budget_control_name_suffix
        if self.n1 != 100:
            base += f"_n1_{self.n1}"
        return base

    @property
    def objective_kind(self) -> str:
        return (
            "mean_target_a_weighted_inverse_probability_minus_one"
            f"_{self.anchor_kind}_alpha_{self.target_alpha:.2f}"
        )

    def _select_target_anchor(
            self,
            event_times: torch.Tensor,
            quantile_est: torch.Tensor,
    ) -> None:
        prior_index = int(
            torch.abs(self.taus_range - self.tau_prior).argmin().item()
        )
        candidate_quantiles = quantile_est[:, :prior_index + 1]
        if self.anchor_kind == "raw_alpha":
            selection_miscoverage = self.taus_range[:prior_index + 1]
        else:
            selection_miscoverage = (
                event_times.reshape(-1, 1) < candidate_quantiles
            ).to(torch.float64).mean(dim=0)
        target = torch.tensor(
            [self.target_alpha],
            dtype=selection_miscoverage.dtype,
            device=selection_miscoverage.device,
        )
        selected = select_calibration_positions(
            selection_miscoverage,
            target,
        )
        self._target_anchor_index = int(selected.item())
        self._target_anchor_selection_miscoverage = float(
            selection_miscoverage[self._target_anchor_index].item()
        )

    def _weights_at_frozen_anchor(
            self,
            event_times: torch.Tensor,
            prior_q: torch.Tensor,
            quantile_est: torch.Tensor | None,
    ) -> tuple[torch.Tensor, int]:
        if quantile_est is None:
            raise ValueError(
                "Target-A-weighted DAPRO requires candidate quantile estimates."
            )
        if self._target_anchor_index is None:
            raise RuntimeError("The target anchor must be selected in Phase I.")
        anchor_q = quantile_est[:, self._target_anchor_index]
        tolerance = 1e-6
        within_prior = bool(torch.all(
            anchor_q.reshape(-1)
            <= prior_q.reshape(-1).to(anchor_q.dtype) + tolerance
        ).item())
        if not within_prior:
            violations = int((
                anchor_q.reshape(-1)
                > prior_q.reshape(-1).to(anchor_q.dtype) + tolerance
            ).sum().item())
            raise ValueError(
                "The target-A anchor exceeds q_prior for "
                f"{violations} rows and is not identifiable."
            )
        weights = (
            event_times.reshape(-1) < anchor_q.reshape(-1)
        ).to(torch.float64)
        return weights, int(within_prior)

    def phase1_objective_weights(
            self,
            event_times: torch.Tensor,
            prior_q: torch.Tensor,
            quantile_est: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if quantile_est is None:
            raise ValueError(
                "Target-A-weighted DAPRO requires candidate quantile estimates."
            )
        if self._target_anchor_index is None:
            self._select_target_anchor(event_times, quantile_est)
        weights, within_prior = self._weights_at_frozen_anchor(
            event_times,
            prior_q,
            quantile_est,
        )
        self._target_anchor_phase1_rate = float(weights.mean().item())
        self._target_anchor_phase1_within_prior = within_prior
        return weights

    def phase2_objective_weights(
            self,
            event_times: torch.Tensor,
            prior_q: torch.Tensor,
            quantile_est: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.phase2_target_indicator(
            event_times,
            prior_q,
            quantile_est,
        )

    def phase2_target_indicator(
            self,
            event_times: torch.Tensor,
            prior_q: torch.Tensor,
            quantile_est: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weights, within_prior = self._weights_at_frozen_anchor(
            event_times,
            prior_q,
            quantile_est,
        )
        self._target_anchor_phase2_rate = float(weights.mean().item())
        self._target_anchor_phase2_within_prior = within_prior
        return weights

    def objective_metadata(self) -> dict:
        if self._target_anchor_index is None:
            return {}
        prior_index = int(
            torch.abs(self.taus_range - self.tau_prior).argmin().item()
        )
        return {
            "target_anchor_kind": self.anchor_kind,
            "target_anchor_alpha": self.target_alpha,
            "target_anchor_index": self._target_anchor_index,
            "target_anchor_tau": float(
                self.taus_range[self._target_anchor_index].item()
            ),
            "target_anchor_prior_index": prior_index,
            "target_anchor_selection_miscoverage": (
                self._target_anchor_selection_miscoverage
            ),
            "target_anchor_phase1_a_rate": self._target_anchor_phase1_rate,
            "target_anchor_phase2_a_rate": self._target_anchor_phase2_rate,
            "target_anchor_phase1_within_prior": (
                self._target_anchor_phase1_within_prior
            ),
            "target_anchor_phase2_within_prior": (
                self._target_anchor_phase2_within_prior
            ),
        }


class RandomAnchoredTargetAWeightedDAPRO(TargetAWeightedDAPRO):
    """Shrink a target-variance policy toward the constant reference policy.

    The fitted direct-time target policy and a constant-continuation policy are
    first calibrated to the same policy-fit budget.  Their cumulative reach
    schedules are then mixed:

        R_mix(t) = gamma R_target(t) + (1 - gamma) R_random(t).

    This preserves temporal monotonicity, positivity, and linear expected-cost
    accounting.  ``gamma=0`` is the constant reference shape and ``gamma=1``
    is raw target-A DAPRO.  Intermediate values trade target efficiency for
    the constant policy's stability.

    With ``fill_random_slack=True``, a lexicographic second stage raises the
    mixture only where the Random schedule is larger, until the policy-fit
    budget is filled.  This cannot reduce target-event reach and prevents a
    sparse target objective's plateau from wasting usable budget.

    When ``budget_control_mode`` is enabled, ``n1`` is the total fully observed
    Phase-I count.  The first ``n1-budget_control_size`` rows learn the policy
    shape, and an independent control fold selects a nested affine scale using
    CRC or a simultaneous Hoeffding UCB.  Only the CRC mode gives the
    distribution-free marginal expected-total-budget guarantee used here.
    """

    def __init__(
            self,
            *args,
            target_policy_fraction: float,
            fill_random_slack: bool = False,
            budget_control_mode: str | None = None,
            budget_control_size: int = 0,
            **kwargs,
    ):
        if not 0 <= target_policy_fraction <= 1:
            raise ValueError(
                "`target_policy_fraction` must lie in [0, 1]."
            )
        kwargs["anchor_kind"] = "raw_alpha"
        kwargs["random_anchor_target_fraction"] = target_policy_fraction
        kwargs["random_anchor_fill_slack"] = fill_random_slack
        kwargs["budget_control_mode"] = budget_control_mode
        kwargs["budget_control_size"] = budget_control_size
        super().__init__(*args, **kwargs)
        self.target_policy_fraction = float(target_policy_fraction)
        self.fill_random_slack = bool(fill_random_slack)

    @staticmethod
    def _format_fraction(value: float) -> str:
        return f"{value:.2f}".replace(".", "p")

    @property
    def name(self) -> str:
        alpha = f"{self.target_alpha:.2f}".replace(".", "p")
        fraction = self._format_fraction(self.target_policy_fraction)
        base = (
            f"projected_optimization_{self.projection}_{self.score}"
            f"_a_target_raw_random_anchor_target_{fraction}_alpha_{alpha}"
        )
        if self.fill_random_slack:
            base += "_random_slack_filled"
        if self.budget_control_mode is not None:
            base += (
                f"_budget_{self.budget_control_mode}"
                f"_control_{self.budget_control_size}"
            )
        if self.n1 != 100:
            base += f"_n1_{self.n1}"
        return base

    @property
    def objective_kind(self) -> str:
        return (
            "random_anchored_target_a_variance_policy"
            f"_target_fraction_{self.target_policy_fraction:.2f}"
        )

    def objective_metadata(self) -> dict:
        metadata = super().objective_metadata()
        metadata.update({
            "random_anchor_target_fraction": self.target_policy_fraction,
            "random_anchor_slack_fill_enabled": int(self.fill_random_slack),
            "risk_budget_control_mode": (
                self.budget_control_mode
                if self.budget_control_mode is not None
                else "none"
            ),
            "risk_budget_control_size": self.budget_control_size,
        })
        return metadata


class RobustTargetAWeightedDAPRO(TargetAWeightedDAPRO):
    """Blend a Phase-I-selected target with the denser raw-alpha target.

    The pure Phase-I target can contain only a handful of events.  Its exact
    optimizer may then saturate those observed endpoints and leave most of the
    budget unused, which is optimal in sample but brittle in Phase II.  This
    method retains the frozen Phase-I target while assigning a small positive
    weight to raw-alpha events.  The normalized blend is fixed before Phase II
    and prevents unseen target-event times from receiving only the floor.
    """

    def __init__(
            self,
            *args,
            robustness_weight: float,
            **kwargs,
    ):
        kwargs["anchor_kind"] = "phase1_unweighted"
        super().__init__(*args, **kwargs)
        if not 0 < robustness_weight <= 1:
            raise ValueError("`robustness_weight` must lie in (0, 1].")
        self.robustness_weight = float(robustness_weight)
        self._raw_anchor_index = None
        self._robust_phase1_raw_rate = np.nan
        self._robust_phase2_raw_rate = np.nan

    @property
    def name(self) -> str:
        alpha = f"{self.target_alpha:.2f}".replace(".", "p")
        gamma = f"{self.robustness_weight:.2f}".replace(".", "p")
        base = (
            f"projected_optimization_{self.projection}_{self.score}"
            f"_a_target_phase1_robust_raw_{gamma}_alpha_{alpha}"
        )
        if self.n1 != 100:
            base += f"_n1_{self.n1}"
        return base

    @property
    def objective_kind(self) -> str:
        return (
            "mean_robust_target_a_weighted_inverse_probability_minus_one"
            f"_raw_{self.robustness_weight:.2f}"
        )

    def _raw_alpha_weights(
            self,
            event_times: torch.Tensor,
            prior_q: torch.Tensor,
            quantile_est: torch.Tensor | None,
    ) -> torch.Tensor:
        if quantile_est is None:
            raise ValueError(
                "Robust target DAPRO requires candidate quantile estimates."
            )
        if self._raw_anchor_index is None:
            prior_index = int(
                torch.abs(self.taus_range - self.tau_prior).argmin().item()
            )
            selected = select_calibration_positions(
                self.taus_range[:prior_index + 1],
                torch.tensor(
                    [self.target_alpha],
                    dtype=self.taus_range.dtype,
                    device=self.taus_range.device,
                ),
            )
            self._raw_anchor_index = int(selected.item())
        raw_q = quantile_est[:, self._raw_anchor_index]
        if torch.any(
                raw_q.reshape(-1)
                > prior_q.reshape(-1).to(raw_q.dtype) + 1e-6
        ):
            raise ValueError(
                "The robust raw-alpha anchor exceeds q_prior."
            )
        return (
            event_times.reshape(-1) < raw_q.reshape(-1)
        ).to(torch.float64)

    def _blend(
            self,
            primary: torch.Tensor,
            raw: torch.Tensor,
    ) -> torch.Tensor:
        return (
            primary + self.robustness_weight * raw
        ) / (1 + self.robustness_weight)

    def phase1_objective_weights(
            self,
            event_times: torch.Tensor,
            prior_q: torch.Tensor,
            quantile_est: torch.Tensor | None = None,
    ) -> torch.Tensor:
        primary = super().phase1_objective_weights(
            event_times,
            prior_q,
            quantile_est,
        )
        raw = self._raw_alpha_weights(
            event_times,
            prior_q,
            quantile_est,
        )
        self._robust_phase1_raw_rate = float(raw.mean().item())
        return self._blend(primary, raw)

    def phase2_objective_weights(
            self,
            event_times: torch.Tensor,
            prior_q: torch.Tensor,
            quantile_est: torch.Tensor | None = None,
    ) -> torch.Tensor:
        primary = super().phase2_objective_weights(
            event_times,
            prior_q,
            quantile_est,
        )
        raw = self._raw_alpha_weights(
            event_times,
            prior_q,
            quantile_est,
        )
        self._robust_phase2_raw_rate = float(raw.mean().item())
        return self._blend(primary, raw)

    def objective_metadata(self) -> dict:
        metadata = super().objective_metadata()
        metadata.update({
            "robustness_weight": self.robustness_weight,
            "robust_raw_anchor_index": self._raw_anchor_index,
            "robust_phase1_raw_a_rate": self._robust_phase1_raw_rate,
            "robust_phase2_raw_a_rate": self._robust_phase2_raw_rate,
        })
        return metadata


class RegularizedTargetAWeightedDAPRO(TargetAWeightedDAPRO):
    """Target-A objective with a small global weight for tail robustness.

    ``A``-only optimization is indifferent to every row with ``A=0``.  If all
    observed target events are early, the exact optimum can consequently stop
    spending once those endpoints have reach probability one.  Adding
    ``delta`` to every objective weight gives

        W_i = (A_i + delta) / (1 + delta),

    preserving target emphasis while ensuring that unused budget still
    improves a well-defined global tail objective.
    """

    def __init__(
            self,
            *args,
            global_regularization: float,
            anchor_kind: str = "raw_alpha",
            **kwargs,
    ):
        kwargs["anchor_kind"] = anchor_kind
        super().__init__(*args, **kwargs)
        if not 0 < global_regularization <= 1:
            raise ValueError(
                "`global_regularization` must lie in (0, 1]."
            )
        self.global_regularization = float(global_regularization)

    @property
    def name(self) -> str:
        alpha = f"{self.target_alpha:.2f}".replace(".", "p")
        delta = f"{self.global_regularization:.3f}".replace(".", "p")
        anchor = (
            "raw"
            if self.anchor_kind == "raw_alpha"
            else "phase1_unweighted"
        )
        base = (
            f"projected_optimization_{self.projection}_{self.score}"
            f"_a_target_{anchor}_regularized_global_{delta}_alpha_{alpha}"
        )
        if self.n1 != 100:
            base += f"_n1_{self.n1}"
        return base

    @property
    def objective_kind(self) -> str:
        return (
            "mean_regularized_target_a_weighted_inverse_probability_minus_one"
            f"_{self.anchor_kind}"
            f"_global_{self.global_regularization:.3f}"
        )

    def _regularize(self, target_weights: torch.Tensor) -> torch.Tensor:
        return (
            target_weights + self.global_regularization
        ) / (1 + self.global_regularization)

    def phase1_objective_weights(
            self,
            event_times: torch.Tensor,
            prior_q: torch.Tensor,
            quantile_est: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._regularize(super().phase1_objective_weights(
            event_times,
            prior_q,
            quantile_est,
        ))

    def phase2_objective_weights(
            self,
            event_times: torch.Tensor,
            prior_q: torch.Tensor,
            quantile_est: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._regularize(super().phase2_objective_weights(
            event_times,
            prior_q,
            quantile_est,
        ))

    def objective_metadata(self) -> dict:
        metadata = super().objective_metadata()
        metadata["global_regularization"] = self.global_regularization
        return metadata


class DefinitiveDAPRO(RegularizedTargetAWeightedDAPRO):
    """Assumption-based, variance-aligned projection DAPRO ablation.

    The policy minimizes a regularized empirical version of

        mean_i A_i * (1 / pi_i - 1),

    where ``A_i`` is the fixed raw-alpha target event.  At every interaction
    time, Phase-I risk scores are divided into two empirical quantile bins and
    the continuation probabilities are optimized jointly under the sequential
    expected-cost constraint.  Deployment is a deterministic bin-table lookup;
    no per-time regression model is fitted.

    ``projection_budget_margin`` reserves cost for Phase-I-to-deployment
    projection error.  If the deployed policy's conditional expected cost is
    at most its corrected Phase-I cost plus this margin, the total expected
    budget is at most the configured budget.  The assumption and its realized
    diagnostic are therefore stated in the same units as the budget.

    This class is retained for the projection-transfer comparison.  The public
    :class:`DAPRO` alias points to :class:`DefinitiveCRCDAPRO`, which replaces
    the projection-accuracy assumption by an independent CRC control fold.
    """

    DEFAULT_N1 = 200
    DEFAULT_TARGET_ALPHA = 0.10
    DEFAULT_GLOBAL_REGULARIZATION = 0.001
    DEFAULT_PROJECTION_BUDGET_MARGIN = 1.0
    SCORE_BIN_COUNT = 2

    def __init__(
            self,
            conditional_grid,
            budget_per_sample,
            taus_range,
            tau_prior,
            m_upper_bound,
            *,
            n1: int = DEFAULT_N1,
            target_alpha: float = DEFAULT_TARGET_ALPHA,
            global_regularization: float = DEFAULT_GLOBAL_REGULARIZATION,
            projection_budget_margin: float = (
                DEFAULT_PROJECTION_BUDGET_MARGIN
            ),
            terminal_pi_min: float = 0.005,
            reach_t_max_is_success: bool = False,
            evaluate_projection: bool = False,
            budget_control_mode: str | None = None,
            budget_control_size: int = 0,
            budget_candidate_count: int = 401,
            risk_candidate_row_cost_cap: float | None = None,
    ):
        super().__init__(
            conditional_grid,
            budget_per_sample,
            taus_range,
            tau_prior,
            m_upper_bound,
            projection="direct_bins_2",
            score="prob",
            reach_t_max_is_success=reach_t_max_is_success,
            n1=n1,
            evaluate_projection=evaluate_projection,
            terminal_pi_min=terminal_pi_min,
            projection_budget_margin=projection_budget_margin,
            budget_control_mode=budget_control_mode,
            budget_control_size=budget_control_size,
            budget_candidate_count=budget_candidate_count,
            risk_candidate_row_cost_cap=risk_candidate_row_cost_cap,
            anchor_kind="raw_alpha",
            target_alpha=target_alpha,
            global_regularization=global_regularization,
        )

    @staticmethod
    def _format_parameter(value: float, digits: int) -> str:
        return f"{value:.{digits}f}".replace(".", "p")

    @property
    def name(self) -> str:
        margin = self._format_parameter(self.projection_budget_margin, 2)
        regularization = self._format_parameter(
            self.global_regularization,
            3,
        )
        alpha = self._format_parameter(self.target_alpha, 2)
        base = (
            "dapro_variance_aligned_bins_2"
            f"_alpha_{alpha}"
            f"_global_{regularization}"
            f"_projection_margin_{margin}"
        )
        return f"{base}_n1_{self.n1}"

    @property
    def objective_kind(self) -> str:
        return "definitive_regularized_target_a_variance"

    def objective_metadata(self) -> dict:
        metadata = super().objective_metadata()
        metadata.update({
            "definitive_dapro": 1,
            "definitive_score_bin_count": self.SCORE_BIN_COUNT,
            "definitive_projection_budget_margin": (
                self.projection_budget_margin
            ),
            "definitive_budget_control_mode": (
                self.budget_control_mode
                if self.budget_control_mode is not None
                else "projection_assumption"
            ),
        })
        return metadata


class DefinitiveCRCDAPRO(DefinitiveDAPRO):
    """Variance-aligned two-bin DAPRO with independent CRC budget control.

    ``n1`` is the total number of fully observed rows.  The first
    ``n1-budget_control_size`` rows learn the target-weighted score-bin table;
    the remaining rows are an independent budget-control fold.  A nested
    affine contraction of cumulative reach toward the terminal floor is fixed
    before the control labels are inspected.  CRC then selects the strongest
    feasible contraction, giving a finite-sample marginal expected-total-
    budget guarantee without a projection-accuracy assumption.
    """

    DEFAULT_N1 = 200
    DEFAULT_BUDGET_CONTROL_SIZE = 100
    DEFAULT_ROW_COST_CAP_MULTIPLIER = 2.0

    def __init__(
            self,
            conditional_grid,
            budget_per_sample,
            taus_range,
            tau_prior,
            m_upper_bound,
            *,
            n1: int = DEFAULT_N1,
            budget_control_size: int = DEFAULT_BUDGET_CONTROL_SIZE,
            target_alpha: float = DefinitiveDAPRO.DEFAULT_TARGET_ALPHA,
            global_regularization: float = (
                DefinitiveDAPRO.DEFAULT_GLOBAL_REGULARIZATION
            ),
            terminal_pi_min: float = 0.005,
            budget_candidate_count: int = 401,
            row_cost_cap_multiplier: float | None = (
                DEFAULT_ROW_COST_CAP_MULTIPLIER
            ),
            reach_t_max_is_success: bool = False,
            evaluate_projection: bool = False,
    ):
        if (
                row_cost_cap_multiplier is not None
                and (
                    not np.isfinite(row_cost_cap_multiplier)
                    or row_cost_cap_multiplier <= 0
                )
        ):
            raise ValueError(
                "`row_cost_cap_multiplier` must be finite and positive."
            )
        self.row_cost_cap_multiplier = (
            None
            if row_cost_cap_multiplier is None
            else float(row_cost_cap_multiplier)
        )
        super().__init__(
            conditional_grid,
            budget_per_sample,
            taus_range,
            tau_prior,
            m_upper_bound,
            n1=n1,
            target_alpha=target_alpha,
            global_regularization=global_regularization,
            projection_budget_margin=0.0,
            terminal_pi_min=terminal_pi_min,
            reach_t_max_is_success=reach_t_max_is_success,
            evaluate_projection=evaluate_projection,
            budget_control_mode="crc",
            budget_control_size=budget_control_size,
            budget_candidate_count=budget_candidate_count,
            risk_candidate_row_cost_cap=(
                None
                if self.row_cost_cap_multiplier is None
                else self.row_cost_cap_multiplier * budget_per_sample
            ),
        )

    @property
    def name(self) -> str:
        regularization = self._format_parameter(
            self.global_regularization,
            3,
        )
        alpha = self._format_parameter(self.target_alpha, 2)
        base = (
            "dapro_variance_aligned_bins_2"
            f"_alpha_{alpha}"
            f"_global_{regularization}"
            "_budget_crc"
            f"_control_{self.budget_control_size}"
        )
        if self.row_cost_cap_multiplier is not None:
            multiplier = self._format_parameter(
                self.row_cost_cap_multiplier,
                2,
            )
            base += f"_row_cap_{multiplier}x_budget"
        return f"{base}_n1_{self.n1}"

    @property
    def objective_kind(self) -> str:
        return "definitive_regularized_target_a_variance_crc"


class DefinitiveCRCUPBDAPRO(DefinitiveCRCDAPRO):
    """Upper-bound counterpart of definitive CRC-DAPRO.

    A 70% UPB uses the fixed calibration event

        A_i = 1{T_i > q_0.70(X_i)}.

    The two-bin policy, row-local cost cap, and independent CRC budget
    selector are unchanged.  The explicit class prevents the lower-tail LPB
    target from being silently reused when constructing an upper bound.
    """

    DEFAULT_TARGET_COVERAGE = 0.70

    def __init__(
            self,
            conditional_grid,
            budget_per_sample,
            taus_range,
            tau_prior,
            m_upper_bound,
            *,
            target_coverage: float = DEFAULT_TARGET_COVERAGE,
            **kwargs,
    ):
        self.target_coverage = float(target_coverage)
        super().__init__(
            conditional_grid,
            budget_per_sample,
            taus_range,
            tau_prior,
            m_upper_bound,
            target_alpha=self.target_coverage,
            reach_t_max_is_success=True,
            **kwargs,
        )

    @property
    def name(self) -> str:
        return super().name.replace(
            "dapro_variance_aligned",
            "dapro_upb_variance_aligned",
            1,
        )

    @property
    def objective_kind(self) -> str:
        return "definitive_regularized_upb_target_a_variance_crc"

    def _weights_at_frozen_anchor(
            self,
            event_times: torch.Tensor,
            prior_q: torch.Tensor,
            quantile_est: torch.Tensor | None,
    ) -> tuple[torch.Tensor, int]:
        if quantile_est is None:
            raise ValueError(
                "Target-A-weighted UPB DAPRO requires candidate quantile "
                "estimates."
            )
        if self._target_anchor_index is None:
            raise RuntimeError("The UPB target anchor must be selected first.")
        anchor_q = quantile_est[:, self._target_anchor_index]
        tolerance = 1e-6
        within_prior = bool(torch.all(
            anchor_q.reshape(-1)
            <= prior_q.reshape(-1).to(anchor_q.dtype) + tolerance
        ).item())
        if not within_prior:
            violations = int((
                anchor_q.reshape(-1)
                > prior_q.reshape(-1).to(anchor_q.dtype) + tolerance
            ).sum().item())
            raise ValueError(
                "The UPB target anchor exceeds q_prior for "
                f"{violations} rows and is not identifiable."
            )
        weights = (
            event_times.reshape(-1) > anchor_q.reshape(-1)
        ).to(torch.float64)
        return weights, int(within_prior)

    def objective_metadata(self) -> dict:
        metadata = super().objective_metadata()
        metadata.update({
            "target_event_side": "upper",
            "target_bound_type": "upb",
            "target_coverage": self.target_coverage,
        })
        return metadata


class BandRegularizedTargetAWeightedDAPRO(AWeightedDAPRO):
    """Average fixed-candidate variance over a target-alpha neighborhood.

    The final weighted calibration rule may select a candidate on either side
    of the nominal 90% target.  Optimizing one raw-alpha event indicator can
    therefore be too brittle.  This variant minimizes the average conditional
    Horvitz-Thompson variance proxy over a prespecified alpha band:

        W_i = (
            mean_k I{T_i < f_{alpha_k}(X_i)} + delta
        ) / (1 + delta).

    The band protects against candidate switching, while ``delta`` supplies a
    secondary global inverse-weight objective and eliminates the zero-weight
    endpoint plateau of a pure sparse-target optimizer.
    """

    def __init__(
            self,
            *args,
            target_alphas: tuple[float, ...] | list[float],
            global_regularization: float,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        alphas = tuple(float(alpha) for alpha in target_alphas)
        if not alphas:
            raise ValueError("`target_alphas` must not be empty.")
        if any(
                not 0 < alpha < self.tau_prior
                for alpha in alphas
        ):
            raise ValueError(
                "Every band alpha must lie strictly inside (0, tau_prior)."
            )
        if tuple(sorted(set(alphas))) != alphas:
            raise ValueError(
                "`target_alphas` must be strictly increasing and unique."
            )
        if not 0 < global_regularization <= 1:
            raise ValueError(
                "`global_regularization` must lie in (0, 1]."
            )
        self.target_alphas = alphas
        self.global_regularization = float(global_regularization)
        self._band_anchor_indices = None
        self._band_phase1_mean_indicator = np.nan
        self._band_phase2_mean_indicator = np.nan
        self._band_phase1_within_prior = 0
        self._band_phase2_within_prior = 0

    @staticmethod
    def _format_probability(value: float, digits: int) -> str:
        return f"{value:.{digits}f}".replace(".", "p")

    @property
    def name(self) -> str:
        low = self._format_probability(self.target_alphas[0], 2)
        high = self._format_probability(self.target_alphas[-1], 2)
        delta = self._format_probability(
            self.global_regularization,
            3,
        )
        base = (
            f"projected_optimization_{self.projection}_{self.score}"
            f"_a_band_{low}_{high}_global_{delta}"
        )
        if self.n1 != 100:
            base += f"_n1_{self.n1}"
        return base

    @property
    def objective_kind(self) -> str:
        return (
            "mean_band_target_a_weighted_inverse_probability_minus_one"
            f"_global_{self.global_regularization:.3f}"
        )

    def _select_band_indices(self) -> None:
        prior_index = int(
            torch.abs(self.taus_range - self.tau_prior).argmin().item()
        )
        selected = select_calibration_positions(
            self.taus_range[:prior_index + 1],
            torch.tensor(
                self.target_alphas,
                dtype=self.taus_range.dtype,
                device=self.taus_range.device,
            ),
        )
        self._band_anchor_indices = tuple(
            int(index)
            for index in selected.reshape(-1).tolist()
        )

    def _band_weights(
            self,
            event_times: torch.Tensor,
            prior_q: torch.Tensor,
            quantile_est: torch.Tensor | None,
    ) -> tuple[torch.Tensor, float, int]:
        if quantile_est is None:
            raise ValueError(
                "Band-target DAPRO requires candidate quantile estimates."
            )
        if self._band_anchor_indices is None:
            self._select_band_indices()
        band_q = quantile_est[:, list(self._band_anchor_indices)]
        tolerance = 1e-6
        within_prior = bool(torch.all(
            band_q
            <= prior_q.reshape(-1, 1).to(band_q.dtype) + tolerance
        ).item())
        if not within_prior:
            violations = int((
                band_q
                > prior_q.reshape(-1, 1).to(band_q.dtype) + tolerance
            ).sum().item())
            raise ValueError(
                "A band-target anchor exceeds q_prior for "
                f"{violations} row/candidate pairs."
            )
        indicators = (
            event_times.reshape(-1, 1) < band_q
        ).to(torch.float64)
        mean_indicator = indicators.mean(dim=1)
        regularized = (
            mean_indicator + self.global_regularization
        ) / (1 + self.global_regularization)
        return (
            regularized,
            float(mean_indicator.mean().item()),
            int(within_prior),
        )

    def phase1_objective_weights(
            self,
            event_times: torch.Tensor,
            prior_q: torch.Tensor,
            quantile_est: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weights, rate, within_prior = self._band_weights(
            event_times,
            prior_q,
            quantile_est,
        )
        self._band_phase1_mean_indicator = rate
        self._band_phase1_within_prior = within_prior
        return weights

    def phase2_objective_weights(
            self,
            event_times: torch.Tensor,
            prior_q: torch.Tensor,
            quantile_est: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weights, rate, within_prior = self._band_weights(
            event_times,
            prior_q,
            quantile_est,
        )
        self._band_phase2_mean_indicator = rate
        self._band_phase2_within_prior = within_prior
        return weights

    def objective_metadata(self) -> dict:
        if self._band_anchor_indices is None:
            return {}
        return {
            "band_target_alpha_low": self.target_alphas[0],
            "band_target_alpha_high": self.target_alphas[-1],
            "band_target_alpha_count": len(self.target_alphas),
            "band_target_anchor_index_low": self._band_anchor_indices[0],
            "band_target_anchor_index_high": self._band_anchor_indices[-1],
            "band_target_phase1_mean_indicator": (
                self._band_phase1_mean_indicator
            ),
            "band_target_phase2_mean_indicator": (
                self._band_phase2_mean_indicator
            ),
            "band_target_phase1_within_prior": (
                self._band_phase1_within_prior
            ),
            "band_target_phase2_within_prior": (
                self._band_phase2_within_prior
            ),
            "global_regularization": self.global_regularization,
        }


# Public algorithm name.  The manuscript and normal experiment factories use
# this definitive implementation; the original objective remains available as
# ``LegacyMeanWeightDAPRO`` for reproducible ablations.
DAPRO = DefinitiveCRCDAPRO


def store(p_val, p_test, test_idxs, val_idxs, t, prior_q, final_C):

    # Time arrangement is usually the same for all, so we only need it once
    time_arrange = torch.arange(0, p_test.shape[-1]).cpu()

    # curr_probabilities = p_test.cpu()  # Move to CPU to save GPU memory
    all_probabilities = torch.ones(prior_q.shape[0], p_test.shape[1]).cpu()
    all_probabilities[test_idxs] = p_test.cpu()
    all_probabilities[val_idxs] = p_val.cpu()
    expected_c1 = all_probabilities * (prior_q.cpu().unsqueeze(1) - time_arrange.unsqueeze(0)) + time_arrange.unsqueeze(0)
    # expected_c2 = ((time_arrange * all_probabilities.cumprod(dim=-1)[:, :-1]) * (1 - all_probabilities[:, -1])).sum(dim=-1) + \
    #               all_probabilities.prod(dim=-1) * prior_q
    N, T = all_probabilities.shape
    device = all_probabilities.device

    log_p = torch.log(all_probabilities + 1e-8).cumsum(dim=-1)
    counts = torch.arange(1, T + 1, device=device).view(1, T)
    p_hat_geometric = torch.exp(log_p / counts)  # (N, T)

    expected_geometric = p_hat_geometric * (prior_q.unsqueeze(1) - time_arrange.unsqueeze(0)) \
                         + time_arrange.unsqueeze(0)

    S = all_probabilities.cumprod(dim=-1)
    S_padded = torch.cat([torch.ones(N, 1, device=device), S], dim=-1)

    S_j = S.unsqueeze(1).expand(N, T, T)
    S_i_minus_1 = S_padded[:, :-1].unsqueeze(2).expand(N, T, T)

    S_rel = torch.triu(S_j / S_i_minus_1)

    ones_matrix = torch.ones(N, T, 1, device=device)
    S_rel_prev = torch.cat([ones_matrix, S_rel[:, :, :-1]], dim=-1)
    S_rel_prev = torch.triu(S_rel_prev)

    p_fail = (1 - all_probabilities).unsqueeze(1).expand(N, T, T)
    prob_die_at_k = S_rel_prev * p_fail

    time_matrix = time_arrange.view(1, 1, T).expand(N, T, T)
    sum_term = (time_matrix * prob_die_at_k).sum(dim=2)

    prob_survive_all = S_rel[:, :, -1]
    expected_c2 = sum_term + (prob_survive_all * prior_q.unsqueeze(1).cpu())


    data_to_save = {
        'time_arrange': time_arrange,
        'expected_c': expected_c1,
        'expected_c2': expected_c2,
        'expected_geometric': expected_geometric,
        'event_times': t.cpu(),
        'prior_q': prior_q.cpu(),
        'final_Cs': final_C.cpu(),
        'test_idxs': torch.Tensor(test_idxs),
        'val_idxs': torch.Tensor(val_idxs),
    }

    # 4. Save to a file
    torch.save(data_to_save, 'projected_optimized_evaluation_plot_data.pt')
