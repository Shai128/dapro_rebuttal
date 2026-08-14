import numpy as np
import torch

from src.predictive_bounds.budget_allocators.budget_allocator import (
    BudgetAllocator,
    BudgetAllocationResult,
    candidate_reach_probabilities,
)
from src.predictive_bounds.budget_allocators.adaptive_optimized_allocator import (
    phase1_empirical_budget_limit,
)
from src.predictive_bounds.budget_allocators.vectorized_adaptive_allocator_patch import (
    _enforce_terminal_probability_floor,
    simulate_process_vectorized,
)
from src.predictive_bounds.calibration.calibration_utils import (
    get_prior,
    select_calibration_positions,
)


_DEFAULT_TERMINAL_PI_MIN = object()


class RandomAdaptiveOptimizedBudgetAllocator(BudgetAllocator):
    """Implement fixed-shape, one-parameter continuation schedules.

    Mechanism:
    1. Uses a fully observed Phase-I set to tune one aggressiveness parameter
       to the remaining budget.
    2. Applies the resulting fixed time schedule to every eligible Phase-II
       sample.
    3. Optionally transforms the cumulative reach path using a terminal
       propensity floor.  The no-floor variant is the only structurally exact
       constant-probability policy; a mixture or binding hard floor changes the
       executed conditional probabilities.

    ``constant`` is the historical constant-continuation reference.  The two
    alternative families are intentionally low capacity, so CRC can tune the
    scalar on Phase I without a separate policy-shape fitting split.
    """

    def __init__(
            self,
            conditional_grid,
            budget_per_sample,
            taus_range,
            tau_prior,
            m_upper_bound,
            reach_t_max_is_success=False,
            terminal_pi_min=_DEFAULT_TERMINAL_PI_MIN,
            terminal_floor_mode="mixture",
            budget_control_mode="empirical",
            schedule_family="constant",
            schedule_alpha=1.0,
            phase1_size=100):
        super().__init__(budget_per_sample, taus_range, tau_prior)
        self.conditional_grid = conditional_grid
        if not np.isfinite(m_upper_bound) or m_upper_bound < 1:
            raise ValueError("`m_upper_bound` must be finite and at least one.")
        self._uses_default_floor = (
            terminal_pi_min is _DEFAULT_TERMINAL_PI_MIN
        )
        self.min_pi = (
            1.0 / float(m_upper_bound)
            if self._uses_default_floor
            else terminal_pi_min
        )
        if self.min_pi is not None and not 0 < self.min_pi <= 1:
            raise ValueError("`terminal_pi_min` must lie in (0, 1].")
        if terminal_floor_mode not in {"mixture", "hard", "none"}:
            raise ValueError(
                "`terminal_floor_mode` must be one of: mixture, hard, none."
            )
        if terminal_floor_mode == "none":
            # A disabled floor must not retain a misleading numerical floor
            # in feasibility checks, names, or stored diagnostics.
            self.min_pi = None
        elif self.min_pi is None:
            terminal_floor_mode = "none"
        self.terminal_floor_mode = terminal_floor_mode
        if budget_control_mode not in {"empirical", "crc"}:
            raise ValueError(
                "`budget_control_mode` must be one of: empirical, crc."
            )
        self.budget_control_mode = budget_control_mode
        if schedule_family not in {
                "constant",
                "complement_power",
                "power_reach",
        }:
            raise ValueError(
                "`schedule_family` must be one of: constant, "
                "complement_power, power_reach."
            )
        if not np.isfinite(schedule_alpha) or schedule_alpha <= 0:
            raise ValueError("`schedule_alpha` must be finite and positive.")
        if not isinstance(phase1_size, (int, np.integer)) or phase1_size <= 0:
            raise ValueError("`phase1_size` must be a positive integer.")
        self.schedule_family = schedule_family
        self.schedule_alpha = float(schedule_alpha)
        self.phase1_size = int(phase1_size)
        self.reach_t_max_is_success = reach_t_max_is_success
        # self.budget_per_sample *= 500

    @property
    def name(self) -> str:
        if self.schedule_family != "constant":
            alpha = (
                f"{self.schedule_alpha:.3f}"
                .rstrip("0")
                .rstrip(".")
                .replace(".", "p")
            )
            base = f"random_schedule_{self.schedule_family}_alpha_{alpha}"
            if self.budget_control_mode != "empirical":
                base += f"_{self.budget_control_mode}"
            if self.phase1_size != 100:
                base += f"_n1_{self.phase1_size}"
            return base
        if (
            self._uses_default_floor
            and self.terminal_floor_mode == "mixture"
        ):
            suffix = (
                ""
                if self.budget_control_mode == "empirical"
                else f"_{self.budget_control_mode}"
            )
            base = f"random_adaptive_optimized{suffix}"
            if self.phase1_size != 100:
                base += f"_n1_{self.phase1_size}"
            return base
        if self.min_pi is None or self.terminal_floor_mode == "none":
            base = "random_adaptive_optimized_no_terminal_floor"
            if self.budget_control_mode != "empirical":
                base += f"_{self.budget_control_mode}"
            if self.phase1_size != 100:
                base += f"_n1_{self.phase1_size}"
            return base
        floor = (
            f"{self.min_pi:.6f}"
            .rstrip("0")
            .rstrip(".")
            .replace(".", "p")
        )
        base = (
            f"random_adaptive_optimized_{self.terminal_floor_mode}"
            f"_terminal_floor_{floor}"
        )
        if self.budget_control_mode != "empirical":
            base += f"_{self.budget_control_mode}"
        if self.phase1_size != 100:
            base += f"_n1_{self.phase1_size}"
        return base

    def _conditional_schedule(
            self,
            aggressiveness: float | torch.Tensor,
            width: int,
            *,
            device: torch.device,
            dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return a one-row conditional schedule, monotone in aggressiveness.

        ``aggressiveness`` always runs from the minimum-cost policy at zero to
        always continue at one.  For ``complement_power`` the parameter in the
        user's notation is ``lambda = 1 - aggressiveness``.  For
        ``power_reach``, cumulative reach is

            R_t = aggressiveness ** t**alpha,

        so the conditional probability is the ratio of consecutive reaches.
        """
        a = torch.as_tensor(aggressiveness, dtype=dtype, device=device).clamp(
            0.0,
            1.0,
        )
        one_based_time = torch.arange(
            1,
            width + 1,
            dtype=dtype,
            device=device,
        )
        if self.schedule_family == "constant":
            return a.expand(1, width)
        if self.schedule_family == "complement_power":
            requested_lambda = 1 - a
            conditional = 1 - torch.pow(
                requested_lambda,
                self.schedule_alpha * one_based_time,
            )
            return conditional.unsqueeze(0)

        previous_power = torch.pow(
            one_based_time - 1,
            self.schedule_alpha,
        )
        current_power = torch.pow(
            one_based_time,
            self.schedule_alpha,
        )
        increments = current_power - previous_power
        return torch.pow(a, increments).unsqueeze(0)

    def allocate_budget(self, probability_est: torch.Tensor, x: torch.Tensor, t: torch.Tensor,
                        quantile_est: torch.Tensor) -> BudgetAllocationResult:
        device = self.conditional_grid.device
        N, T_max_curr, T_max_future = self.conditional_grid.shape
        total_budget = self.budget_per_sample * N
        val_size = self.phase1_size
        if val_size >= N:
            raise ValueError("`phase1_size` must be smaller than the sample count.")
        perm = np.random.permutation(N)
        val_idxs = perm[:val_size]
        test_idxs = perm[val_size:]
        prior_q = get_prior(
            quantile_est,
            self.taus_range,
            self.tau_prior,
        ).clamp(max=T_max_curr)

        # --- Data Splitting ---
        # Validation Set: Used to learn the optimal policy parameters (lambda)
        val_prior_q = prior_q[val_idxs]
        t_val = t[val_idxs]
        val_budget_used = torch.minimum(t_val, val_prior_q).sum().item()
        if total_budget < val_budget_used:
            raise ValueError("Total budget is too small")

        # Test Set: The data we need to mine
        test_prior_q = prior_q[test_idxs]
        t_test = t[test_idxs]

        # Global Target Budget for the test set
        # We assume the budget density (budget per sample) should be consistent
        target_budget_avg = (self.budget_per_sample * N - val_budget_used) / (N - val_size)
        # The constant-policy reference uses only the grid shape and dtype.
        # Avoid advanced-index copies of the full (N,T,T) conditional grid.
        val_expected_remaining = torch.ones(
            (val_size, T_max_curr),
            dtype=self.conditional_grid.dtype,
            device=device,
        )
        test_expected_remaining = torch.ones(
            (N - val_size, T_max_curr),
            dtype=self.conditional_grid.dtype,
            device=device,
        )

        def schedule(aggressiveness):
            return self._conditional_schedule(
                aggressiveness,
                T_max_curr,
                device=device,
                dtype=val_expected_remaining.dtype,
            )

        val_p_func = lambda x: schedule(x).expand(val_size, -1)
        test_p_func = lambda x: schedule(x).expand(N - val_size, -1)

        def validation_expected_cost(probability):
            return simulate_process_vectorized(
                val_expected_remaining,
                val_prior_q,
                t_val,
                probability,
                stochastic=False,
                reach_t_max_is_success=self.reach_t_max_is_success,
                pi_func=val_p_func,
                terminal_pi_min=self.min_pi,
                terminal_floor_mode=self.terminal_floor_mode,
            )[2]

        active_lengths = torch.minimum(
            t_val.to(torch.long),
            val_prior_q.to(torch.long),
        ).clamp(min=0, max=T_max_curr)
        floor_minimum_cost = (
            0.0
            if self.min_pi is None or self.terminal_floor_mode == "none"
            else self.min_pi * active_lengths.sum().item()
        )
        empirical_budget_limit = phase1_empirical_budget_limit(
            target_budget_avg,
            val_size,
            T_max_curr,
            self.budget_control_mode,
            phase2_sample_count=len(test_idxs),
        )
        if empirical_budget_limit < 0:
            raise ValueError(
                "The finite-sample Random budget correction is infeasible: "
                f"its Phase-I empirical-cost limit is "
                f"{empirical_budget_limit:.6g} per sample. Increase the "
                "budget or Phase-I sample count."
            )
        target_validation_cost = empirical_budget_limit * val_size
        tolerance = 1e-7 * max(1.0, target_validation_cost)
        if floor_minimum_cost > target_validation_cost + tolerance:
            raise ValueError(
                "The terminal probability floor makes the Random expected "
                "budget infeasible: minimum Phase-I cost "
                f"{floor_minimum_cost / val_size:.6g} exceeds target "
                f"{empirical_budget_limit:.6g} per sample."
            )

        lam_low, lam_high = 0.0, 1.0
        low_cost = validation_expected_cost(lam_low)
        high_cost = validation_expected_cost(lam_high)
        if low_cost > target_validation_cost + tolerance:
            raise ValueError(
                "Could not construct a Random policy satisfying the expected "
                "budget. Reduce the terminal floor or increase the budget."
            )
        if high_cost <= target_validation_cost + tolerance:
            # Following every eligible interaction is feasible.
            best_lambda = lam_high
        else:
            for _ in range(50):
                mid = (lam_low + lam_high) / 2
                val_expected_cost = validation_expected_cost(mid)
                if val_expected_cost <= target_validation_cost:
                    lam_low = mid
                else:
                    lam_high = mid

            # Every family is parameterized so cost increases with
            # aggressiveness; the lower endpoint is known feasible.
            best_lambda = lam_low

        phase2_active_lengths = torch.minimum(
            t_test.to(torch.long),
            test_prior_q.to(torch.long),
        ).clamp(min=0, max=T_max_curr)
        if (
                best_lambda <= 0
                and self.min_pi is None
                and torch.any(phase2_active_lengths > 0)
        ):
            raise ValueError(
                "The available budget selects zero continuation probability, "
                "which violates the positivity required by inverse-propensity "
                "calibration. Increase the budget."
            )

        _, _, tuned_validation_expected_cost = simulate_process_vectorized(
            val_expected_remaining,
            val_prior_q,
            t_val,
            best_lambda,
            stochastic=False,
            reach_t_max_is_success=self.reach_t_max_is_success,
            pi_func=val_p_func,
            terminal_pi_min=self.min_pi,
            terminal_floor_mode=self.terminal_floor_mode,
        )
        # ``t_test`` appears below only because this offline simulator must stop
        # charging after an event is revealed.  The already-selected scalar
        # probability and ``test_p_func`` do not depend on Phase-II labels.
        _, _, test_expected_cost = simulate_process_vectorized(
            test_expected_remaining,
            test_prior_q,
            t_test,
            best_lambda,
            stochastic=False,
            reach_t_max_is_success=self.reach_t_max_is_success,
            pi_func=test_p_func,
            terminal_pi_min=self.min_pi,
            terminal_floor_mode=self.terminal_floor_mode,
        )
        acquisition_uniforms = self.get_acquisition_uniforms(
            N,
            T_max_curr,
            device=device,
            dtype=test_expected_remaining.dtype,
        )
        test_uniforms = (
            None
            if acquisition_uniforms is None
            else acquisition_uniforms[test_idxs]
        )
        if acquisition_uniforms is None:
            self.reset_acquisition_rng()
        test_C, test_C_probs, test_total_used = simulate_process_vectorized(
            test_expected_remaining,
            test_prior_q,
            t_test,
            best_lambda,
            stochastic=True,
            reach_t_max_is_success=self.reach_t_max_is_success,
            pi_func=test_p_func,
            terminal_pi_min=self.min_pi,
            terminal_floor_mode=self.terminal_floor_mode,
            uniforms=test_uniforms,
        )

        test_avg_cost = test_total_used / len(test_C)
        # print(f"err: {val_avg_cost - test_avg_cost}")
        total_budget_used = test_total_used + val_budget_used

        # Reconstruct full-size tensors
        # Val C: Standard logic (stop at event or horizon)
        val_C = val_prior_q
        # For Validation set, we don't care about C_probs (set to 1.0 or dummy)
        val_C_probs = torch.ones(val_size, device=device)

        # Concatenate
        final_C = torch.empty(
            N,
            device=device,
            dtype=val_C.dtype,
        )
        final_C[val_idxs] = val_C
        final_C[test_idxs] = test_C.to(final_C.dtype)

        # final_C[final_C > t] = torch.max(prior_q[final_C > t], final_C[final_C > t])
        final_C_probs = torch.empty(N, device=device, dtype=test_C_probs.dtype)
        final_C_probs[val_idxs] = val_C_probs.to(final_C_probs.dtype)
        final_C_probs[test_idxs] = test_C_probs.to(final_C_probs.dtype)
        mean_val_weight = (1/final_C_probs).mean().item()
        max_val_weight = (1/final_C_probs).max().item()

        total_expected_budget = val_budget_used + test_expected_cost
        crc_rho = (val_size + 1) / len(test_idxs)
        crc_envelope_upper_bound = (1 + crc_rho) * T_max_curr
        crc_selector_left_side = (
            val_size / (val_size + 1)
            * (tuned_validation_expected_cost / val_size)
            + crc_envelope_upper_bound / (val_size + 1)
        )
        raw_schedule = schedule(best_lambda).to(torch.float64)
        phase1_active_mask = (
            torch.arange(T_max_curr, device=device).unsqueeze(0)
            < active_lengths.unsqueeze(1)
        )
        phase1_raw_terminal_probability = torch.exp(torch.where(
            phase1_active_mask,
            torch.log(raw_schedule.clamp_min(
                torch.finfo(torch.float64).tiny
            )),
            torch.zeros((), dtype=torch.float64, device=device),
        ).sum(dim=1))
        phase2_active_mask = (
            torch.arange(T_max_curr, device=device).unsqueeze(0)
            < phase2_active_lengths.unsqueeze(1)
        )
        phase2_raw_terminal_probability = torch.exp(torch.where(
            phase2_active_mask,
            torch.log(raw_schedule.clamp_min(
                torch.finfo(torch.float64).tiny
            )),
            torch.zeros((), dtype=torch.float64, device=device),
        ).sum(dim=1))
        constant_test_probabilities = raw_schedule.expand(
            len(test_idxs),
            -1,
        )
        executed_test_probabilities = _enforce_terminal_probability_floor(
            constant_test_probabilities,
            test_prior_q,
            self.min_pi,
            self.terminal_floor_mode,
        )
        focus_time = torch.arange(T_max_curr, device=device).unsqueeze(0)
        focus_active = focus_time < phase2_active_lengths.unsqueeze(1)
        focus_path = torch.where(
            focus_active,
            executed_test_probabilities,
            torch.ones((), dtype=torch.float64, device=device),
        ).cumprod(dim=1)
        focus_row_expected_cost = (
            focus_path * focus_active.to(torch.float64)
        ).sum(dim=1).detach().cpu().numpy()
        focus_terminal_probability = torch.exp(torch.where(
            focus_active,
            torch.log(executed_test_probabilities.clamp_min(
                torch.finfo(torch.float64).tiny
            )),
            torch.zeros((), dtype=torch.float64, device=device),
        ).sum(dim=1)).detach().cpu().numpy()

        eligible_target_indices = torch.nonzero(
            self.taus_range <= 0.10 + 1e-12,
            as_tuple=False,
        ).reshape(-1)
        if len(eligible_target_indices) > 0 and self.tau_prior > 0.10:
            target_index = int(select_calibration_positions(
                self.taus_range[:int(eligible_target_indices[-1]) + 1],
                torch.tensor(
                    [0.10],
                    dtype=self.taus_range.dtype,
                    device=self.taus_range.device,
                ),
            ).item())
            focus_a = (
                t_test.reshape(-1)
                < quantile_est[test_idxs, target_index].reshape(-1)
            ).detach().cpu().numpy().astype(bool)
            focus_rate = float(np.mean(focus_a))
            focus_query_share = (
                float(np.sum(focus_row_expected_cost[focus_a]))
                / max(
                    float(np.sum(focus_row_expected_cost)),
                    np.finfo(np.float64).tiny,
                )
            )

            def focus_mean(values, mask):
                return (
                    float(np.mean(values[mask])) if np.any(mask) else np.nan
                )

            if (
                    np.std(focus_a.astype(np.float64)) > 0
                    and np.std(focus_terminal_probability) > 0
            ):
                focus_pi_correlation = float(np.corrcoef(
                    focus_a.astype(np.float64),
                    focus_terminal_probability,
                )[0, 1])
            else:
                focus_pi_correlation = np.nan
            focus_metrics = {
                "phase2_focus_a_rate": focus_rate,
                "phase2_focus_expected_query_share": focus_query_share,
                "phase2_focus_expected_query_lift": (
                    focus_query_share / focus_rate
                    if focus_rate > 0
                    else np.nan
                ),
                "phase2_focus_mean_expected_queries": focus_mean(
                    focus_row_expected_cost,
                    focus_a,
                ),
                "phase2_nonfocus_mean_expected_queries": focus_mean(
                    focus_row_expected_cost,
                    ~focus_a,
                ),
                "phase2_focus_mean_terminal_probability": focus_mean(
                    focus_terminal_probability,
                    focus_a,
                ),
                "phase2_nonfocus_mean_terminal_probability": focus_mean(
                    focus_terminal_probability,
                    ~focus_a,
                ),
                "phase2_focus_terminal_probability_correlation": (
                    focus_pi_correlation
                ),
            }
        else:
            focus_metrics = {}

        def floor_changes_terminal(raw_probability):
            if self.min_pi is None or self.terminal_floor_mode == "none":
                return torch.zeros_like(raw_probability, dtype=torch.bool)
            if self.terminal_floor_mode == "mixture":
                transformed = (
                    self.min_pi
                    + (1 - self.min_pi) * raw_probability
                )
            else:
                transformed = raw_probability.clamp_min(self.min_pi)
            return ~torch.isclose(
                transformed,
                raw_probability,
                rtol=1e-12,
                atol=1e-15,
            )

        phase1_floor_changes = floor_changes_terminal(
            phase1_raw_terminal_probability
        )
        phase2_floor_changes = floor_changes_terminal(
            phase2_raw_terminal_probability
        )
        additional_metrics = {
            "phase1_tuned_expected_cost_per_sample": (
                tuned_validation_expected_cost / val_size
            ),
            "phase1_empirical_budget_limit_per_sample": (
                empirical_budget_limit
            ),
            "budget_control_mode": self.budget_control_mode,
            "crc_finite_sample_penalty_per_sample": (
                target_budget_avg - empirical_budget_limit
            ),
            "crc_selector_left_side_per_sample": crc_selector_left_side,
            "crc_distribution_free_envelope_upper_bound": (
                crc_envelope_upper_bound
            ),
            "crc_transformed_loss_rho": crc_rho,
            "crc_distribution_free_envelope_used": int(
                self.budget_control_mode == "crc"
            ),
            "budget_guarantee_kind": (
                "crc_marginal_expected_total_budget"
                if self.budget_control_mode == "crc"
                else "empirical_budget_fit_only"
            ),
            "crc_selector_valid": int(
                self.budget_control_mode != "crc"
                or crc_selector_left_side <= target_budget_avg + 1e-7
            ),
            "phase1_tuned_expected_cost_total": (
                tuned_validation_expected_cost
            ),
            "phase1_realized_cost_per_sample": (
                val_budget_used / val_size
            ),
            "phase1_realized_cost_total": val_budget_used,
            "phase2_target_budget_per_sample": target_budget_avg,
            "phase2_expected_cost_per_sample": (
                test_expected_cost / len(test_idxs)
            ),
            "phase2_expected_cost_total": test_expected_cost,
            "phase2_expected_budget_gap_per_sample": (
                test_expected_cost / len(test_idxs) - target_budget_avg
            ),
            "phase2_expected_budget_valid": int(
                test_expected_cost / len(test_idxs)
                <= target_budget_avg + 1e-7
            ),
            "phase2_realized_cost_per_sample": test_avg_cost,
            **focus_metrics,
            "total_expected_budget": total_expected_budget,
            "total_expected_budget_per_sample": (
                total_expected_budget / N
            ),
            "total_expected_budget_gap": (
                total_expected_budget - total_budget
            ),
            "total_expected_budget_gap_per_sample": (
                total_expected_budget / N - self.budget_per_sample
            ),
            "total_expected_budget_valid": int(
                total_expected_budget <= total_budget + 1e-7 * max(1.0, N)
            ),
            "configured_total_budget": total_budget,
            "random_constant_probability": best_lambda,
            "random_constant_continuation_probability": best_lambda,
            "random_shadow_parameter": best_lambda,
            "random_schedule_family": self.schedule_family,
            "random_schedule_alpha": self.schedule_alpha,
            "random_schedule_aggressiveness": best_lambda,
            "random_schedule_requested_lambda": (
                1 - best_lambda
                if self.schedule_family == "complement_power"
                else np.nan
            ),
            "random_executed_policy_is_constant": int(
                self.schedule_family == "constant"
                and not phase2_floor_changes.any().item()
            ),
            "random_policy_uses_phase2_event_times": 0,
            "random_parameter_semantics": (
                "conditional_continuation_probability_not_lagrange_multiplier"
            ),
            "phase1_terminal_floor_changed_fraction": (
                phase1_floor_changes.to(torch.float64).mean().item()
            ),
            "phase2_terminal_floor_changed_fraction": (
                phase2_floor_changes.to(torch.float64).mean().item()
            ),
            "phase2_max_unfloored_inverse_probability": (
                1
                / phase2_raw_terminal_probability.clamp_min(
                    torch.finfo(torch.float64).tiny
                ).min().item()
            ),
            "terminal_pi_min": (
                self.min_pi if self.min_pi is not None else np.nan
            ),
            "terminal_floor_mode": self.terminal_floor_mode,
            "floor_minimum_expected_cost_per_sample": (
                floor_minimum_cost / val_size
            ),
            "phase1_sample_count": val_size,
            "phase2_sample_count": len(test_idxs),
            "phase2_mean_inverse_probability": (
                (1 / test_C_probs).mean().item()
            ),
            "phase2_variance_inverse_probability": (
                (1 / test_C_probs.to(torch.float64))
                .var(unbiased=False)
                .item()
            ),
        }
        all_conditionals = torch.ones(
            (N, T_max_curr), dtype=torch.float64, device=device
        )
        all_conditionals[test_idxs] = executed_test_probabilities
        return BudgetAllocationResult(
            quantile_est,
            final_C,
            final_C_probs,
            total_budget_used,
            mean_weight=mean_val_weight,
            max_weight=max_val_weight,
            additional_metrics=additional_metrics,
            candidate_C_probs=candidate_reach_probabilities(
                all_conditionals,
                quantile_est,
                infinity_value=T_max_curr + 1,
            ),
        )


class ConstantCRCBudgetAllocator(RandomAdaptiveOptimizedBudgetAllocator):
    """Stable constant-probability allocation with CRC budget control.

    Phase I is fully observed and is the only part of the calibration sample
    used to choose the scalar continuation probability.  Every eligible
    Phase-II interaction uses that same core probability, independently of the
    sample, time, model output, and latent event time.

    A row-level always-follow mixture of probability ``1 / m_upper_bound`` is
    applied to the core policy.  Without it, an event at time ``t`` has reach
    probability ``p ** t``; at the real-data horizon of 200 this produced
    inverse-propensity weights as large as 1e29.  The Horvitz--Thompson
    estimator remained unbiased in theory, but its expectation was carried by
    events far too rare to appear in 50 experiment seeds.  The mixture keeps
    the exact executed propensities in ``C_probs``, preserves unbiasedness, and
    bounds every inverse-propensity weight by ``m_upper_bound``.  CRC tuning
    accounts for the mixture's acquisition cost.

    The parent class remains configurable for named ablations.  This narrow
    wrapper prevents the paper baseline from accidentally being constructed
    with empirical budget tuning, a time-varying core schedule, or an
    unbounded terminal propensity.
    """

    def __init__(
            self,
            conditional_grid,
            budget_per_sample,
            taus_range,
            tau_prior,
            m_upper_bound,
            reach_t_max_is_success=False,
            phase1_size=100):
        super().__init__(
            conditional_grid=conditional_grid,
            budget_per_sample=budget_per_sample,
            taus_range=taus_range,
            tau_prior=tau_prior,
            m_upper_bound=m_upper_bound,
            reach_t_max_is_success=reach_t_max_is_success,
            terminal_pi_min=1.0 / float(m_upper_bound),
            terminal_floor_mode="mixture",
            budget_control_mode="crc",
            schedule_family="constant",
            schedule_alpha=1.0,
            phase1_size=phase1_size,
        )
