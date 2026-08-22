import re

import numpy as np

from src.predictive_bounds.budget_allocators.budget_allocator import (
    BudgetAllocationResult,
    BudgetAllocator,
    candidate_reach_probabilities,
    summarize_expected_budget,
)
from src.predictive_bounds.budget_allocators.dapro_projection_metrics import (
    compute_dapro_projection_metrics,
)
from src.predictive_bounds.budget_allocators.dapro_objectives import (
    history_soft_objective_coefficients,
    realized_target_weights,
)
from src.predictive_bounds.budget_allocators.optimization_solver_utils import (
    solve_binned_deployable_policy,
    solve_exact_fast,
    solve_time_only_cumulative_policy,
)
from src.predictive_bounds.budget_allocators.metric_optimal_allocator import (
    antitonic_pav_bases,
    solve_common_scale,
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
    select_upb_calibration_positions,
)

import torch

from src.predictive_bounds.survival_utils.compute_mean_time_given_pmf import compute_quantile_survival_time


CAUSAL_SHARED_PAV_CAP_VERSION = "causal_shared_pav_v1"


def _apply_causal_cumulative_row_cost_cap(
        base_conditionals: torch.Tensor,
        row_prior_q: torch.Tensor,
        row_cost_cap: float,
        terminal_pi_min: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cap every cumulative-reach path without looking into its future.

    A row-local affine contraction chosen from the *full* cumulative path is
    not deployable for a history-adaptive policy: its first-turn probability
    changes when an as-yet-unobserved future score changes.  This projection
    instead maintains an excess-reach token bucket.  At turn ``t`` it spends
    as much of the remaining bucket as the base cumulative reach requests,
    and therefore depends only on the base path through ``t``.

    If ``q_i`` is the known acquisition horizon and ``epsilon`` is the
    terminal propensity floor, the initial bucket is

        row_cost_cap - epsilon * q_i.

    Consequently every possible future path satisfies

        sum_{t <= q_i} rho_it <= row_cost_cap,

    while cumulative reach stays non-increasing, no larger than the base
    path, and at least ``epsilon``.  The returned Boolean vector identifies
    rows whose active cumulative path was changed.
    """
    probabilities = base_conditionals.to(torch.float64)
    if probabilities.ndim != 2:
        raise ValueError("`base_conditionals` must be a matrix.")
    if not bool(torch.isfinite(probabilities).all()):
        raise ValueError("Base continuation probabilities must be finite.")
    if bool(((probabilities < 0) | (probabilities > 1)).any()):
        raise ValueError("Base continuation probabilities must lie in [0, 1].")
    if not np.isfinite(row_cost_cap) or row_cost_cap <= 0:
        raise ValueError("`row_cost_cap` must be finite and positive.")
    if not 0 < terminal_pi_min <= 1:
        raise ValueError("`terminal_pi_min` must lie in (0, 1].")

    row_q = row_prior_q.to(
        device=probabilities.device,
        dtype=torch.long,
    ).reshape(-1).clamp(min=0, max=probabilities.shape[1])
    if len(row_q) != len(probabilities):
        raise ValueError("`row_prior_q` must have one value per policy row.")

    base_cumulative = probabilities.cumprod(dim=1)
    active = (
        torch.arange(
            probabilities.shape[1],
            device=probabilities.device,
        ).unsqueeze(0)
        < row_q.unsqueeze(1)
    )
    if bool((base_cumulative[active] < terminal_pi_min - 1e-10).any()):
        raise ValueError(
            "The active base cumulative path must respect the terminal floor."
        )
    floor_cost = terminal_pi_min * row_q.to(torch.float64)
    if torch.any(floor_cost > row_cost_cap + 1e-10):
        raise ValueError(
            "The terminal floor is incompatible with the risk-candidate "
            "row cost cap."
        )

    remaining_excess = (row_cost_cap - floor_cost).clamp_min(0.0)
    capped_cumulative = torch.ones_like(base_cumulative)
    previous = torch.ones(
        len(probabilities),
        dtype=torch.float64,
        device=probabilities.device,
    )
    for step in range(probabilities.shape[1]):
        active = step < row_q
        desired_excess = (
            base_cumulative[:, step] - terminal_pi_min
        ).clamp_min(0.0)
        spent = torch.minimum(desired_excess, remaining_excess)
        current = terminal_pi_min + spent
        current = torch.minimum(current, previous)
        current = torch.minimum(current, base_cumulative[:, step])
        current = torch.where(active, current, torch.ones_like(current))
        capped_cumulative[:, step] = current
        active_spent = torch.where(
            active,
            (current - terminal_pi_min).clamp_min(0.0),
            torch.zeros_like(current),
        )
        remaining_excess = (remaining_excess - active_spent).clamp_min(0.0)
        previous = torch.where(active, current, previous)

    time = torch.arange(
        probabilities.shape[1],
        device=probabilities.device,
    ).unsqueeze(0)
    active = time < row_q.unsqueeze(1)
    previous_cumulative = torch.cat(
        [
            torch.ones(
                (len(probabilities), 1),
                dtype=torch.float64,
                device=probabilities.device,
            ),
            capped_cumulative[:, :-1],
        ],
        dim=1,
    )
    capped_conditionals = (
        capped_cumulative
        / previous_cumulative.clamp_min(torch.finfo(torch.float64).tiny)
    ).clamp(max=1.0)
    capped_conditionals = torch.where(
        active,
        capped_conditionals,
        torch.ones_like(capped_conditionals),
    )
    changed = (
        (capped_cumulative - torch.where(
            active,
            base_cumulative,
            torch.ones_like(base_cumulative),
        )).abs() > 1e-10
    ).any(dim=1)
    return capped_conditionals, changed


def _solve_shared_causal_row_cap_envelope(
        active_lengths: torch.Tensor,
        row_cost_cap: float,
        terminal_pi_min: float,
        width: int,
        *,
        objective_weights: torch.Tensor | np.ndarray | None = None,
        objective_masses: torch.Tensor | np.ndarray | None = None,
) -> tuple[torch.Tensor, dict]:
    """Fit the target-optimal shared full-horizon CRC cap envelope.

    The envelope is learned only from the policy-fit objective coefficients.
    It minimizes their pooled inverse-reach objective subject to a worst-path
    horizon-cost cap.  Taking the pointwise minimum of a causal base path and
    this frozen envelope remains causal and gives every candidate the support
    bound required by CRC.
    """
    lengths = np.asarray(
        active_lengths.detach().cpu(),
        dtype=np.int64,
    ).reshape(-1)
    if np.any(lengths < 0) or np.any(lengths > width):
        raise ValueError("Active lengths must lie in the envelope width.")
    if objective_weights is not None and objective_masses is not None:
        raise ValueError(
            "Pass either envelope objective weights or masses, not both."
        )
    if objective_masses is not None:
        masses = np.asarray(
            (
                objective_masses.detach().cpu()
                if torch.is_tensor(objective_masses)
                else objective_masses
            ),
            dtype=np.float64,
        )
        if masses.shape != (len(lengths), width):
            raise ValueError("Envelope objective masses have the wrong shape.")
        active = np.arange(width)[None, :] < lengths[:, None]
        masses = masses * active
    else:
        weights = np.ones(len(lengths), dtype=np.float64)
        if objective_weights is not None:
            weights = np.asarray(
                (
                    objective_weights.detach().cpu()
                    if torch.is_tensor(objective_weights)
                    else objective_weights
                ),
                dtype=np.float64,
            ).reshape(-1)
        if weights.shape != (len(lengths),):
            raise ValueError("Envelope objective weights have the wrong shape.")
        masses = np.zeros((len(lengths), width), dtype=np.float64)
        positive = lengths > 0
        masses[np.flatnonzero(positive), lengths[positive] - 1] = (
            weights[positive]
        )
    if not np.all(np.isfinite(masses)) or np.any(masses < 0):
        raise ValueError("Envelope objective masses must be nonnegative.")

    pooled_mass = masses.mean(axis=0, keepdims=True)
    horizon_cost = np.ones_like(pooled_mass)
    bases, block_counts = antitonic_pav_bases(
        pooled_mass,
        horizon_cost,
    )
    envelope, scale, boundary = solve_common_scale(
        bases,
        horizon_cost,
        row_cost_cap,
        terminal_pi_min,
    )
    return (
        torch.as_tensor(
            envelope[0],
            dtype=torch.float64,
            device=active_lengths.device,
        ),
        {
            "risk_budget_row_cost_cap_envelope_scale": scale,
            "risk_budget_row_cost_cap_envelope_boundary": boundary,
            "risk_budget_row_cost_cap_envelope_pav_blocks": int(
                block_counts[0]
            ),
            "risk_budget_row_cost_cap_envelope_horizon_cost": float(
                envelope[0].sum()
            ),
        },
    )


def _apply_shared_cumulative_row_cost_envelope(
        base_conditionals: torch.Tensor,
        row_prior_q: torch.Tensor,
        envelope: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Intersect a causal cumulative policy with a frozen shared envelope."""
    probabilities = base_conditionals.to(torch.float64)
    row_q = row_prior_q.to(
        device=probabilities.device,
        dtype=torch.long,
    ).reshape(-1).clamp(min=0, max=probabilities.shape[1])
    cap = envelope.to(
        device=probabilities.device,
        dtype=torch.float64,
    ).reshape(-1)
    if len(row_q) != len(probabilities) or len(cap) != probabilities.shape[1]:
        raise ValueError("Policy rows, horizons, and envelope must agree.")
    if not bool(torch.isfinite(cap).all()) or bool(((cap < 0) | (cap > 1)).any()):
        raise ValueError("The cumulative envelope must lie in [0, 1].")
    if bool((cap[1:] > cap[:-1] + 1e-12).any()):
        raise ValueError("The cumulative envelope must be non-increasing.")

    base_cumulative = probabilities.cumprod(dim=1)
    active = (
        torch.arange(
            probabilities.shape[1],
            device=probabilities.device,
        ).unsqueeze(0)
        < row_q.unsqueeze(1)
    )
    capped_cumulative = torch.minimum(base_cumulative, cap.unsqueeze(0))
    capped_cumulative = torch.where(
        active,
        capped_cumulative,
        torch.ones_like(capped_cumulative),
    )
    previous = torch.cat(
        [
            torch.ones(
                (len(probabilities), 1),
                dtype=torch.float64,
                device=probabilities.device,
            ),
            capped_cumulative[:, :-1],
        ],
        dim=1,
    )
    capped_conditionals = (capped_cumulative / previous.clamp_min(
        torch.finfo(torch.float64).tiny
    )).clamp(max=1.0)
    capped_conditionals = torch.where(
        active,
        capped_conditionals,
        torch.ones_like(capped_conditionals),
    )
    changed = (
        (capped_cumulative - torch.where(
            active,
            base_cumulative,
            torch.ones_like(base_cumulative),
        )).abs() > 1e-10
    ).any(dim=1)
    return capped_conditionals, changed


def _mean_soft_mass_variance_proxy(
        objective_masses: torch.Tensor | np.ndarray,
        conditional_probabilities: torch.Tensor | np.ndarray,
) -> float:
    """Evaluate the prefix-mass variance proxy under a fixed policy.

    For cumulative reach ``rho_it = prod_{s<=t} p_is``, this returns

        mean_i sum_t m_it * (1 / rho_it - 1).

    This is the exact objective optimized when DAPRO receives
    ``objective_masses``.  Keeping it separate from terminal hard-weight
    diagnostics prevents the soft objective from being mislabeled as the
    legacy endpoint proxy.
    """
    masses = torch.as_tensor(objective_masses, dtype=torch.float64)
    conditionals = torch.as_tensor(
        conditional_probabilities,
        dtype=torch.float64,
        device=masses.device,
    )
    if masses.ndim != 2 or conditionals.shape != masses.shape:
        raise ValueError(
            "Soft objective masses and policy conditionals must be matrices "
            "with the same shape."
        )
    if not bool(torch.isfinite(masses).all()) or bool((masses < 0).any()):
        raise ValueError("Soft objective masses must be finite and nonnegative.")
    if (
            not bool(torch.isfinite(conditionals).all())
            or bool(((conditionals < 0) | (conditionals > 1)).any())
    ):
        raise ValueError("Policy conditionals must lie in [0, 1].")
    if len(masses) == 0:
        return np.nan
    cumulative_reach = conditionals.cumprod(dim=1).clamp_min(
        torch.finfo(torch.float64).tiny
    )
    row_proxy = (
        masses * (cumulative_reach.reciprocal() - 1.0)
    ).sum(dim=1)
    return float(row_proxy.mean().item())


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
        direct_bins_match = re.fullmatch(r"direct_bins_([1-9][0-9]*)", projection)
        risk_compatible_projections = {"direct_time"}
        if direct_bins_match is not None:
            risk_compatible_projections.add(projection)
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
        ] and direct_bins_match is None:
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
            suffix += f"_{CAUSAL_SHARED_PAV_CAP_VERSION}"
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

    def phase1_objective_masses(
            self,
            event_times: torch.Tensor,
            prior_q: torch.Tensor,
            quantile_est: torch.Tensor,
            conditional_grid: torch.Tensor,
    ) -> torch.Tensor | None:
        """Optional soft event/influence mass at every observed prefix.

        Existing hard-label DAPRO variants return ``None`` and use terminal
        ``phase1_objective_weights``. Generalized soft DAPRO overrides this
        hook with model-integrated prefix masses, keeping the policy fitting,
        projection, and deployment machinery shared.
        """
        del event_times, prior_q, quantile_est, conditional_grid
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

    def phase2_objective_masses(
            self,
            event_times: torch.Tensor,
            prior_q: torch.Tensor,
            quantile_est: torch.Tensor,
            conditional_grid: torch.Tensor,
    ) -> torch.Tensor | None:
        """Audit frozen Phase-I soft coefficients on the Phase-II rows."""
        return self.phase1_objective_masses(
            event_times,
            prior_q,
            quantile_est,
            conditional_grid,
        )

    def phase2_target_indicator(
            self,
            event_times: torch.Tensor,
            prior_q: torch.Tensor,
            quantile_est: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the binary event whose acquisition variance is audited."""
        return realized_target_weights(
            event_times,
            prior_q,
            strict=True,
        )

    def objective_metadata(self) -> dict:
        return {}

    def policy_scores(
            self,
            quantile_est: torch.Tensor,
    ) -> torch.Tensor:
        """Return the predictable scalar score available at every prefix."""
        del quantile_est
        width = self.conditional_grid.shape[1]
        step = torch.arange(width, device=self.conditional_grid.device)
        return self.conditional_grid[:, step, step]

    def policy_scores_for_allocation(
            self,
            quantile_est: torch.Tensor,
            event_times: torch.Tensor,
    ) -> torch.Tensor:
        """Score hook with latent times available to explicit oracle ablations.

        Production policies ignore ``event_times`` and remain predictable.
        A separate hook avoids weakening the causal signature of
        :meth:`policy_scores`, while allowing a clearly marked full-information
        score-quality upper anchor in controlled experiments.
        """
        del event_times
        return self.policy_scores(quantile_est)

    def allocate_budget(self, probability_est: torch.Tensor, x: torch.Tensor, t: torch.Tensor,
                        quantile_est: torch.Tensor) -> BudgetAllocationResult:
        device = self.conditional_grid.device
        N, T_max_curr, T_max_future = self.conditional_grid.shape
        # Quantile value width+1 is the UPB infinity/no-event sentinel.  It is
        # a valid prediction target but not an executable acquisition turn.
        prior_q = get_prior(
            quantile_est, self.taus_range, self.tau_prior
        ).clamp(max=T_max_curr)
        if self.score == 'prob':
            scores = self.policy_scores_for_allocation(quantile_est, t)
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
            policy_fit_masses = self.phase1_objective_masses(
                t_val,
                val_prior_q,
                val_quantile_est,
                val_grid,
            )
        else:
            policy_fit_count = len(t_val) - self.budget_control_size
            policy_fit_weights = self.phase1_objective_weights(
                t_val[:policy_fit_count],
                val_prior_q[:policy_fit_count],
                val_quantile_est[:policy_fit_count],
            )
            policy_fit_masses = self.phase1_objective_masses(
                t_val[:policy_fit_count],
                val_prior_q[:policy_fit_count],
                val_quantile_est[:policy_fit_count],
                val_grid[:policy_fit_count],
            )
            # The full-fold objective is computed after the policy has been
            # frozen.  This keeps the independent control labels out of both
            # the target-anchor choice and the learned time-policy shape.
            phase1_weights = None
        solver_policy_fit_weights = (
            None if policy_fit_masses is not None else policy_fit_weights
        )
        raw_policy_fit_conditionals = None
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
        direct_bins_match = re.fullmatch(
            r"direct_bins_([1-9][0-9]*)", self.projection
        )
        direct_bin_count = (
            int(direct_bins_match.group(1))
            if direct_bins_match is not None
            else None
        )
        cumulative_projection = self.projection == 'cumulative_platt'
        if policy_fit_masses is not None and direct_time_projection:
            raise ValueError(
                "Soft prefix-mass objectives require a score-adaptive or "
                "row-adaptive DAPRO projection, not direct_time."
            )
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
                    solver_policy_fit_weights,
                    direct_bin_count,
                    objective_masses=policy_fit_masses,
                    smooth_rank_lookup=getattr(
                        self, "smooth_score_rank_map", False
                    ),
                )
                raw_policy_fit_conditionals = optimal_fit
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
                    "risk_budget_row_cost_cap_kind": "none",
                    "risk_budget_row_cost_cap_policy_version": "none",
                    "risk_budget_row_cost_cap_uses_future_prefixes": 0,
                    "risk_budget_row_cost_cap_fit_changed_fraction": 0.0,
                    "risk_budget_row_cost_cap_remaining_changed_fraction": (
                        0.0
                    ),
                }
                if self.risk_candidate_row_cost_cap is not None:
                    row_cost_cap = self.risk_candidate_row_cost_cap
                    envelope, envelope_metrics = (
                        _solve_shared_causal_row_cap_envelope(
                            val_max_steps[:policy_fit_count],
                            row_cost_cap,
                            self.terminal_pi_min,
                            T_max_curr,
                            objective_weights=solver_policy_fit_weights,
                            objective_masses=policy_fit_masses,
                        )
                    )
                    p_fit_base, fit_cap_changed = (
                        _apply_shared_cumulative_row_cost_envelope(
                            p_fit_base,
                            val_prior_q[:policy_fit_count],
                            envelope,
                        )
                    )
                    (
                        p_remaining_base,
                        remaining_cap_changed,
                    ) = _apply_shared_cumulative_row_cost_envelope(
                        p_remaining_base,
                        remaining_prior_q,
                        envelope,
                    )
                    row_cap_metrics = {
                        "risk_budget_row_cost_cap_enabled": 1,
                        "risk_budget_row_cost_cap_per_sample": row_cost_cap,
                        "risk_budget_row_cost_cap_kind": (
                            "causal_shared_target_pav_envelope"
                        ),
                        "risk_budget_row_cost_cap_policy_version": (
                            CAUSAL_SHARED_PAV_CAP_VERSION
                        ),
                        "risk_budget_row_cost_cap_uses_future_prefixes": 0,
                        (
                            "risk_budget_row_cost_cap_fit_changed_fraction"
                        ): float(fit_cap_changed.to(
                            torch.float64
                        ).mean().item()),
                        (
                            "risk_budget_row_cost_cap_remaining_changed_"
                            "fraction"
                        ): float(remaining_cap_changed.to(
                            torch.float64
                        ).mean().item()),
                        **envelope_metrics,
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
                    # The score-bin map is data adaptive.  Without the frozen
                    # shared causal envelope, its distribution-free support
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
                    solver_policy_fit_weights,
                    direct_bin_count,
                    objective_masses=policy_fit_masses,
                    smooth_rank_lookup=getattr(
                        self, "smooth_score_rank_map", False
                    ),
                )
                raw_policy_fit_conditionals = optimal_P
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
                    objective_weights=solver_policy_fit_weights,
                    objective_masses=policy_fit_masses,
                    terminal_pi_min=None,
                    verbose=False,
                )
                raw_policy_fit_conditionals = optimal_P
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

        # The legacy objective columns below intentionally retain their
        # terminal hard-weight semantics.  SoftTargetDAPRO instead optimizes
        # prefix masses, so report that exact variance proxy under three
        # clearly separated policies without repurposing any existing field.
        if policy_fit_masses is None:
            phase1_selected_masses = None
            phase2_frozen_masses = None
        else:
            phase1_selected_masses = self.phase1_objective_masses(
                t_val,
                val_prior_q,
                val_quantile_est,
                val_grid,
            )
            phase2_frozen_masses = self.phase2_objective_masses(
                t_test,
                test_prior_q,
                test_quantile_est,
                test_grid,
            )

        soft_mass_raw_available = int(
            policy_fit_masses is not None
            and raw_policy_fit_conditionals is not None
        )
        soft_mass_selected_available = int(
            phase1_selected_masses is not None
        )
        soft_mass_phase2_available = int(
            phase2_frozen_masses is not None
        )
        soft_mass_diagnostics = {
            "soft_mass_variance_proxy_available": int(
                soft_mass_raw_available
                and soft_mass_selected_available
                and soft_mass_phase2_available
            ),
            "soft_mass_variance_proxy_semantics": (
                "mean_rows_sum_prefix_mass_times_inverse_cumulative_reach_"
                "minus_one"
                if policy_fit_masses is not None
                else "unavailable_no_prefix_objective_masses"
            ),
            "soft_mass_phase1_raw_policy_fit_available": (
                soft_mass_raw_available
            ),
            "soft_mass_phase1_raw_policy_fit_sample_count": (
                policy_fit_count if soft_mass_raw_available else 0
            ),
            "soft_mass_phase1_raw_policy_fit_mean_variance_proxy": (
                _mean_soft_mass_variance_proxy(
                    policy_fit_masses,
                    raw_policy_fit_conditionals,
                )
                if soft_mass_raw_available
                else np.nan
            ),
            "soft_mass_phase1_selected_full_fold_available": (
                soft_mass_selected_available
            ),
            "soft_mass_phase1_selected_full_fold_sample_count": (
                len(t_val) if soft_mass_selected_available else 0
            ),
            "soft_mass_phase1_selected_full_fold_mean_variance_proxy": (
                _mean_soft_mass_variance_proxy(
                    phase1_selected_masses,
                    p_val,
                )
                if soft_mass_selected_available
                else np.nan
            ),
            "soft_mass_phase2_frozen_policy_available": (
                soft_mass_phase2_available
            ),
            "soft_mass_phase2_frozen_policy_sample_count": (
                len(t_test) if soft_mass_phase2_available else 0
            ),
            "soft_mass_phase2_frozen_policy_mean_variance_proxy": (
                _mean_soft_mass_variance_proxy(
                    phase2_frozen_masses,
                    p_test,
                )
                if soft_mass_phase2_available
                else np.nan
            ),
        }

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
            **soft_mass_diagnostics,
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
        all_conditionals = torch.ones(
            (N, T_max_curr), dtype=torch.float64, device=device
        )
        # Phase-I rows are fully observed by design, so their inclusion
        # propensity for every finite candidate is one.
        all_conditionals[test_idxs] = p_test.to(torch.float64)
        candidate_C_probs = candidate_reach_probabilities(
            all_conditionals,
            quantile_est,
            infinity_value=T_max_curr + 1,
        )
        return BudgetAllocationResult(
            quantile_est,
            final_C,
            final_C_probs,
            total_budget_used,
            additional_metrics=additional_metrics,
            candidate_C_probs=candidate_C_probs,
            continuation_probabilities=all_conditionals,
        )


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
        return realized_target_weights(
            event_times,
            prior_q,
            strict=True,
        )


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
            metric_estimation_horizon: int | None = None,
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
        if metric_estimation_horizon is not None and metric_estimation_horizon <= 0:
            raise ValueError("`metric_estimation_horizon` must be positive.")
        self.metric_estimation_horizon = (
            None
            if metric_estimation_horizon is None
            else int(metric_estimation_horizon)
        )
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
        if self.metric_estimation_horizon is not None:
            return "mean_metric_event_weighted_inverse_probability_minus_one"
        return (
            "mean_target_a_weighted_inverse_probability_minus_one"
            f"_{self.anchor_kind}_alpha_{self.target_alpha:.2f}"
        )

    def _select_target_anchor(
            self,
            event_times: torch.Tensor,
            quantile_est: torch.Tensor,
    ) -> None:
        if self.metric_estimation_horizon is not None:
            return
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
        if self.metric_estimation_horizon is not None:
            return (
                realized_target_weights(
                    event_times,
                    self.metric_estimation_horizon,
                    strict=False,
                ),
                int(bool(torch.all(
                    prior_q.reshape(-1) >= self.metric_estimation_horizon
                ).item())),
            )
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
        weights = realized_target_weights(
            event_times,
            anchor_q,
            strict=True,
        )
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
        if self.metric_estimation_horizon is not None:
            return {
                "target_event_kind": "metric_event_by_fixed_horizon",
                "target_event_side": "lower_or_equal",
                "target_metric": "unsafe_event_rate",
                "target_metric_horizon": self.metric_estimation_horizon,
                "target_anchor_phase1_a_rate": self._target_anchor_phase1_rate,
                "target_anchor_phase2_a_rate": self._target_anchor_phase2_rate,
                "target_anchor_phase1_within_prior": (
                    self._target_anchor_phase1_within_prior
                ),
                "target_anchor_phase2_within_prior": (
                    self._target_anchor_phase2_within_prior
                ),
            }
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
        base += self.budget_control_name_suffix
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
        base += self.budget_control_name_suffix
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
        if not 0 <= global_regularization <= 1:
            raise ValueError(
                "`global_regularization` must lie in [0, 1]."
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
        base += self.budget_control_name_suffix
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
            metric_estimation_horizon: int | None = None,
            global_regularization: float = DEFAULT_GLOBAL_REGULARIZATION,
            score_bin_count: int = SCORE_BIN_COUNT,
            smooth_score_rank_map: bool = False,
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
        if not isinstance(score_bin_count, int) or score_bin_count < 1:
            raise ValueError("`score_bin_count` must be a positive integer.")
        self.score_bin_count = int(score_bin_count)
        self.smooth_score_rank_map = bool(smooth_score_rank_map)
        if self.smooth_score_rank_map and self.score_bin_count < 2:
            raise ValueError(
                "A continuous smooth-rank map requires at least two knots."
            )
        super().__init__(
            conditional_grid,
            budget_per_sample,
            taus_range,
            tau_prior,
            m_upper_bound,
            projection=f"direct_bins_{self.score_bin_count}",
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
            metric_estimation_horizon=metric_estimation_horizon,
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
            f"dapro_variance_aligned_bins_{self.score_bin_count}"
            f"_alpha_{alpha}"
            f"_global_{regularization}"
            f"_projection_margin_{margin}"
        )
        if self.smooth_score_rank_map:
            base += "_continuous_rank"
        base += self.budget_control_name_suffix
        return f"{base}_n1_{self.n1}"

    @property
    def objective_kind(self) -> str:
        if self.metric_estimation_horizon is not None:
            return "definitive_regularized_metric_event_variance"
        return "definitive_regularized_target_a_variance"

    def objective_metadata(self) -> dict:
        metadata = super().objective_metadata()
        metadata.update({
            "definitive_dapro": 1,
            "definitive_score_bin_count": self.score_bin_count,
            "definitive_projection_budget_margin": (
                self.projection_budget_margin
            ),
            "definitive_smooth_score_rank_map": int(
                self.smooth_score_rank_map
            ),
            "definitive_budget_control_mode": (
                self.budget_control_mode
                if self.budget_control_mode is not None
                else "projection_assumption"
            ),
        })
        return metadata


class SoftTargetDAPRO(DefinitiveDAPRO):
    """Generalized DAPRO with Rao--Blackwellized prefix event masses.

    Target-A and Definitive DAPRO put a hard binary target weight at each
    fully observed Phase-I row's realized endpoint. This variant optimizes the
    same target-specific HT variance functional after replacing that noisy
    terminal indicator by conditional event mass at every causal prefix:

        a_it = P_hat(T_i=t | T_i>=t, X_it).

    The existing time/score-bin DAPRO backend then learns a continuation map
    from the policy-fit fold and applies it causally to Phase II as ``X_it`` is
    observed. ``metric_estimation_horizon`` selects the unsafe-event-rate
    target; leaving it ``None`` selects the same raw-alpha LPB target as
    Definitive DAPRO. Thus hard Target-A, soft history-adaptive DAPRO, and the
    pre-run initial-PMF allocator share one event-mass/cost-mass objective but
    use different coefficient estimators and policy classes.
    """

    @property
    def name(self) -> str:
        regularization = self._format_parameter(
            self.global_regularization,
            3,
        )
        margin = self._format_parameter(
            self.projection_budget_margin,
            2,
        )
        if self.metric_estimation_horizon is None:
            alpha = self._format_parameter(self.target_alpha, 2)
            target = f"lpb_alpha_{alpha}"
        else:
            target = f"metric_horizon_{self.metric_estimation_horizon}"
        base = (
            f"dapro_soft_prefix_bins_{self.score_bin_count}_"
            f"{target}_global_{regularization}"
            f"_projection_margin_{margin}"
        )
        if self.smooth_score_rank_map:
            base += "_continuous_rank"
        base += self.budget_control_name_suffix
        return f"{base}_n1_{self.n1}"

    @property
    def objective_kind(self) -> str:
        target = (
            "unsafe_event_rate"
            if self.metric_estimation_horizon is not None
            else "lpb_raw_alpha"
        )
        return f"soft_prefix_hazard_variance_{target}"

    def phase1_objective_masses(
            self,
            event_times: torch.Tensor,
            prior_q: torch.Tensor,
            quantile_est: torch.Tensor,
            conditional_grid: torch.Tensor,
    ) -> torch.Tensor:
        active_lengths = torch.minimum(
            event_times.reshape(-1).to(torch.long),
            prior_q.reshape(-1).to(torch.long),
        )
        if self.metric_estimation_horizon is not None:
            horizons = self.metric_estimation_horizon
            strict = False
            target_kind = "unsafe_event_rate"
        else:
            if self._target_anchor_index is None:
                self._select_target_anchor(event_times, quantile_est)
            horizons = quantile_est[:, self._target_anchor_index]
            strict = True
            target_kind = "lpb_raw_alpha"
        coefficients = history_soft_objective_coefficients(
            conditional_grid,
            active_lengths,
            horizons,
            strict=strict,
            target_kind=target_kind,
            global_regularization=self.global_regularization,
        )
        return torch.as_tensor(
            coefficients.event_mass,
            dtype=torch.float64,
            device=conditional_grid.device,
        )

    def objective_metadata(self) -> dict:
        metadata = super().objective_metadata()
        metadata.update({
            "generalized_dapro": 1,
            "generalized_dapro_coefficient_estimator": (
                "history_prefix_hazard_model_integrated"
            ),
            "generalized_dapro_policy_class": "time_score_bin_dynamic",
            "generalized_dapro_uses_current_prefix_x_it": 1,
            "generalized_dapro_uses_initial_x_i0_only": 0,
        })
        return metadata


class DefinitiveCRCDAPRO(DefinitiveDAPRO):
    """Variance-aligned two-bin DAPRO with independent CRC budget control.

    ``n1`` is the total number of fully observed rows.  The first
    ``n1-budget_control_size`` rows learn the target-weighted score-bin table;
    the remaining rows are an independent budget-control fold.  When enabled,
    a shared cumulative PAV envelope is fitted from policy-fit objective
    masses and frozen; pointwise intersection with each causal base path then
    supplies the pathwise row-cost bound without future-prefix access.  A
    nested affine contraction of that capped reach toward the terminal floor
    is fixed before the control labels are inspected.  CRC then selects the
    strongest feasible contraction, giving a finite-sample marginal expected-
    total-budget guarantee without a projection-accuracy assumption.
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
            metric_estimation_horizon: int | None = None,
            global_regularization: float = (
                DefinitiveDAPRO.DEFAULT_GLOBAL_REGULARIZATION
            ),
            score_bin_count: int = DefinitiveDAPRO.SCORE_BIN_COUNT,
            smooth_score_rank_map: bool = False,
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
            metric_estimation_horizon=metric_estimation_horizon,
            global_regularization=global_regularization,
            score_bin_count=score_bin_count,
            smooth_score_rank_map=smooth_score_rank_map,
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
            f"dapro_variance_aligned_bins_{self.score_bin_count}"
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
            base += f"_{CAUSAL_SHARED_PAV_CAP_VERSION}"
        return f"{base}_n1_{self.n1}"

    @property
    def objective_kind(self) -> str:
        if self.metric_estimation_horizon is not None:
            return "definitive_regularized_metric_event_variance_crc"
        return "definitive_regularized_target_a_variance_crc"


class SoftTargetCRCDAPRO(SoftTargetDAPRO):
    """Soft-prefix Generalized DAPRO with independent CRC budget control.

    The first ``n1 - budget_control_size`` fully observed rows learn the
    causal time/score-bin policy from soft prefix event masses.  The remaining
    fully observed rows are used only by the existing nested-family CRC
    selector.  Hence this class changes the budget controller, not the target
    or coefficient estimator, relative to :class:`SoftTargetDAPRO`.
    """

    DEFAULT_N1 = 50
    DEFAULT_BUDGET_CONTROL_SIZE = 25
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
            metric_estimation_horizon: int | None = None,
            global_regularization: float = (
                DefinitiveDAPRO.DEFAULT_GLOBAL_REGULARIZATION
            ),
            score_bin_count: int = DefinitiveDAPRO.SCORE_BIN_COUNT,
            smooth_score_rank_map: bool = False,
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
            metric_estimation_horizon=metric_estimation_horizon,
            global_regularization=global_regularization,
            score_bin_count=score_bin_count,
            smooth_score_rank_map=smooth_score_rank_map,
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
        if self.metric_estimation_horizon is None:
            alpha = self._format_parameter(self.target_alpha, 2)
            target = f"lpb_alpha_{alpha}"
        else:
            target = f"metric_horizon_{self.metric_estimation_horizon}"
        base = (
            f"dapro_soft_prefix_bins_{self.score_bin_count}_"
            f"{target}_global_{regularization}_budget_crc"
            f"_control_{self.budget_control_size}"
        )
        if self.smooth_score_rank_map:
            base += "_continuous_rank"
        if self.row_cost_cap_multiplier is not None:
            multiplier = self._format_parameter(
                self.row_cost_cap_multiplier,
                2,
            )
            base += f"_row_cap_{multiplier}x_budget"
            base += f"_{CAUSAL_SHARED_PAV_CAP_VERSION}"
        return f"{base}_n1_{self.n1}"

    @property
    def objective_kind(self) -> str:
        return f"{super().objective_kind}_crc"

    def objective_metadata(self) -> dict:
        metadata = super().objective_metadata()
        metadata.update({
            "generalized_dapro_budget_control_mode": "crc",
            "generalized_dapro_crc_control_size": self.budget_control_size,
            "generalized_dapro_crc_row_cost_cap_multiplier": (
                self.row_cost_cap_multiplier
                if self.row_cost_cap_multiplier is not None
                else np.nan
            ),
        })
        return metadata


def _conditional_lower_target_probability(
        pmf: torch.Tensor,
        horizons: torch.Tensor,
        *,
        strict: bool,
) -> torch.Tensor:
    """Evaluate a row-specific lower-event probability from a PMF row."""
    values = pmf.to(torch.float64).clamp_min(0.0)
    values = values / values.sum(dim=1, keepdim=True).clamp_min(1e-15)
    h = horizons.to(device=values.device, dtype=torch.long).reshape(-1)
    outcome = torch.arange(1, values.shape[1] + 1, device=values.device)
    mask = outcome[None, :] < h[:, None] if strict else outcome[None, :] <= h[:, None]
    return (values * mask.to(values.dtype)).sum(dim=1)


class _LowerTargetSequentialAHTDAPRO(SoftTargetDAPRO):
    """Shared dynamic schedule for LPB and metric sequential AHT."""

    aht_estimator_kind = "sequential"
    schedule_objective = "information_gain"

    def _target_horizons(
            self,
            event_times: torch.Tensor,
            quantile_est: torch.Tensor,
    ) -> tuple[torch.Tensor, bool, str]:
        if self.metric_estimation_horizon is not None:
            return (
                torch.full(
                    (len(event_times),),
                    int(self.metric_estimation_horizon),
                    dtype=torch.long,
                    device=event_times.device,
                ),
                False,
                "metric",
            )
        if self._target_anchor_index is None:
            self._select_target_anchor(event_times, quantile_est)
        return (
            quantile_est[:, self._target_anchor_index].to(torch.long),
            True,
            "lpb",
        )

    def _schedule_masses(
            self,
            event_times: torch.Tensor,
            prior_q: torch.Tensor,
            quantile_est: torch.Tensor,
            conditional_grid: torch.Tensor,
    ) -> torch.Tensor:
        grid = conditional_grid.to(torch.float64)
        n, width, _ = grid.shape
        times = event_times.reshape(-1).to(device=grid.device, dtype=torch.long)
        prior = prior_q.reshape(-1).to(device=grid.device, dtype=torch.long)
        horizons, strict, _ = self._target_horizons(times, quantile_est)
        horizons = horizons.to(grid.device)
        target = (
            times < horizons if strict else times <= horizons
        ).to(torch.float64)
        previous = _conditional_lower_target_probability(
            grid[:, 0, :], horizons, strict=strict
        )
        masses = torch.zeros((n, width), dtype=torch.float64, device=grid.device)
        for turn in range(1, width + 1):
            active = (turn <= times) & (turn <= prior)
            if not torch.any(active):
                break
            if self.schedule_objective == "residual":
                masses[active, turn - 1] = (
                    target[active] - previous[active]
                ).square()
            post = previous.clone()
            event_now = active & (times == turn)
            post[event_now] = target[event_now]
            survived = active & (times > turn)
            if torch.any(survived):
                if turn < width:
                    next_probability = _conditional_lower_target_probability(
                        grid[:, turn, :], horizons, strict=strict
                    )
                    post[survived] = next_probability[survived]
                else:
                    post[survived] = target[survived]
            if self.schedule_objective == "information_gain":
                masses[active, turn - 1] = (
                    post[active] - previous[active]
                ).square()
            previous[active] = post[active]

        if self.global_regularization > 0:
            active_lengths = torch.minimum(times, prior)
            exploration = history_soft_objective_coefficients(
                conditional_grid,
                active_lengths,
                width,
                strict=False,
                target_kind="all_observable_events",
                global_regularization=0.0,
            ).event_mass
            masses = (
                masses
                + self.global_regularization * torch.as_tensor(
                    exploration, dtype=torch.float64, device=grid.device
                )
            ) / (1.0 + self.global_regularization)
        return masses

    def phase1_objective_masses(
            self,
            event_times: torch.Tensor,
            prior_q: torch.Tensor,
            quantile_est: torch.Tensor,
            conditional_grid: torch.Tensor,
    ) -> torch.Tensor:
        return self._schedule_masses(
            event_times, prior_q, quantile_est, conditional_grid
        )

    def policy_scores(self, quantile_est: torch.Tensor) -> torch.Tensor:
        """Causal target residual variance per predicted remaining cost."""
        grid = self.conditional_grid.to(torch.float64)
        n, width, _ = grid.shape
        dummy_times = torch.full(
            (n,), width, dtype=torch.long, device=grid.device
        )
        horizons, strict, _ = self._target_horizons(dummy_times, quantile_est)
        horizons = horizons.to(grid.device).clamp(min=1, max=width)
        scores = torch.zeros((n, width), dtype=torch.float64, device=grid.device)
        row = torch.arange(n, device=grid.device)
        for current in range(width):
            pmf = grid[:, current, :].clamp_min(0.0)
            pmf = pmf / pmf.sum(dim=1, keepdim=True).clamp_min(1e-15)
            probability = _conditional_lower_target_probability(
                pmf, horizons, strict=strict
            )
            tail = torch.flip(
                torch.cumsum(torch.flip(pmf, dims=(1,)), dim=1), dims=(1,)
            )[:, :width]
            prefix = torch.cumsum(tail, dim=1)
            endpoint = horizons - 1
            endpoint_cost = prefix[row, endpoint]
            past_cost = (
                prefix[:, current - 1]
                if current > 0
                else torch.zeros(n, dtype=prefix.dtype, device=prefix.device)
            )
            remaining = (endpoint_cost - past_cost).clamp_min(1e-12)
            unresolved = horizons > current
            scores[:, current] = torch.where(
                unresolved,
                torch.sqrt(probability * (1.0 - probability) / remaining),
                torch.zeros_like(probability),
            )
        return scores

    @property
    def name(self) -> str:
        target = (
            f"metric_m{self.metric_estimation_horizon}"
            if self.metric_estimation_horizon is not None
            else f"lpb_c{1.0 - self.target_alpha:.2f}".replace(".", "p")
        )
        margin = self._format_parameter(self.projection_budget_margin, 2)
        base = (
            f"dapro_{self.schedule_objective}_sequential_aht_"
            f"{target}_bins_{self.score_bin_count}_raw_margin_{margin}"
        )
        base += self.budget_control_name_suffix
        return f"{base}_n1_{self.n1}"

    @property
    def objective_kind(self) -> str:
        return f"dynamic_{self.schedule_objective}_schedule_sequential_aht"

    def objective_metadata(self) -> dict:
        metadata = super().objective_metadata()
        metadata.update({
            "aht_estimator_kind": "sequential",
            "dynamic_schedule_objective": self.schedule_objective,
            "dynamic_schedule_score": "target_residual_variance_per_remaining_cost",
        })
        return metadata


class InformationGainDAPRO(_LowerTargetSequentialAHTDAPRO):
    """K2 history-adaptive schedule fitted to squared prediction updates."""

    schedule_objective = "information_gain"


class ResidualDAPRO(_LowerTargetSequentialAHTDAPRO):
    """K2 history-adaptive schedule fitted to squared remaining residuals."""

    schedule_objective = "residual"


class _SequentialAHTCRCMixin:
    """Independent CRC constructor shared by the sequential AHT schedules."""

    def __init__(
            self, *args, n1: int = 200, budget_control_size: int = 100,
            row_cost_cap_multiplier: float | None = 2.0, **kwargs,
    ):
        kwargs.pop("projection_budget_margin", None)
        cap = (
            None
            if row_cost_cap_multiplier is None
            else float(row_cost_cap_multiplier) * float(args[1] if len(args) > 1 else kwargs["budget_per_sample"])
        )
        super().__init__(
            *args,
            n1=n1,
            projection_budget_margin=0.0,
            budget_control_mode="crc",
            budget_control_size=budget_control_size,
            risk_candidate_row_cost_cap=cap,
            **kwargs,
        )


class InformationGainCRCDAPRO(_SequentialAHTCRCMixin, InformationGainDAPRO):
    pass


class ResidualCRCDAPRO(_SequentialAHTCRCMixin, ResidualDAPRO):
    pass


class SoftTargetUPBDAPRO(SoftTargetDAPRO):
    """Soft-prefix Generalized DAPRO for upper-bound coverage.

    At the frozen UPB anchor ``f_c`` the stochastic calibration target is the
    miscoverage event ``A_i=1{T_i>f_c(X_i), f_c(X_i)<201}``.  It is resolved
    by reaching the candidate bound itself.  Conditional on being at risk at
    that endpoint, its Rao--Blackwellized mass is one minus the current event
    hazard.  Rows assigned 201 have deterministic zero miscoverage and carry
    no target acquisition variance.  For this endpoint target the efficient
    soft-prefix action is a block reveal: sample at ``X_0`` and, if selected,
    follow through the candidate.  The K2 score map is therefore the
    endpoint/block specialization of the same Generalized-DAPRO objective,
    rather than a generic per-turn policy whose propensities decay before the
    only informative endpoint.
    """

    DEFAULT_TARGET_COVERAGE = 0.70
    requires_unclipped_upb_quantiles = True
    upb_endpoint_block_action = True

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
        if not 0 < target_coverage < 1:
            raise ValueError("`target_coverage` must lie in (0, 1).")
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

    def _select_target_anchor(
            self,
            event_times: torch.Tensor,
            quantile_est: torch.Tensor,
    ) -> None:
        del event_times
        prior_index = int(
            torch.abs(self.taus_range - self.tau_prior).argmin().item()
        )
        candidates = quantile_est[:, :prior_index + 1]
        pmf = self.conditional_grid[:, 0, :].to(torch.float64).clamp_min(0.0)
        pmf = pmf / pmf.sum(dim=1, keepdim=True).clamp_min(1e-15)
        tail = torch.flip(
            torch.cumsum(torch.flip(pmf, dims=(1,)), dim=1), dims=(1,)
        )
        width = self.conditional_grid.shape[1]
        finite = candidates <= width
        indices = candidates.to(torch.long).clamp(min=1, max=width)
        model_miscoverage = tail.gather(1, indices)
        model_miscoverage = torch.where(
            finite, model_miscoverage, torch.zeros_like(model_miscoverage)
        )
        # The entire target is a frozen function of X_0 and the pretrained
        # conditional PMF.  No trajectory label is spent to fit the UPB policy.
        # This is the Rao--Blackwellized soft-prefix counterpart of choosing a
        # hard Phase-I anchor, and avoids injecting an unnecessary random split
        # into the final calibrated UPB.
        model_coverages = 1.0 - model_miscoverage.mean(dim=0)
        selected = select_upb_calibration_positions(
            model_coverages,
            torch.tensor(
                [self.target_coverage],
                dtype=model_coverages.dtype,
                device=model_coverages.device,
            ),
        )
        self._target_anchor_index = int(selected.item())
        self._target_anchor_selection_miscoverage = float(
            model_miscoverage[:, self._target_anchor_index].mean().item()
        )

    def policy_scores(
            self,
            quantile_est: torch.Tensor,
    ) -> torch.Tensor:
        """Causal probability of the UPB miscoverage event at each prefix.

        The score horizon is the fixed nominal target-coverage quantile, so it
        is chosen without Phase-I or Phase-II labels.  At prefix ``X_it`` the
        score sums the conditional PMF beyond that row-specific horizon.  A
        larger score therefore means a larger chance of ``T>f`` and correctly
        receives weakly larger K2 continuation probability.
        """
        target_index = int(
            torch.abs(
                self.taus_range - self.target_coverage
            ).argmin().item()
        )
        horizons = quantile_est[:, target_index].to(torch.long)
        n, width, _ = self.conditional_grid.shape
        scores = torch.zeros(
            (n, width),
            dtype=self.conditional_grid.dtype,
            device=self.conditional_grid.device,
        )
        for horizon in torch.unique(horizons).tolist():
            rows = horizons == int(horizon)
            if 1 <= horizon <= width:
                # PMF outcome index h is T=h+1, hence indices h,... are T>h.
                scores[rows] = self.conditional_grid[rows, :, int(horizon):].sum(
                    dim=-1
                )
            # f=201 has deterministic zero miscoverage and retains score zero.
        return scores.clamp(0.0, 1.0)

    def _initial_upb_risk_and_cost(
            self,
            quantile_est: torch.Tensor,
            prior_q: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._target_anchor_index is None:
            raise RuntimeError("The UPB target anchor must be frozen first.")
        pmf = self.conditional_grid[:, 0, :].to(torch.float64)
        pmf = pmf.clamp_min(0.0)
        pmf = pmf / pmf.sum(dim=1, keepdim=True).clamp_min(1e-15)
        anchor = quantile_est[:, self._target_anchor_index].to(torch.long)
        width = self.conditional_grid.shape[1]
        tail = torch.flip(
            torch.cumsum(torch.flip(pmf, dims=(1,)), dim=1), dims=(1,)
        )
        finite_anchor = (anchor >= 1) & (anchor <= width)
        anchor_index = anchor.clamp(min=1, max=width)
        miscoverage_probability = tail.gather(
            1, anchor_index[:, None]
        ).squeeze(1)
        miscoverage_probability = torch.where(
            finite_anchor,
            miscoverage_probability,
            torch.zeros_like(miscoverage_probability),
        )
        # The augmented-HT UPB estimator weights the residual A-m rather than
        # the raw indicator A.  Under the fitted Bernoulli model its expected
        # squared residual is m(1-m), the exact soft target for acquisition
        # variance.  Misspecification cannot affect design unbiasedness.
        risk = miscoverage_probability * (1.0 - miscoverage_probability)
        time = torch.arange(width, device=pmf.device).unsqueeze(0)
        at_risk = tail[:, :width]
        predicted_cost = (
            at_risk * (time < prior_q.to(torch.long)[:, None])
        ).sum(dim=1)
        value_score = torch.sqrt(
            risk / predicted_cost.clamp_min(1e-12)
        )
        return risk, predicted_cost, value_score

    @staticmethod
    def _k2_base_probabilities(
            fit_risk: torch.Tensor,
            fit_cost: torch.Tensor,
            fit_score: torch.Tensor,
            all_score: torch.Tensor,
    ) -> tuple[torch.Tensor, float, torch.Tensor]:
        threshold = float(torch.median(fit_score).item())
        fit_high = fit_score >= threshold
        all_high = all_score >= threshold
        bases = []
        for mask in (~fit_high, fit_high):
            if not torch.any(mask):
                bases.append(torch.tensor(0.0, dtype=torch.float64, device=fit_score.device))
                continue
            a = fit_risk[mask].sum()
            d = fit_cost[mask].sum().clamp_min(1e-12)
            bases.append(torch.sqrt(a / d))
        base = torch.stack(bases)
        # Enforce the K2 monotone score map.  Pool a reversed pair exactly as
        # the generalized-PAV step in ordinary DAPRO would.
        if base[0] > base[1]:
            pooled = torch.sqrt(
                fit_risk.sum() / fit_cost.sum().clamp_min(1e-12)
            )
            base[:] = pooled
        row_base = torch.where(all_high, base[1], base[0])
        if row_base.max() <= 0:
            row_base = torch.ones_like(row_base)
        else:
            row_base = row_base / row_base.max()
        return row_base, threshold, base

    def allocate_budget(
            self,
            probability_est: torch.Tensor,
            x: torch.Tensor,
            t: torch.Tensor,
            quantile_est: torch.Tensor,
    ) -> BudgetAllocationResult:
        """Fit the exact K2 endpoint/block-action UPB specialization.

        A finite UPB miscoverage indicator is revealed only by reaching its
        endpoint.  Conditional partial trajectories have no direct HT
        contribution, so the block policy samples once at ``X_i0`` and then
        follows selected trajectories to the event or q_prior.  This avoids
        the exponential propensity decay of a generic turn-by-turn policy.
        """
        del probability_est, x
        device = self.conditional_grid.device
        n, width, _ = self.conditional_grid.shape
        prior_q = get_prior(
            quantile_est, self.taus_range, self.tau_prior
        ).to(torch.long).clamp(min=1, max=width)
        permutation = np.random.permutation(n)
        control = self.budget_control_size if self.budget_control_mode else 0
        control_rows = permutation[:control]
        phase2 = permutation[control:]
        phase1 = control_rows
        if len(phase2) <= 0:
            raise ValueError("UPB DAPRO needs a non-empty deployment fold.")

        self._select_target_anchor(t, quantile_est)
        risk, predicted_cost, value_score = self._initial_upb_risk_and_cost(
            quantile_est, prior_q
        )
        regularized_risk = (
            risk + self.global_regularization * risk.mean()
        ) / (1.0 + self.global_regularization)
        row_base, threshold, bin_bases = self._k2_base_probabilities(
            regularized_risk,
            predicted_cost,
            value_score,
            value_score,
        )
        lengths = torch.minimum(
            t.reshape(-1).to(torch.long), prior_q
        ).to(torch.float64)
        phase1_cost = float(lengths[control_rows].sum().item())
        total_budget = float(self.budget_per_sample * n)
        epsilon = float(self.terminal_pi_min or 0.005)
        row_base = row_base.clamp(min=epsilon, max=1.0)
        if total_budget <= phase1_cost:
            raise ValueError("Fully observed UPB Phase I exhausts the budget.")

        selector_metrics = {}
        if self.budget_control_mode is None:
            target = (
                (total_budget - phase1_cost) / len(phase2)
                - self.projection_budget_margin
            )
            if target < epsilon * predicted_cost[phase2].mean().item():
                raise ValueError("UPB DAPRO positivity floor is budget-infeasible.")

            def probabilities(scale: float) -> torch.Tensor:
                return torch.clamp(scale * row_base, min=epsilon, max=1.0)

            low, high = 0.0, 1.0
            while (
                (probabilities(high)[phase2] * predicted_cost[phase2]).mean().item()
                < target and high < 1e8
            ):
                high *= 2.0
            for _ in range(80):
                middle = (low + high) / 2.0
                cost = (
                    probabilities(middle)[phase2] * predicted_cost[phase2]
                ).mean().item()
                if cost <= target:
                    low = middle
                else:
                    high = middle
            selected_probabilities = probabilities(low)
            selector_metrics = {
                "upb_block_model_budget_scale": low,
                "expected_budget_guarantee_kind": "projection_transfer_assumption",
                "expected_budget_guarantee_requires_projection_accuracy": 1,
                "expected_budget_guarantee_is_marginal_finite_sample": 0,
            }
        else:
            alpha = torch.linspace(
                1.0, 0.0, self.budget_candidate_count,
                dtype=torch.float64, device=device,
            )
            base = row_base[:, None]
            family = epsilon + alpha[None, :] * (base - epsilon)
            control_costs = (
                lengths[control_rows, None] * family[control_rows]
            ).detach().cpu().numpy()
            selection = select_crc_budget_candidate(
                control_costs,
                lengths[control_rows].detach().cpu().numpy(),
                total_budget_after_policy_fit=(
                    total_budget
                ),
                deployment_sample_count=len(phase2),
                maximum_cost_per_sample=float(width),
                maximum_candidate_cost_per_sample=float(width),
                maximum_pilot_cost_per_sample=float(width),
            )
            selected_probabilities = family[:, selection.selected_index]
            selector_metrics = {
                "risk_budget_selector_valid": 1,
                "risk_budget_selected_candidate_index": selection.selected_index,
                "risk_budget_selector_left_side_per_sample": selection.selector_left_side_per_sample,
                "risk_budget_guarantee_kind": selection.guarantee_kind,
                "expected_budget_guarantee_kind": selection.guarantee_kind,
                "expected_budget_guarantee_requires_projection_accuracy": 0,
                "expected_budget_guarantee_is_marginal_finite_sample": 1,
            }

        selected_probabilities = selected_probabilities.to(torch.float64)
        acquisition_uniforms = self.get_acquisition_uniforms(
            n, width, device=device, dtype=torch.float64
        )
        if acquisition_uniforms is None:
            self.reset_acquisition_rng()
            first_uniform = torch.rand(n, device=device, dtype=torch.float64)
        else:
            first_uniform = acquisition_uniforms[:, 0]
        selected = first_uniform[phase2] < selected_probabilities[phase2]
        C = torch.zeros(n, dtype=torch.long, device=device)
        C[phase1] = prior_q[phase1]
        C[phase2] = torch.where(
            selected, prior_q[phase2], torch.zeros_like(prior_q[phase2])
        )
        terminal_pi = selected_probabilities.clone()
        terminal_pi[phase1] = 1.0
        candidate_pi = terminal_pi[:, None].expand_as(quantile_est).clone()
        candidate_pi = torch.where(
            quantile_est == width + 1,
            torch.ones_like(candidate_pi),
            candidate_pi,
        )
        realized_cost = phase1_cost + float(
            lengths[phase2][selected].sum().item()
        )
        expected_total = phase1_cost + float(
            (lengths[phase2] * selected_probabilities[phase2]).sum().item()
        )
        expected_metrics = summarize_expected_budget(
            expected_total,
            n,
            self.budget_per_sample,
            cost_semantics="phase1_full_plus_k2_endpoint_block_sampling",
        )
        target_weights, _ = self._weights_at_frozen_anchor(
            t, prior_q, quantile_est
        )
        objective_proxy = (
            target_weights * (terminal_pi.reciprocal() - 1.0)
        )
        additional = {
            "objective_kind": self.objective_kind,
            "upb_endpoint_block_action": 1,
            "upb_block_score": "sqrt_initial_upb_residual_variance_per_expected_cost",
            "upb_block_k2_threshold": threshold,
            "upb_block_low_base": float(bin_bases[0].item()),
            "upb_block_high_base": float(bin_bases[1].item()),
            "phase1_sample_count": control,
            "policy_fit_label_count": 0,
            "policy_fit_kind": "model_only_soft_upb_target",
            "crc_control_sample_count": control,
            "phase2_sample_count": len(phase2),
            "phase1_expected_cost_total": phase1_cost,
            "phase2_expected_cost_total": expected_total - phase1_cost,
            "phase2_expected_cost_per_sample": (
                (expected_total - phase1_cost) / len(phase2)
            ),
            "phase2_mean_objective_variance_proxy": float(
                objective_proxy[phase2].mean().item()
            ),
            "terminal_pi_min": epsilon,
            **expected_metrics,
            **selector_metrics,
            **self.objective_metadata(),
        }
        return BudgetAllocationResult(
            f=quantile_est,
            C=C,
            C_probs=terminal_pi,
            total_budget_used=realized_cost,
            mean_weight=float(terminal_pi.reciprocal().mean().item()),
            max_weight=float(terminal_pi.reciprocal().max().item()),
            additional_metrics=additional,
            candidate_C_probs=candidate_pi,
        )

    def _weights_at_frozen_anchor(
            self,
            event_times: torch.Tensor,
            prior_q: torch.Tensor,
            quantile_est: torch.Tensor | None,
    ) -> tuple[torch.Tensor, int]:
        if quantile_est is None:
            raise ValueError("UPB DAPRO requires candidate quantile estimates.")
        if self._target_anchor_index is None:
            raise RuntimeError("The UPB target anchor must be selected first.")
        anchor_q = quantile_est[:, self._target_anchor_index]
        finite = anchor_q < 201
        tolerance = 1e-6
        within_prior = bool(torch.all(
            (~finite) | (
                anchor_q.reshape(-1)
                <= prior_q.reshape(-1).to(anchor_q.dtype) + tolerance
            )
        ).item())
        if not within_prior:
            raise ValueError(
                "A finite UPB target anchor exceeds the executable prior."
            )
        weights = (
            finite
            & (event_times.reshape(-1) > anchor_q.reshape(-1))
        ).to(torch.float64)
        return weights, int(within_prior)

    def phase1_objective_masses(
            self,
            event_times: torch.Tensor,
            prior_q: torch.Tensor,
            quantile_est: torch.Tensor,
            conditional_grid: torch.Tensor,
    ) -> torch.Tensor:
        if self._target_anchor_index is None:
            self._select_target_anchor(event_times, quantile_est)
        anchor_q = quantile_est[:, self._target_anchor_index]
        active_lengths = torch.minimum(
            event_times.reshape(-1).to(torch.long),
            prior_q.reshape(-1).to(torch.long),
        )
        n, width, _ = conditional_grid.shape
        named = np.zeros((n, width), dtype=np.float64)
        endpoints = np.asarray(anchor_q.detach().cpu(), dtype=np.int64)
        lengths = np.asarray(active_lengths.detach().cpu(), dtype=np.int64)
        finite = (endpoints >= 1) & (endpoints <= width)
        resolvable = finite & (lengths >= endpoints)
        rows = np.flatnonzero(resolvable)
        if len(rows):
            columns = endpoints[rows] - 1
            hazards = np.asarray(
                conditional_grid[
                    torch.as_tensor(rows, device=conditional_grid.device),
                    torch.as_tensor(columns, device=conditional_grid.device),
                    torch.as_tensor(columns, device=conditional_grid.device),
                ].detach().cpu(),
                dtype=np.float64,
            )
            named[rows, columns] = 1.0 - np.clip(hazards, 0.0, 1.0)
        if self.global_regularization > 0:
            all_event = history_soft_objective_coefficients(
                conditional_grid,
                active_lengths,
                conditional_grid.shape[1],
                strict=False,
                target_kind="all_observable_events",
                global_regularization=0.0,
            ).event_mass
            named = (
                named + self.global_regularization * all_event
            ) / (1.0 + self.global_regularization)
        return torch.as_tensor(
            named,
            dtype=torch.float64,
            device=conditional_grid.device,
        )

    @property
    def name(self) -> str:
        coverage = self._format_parameter(self.target_coverage, 2)
        regularization = self._format_parameter(
            self.global_regularization, 3
        )
        margin = self._format_parameter(
            self.projection_budget_margin, 2
        )
        base = (
            f"dapro_soft_prefix_bins_{self.score_bin_count}_"
            f"upb_residual_aht_coverage_{coverage}_model_anchor_"
            f"global_{regularization}"
            f"_projection_margin_{margin}"
        )
        base += self.budget_control_name_suffix
        return base

    @property
    def objective_kind(self) -> str:
        return "soft_prefix_augmented_ht_residual_variance_upb_coverage"

    def objective_metadata(self) -> dict:
        metadata = super().objective_metadata()
        metadata.update({
            "target_bound_type": "upb",
            "target_event_side": "upper",
            "target_event_kind": "upb_miscoverage_t_gt_f",
            "target_coverage": self.target_coverage,
            "policy_target_coverage": self.target_coverage,
            "target_anchor_kind": "model_soft_upb_coverage",
            "target_anchor_label_rates_available": 0,
            "target_anchor_phase1_a_rate": np.nan,
            "target_anchor_phase2_a_rate": np.nan,
            "target_anchor_phase1_within_prior": 1,
            "target_anchor_phase2_within_prior": 1,
            "generalized_dapro_upb_action": "endpoint_block_from_x_i0",
            "generalized_dapro_policy_fit_uses_labels": 0,
            "generalized_dapro_score": "upb_residual_variance_per_expected_cost",
            "upb_infinity_value": 201,
            "upb_infinity_contribution_is_deterministic": 1,
        })
        return metadata


class SoftTargetCRCUPBDAPRO(SoftTargetUPBDAPRO):
    """UPB soft-prefix DAPRO with CRC and no shared-PAV row cap.

    The nested CRC family uses the full executable horizon (200) as its
    candidate-cost bound.  This deliberately avoids the causal shared-PAV
    envelope requested for other experiments; it is valid but can be more
    conservative because the CRC support bound is larger.
    """

    DEFAULT_N1 = 200
    DEFAULT_BUDGET_CONTROL_SIZE = 100

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
            budget_candidate_count: int = 401,
            **kwargs,
    ):
        super().__init__(
            conditional_grid,
            budget_per_sample,
            taus_range,
            tau_prior,
            m_upper_bound,
            n1=n1,
            projection_budget_margin=0.0,
            budget_control_mode="crc",
            budget_control_size=budget_control_size,
            budget_candidate_count=budget_candidate_count,
            risk_candidate_row_cost_cap=None,
            **kwargs,
        )

    @property
    def name(self) -> str:
        coverage = self._format_parameter(self.target_coverage, 2)
        regularization = self._format_parameter(
            self.global_regularization, 3
        )
        return (
            f"dapro_soft_prefix_bins_{self.score_bin_count}_"
            f"upb_residual_aht_coverage_{coverage}_model_anchor_"
            f"global_{regularization}"
            f"_budget_crc_control_{self.budget_control_size}"
        )

    @property
    def objective_kind(self) -> str:
        return "soft_prefix_augmented_ht_residual_variance_upb_coverage_crc"

    def objective_metadata(self) -> dict:
        metadata = super().objective_metadata()
        metadata.update({
            "generalized_dapro_budget_control_mode": "crc",
            "generalized_dapro_crc_control_size": self.budget_control_size,
            "generalized_dapro_crc_row_cost_cap_multiplier": np.nan,
            "risk_budget_row_cost_cap_policy_version": "none",
        })
        return metadata


class HistoryAdaptiveUPBDAPROMixin:
    """Execute the shared DAPRO solver as a genuinely sequential UPB policy."""

    upb_endpoint_block_action = False

    def allocate_budget(
            self,
            probability_est: torch.Tensor,
            x: torch.Tensor,
            t: torch.Tensor,
            quantile_est: torch.Tensor,
    ) -> BudgetAllocationResult:
        # The model-soft anchor is frozen before the score table is evaluated.
        self._select_target_anchor(t, quantile_est)
        return SoftTargetDAPRO.allocate_budget(
            self, probability_est, x, t, quantile_est
        )

    def policy_scores(self, quantile_est: torch.Tensor) -> torch.Tensor:
        """Current-prefix residual value per predicted remaining reveal cost."""
        if self._target_anchor_index is None:
            raise RuntimeError("The UPB target anchor must be frozen first.")
        grid = self.conditional_grid.to(torch.float64)
        n, width, _ = grid.shape
        horizons = quantile_est[:, self._target_anchor_index].to(torch.long)
        scores = torch.zeros((n, width), dtype=torch.float64, device=grid.device)
        row_index = torch.arange(n, device=grid.device)
        for current in range(width):
            pmf = grid[:, current, :].clamp_min(0.0)
            pmf = pmf / pmf.sum(dim=1, keepdim=True).clamp_min(1e-15)
            tail = torch.flip(
                torch.cumsum(torch.flip(pmf, dims=(1,)), dim=1), dims=(1,)
            )
            finite_future = (horizons > current) & (horizons <= width)
            if not torch.any(finite_future):
                continue
            endpoint = horizons.clamp(min=1, max=width)
            miscoverage = tail[row_index, endpoint]
            # tail[k] is P(T>=k+1); summing k=current,...,f-1 is
            # the expected number of additional turns needed to resolve A(f).
            prefix = torch.cumsum(tail[:, :width], dim=1)
            endpoint_cost = prefix[row_index, endpoint - 1]
            prior_cost = (
                prefix[:, current - 1]
                if current > 0
                else torch.zeros(n, dtype=prefix.dtype, device=prefix.device)
            )
            remaining_cost = (endpoint_cost - prior_cost).clamp_min(1e-12)
            residual_risk = miscoverage * (1.0 - miscoverage)
            scores[:, current] = torch.where(
                finite_future,
                torch.sqrt(residual_risk / remaining_cost),
                torch.zeros_like(residual_risk),
            )
        return scores

    def objective_metadata(self) -> dict:
        metadata = super().objective_metadata()
        metadata.update({
            "upb_endpoint_block_action": 0,
            "generalized_dapro_upb_action": (
                "history_adaptive_prefix_continuation"
            ),
            "generalized_dapro_policy_fit_uses_labels": 1,
            "generalized_dapro_score": (
                "current_upb_residual_variance_per_predicted_remaining_cost"
            ),
            "generalized_dapro_uses_current_prefix_x_it": 1,
            "generalized_dapro_uses_initial_x_i0_only": 0,
        })
        return metadata


class SoftPrefixEndpointUPBDAPRO(
        HistoryAdaptiveUPBDAPROMixin, SoftTargetUPBDAPRO
):
    """History-adaptive K2 UPB DAPRO with the endpoint soft-mass target.

    This is the direct sequential counterpart of the historical soft-prefix
    policy.  It retains the endpoint survival mass used by the original UPB
    Generalized-DAPRO formulation, while the calibrator uses every reached
    model-prediction update through sequential augmented HT.
    """

    # This history-adaptive policy exposes a propensity for every reached
    # prefix.  Using ordinary endpoint HT here silently discarded those
    # observations and made the method incomparable to the residual-AHT
    # Static baseline.  The class documentation and allocator name have
    # always described a sequential-AHT method, so make that contract real.
    upb_estimator_kind = "sequential"

    @property
    def name(self) -> str:
        coverage = self._format_parameter(self.target_coverage, 2)
        regularization = self._format_parameter(self.global_regularization, 3)
        margin = self._format_parameter(self.projection_budget_margin, 2)
        base = (
            f"dapro_soft_prefix_bins_{self.score_bin_count}_"
            f"upb_endpoint_dynamic_aht_seq_estimator_v2_coverage_{coverage}_"
            f"global_{regularization}_projection_margin_{margin}"
        )
        base += self.budget_control_name_suffix
        return f"{base}_n1_{self.n1}"

    @property
    def objective_kind(self) -> str:
        return "soft_prefix_upb_endpoint_mass_sequential_aht"


class InformationGainUPBDAPRO(
        HistoryAdaptiveUPBDAPROMixin, SoftTargetUPBDAPRO
):
    """History-adaptive K2 DAPRO fitted to squared UPB prediction updates."""

    upb_estimator_kind = "sequential"

    def phase1_objective_masses(
            self,
            event_times: torch.Tensor,
            prior_q: torch.Tensor,
            quantile_est: torch.Tensor,
            conditional_grid: torch.Tensor,
    ) -> torch.Tensor:
        if self._target_anchor_index is None:
            self._select_target_anchor(event_times, quantile_est)
        grid = conditional_grid.to(torch.float64)
        n, width, _ = grid.shape
        horizons = quantile_est[:, self._target_anchor_index].to(torch.long)
        times = event_times.reshape(-1).to(device=grid.device, dtype=torch.long)
        prior = prior_q.reshape(-1).to(device=grid.device, dtype=torch.long)
        finite = (horizons >= 1) & (horizons <= width)
        row_index = torch.arange(n, device=grid.device)

        def tail_at(current: int) -> torch.Tensor:
            pmf = grid[:, current, :].clamp_min(0.0)
            pmf = pmf / pmf.sum(dim=1, keepdim=True).clamp_min(1e-15)
            return torch.flip(
                torch.cumsum(torch.flip(pmf, dims=(1,)), dim=1), dims=(1,)
            )

        initial = tail_at(0)
        endpoint = horizons.clamp(min=1, max=width)
        previous = initial[row_index, endpoint]
        previous = torch.where(finite, previous, torch.zeros_like(previous))
        masses = torch.zeros((n, width), dtype=torch.float64, device=grid.device)
        for turn in range(1, width + 1):
            active = finite & (turn <= times) & (turn <= prior) & (turn <= horizons)
            if not torch.any(active):
                continue
            post = previous.clone()
            event_now = active & (times == turn)
            resolves_at_bound = active & (times > turn) & (horizons == turn)
            continues = active & (times > turn) & (horizons > turn)
            post[event_now] = 0.0
            post[resolves_at_bound] = 1.0
            if torch.any(continues):
                next_tail = tail_at(turn)
                post[continues] = next_tail[continues, endpoint[continues]]
            masses[active, turn - 1] = (post[active] - previous[active]).square()
            previous[active] = post[active]

        if self.global_regularization > 0:
            active_lengths = torch.minimum(times, prior)
            exploration = history_soft_objective_coefficients(
                conditional_grid,
                active_lengths,
                width,
                strict=False,
                target_kind="all_observable_events",
                global_regularization=0.0,
            ).event_mass
            masses = (
                masses
                + self.global_regularization * torch.as_tensor(
                    exploration, dtype=torch.float64, device=grid.device
                )
            ) / (1.0 + self.global_regularization)
        return masses

    @property
    def name(self) -> str:
        coverage = self._format_parameter(self.target_coverage, 2)
        regularization = self._format_parameter(self.global_regularization, 3)
        margin = self._format_parameter(self.projection_budget_margin, 2)
        base = (
            f"dapro_information_gain_bins_{self.score_bin_count}_"
            f"upb_sequential_aht_coverage_{coverage}_"
            f"global_{regularization}_projection_margin_{margin}"
        )
        base += self.budget_control_name_suffix
        return f"{base}_n1_{self.n1}"

    @property
    def objective_kind(self) -> str:
        return "soft_prefix_upb_squared_prediction_update_sequential_aht"

    def objective_metadata(self) -> dict:
        metadata = super().objective_metadata()
        metadata.update({
            "generalized_dapro_coefficient_estimator": (
                "policy_fit_realized_squared_upb_prediction_update"
            ),
            "upb_information_gain_objective": 1,
        })
        return metadata


class ResidualUPBDAPRO(InformationGainUPBDAPRO):
    """History-adaptive K2 schedule for the UPB residual-tail bound."""

    upb_estimator_kind = "sequential"

    def phase1_objective_masses(
            self,
            event_times: torch.Tensor,
            prior_q: torch.Tensor,
            quantile_est: torch.Tensor,
            conditional_grid: torch.Tensor,
    ) -> torch.Tensor:
        if self._target_anchor_index is None:
            self._select_target_anchor(event_times, quantile_est)
        grid = conditional_grid.to(torch.float64)
        n, width, _ = grid.shape
        horizons = quantile_est[:, self._target_anchor_index].to(torch.long)
        times = event_times.reshape(-1).to(device=grid.device, dtype=torch.long)
        prior = prior_q.reshape(-1).to(device=grid.device, dtype=torch.long)
        finite = (horizons >= 1) & (horizons <= width)
        row_index = torch.arange(n, device=grid.device)

        def tail_at(current: int) -> torch.Tensor:
            pmf = grid[:, current, :].clamp_min(0.0)
            pmf = pmf / pmf.sum(dim=1, keepdim=True).clamp_min(1e-15)
            return torch.flip(
                torch.cumsum(torch.flip(pmf, dims=(1,)), dim=1), dims=(1,)
            )

        endpoint = horizons.clamp(min=1, max=width)
        previous = tail_at(0)[row_index, endpoint]
        previous = torch.where(finite, previous, torch.zeros_like(previous))
        target = (finite & (times > horizons)).to(torch.float64)
        masses = torch.zeros((n, width), dtype=torch.float64, device=grid.device)
        for turn in range(1, width + 1):
            active = finite & (turn <= times) & (turn <= prior) & (turn <= horizons)
            if not torch.any(active):
                continue
            masses[active, turn - 1] = (
                target[active] - previous[active]
            ).square()
            post = previous.clone()
            event_now = active & (times == turn)
            resolves_at_bound = active & (times > turn) & (horizons == turn)
            continues = active & (times > turn) & (horizons > turn)
            post[event_now] = 0.0
            post[resolves_at_bound] = 1.0
            if torch.any(continues):
                next_tail = tail_at(turn)
                post[continues] = next_tail[continues, endpoint[continues]]
            previous[active] = post[active]
        if self.global_regularization > 0:
            active_lengths = torch.minimum(times, prior)
            exploration = history_soft_objective_coefficients(
                conditional_grid,
                active_lengths,
                width,
                strict=False,
                target_kind="all_observable_events",
                global_regularization=0.0,
            ).event_mass
            masses = (
                masses + self.global_regularization * torch.as_tensor(
                    exploration, dtype=torch.float64, device=grid.device
                )
            ) / (1.0 + self.global_regularization)
        return masses

    @property
    def name(self) -> str:
        coverage = self._format_parameter(self.target_coverage, 2)
        margin = self._format_parameter(self.projection_budget_margin, 2)
        base = (
            f"dapro_residual_sequential_aht_upb_c{coverage}_"
            f"bins_{self.score_bin_count}_raw_margin_{margin}"
        )
        base += self.budget_control_name_suffix
        return f"{base}_n1_{self.n1}"

    @property
    def objective_kind(self) -> str:
        return "dynamic_residual_schedule_sequential_aht_upb"

    def objective_metadata(self) -> dict:
        metadata = super().objective_metadata()
        metadata.update({
            "generalized_dapro_coefficient_estimator": (
                "policy_fit_realized_upb_squared_remaining_residual"
            ),
            "upb_information_gain_objective": 0,
            "upb_residual_tail_objective": 1,
        })
        return metadata


class SoftPrefixEndpointCRCUPBDAPRO(SoftPrefixEndpointUPBDAPRO):
    """Independent-CRC controller for dynamic endpoint-mass UPB DAPRO."""

    def __init__(
            self, *args, n1: int = 50, budget_control_size: int = 25,
            budget_candidate_count: int = 401, **kwargs,
    ):
        super().__init__(
            *args,
            n1=n1,
            projection_budget_margin=0.0,
            budget_control_mode="crc",
            budget_control_size=budget_control_size,
            budget_candidate_count=budget_candidate_count,
            risk_candidate_row_cost_cap=None,
            **kwargs,
        )

    @property
    def objective_kind(self) -> str:
        return f"{super().objective_kind}_crc"


class InformationGainCRCUPBDAPRO(InformationGainUPBDAPRO):
    """Independent-CRC controller for information-gain UPB DAPRO."""

    def __init__(
            self, *args, n1: int = 50, budget_control_size: int = 25,
            budget_candidate_count: int = 401, **kwargs,
    ):
        super().__init__(
            *args,
            n1=n1,
            projection_budget_margin=0.0,
            budget_control_mode="crc",
            budget_control_size=budget_control_size,
            budget_candidate_count=budget_candidate_count,
            risk_candidate_row_cost_cap=None,
            **kwargs,
        )

    @property
    def objective_kind(self) -> str:
        return f"{super().objective_kind}_crc"


class ResidualCRCUPBDAPRO(ResidualUPBDAPRO):
    """Independent-CRC controller for residual-tail UPB DAPRO."""

    def __init__(
            self, *args, n1: int = 50, budget_control_size: int = 25,
            budget_candidate_count: int = 401, **kwargs,
    ):
        super().__init__(
            *args,
            n1=n1,
            projection_budget_margin=0.0,
            budget_control_mode="crc",
            budget_control_size=budget_control_size,
            budget_candidate_count=budget_candidate_count,
            risk_candidate_row_cost_cap=None,
            **kwargs,
        )

    @property
    def objective_kind(self) -> str:
        return f"{super().objective_kind}_crc"


class DefinitiveCRCUPBDAPRO(DefinitiveCRCDAPRO):
    """Upper-bound counterpart of definitive CRC-DAPRO.

    A 70% UPB uses the fixed calibration event

        A_i = 1{T_i > q_0.70(X_i)}.

    The two-bin policy, shared causal PAV cost envelope, and independent CRC
    budget selector are unchanged.  The explicit class prevents the
    lower-tail LPB target from being silently reused when constructing an
    upper bound.
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
        base += self.budget_control_name_suffix
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
