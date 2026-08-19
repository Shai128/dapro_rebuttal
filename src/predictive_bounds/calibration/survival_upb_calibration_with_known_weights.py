"""IPW calibration of upper predictive bounds under sequential acquisition."""

from typing import Dict, Union

import numpy as np
import torch

from src.predictive_bounds.budget_allocators.budget_allocator import (
    BudgetAllocator,
    BudgetAllocationResult,
)
from src.predictive_bounds.calibration.abstract_calibration import (
    SurvivalUPBCalibration,
)
from src.predictive_bounds.calibration.calibration_utils import (
    get_prior,
    indexed_tensor_metrics,
    select_upb_calibration_positions,
)
from src.train_model.models.utils import ModelPrediction, SurvivalModelPrediction


UPB_INFINITY_VALUE = 201


def get_gamma(m_upper_bound: float, budget_per_sample: float):
    return m_upper_bound / budget_per_sample


def get_coverage_rate_deviation(gamma: float, cal_size: int, delta: float):
    return np.sqrt((2 * gamma ** 2 + 5) / cal_size * np.log(1 / delta))


class SurvivalUPBCalibrationWithKnownWeights(SurvivalUPBCalibration):
    """Calibrate a UPB from unbiased estimates of its miscoverage curve.

    For every finite candidate ``f`` the target is

        A_i(f) = 1{T_i > f(X_i)}.

    It is observed by reaching the candidate bound and is reweighted by that
    candidate-specific reach propensity.  A single event/prior terminal
    propensity is not valid for a history-adaptive policy.  The value 201
    denotes infinity/no event through turn 200 and has deterministic zero
    miscoverage.
    """

    def __init__(
            self,
            budget_allocator: BudgetAllocator,
            taus_range: torch.Tensor,
            tau_prior: float,
    ):
        super().__init__()
        self.budget_allocator = budget_allocator
        self.coverage = None
        self.miscoverage = None
        self.taus_range = taus_range
        self.allocation_result: Union[BudgetAllocationResult, None] = None
        self.t_cal = None
        self.tau_prior = tau_prior

    @staticmethod
    def _miscoverage_contributions(
            event_times: torch.Tensor,
            candidates: torch.Tensor,
            acquisition_horizons: torch.Tensor,
            candidate_propensities: torch.Tensor,
    ) -> torch.Tensor:
        times = event_times.reshape(-1, 1).to(candidates.device)
        horizons = acquisition_horizons.reshape(-1, 1).to(candidates.device)
        propensities = candidate_propensities.to(
            device=candidates.device,
            dtype=torch.float64,
        )
        if propensities.shape != candidates.shape:
            raise ValueError("Candidate propensities must match UPB candidates.")
        if torch.any(propensities <= 0) or not torch.all(
                torch.isfinite(propensities)
        ):
            raise ValueError("UPB terminal propensities must be finite and positive.")
        finite_miscoverage_observed = (
            (candidates < UPB_INFINITY_VALUE)
            & (times > candidates)
            & (horizons >= candidates)
        )
        return finite_miscoverage_observed.to(torch.float64) / propensities

    @staticmethod
    def _model_miscoverage_probabilities(
            probability_est: torch.Tensor,
            candidates: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``P_hat(T > f | X_0)`` for every UPB candidate.

        Real-data caches contain an ``N x M x (M+1)`` conditional-PMF grid;
        only its prefix-zero row is used here.  The two-dimensional fallback
        accepts the historical discrete hazard representation used by the
        synthetic tests.  The 201/no-event sentinel has zero miscoverage by
        definition, independently of the fitted model.
        """
        values = probability_est.to(torch.float64)
        if values.ndim == 3:
            pmf = values[:, 0, :].clamp_min(0.0)
            pmf = pmf / pmf.sum(dim=1, keepdim=True).clamp_min(1e-15)
            tail = torch.flip(
                torch.cumsum(torch.flip(pmf, dims=(1,)), dim=1), dims=(1,)
            )
            width = values.shape[1]
            indices = candidates.to(torch.long).clamp(min=1, max=width)
            model_miscoverage = tail.gather(1, indices)
        elif values.ndim == 2:
            hazards = values.clamp(0.0, 1.0)
            survival = torch.cumprod(1.0 - hazards, dim=1)
            width = hazards.shape[1]
            indices = candidates.to(torch.long).clamp(min=1, max=width) - 1
            model_miscoverage = survival.gather(1, indices)
        else:
            raise ValueError(
                "UPB augmentation expects a hazard matrix or conditional-PMF grid."
            )
        finite = candidates < UPB_INFINITY_VALUE
        return torch.where(
            finite,
            model_miscoverage.clamp(0.0, 1.0),
            torch.zeros_like(model_miscoverage),
        )

    @staticmethod
    def _augmented_miscoverage_contributions(
            event_times: torch.Tensor,
            candidates: torch.Tensor,
            acquisition_horizons: torch.Tensor,
            candidate_propensities: torch.Tensor,
            terminal_propensities: torch.Tensor,
            model_miscoverage: torch.Tensor,
    ) -> torch.Tensor:
        """Sequential augmented-HT contributions for ``1{T > f}``.

        If ``T <= f``, observing the event resolves the indicator at the event
        propensity.  If ``T > f``, reaching ``f`` resolves it at the candidate
        propensity.  Thus ``R_i(f)`` observes ``A_i(f)`` at
        ``min(T_i, f_i)`` and, conditionally on the complete benchmark,

        ``m_i(f) + R_i(f) [A_i(f)-m_i(f)] / pi_i(min(T_i,f_i))``

        has expectation ``A_i(f)`` for every frozen (possibly misspecified)
        model prediction ``m_i(f)``.
        """
        times = event_times.reshape(-1, 1).to(candidates.device)
        horizons = acquisition_horizons.reshape(-1, 1).to(candidates.device)
        candidate_pi = candidate_propensities.to(
            device=candidates.device, dtype=torch.float64
        )
        terminal_pi = terminal_propensities.reshape(-1, 1).to(
            device=candidates.device, dtype=torch.float64
        )
        if candidate_pi.shape != candidates.shape:
            raise ValueError("Candidate propensities must match UPB candidates.")
        finite = candidates < UPB_INFINITY_VALUE
        event_resolves = finite & (times <= candidates) & (horizons >= times)
        bound_resolves = finite & (times > candidates) & (horizons >= candidates)
        observed = event_resolves | bound_resolves
        observation_pi = torch.where(times <= candidates, terminal_pi, candidate_pi)
        if torch.any(observation_pi <= 0) or not torch.all(torch.isfinite(observation_pi)):
            raise ValueError("UPB observation propensities must be finite and positive.")
        target = (finite & (times > candidates)).to(torch.float64)
        augmented = model_miscoverage + observed.to(torch.float64) * (
            target - model_miscoverage
        ) / observation_pi
        return torch.where(finite, augmented, torch.zeros_like(augmented))

    @staticmethod
    def _sequential_augmented_miscoverage_contributions(
            event_times: torch.Tensor,
            candidates: torch.Tensor,
            acquisition_horizons: torch.Tensor,
            continuation_probabilities: torch.Tensor,
            conditional_grid: torch.Tensor,
    ) -> torch.Tensor:
        """Use every reached prefix to estimate ``1{T > f}``.

        For a finite integer horizon ``f``, let

        ``m_t(f) = P_hat(T > f | H_t)``

        before resolution and set the terminal prediction to the realized
        indicator when an event occurs or the row survives through ``f``.
        The returned contribution is

        ``m_0(f) + sum_t R_t [m_t(f)-m_{t-1}(f)] / rho_t``.

        It is design-unbiased for every frozen prediction path.  Computing the
        curve on the M possible integer horizons and gathering the row-specific
        quantile candidates avoids materializing an N-by-J-by-M tensor.
        """
        grid = conditional_grid.to(torch.float64)
        conditionals = continuation_probabilities.to(
            device=grid.device, dtype=torch.float64
        )
        if grid.ndim != 3:
            raise ValueError(
                "Sequential UPB AHT requires an N-by-current-time-by-outcome "
                "conditional-PMF grid."
            )
        n, width, outcomes = grid.shape
        if outcomes < width + 1:
            raise ValueError(
                "Sequential UPB AHT requires event outcomes plus the no-event "
                "tail class."
            )
        if conditionals.shape != (n, width):
            raise ValueError(
                "Continuation probabilities must have shape (N, horizon)."
            )
        if torch.any(conditionals <= 0) or torch.any(conditionals > 1):
            raise ValueError("Continuation probabilities must lie in (0, 1].")
        if candidates.ndim != 2 or len(candidates) != n:
            raise ValueError("UPB candidates must have one matrix row per sample.")

        times = event_times.reshape(-1).to(device=grid.device, dtype=torch.long)
        acquired = acquisition_horizons.reshape(-1).to(
            device=grid.device, dtype=torch.long
        )
        cumulative = conditionals.cumprod(dim=1)

        def normalized_tail(row: torch.Tensor) -> torch.Tensor:
            pmf = row.clamp_min(0.0)
            pmf = pmf / pmf.sum(dim=1, keepdim=True).clamp_min(1e-15)
            return torch.flip(
                torch.cumsum(torch.flip(pmf, dims=(1,)), dim=1),
                dims=(1,),
            )

        # Column h-1 represents the finite bound f=h.  Outcome index h is
        # T=h+1, so tail[h] is P(T>h).
        initial_tail = normalized_tail(grid[:, 0, :])
        previous = initial_tail[:, 1:width + 1].clone()
        estimate = previous.clone()

        for turn in range(1, width + 1):
            horizon_slice = slice(turn - 1, width)
            old = previous[:, horizon_slice]
            post = old.clone()
            event_now = times == turn
            survived = times > turn
            if torch.any(event_now):
                post[event_now] = 0.0
            if torch.any(survived):
                # Surviving through the candidate f=turn resolves A(f)=1.
                post[survived, 0] = 1.0
                if turn < width:
                    next_tail = normalized_tail(grid[:, turn, :])
                    post[survived, 1:] = next_tail[
                        survived, turn + 1:width + 1
                    ]
            delta = post - old
            observed = (acquired >= turn).to(torch.float64)
            estimate[:, horizon_slice] += (
                observed / cumulative[:, turn - 1]
            )[:, None] * delta
            previous[:, horizon_slice] = post

        finite = candidates < UPB_INFINITY_VALUE
        indices = candidates.to(torch.long).clamp(min=1, max=width) - 1
        gathered = estimate.gather(1, indices)
        return torch.where(finite, gathered, torch.zeros_like(gathered))

    @staticmethod
    def _sequential_augmented_path_variance(
            event_times: torch.Tensor,
            candidates: torch.Tensor,
            continuation_probabilities: torch.Tensor,
            conditional_grid: torch.Tensor,
    ) -> torch.Tensor:
        """Exact conditional acquisition variance for selected UPB candidates."""
        grid = conditional_grid.to(torch.float64)
        conditionals = continuation_probabilities.to(
            device=grid.device, dtype=torch.float64
        )
        n, width, _ = grid.shape
        if conditionals.shape != (n, width):
            raise ValueError("Continuation probabilities must match the PMF grid.")
        times = event_times.reshape(-1).to(device=grid.device, dtype=torch.long)
        finite = candidates < UPB_INFINITY_VALUE
        endpoint = candidates.to(torch.long).clamp(min=1, max=width)
        cumulative = conditionals.cumprod(dim=1)

        def normalized_tail(current: int) -> torch.Tensor:
            pmf = grid[:, current, :].clamp_min(0.0)
            pmf = pmf / pmf.sum(dim=1, keepdim=True).clamp_min(1e-15)
            return torch.flip(
                torch.cumsum(torch.flip(pmf, dims=(1,)), dim=1), dims=(1,)
            )

        row_index = torch.arange(n, device=grid.device)[:, None]
        initial_tail = normalized_tail(0)
        previous = initial_tail[row_index, endpoint]
        previous = torch.where(finite, previous, torch.zeros_like(previous))
        target = (finite & (times[:, None] > candidates)).to(torch.float64)
        variance = torch.zeros_like(previous)
        for turn in range(1, width + 1):
            active = finite & (turn <= times[:, None]) & (turn <= endpoint)
            if not torch.any(active):
                continue
            inverse_now = cumulative[:, turn - 1].reciprocal()[:, None]
            inverse_before = (
                torch.ones_like(inverse_now)
                if turn == 1
                else cumulative[:, turn - 2].reciprocal()[:, None]
            )
            variance += active.to(torch.float64) * (
                inverse_now - inverse_before
            ) * (target - previous).square()

            post = previous.clone()
            event_now = active & (times[:, None] == turn)
            resolves_at_bound = active & (times[:, None] > turn) & (
                endpoint == turn
            )
            continues = active & (times[:, None] > turn) & (endpoint > turn)
            post[event_now] = 0.0
            post[resolves_at_bound] = 1.0
            if torch.any(continues):
                next_tail = normalized_tail(turn)
                next_value = next_tail[row_index, endpoint]
                post[continues] = next_value[continues]
            previous[active] = post[active]
        return torch.where(finite, variance, torch.zeros_like(variance))

    def calibrate(
            self,
            x_cal: torch.Tensor,
            t_cal: torch.Tensor,
            model_prediction_cal: ModelPrediction,
    ):
        if not isinstance(model_prediction_cal, SurvivalModelPrediction):
            raise TypeError("UPB calibration needs survival predictions.")

        full_candidates = model_prediction_cal.quantile_est
        horizon = int(model_prediction_cal.probability_est.shape[1])
        # Ordinary allocators need only the executable 200-turn acquisition
        # horizon.  UPB-aware DAPRO explicitly opts into the unmodified
        # candidates because its soft target must distinguish 200 from 201;
        # its own acquisition prior is still clipped to the grid width.
        if getattr(
                self.budget_allocator,
                "requires_unclipped_upb_quantiles",
                False,
        ):
            allocation_candidates = full_candidates
        else:
            allocation_candidates = full_candidates.clamp(max=horizon)
        allocation_result = self.budget_allocator.allocate_budget(
            model_prediction_cal.probability_est,
            x_cal,
            t_cal,
            allocation_candidates,
        )
        # Calibration always acts on the original UPB candidates, including
        # the 201 infinity sentinel, independently of acquisition trimming.
        allocation_result.f = full_candidates
        self.allocation_result = allocation_result
        self.t_cal = t_cal
        if allocation_result.candidate_C_probs is None:
            raise ValueError(
                f"Allocator {self.budget_allocator.name!r} does not expose "
                "candidate-specific UPB reach propensities."
            )
        prediction_grid = getattr(
            self.budget_allocator,
            "conditional_grid",
            model_prediction_cal.probability_est,
        )
        model_miscoverage = self._model_miscoverage_probabilities(
            prediction_grid,
            full_candidates,
        )
        estimator_kind = getattr(
            self.budget_allocator,
            "upb_estimator_kind",
            (
                "terminal_residual"
                if allocation_result.continuation_probabilities is None
                else "sequential"
            ),
        )
        if estimator_kind == "ordinary_ht":
            contributions = self._miscoverage_contributions(
                t_cal,
                full_candidates,
                allocation_result.C,
                allocation_result.candidate_C_probs,
            )
            sequential_prefix_updates = 0
        elif estimator_kind == "terminal_residual":
            contributions = self._augmented_miscoverage_contributions(
                t_cal,
                full_candidates,
                allocation_result.C,
                allocation_result.candidate_C_probs,
                allocation_result.C_probs,
                model_miscoverage,
            )
            sequential_prefix_updates = 0
        elif estimator_kind == "sequential":
            contributions = self._sequential_augmented_miscoverage_contributions(
                t_cal,
                full_candidates,
                allocation_result.C,
                allocation_result.continuation_probabilities,
                prediction_grid,
            )
            sequential_prefix_updates = 1
        else:
            raise ValueError(f"Unknown UPB estimator kind: {estimator_kind!r}.")
        if allocation_result.additional_metrics is None:
            allocation_result.additional_metrics = {}
        allocation_result.additional_metrics.update({
            "upb_sequential_prefix_updates": sequential_prefix_updates,
            "upb_aht_estimator_path": (
                "all_reached_prediction_updates"
                if estimator_kind == "sequential"
                else estimator_kind
            ),
        })
        self.model_miscoverage = model_miscoverage
        self.miscoverage_contributions = contributions
        self.miscoverage = contributions.mean(dim=0)
        self.coverage = 1.0 - self.miscoverage

    def _selected_positions(self, target_coverages: torch.Tensor) -> torch.Tensor:
        prior_index = int(
            torch.abs(self.taus_range - self.tau_prior).argmin().item()
        )
        # Only finite candidates no larger than q_prior are acquired by every
        # policy.  Append the final tau=1/UPB=201 candidate as a deterministic
        # always-valid fallback.
        curve = torch.cat([
            self.coverage[:prior_index + 1],
            self.coverage[-1:],
        ])
        local = select_upb_calibration_positions(curve, target_coverages)
        return torch.where(
            local == prior_index + 1,
            torch.full_like(local, len(self.coverage) - 1),
            local,
        )

    def get_calibrated_upb(
            self,
            target_taus: torch.Tensor,
            x: torch.Tensor,
            model_prediction: ModelPrediction,
    ):
        del x
        if not isinstance(model_prediction, SurvivalModelPrediction):
            raise TypeError("UPB calibration needs survival predictions.")
        positions = self._selected_positions(target_taus)
        calibrated = model_prediction.quantile_est[:, positions]
        if hasattr(self.budget_allocator, "max_estimator"):
            # 201 is a semantic infinity marker, not a value to trim to the
            # acquisition horizon.  Finite values remain within the horizon.
            finite = calibrated < UPB_INFINITY_VALUE
            calibrated = torch.where(
                finite,
                torch.minimum(
                    calibrated,
                    torch.as_tensor(
                        self.budget_allocator.max_estimator,
                        dtype=calibrated.dtype,
                        device=calibrated.device,
                    ),
                ),
                calibrated,
            )
        return calibrated.squeeze()

    @property
    def name(self) -> str:
        return f"calibration_{self.budget_allocator.name}_allocation"

    def compute_metrics(self, model_prediction, target_taus) -> Dict[str, float]:
        t = self.t_cal.reshape(-1, 1)
        f = self.allocation_result.f
        C = self.allocation_result.C.reshape(-1, 1)
        C_probs = self.allocation_result.C_probs.reshape(-1).to(torch.float64)
        inverse_probability = C_probs.reciprocal()
        positions = self._selected_positions(target_taus)
        selected_f = f[:, positions]
        infinite = selected_f == UPB_INFINITY_VALUE
        finite_a = (
            (selected_f < UPB_INFINITY_VALUE)
            & (t.to(selected_f.device) > selected_f)
        )
        selected_pi = self.allocation_result.candidate_C_probs[:, positions].to(
            torch.float64
        )
        weighted = finite_a.to(torch.float64) / selected_pi
        selected_model_miscoverage = self.model_miscoverage[:, positions]
        selected_observation_pi = torch.where(
            t.to(selected_f.device) <= selected_f,
            C_probs[:, None].expand_as(selected_pi),
            selected_pi,
        )
        residual = finite_a.to(torch.float64) - selected_model_miscoverage
        variance_proxy = residual.square() * (
            selected_observation_pi.reciprocal() - 1.0
        )
        estimator_kind = getattr(
            self.budget_allocator,
            "upb_estimator_kind",
            (
                "terminal_residual"
                if self.allocation_result.continuation_probabilities is None
                else "sequential"
            ),
        )
        if estimator_kind == "ordinary_ht":
            exact_path_variance = finite_a.to(torch.float64) * (
                selected_pi.reciprocal() - 1.0
            )
            exact_variance_kind = "ordinary_ht"
        elif estimator_kind == "terminal_residual":
            exact_path_variance = variance_proxy
            exact_variance_kind = "terminal_residual_aht"
        else:
            exact_path_variance = self._sequential_augmented_path_variance(
                self.t_cal,
                selected_f,
                self.allocation_result.continuation_probabilities,
                getattr(
                    self.budget_allocator,
                    "conditional_grid",
                    model_prediction.probability_est,
                ),
            )
            exact_variance_kind = "all_prefix_sequential_aht"
        observed_positive = finite_a & (selected_f.to(C.device) <= C)

        mean_weight = self.allocation_result.mean_weight
        max_weight = self.allocation_result.max_weight
        if mean_weight is None:
            mean_weight = inverse_probability.mean().item()
        if max_weight is None:
            max_weight = inverse_probability.max().item()
        allocator_reported_budget = self.allocation_result.total_budget_used
        budget_used = C.to(torch.float64).sum().item()
        observation_horizon = float(
            getattr(self.budget_allocator, "m_upper_bound", 200)
        )
        actual_event_stopped_budget = torch.minimum(
            self.t_cal.reshape(-1).to(torch.float64).clamp(
                max=observation_horizon
            ),
            C.reshape(-1).to(torch.float64),
        ).sum().item()

        f_prior = get_prior(f, self.taus_range, self.tau_prior)
        prior_index = int(
            torch.abs(self.taus_range - self.tau_prior).argmin().item()
        )
        tau_point_one_index = int(
            torch.abs(self.taus_range - 0.10).argmin().item()
        )
        tau_point_one_f = f[:, tau_point_one_index]
        candidate_propensities = self.allocation_result.candidate_C_probs.to(
            torch.float64
        )
        prior_pi = candidate_propensities[:, prior_index]
        tau_point_one_pi = candidate_propensities[:, tau_point_one_index]
        prior_infinite = f_prior == UPB_INFINITY_VALUE
        prior_miscovered = (~prior_infinite) & (
            t.reshape(-1).to(f_prior.device) > f_prior
        )
        tau_point_one_a = (
            (tau_point_one_f < UPB_INFINITY_VALUE)
            & (t.reshape(-1).to(tau_point_one_f.device) > tau_point_one_f)
        ).to(torch.float64)
        prior_a_weighted = (
            prior_miscovered.to(torch.float64) / prior_pi
        )
        tau_point_one_a_weighted = tau_point_one_a / tau_point_one_pi
        prior_observable = prior_infinite | (
            t.reshape(-1).to(C.device) <= C.reshape(-1)
        )
        additional = dict(self.allocation_result.additional_metrics or {})
        indexed = indexed_tensor_metrics({
            "all_observed_jailbreaks": finite_a.to(f.dtype).sum(dim=0),
            "all_f_lower_c": (
                infinite | (selected_f <= C.to(selected_f.device))
            ).to(f.dtype).sum(dim=0),
            "all_observed_both": observed_positive.to(f.dtype).sum(dim=0),
            "alpha_hat_per_tau": self.miscoverage[positions],
            "coverage_hat_per_tau": self.coverage[positions],
            "mean_a_weighted_inverse_probability_minus_one": (
                (
                    finite_a.to(torch.float64)
                    * (selected_pi.reciprocal() - 1.0)
                ).mean(dim=0)
            ),
            "mean_upb_residual_squared_inverse_probability_minus_one": (
                variance_proxy.mean(dim=0)
            ),
            "mean_upb_exact_sequential_aht_path_variance": (
                exact_path_variance.mean(dim=0)
            ),
            "estimated_conditional_variance_upb_coverage_estimator": (
                10000.0 * exact_path_variance.mean(dim=0) / len(t)
            ),
            "mean_a_weighted_inverse_probability": weighted.mean(dim=0),
            "mean_calibrated_a_weighted_inverse_probability": (
                weighted.mean(dim=0)
            ),
            "variance_a_weighted_inverse_probability": weighted.var(
                dim=0, unbiased=False
            ),
            "selected_infinite_bound_rate": infinite.to(f.dtype).mean(dim=0),
        })
        return {
            "coverage_deviation": None,
            "prior_observed_jailbreaks": prior_miscovered.to(f.dtype).sum().item(),
            "prior_observed_f_lower_c": prior_observable.to(f.dtype).sum().item(),
            "prior_observed_both": (
                prior_miscovered & prior_observable
            ).to(f.dtype).sum().item(),
            "n_observed_events": (
                (t <= C) & (t <= observation_horizon)
            ).sum().item(),
            "n_achieved_q_prior1": prior_observable.to(f.dtype).sum().item(),
            "n_achieved_q_prior2": prior_observable.to(f.dtype).sum().item(),
            "budget_used": budget_used,
            "reported_assigned_budget_total": budget_used,
            "reported_assigned_budget_per_sample": budget_used / len(t),
            "actual_event_stopped_budget_total": actual_event_stopped_budget,
            "actual_event_stopped_budget_per_sample": (
                actual_event_stopped_budget / len(t)
            ),
            "allocator_reported_budget_total": allocator_reported_budget,
            "reported_budget_semantics": "sum_assigned_C_i",
            "mean_weight": mean_weight,
            "max_weight": max_weight,
            "upb_infinity_value": UPB_INFINITY_VALUE,
            "upb_calibration_target": "miscoverage_event_t_gt_f",
            "upb_calibration_estimator": {
                "ordinary_ht": "ordinary_horvitz_thompson",
                "terminal_residual": "terminal_residual_augmented_ht",
                "sequential": "sequential_augmented_horvitz_thompson",
            }[estimator_kind],
            "upb_exact_variance_diagnostic_kind": exact_variance_kind,
            "mean_prior_a_weighted_inverse_probability": (
                prior_a_weighted.mean().item()
            ),
            "variance_prior_a_weighted_inverse_probability": (
                prior_a_weighted.var(unbiased=False).item()
            ),
            "mean_tau_0p10_a_weighted_inverse_probability": (
                tau_point_one_a_weighted.mean().item()
            ),
            "variance_tau_0p10_a_weighted_inverse_probability": (
                tau_point_one_a_weighted.var(unbiased=False).item()
            ),
            "tau_0p10_target_a_rate": tau_point_one_a.mean().item(),
            **indexed,
            **additional,
        }
