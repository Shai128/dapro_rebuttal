"""Model-only endpoint/block schedules paired with terminal residual AHT."""

from __future__ import annotations

import numpy as np
import torch

from src.predictive_bounds.budget_allocators.budget_allocator import (
    BudgetAllocationResult,
    BudgetAllocator,
    summarize_expected_budget,
)
from src.predictive_bounds.budget_allocators.risk_controlled_budget import (
    select_crc_budget_candidate,
)
from src.predictive_bounds.calibration.calibration_utils import get_prior


class EndpointResidualAHTAllocator(BudgetAllocator):
    """One initial Bernoulli decision followed by a complete reveal block.

    The score is the initial model residual variance ``m(1-m)`` per predicted
    reveal cost.  Two median score bins share probabilities.  The raw variant
    scales those probabilities to the model-predicted budget with no
    projection-error reserve; the CRC variant uses a fully observed control
    fold and a fixed nested scale family.
    """

    aht_estimator_kind = "terminal_residual"

    def __init__(
            self,
            conditional_grid: torch.Tensor,
            budget_per_sample: float,
            taus_range: torch.Tensor,
            tau_prior: float,
            m_upper_bound: int,
            *,
            target_kind: str,
            target_alpha: float = 0.10,
            crc_control_size: int = 0,
            candidate_count: int = 401,
            terminal_pi_min: float = 0.005,
    ):
        super().__init__(budget_per_sample, taus_range, tau_prior)
        if target_kind not in {"metric", "lpb"}:
            raise ValueError("Endpoint residual allocation supports metric or LPB targets.")
        self.conditional_grid = conditional_grid
        self.m_upper_bound = int(m_upper_bound)
        self.target_kind = target_kind
        self.target_alpha = float(target_alpha)
        self.crc_control_size = int(crc_control_size)
        self.candidate_count = int(candidate_count)
        self.terminal_pi_min = float(terminal_pi_min)
        if not 0 < self.terminal_pi_min <= 1:
            raise ValueError("The propensity floor must lie in (0, 1].")

    @property
    def name(self) -> str:
        target = (
            f"metric_m{self.m_upper_bound}"
            if self.target_kind == "metric"
            else f"lpb_c{1.0 - self.target_alpha:.2f}".replace(".", "p")
        )
        suffix = (
            f"_crc_control_{self.crc_control_size}"
            if self.crc_control_size
            else "_raw_model_budget"
        )
        return f"endpoint_block_terminal_residual_aht_{target}{suffix}"

    def _initial_target_probability(
            self, quantile_est: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        grid = self.conditional_grid.to(torch.float64)
        pmf = grid[:, 0, :].clamp_min(0.0)
        pmf = pmf / pmf.sum(dim=1, keepdim=True).clamp_min(1e-15)
        n, width, _ = grid.shape
        if self.target_kind == "metric":
            horizons = torch.full(
                (n,), self.m_upper_bound, dtype=torch.long, device=grid.device
            )
            strict = False
        else:
            anchor = int(
                torch.abs(self.taus_range - self.target_alpha).argmin().item()
            )
            horizons = quantile_est[:, anchor].to(torch.long).clamp(
                min=1, max=width
            )
            strict = True
        outcome = torch.arange(1, pmf.shape[1] + 1, device=grid.device)
        mask = outcome[None, :] < horizons[:, None] if strict else outcome[None, :] <= horizons[:, None]
        probability = (pmf * mask.to(pmf.dtype)).sum(dim=1)
        return probability.clamp(0.0, 1.0), horizons

    def _model_cost(self, prior: torch.Tensor) -> torch.Tensor:
        pmf = self.conditional_grid[:, 0, :].to(torch.float64).clamp_min(0.0)
        pmf = pmf / pmf.sum(dim=1, keepdim=True).clamp_min(1e-15)
        tail = torch.flip(
            torch.cumsum(torch.flip(pmf, dims=(1,)), dim=1), dims=(1,)
        )[:, :self.m_upper_bound]
        active = torch.arange(
            self.m_upper_bound, device=pmf.device
        )[None, :] < prior[:, None]
        return (tail * active).sum(dim=1).clamp_min(1e-12)

    @staticmethod
    def _k2_base(
            risk: torch.Tensor, cost: torch.Tensor, score: torch.Tensor
    ) -> tuple[torch.Tensor, float, torch.Tensor]:
        threshold = float(torch.median(score).item())
        high = score >= threshold
        values = []
        for mask in (~high, high):
            if torch.any(mask):
                values.append(torch.sqrt(
                    risk[mask].sum() / cost[mask].sum().clamp_min(1e-12)
                ))
            else:
                values.append(torch.tensor(0.0, dtype=torch.float64, device=score.device))
        bases = torch.stack(values)
        if bases[0] > bases[1]:
            bases[:] = torch.sqrt(risk.sum() / cost.sum().clamp_min(1e-12))
        row = torch.where(high, bases[1], bases[0])
        row = row / row.max().clamp_min(1e-12)
        return row, threshold, bases

    def allocate_budget(
            self,
            probability_est: torch.Tensor,
            x: torch.Tensor,
            t: torch.Tensor,
            quantile_est: torch.Tensor,
    ) -> BudgetAllocationResult:
        del probability_est, x
        n, width, _ = self.conditional_grid.shape
        prior = get_prior(quantile_est, self.taus_range, self.tau_prior).to(
            torch.long
        ).clamp(min=1, max=width)
        probability, _ = self._initial_target_probability(quantile_est)
        risk = probability * (1.0 - probability)
        model_cost = self._model_cost(prior)
        score = torch.sqrt(risk / model_cost)
        row_base, threshold, bin_bases = self._k2_base(
            risk, model_cost, score
        )
        epsilon = self.terminal_pi_min
        row_base = row_base.clamp(min=epsilon, max=1.0)
        permutation = np.random.permutation(n)
        control = permutation[:self.crc_control_size]
        deployment = permutation[self.crc_control_size:]
        if len(deployment) == 0:
            raise ValueError("The CRC control fold must be smaller than N.")
        lengths = torch.minimum(t.reshape(-1).to(torch.long), prior).to(torch.float64)
        total_budget = float(self.budget_per_sample * n)
        pilot_cost = float(lengths[control].sum().item())

        selector_metrics = {}
        if self.crc_control_size == 0:
            def candidate(scale: float) -> torch.Tensor:
                return torch.clamp(scale * row_base, min=epsilon, max=1.0)

            low, high_scale = 0.0, 1.0
            while float((candidate(high_scale) * model_cost).mean().item()) < self.budget_per_sample and high_scale < 1e12:
                high_scale *= 2.0
            for _ in range(80):
                middle = (low + high_scale) / 2.0
                if float((candidate(middle) * model_cost).mean().item()) <= self.budget_per_sample:
                    low = middle
                else:
                    high_scale = middle
            selected_probability = candidate(low)
            selector_metrics = {
                "endpoint_block_common_scale": low,
                "expected_budget_guarantee_kind": "model_predicted_raw_no_projection_margin",
                "expected_budget_guarantee_requires_projection_error_bound": 0,
            }
        else:
            alpha = torch.linspace(
                1.0, 0.0, self.candidate_count,
                dtype=torch.float64, device=t.device,
            )
            family = epsilon + alpha[None, :] * (row_base[:, None] - epsilon)
            selection = select_crc_budget_candidate(
                (lengths[control, None] * family[control]).detach().cpu().numpy(),
                lengths[control].detach().cpu().numpy(),
                total_budget_after_policy_fit=total_budget,
                deployment_sample_count=len(deployment),
                maximum_cost_per_sample=float(width),
                maximum_candidate_cost_per_sample=float(width),
                maximum_pilot_cost_per_sample=float(width),
            )
            selected_probability = family[:, selection.selected_index]
            selector_metrics = {
                "risk_budget_selector_valid": 1,
                "risk_budget_selected_candidate_index": selection.selected_index,
                "expected_budget_guarantee_kind": selection.guarantee_kind,
                "expected_budget_guarantee_requires_projection_error_bound": 0,
            }

        uniforms = self.get_acquisition_uniforms(
            n, width, device=t.device, dtype=torch.float64
        )
        if uniforms is None:
            self.reset_acquisition_rng()
            first = torch.rand(n, device=t.device, dtype=torch.float64)
        else:
            first = uniforms[:, 0]
        chosen = first[deployment] < selected_probability[deployment]
        C = torch.zeros(n, dtype=torch.long, device=t.device)
        C[control] = prior[control]
        C[deployment] = torch.where(
            chosen, prior[deployment], torch.zeros_like(prior[deployment])
        )
        terminal_pi = selected_probability.clone()
        terminal_pi[control] = 1.0
        candidate_pi = terminal_pi[:, None].expand_as(quantile_est).clone()
        candidate_pi = torch.where(
            quantile_est == width + 1,
            torch.ones_like(candidate_pi),
            candidate_pi,
        )
        realized = float(torch.minimum(t.reshape(-1), C).sum().item())
        expected = pilot_cost + float(
            (lengths[deployment] * terminal_pi[deployment]).sum().item()
        )
        metrics = {
            "objective_kind": "endpoint_block_terminal_residual_aht",
            "aht_estimator_kind": "terminal_residual",
            "endpoint_block_target_kind": self.target_kind,
            "endpoint_block_score": "sqrt_initial_residual_variance_per_expected_cost",
            "endpoint_block_k2_threshold": threshold,
            "endpoint_block_low_base": float(bin_bases[0].item()),
            "endpoint_block_high_base": float(bin_bases[1].item()),
            "policy_fit_label_count": 0,
            "crc_control_sample_count": self.crc_control_size,
            **summarize_expected_budget(
                expected,
                n,
                self.budget_per_sample,
                cost_semantics="control_full_plus_endpoint_block_event_stopped",
            ),
            **selector_metrics,
        }
        return BudgetAllocationResult(
            quantile_est,
            C,
            terminal_pi,
            total_budget_used=realized,
            mean_weight=float(terminal_pi.reciprocal().mean().item()),
            max_weight=float(terminal_pi.reciprocal().max().item()),
            additional_metrics=metrics,
            candidate_C_probs=candidate_pi,
        )


__all__ = ["EndpointResidualAHTAllocator"]
