import numpy as np
import torch

from src.predictive_bounds.budget_allocators.oracle_dapro_allocator import (
    CRCOracleTargetADAPRO,
    GlobalOracleTargetADAPRO,
    SplitOracleTargetADAPRO,
    solve_unrestricted_oracle_policy,
)


def _problem(n=300, width=10):
    grid = torch.ones(n, width, width)
    taus = torch.tensor([0.10, 0.56])
    quantiles = torch.stack(
        [
            torch.full((n,), 5),
            torch.full((n,), width),
        ],
        dim=1,
    )
    event_times = (torch.arange(n) % width + 1).to(torch.long)
    return grid, taus, quantiles, event_times


def test_unrestricted_oracle_has_closed_form_one_step_solution():
    result = solve_unrestricted_oracle_policy(
        active_lengths=torch.tensor([1, 1]),
        objective_weights=torch.tensor([1.0, 0.0]),
        budget_per_sample=0.4,
        width=1,
        minimum_terminal_probability=1e-12,
    )

    torch.testing.assert_close(
        result.terminal_probabilities,
        torch.tensor([0.8, 1e-12], dtype=torch.float64),
        atol=2e-10,
        rtol=0,
    )
    assert result.expected_cost_per_sample <= 0.4 + 1e-10


def test_oracle_uses_flat_cumulative_reach_at_minimum_cost():
    lengths = torch.tensor([2, 4, 5])
    weights = torch.tensor([1.0, 1.0, 0.0])
    result = solve_unrestricted_oracle_policy(
        lengths,
        weights,
        budget_per_sample=0.7,
        width=5,
    )
    cumulative = result.continuation_probabilities.cumprod(dim=1)

    for row, length in enumerate(lengths.tolist()):
        torch.testing.assert_close(
            cumulative[row, :length],
            result.terminal_probabilities[row].expand(length),
        )
    expected = (
        lengths.to(torch.float64) * result.terminal_probabilities
    ).mean().item()
    assert abs(expected - result.expected_cost_per_sample) <= 1e-12


def test_split_oracle_fully_observes_only_phase1_and_meets_budget():
    grid, taus, quantiles, event_times = _problem()
    allocator = SplitOracleTargetADAPRO(
        grid,
        budget_per_sample=2.0,
        taus_range=taus,
        tau_prior=0.56,
        m_upper_bound=200,
        n1=100,
    )
    allocator.acquisition_seed = 4
    np.random.seed(7)
    result = allocator.allocate_budget(
        torch.empty(len(grid)),
        torch.empty(len(grid)),
        event_times,
        quantiles,
    )

    phase1 = allocator.last_phase1_indices
    phase2 = allocator.last_phase2_indices
    torch.testing.assert_close(
        allocator.last_continuation_probabilities[phase1],
        torch.ones((len(phase1), grid.shape[1]), dtype=torch.float64),
    )
    torch.testing.assert_close(
        result.C_probs[phase1],
        torch.ones(len(phase1), dtype=torch.float64),
    )
    assert len(phase1) == 100
    assert len(phase2) == 200
    assert result.additional_metrics["total_expected_budget_valid"] == 1
    assert result.additional_metrics["phase1_all_probabilities_one"] == 1
    assert result.additional_metrics["oracle_uses_full_trajectories"] == 1


def test_crc_oracle_has_three_splits_and_valid_crc_selector():
    grid, taus, quantiles, event_times = _problem()
    allocator = CRCOracleTargetADAPRO(
        grid,
        budget_per_sample=2.0,
        taus_range=taus,
        tau_prior=0.56,
        m_upper_bound=200,
        n1=100,
        budget_control_size=50,
        budget_candidate_count=501,
    )
    allocator.acquisition_seed = 8
    np.random.seed(11)
    result = allocator.allocate_budget(
        torch.empty(len(grid)),
        torch.empty(len(grid)),
        event_times,
        quantiles,
    )
    metrics = result.additional_metrics
    phase1 = allocator.last_phase1_indices
    control = allocator.last_control_indices

    assert set(control).issubset(set(phase1))
    assert len(phase1) == 100
    assert len(control) == 50
    assert len(allocator.last_phase2_indices) == 200
    torch.testing.assert_close(
        allocator.last_continuation_probabilities[phase1],
        torch.ones((100, grid.shape[1]), dtype=torch.float64),
    )
    assert metrics["crc_split_all_probabilities_one"] == 1
    assert metrics["risk_budget_control_mode"] == "crc"
    assert metrics["risk_budget_selector_valid"] == 1
    assert metrics["risk_budget_guarantee_kind"] == (
        "crc_marginal_expected_total_budget"
    )
    assert metrics["total_expected_budget_valid"] == 1


def test_global_oracle_has_no_fully_observed_split_and_controls_budget():
    grid, taus, quantiles, event_times = _problem()
    allocator = GlobalOracleTargetADAPRO(
        grid,
        budget_per_sample=0.8,
        taus_range=taus,
        tau_prior=0.56,
        m_upper_bound=200,
    )
    allocator.acquisition_seed = 9
    result = allocator.allocate_budget(
        torch.empty(len(grid)),
        torch.empty(len(grid)),
        event_times,
        quantiles,
    )
    metrics = result.additional_metrics

    assert len(allocator.last_phase1_indices) == 0
    assert len(allocator.last_phase2_indices) == len(grid)
    assert metrics["phase1_sample_count"] == 0
    assert metrics["oracle_split_mode"] == "none"
    assert metrics["total_expected_budget_per_sample"] <= 0.8 + 1e-9
    assert metrics["total_expected_budget_valid"] == 1
