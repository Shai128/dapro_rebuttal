import numpy as np
import pytest
import torch

from src.predictive_bounds.budget_allocators.random_adaptive_optimized_allocator import (
    ConstantCRCBudgetAllocator,
    RandomAdaptiveOptimizedBudgetAllocator,
)
from src.predictive_bounds.budget_allocators.vectorized_adaptive_allocator_patch import (
    simulate_process_vectorized,
)


def _allocator(*, terminal_pi_min="default", terminal_floor_mode="mixture"):
    kwargs = {}
    if terminal_pi_min != "default":
        kwargs["terminal_pi_min"] = terminal_pi_min
    return RandomAdaptiveOptimizedBudgetAllocator(
        conditional_grid=torch.ones(200, 10, 10),
        budget_per_sample=6.0,
        taus_range=torch.tensor([0.56]),
        tau_prior=0.56,
        m_upper_bound=200,
        terminal_floor_mode=terminal_floor_mode,
        **kwargs,
    )


def test_random_floor_variant_names_are_unambiguous():
    assert _allocator().name == "random_adaptive_optimized"
    assert (
        _allocator(
            terminal_pi_min=0.005,
            terminal_floor_mode="mixture",
        ).name
        == "random_adaptive_optimized_mixture_terminal_floor_0p005"
    )
    assert (
        _allocator(
            terminal_pi_min=0.005,
            terminal_floor_mode="hard",
        ).name
        == "random_adaptive_optimized_hard_terminal_floor_0p005"
    )
    assert (
        _allocator(terminal_pi_min=None).name
        == "random_adaptive_optimized_no_terminal_floor"
    )
    crc = RandomAdaptiveOptimizedBudgetAllocator(
        conditional_grid=torch.ones(200, 10, 10),
        budget_per_sample=6.0,
        taus_range=torch.tensor([0.56]),
        tau_prior=0.56,
        m_upper_bound=200,
        terminal_pi_min=0.005,
        terminal_floor_mode="hard",
        budget_control_mode="crc",
    )
    assert (
        crc.name
        == "random_adaptive_optimized_hard_terminal_floor_0p005_crc"
    )
    constant_crc = RandomAdaptiveOptimizedBudgetAllocator(
        conditional_grid=torch.ones(200, 10, 10),
        budget_per_sample=6.0,
        taus_range=torch.tensor([0.56]),
        tau_prior=0.56,
        m_upper_bound=200,
        terminal_pi_min=None,
        terminal_floor_mode="none",
        budget_control_mode="crc",
    )
    assert (
        constant_crc.name
        == "random_adaptive_optimized_no_terminal_floor_crc"
    )
    explicit_none = RandomAdaptiveOptimizedBudgetAllocator(
        conditional_grid=torch.ones(200, 10, 10),
        budget_per_sample=6.0,
        taus_range=torch.tensor([0.56]),
        tau_prior=0.56,
        m_upper_bound=200,
        terminal_pi_min=0.005,
        terminal_floor_mode="none",
    )
    assert (
        explicit_none.name
        == "random_adaptive_optimized_no_terminal_floor"
    )
    assert explicit_none.min_pi is None


def test_canonical_constant_allocator_cannot_drift_from_crc_policy():
    allocator = ConstantCRCBudgetAllocator(
        conditional_grid=torch.ones(200, 10, 10),
        budget_per_sample=6.0,
        taus_range=torch.tensor([0.56]),
        tau_prior=0.56,
        m_upper_bound=200,
    )

    assert (
        allocator.name
        == "random_adaptive_optimized_mixture_terminal_floor_0p005_crc"
    )
    assert allocator.schedule_family == "constant"
    assert allocator.schedule_alpha == 1.0
    assert allocator.budget_control_mode == "crc"
    assert allocator.terminal_floor_mode == "mixture"
    assert allocator.min_pi == 0.005


def test_nondefault_phase1_size_is_encoded_in_random_method_name():
    allocator = ConstantCRCBudgetAllocator(
        conditional_grid=torch.ones(240, 10, 10),
        budget_per_sample=6.0,
        taus_range=torch.tensor([0.56]),
        tau_prior=0.56,
        m_upper_bound=200,
        phase1_size=120,
    )

    assert allocator.name.endswith("_crc_n1_120")


def test_random_floor_paths_have_expected_constant_policy_semantics():
    n, width = 3, 8
    p = 0.61
    expected_remaining = torch.ones(n, width)
    endpoints = torch.tensor([1, 4, 8])
    pi_func = lambda probability: probability * expected_remaining

    _, raw, _ = simulate_process_vectorized(
        expected_remaining,
        endpoints,
        endpoints,
        p,
        stochastic=False,
        pi_func=pi_func,
        terminal_pi_min=None,
    )
    _, mixture, _ = simulate_process_vectorized(
        expected_remaining,
        endpoints,
        endpoints,
        p,
        stochastic=False,
        pi_func=pi_func,
        terminal_pi_min=0.05,
        terminal_floor_mode="mixture",
    )
    _, hard, _ = simulate_process_vectorized(
        expected_remaining,
        endpoints,
        endpoints,
        p,
        stochastic=False,
        pi_func=pi_func,
        terminal_pi_min=0.05,
        terminal_floor_mode="hard",
    )

    expected_raw = p ** endpoints.to(torch.float64)
    torch.testing.assert_close(raw, expected_raw)
    torch.testing.assert_close(
        mixture,
        0.05 + 0.95 * expected_raw,
    )
    torch.testing.assert_close(
        hard,
        expected_raw.clamp_min(0.05),
    )


def test_complement_power_alpha_only_reparameterizes_the_same_family():
    width = 12
    beta = 0.64
    schedules = []
    for alpha in [0.5, 1.0, 2.0]:
        allocator = RandomAdaptiveOptimizedBudgetAllocator(
            conditional_grid=torch.ones(200, width, width),
            budget_per_sample=6.0,
            taus_range=torch.tensor([0.56]),
            tau_prior=0.56,
            m_upper_bound=200,
            schedule_family="complement_power",
            schedule_alpha=alpha,
        )
        requested_lambda = beta ** (1 / alpha)
        schedules.append(allocator._conditional_schedule(
            1 - requested_lambda,
            width,
            device=torch.device("cpu"),
            dtype=torch.float64,
        ))

    for candidate in schedules[1:]:
        torch.testing.assert_close(candidate, schedules[0])


def test_power_reach_schedule_has_requested_cumulative_shape():
    width = 10
    alpha = 1.7
    aggressiveness = 0.91
    allocator = RandomAdaptiveOptimizedBudgetAllocator(
        conditional_grid=torch.ones(200, width, width),
        budget_per_sample=6.0,
        taus_range=torch.tensor([0.56]),
        tau_prior=0.56,
        m_upper_bound=200,
        schedule_family="power_reach",
        schedule_alpha=alpha,
    )
    conditionals = allocator._conditional_schedule(
        aggressiveness,
        width,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    expected = aggressiveness ** torch.pow(
        torch.arange(1, width + 1, dtype=torch.float64),
        alpha,
    )
    torch.testing.assert_close(conditionals.cumprod(dim=1)[0], expected)


def test_large_random_weights_are_real_only_without_the_terminal_floor():
    width = 200
    p = 0.94
    expected_remaining = torch.ones(1, width, dtype=torch.float64)
    endpoint = torch.tensor([width])
    pi_func = lambda probability: probability * expected_remaining

    _, raw, _ = simulate_process_vectorized(
        expected_remaining,
        endpoint,
        endpoint,
        p,
        stochastic=False,
        pi_func=pi_func,
        terminal_pi_min=None,
    )
    _, hard, _ = simulate_process_vectorized(
        expected_remaining,
        endpoint,
        endpoint,
        p,
        stochastic=False,
        pi_func=pi_func,
        terminal_pi_min=0.005,
        terminal_floor_mode="hard",
    )
    _, mixture, _ = simulate_process_vectorized(
        expected_remaining,
        endpoint,
        endpoint,
        p,
        stochastic=False,
        pi_func=pi_func,
        terminal_pi_min=0.005,
        terminal_floor_mode="mixture",
    )

    assert (1 / raw).item() > 100_000
    assert (1 / hard).item() <= 200 + 1e-10
    assert (1 / mixture).item() <= 200 + 1e-10


@pytest.mark.parametrize(
    ("terminal_pi_min", "terminal_floor_mode", "constant_flag"),
    [
        (None, "none", 1),
        (0.05, "hard", 0),
        (0.05, "mixture", 0),
    ],
)
def test_random_allocator_tunes_a_feasible_expected_budget(
        terminal_pi_min,
        terminal_floor_mode,
        constant_flag):
    allocator = _allocator(
        terminal_pi_min=terminal_pi_min,
        terminal_floor_mode=terminal_floor_mode,
    )
    allocator.acquisition_seed = 19
    t = torch.full((200,), 10)
    quantiles = torch.full((200, 1), 10)

    np.random.seed(7)
    result = allocator.allocate_budget(
        probability_est=torch.empty(200),
        x=torch.empty(200),
        t=t,
        quantile_est=quantiles,
    )
    metrics = result.additional_metrics

    assert metrics["phase1_realized_cost_total"] == 1000
    assert metrics["configured_total_budget"] == 1200
    assert metrics["phase1_tuned_expected_cost_per_sample"] <= 2 + 1e-7
    assert metrics["phase2_expected_cost_per_sample"] <= 2 + 1e-7
    assert abs(metrics["total_expected_budget_per_sample"] - 6) <= 1e-6
    assert metrics["total_expected_budget_valid"] == 1
    assert metrics["random_executed_policy_is_constant"] == constant_flag
    assert 0 <= metrics["random_constant_probability"] <= 1
    if terminal_pi_min is not None:
        assert result.max_weight <= 1 / terminal_pi_min + 1e-7


def test_random_allocator_rejects_an_infeasible_floor_budget():
    allocator = RandomAdaptiveOptimizedBudgetAllocator(
        conditional_grid=torch.ones(200, 10, 10),
        budget_per_sample=5.1,
        taus_range=torch.tensor([0.56]),
        tau_prior=0.56,
        m_upper_bound=200,
        terminal_pi_min=0.05,
        terminal_floor_mode="hard",
    )
    t = torch.full((200,), 10)
    quantiles = torch.full((200, 1), 10)

    with pytest.raises(ValueError, match="floor makes the Random"):
        allocator.allocate_budget(
            probability_est=torch.empty(200),
            x=torch.empty(200),
            t=t,
            quantile_est=quantiles,
        )


def test_random_crc_selector_is_integrated_with_full_budget_metrics():
    n, width = 200, 10
    allocator = RandomAdaptiveOptimizedBudgetAllocator(
        conditional_grid=torch.ones(n, width, width),
        budget_per_sample=6.0,
        taus_range=torch.tensor([0.56]),
        tau_prior=0.56,
        m_upper_bound=200,
        terminal_pi_min=0.005,
        terminal_floor_mode="hard",
        budget_control_mode="crc",
    )
    allocator.acquisition_seed = 19
    event_times = torch.full((n,), width)
    quantiles = torch.full((n, 1), width)

    np.random.seed(7)
    result = allocator.allocate_budget(
        probability_est=torch.empty(n),
        x=torch.empty(n),
        t=event_times,
        quantile_est=quantiles,
    )
    metrics = result.additional_metrics
    expected_envelope = (1 + 101 / 100) * width
    expected_limit = ((101 * 2.0) - expected_envelope) / 100

    assert abs(
        metrics["phase1_empirical_budget_limit_per_sample"]
        - expected_limit
    ) <= 1e-12
    assert (
        metrics["phase1_tuned_expected_cost_per_sample"]
        <= expected_limit + 1e-7
    )
    assert metrics["crc_selector_left_side_per_sample"] <= 2 + 1e-7
    assert (
        metrics["crc_distribution_free_envelope_upper_bound"]
        == expected_envelope
    )
    assert metrics["crc_distribution_free_envelope_used"] == 1
    assert metrics["crc_selector_valid"] == 1
    assert metrics["total_expected_budget_per_sample"] <= 6 + 1e-7
    assert metrics["total_expected_budget_valid"] == 1
    assert metrics["configured_total_budget"] == 1200


def test_constant_crc_probability_does_not_use_phase2_event_times():
    """Changing unrevealed Phase-II labels cannot change the policy."""
    n, width = 240, 8
    allocator_kwargs = dict(
        conditional_grid=torch.ones(n, width, width),
        budget_per_sample=5.0,
        taus_range=torch.tensor([0.56]),
        tau_prior=0.56,
        m_upper_bound=200,
    )
    original_times = (torch.arange(n) % width + 1).to(torch.long)
    changed_times = original_times.clone()
    quantiles = torch.full((n, 1), width)

    np.random.seed(31)
    permutation = np.random.permutation(n)
    phase2_indices = permutation[100:]
    changed_times[phase2_indices] = torch.where(
        changed_times[phase2_indices] == 1,
        torch.tensor(width),
        torch.tensor(1),
    )

    results = []
    for event_times in [original_times, changed_times]:
        allocator = ConstantCRCBudgetAllocator(**allocator_kwargs)
        allocator.acquisition_seed = 17
        np.random.seed(31)
        results.append(allocator.allocate_budget(
            probability_est=torch.empty(n),
            x=torch.empty(n),
            t=event_times,
            quantile_est=quantiles,
        ))

    first, second = [result.additional_metrics for result in results]
    assert (
        first["random_constant_continuation_probability"]
        == second["random_constant_continuation_probability"]
    )
    assert (
        first["phase1_tuned_expected_cost_per_sample"]
        == second["phase1_tuned_expected_cost_per_sample"]
    )
    assert first["random_executed_policy_is_constant"] == 1
    assert second["random_executed_policy_is_constant"] == 1
    assert first["random_policy_uses_phase2_event_times"] == 0
    assert first["budget_guarantee_kind"] == (
        "crc_marginal_expected_total_budget"
    )


def test_constant_crc_returns_exact_mixture_path_propensities():
    n, width = 220, 8
    allocator = ConstantCRCBudgetAllocator(
        conditional_grid=torch.ones(n, width, width),
        budget_per_sample=5.0,
        taus_range=torch.tensor([0.56]),
        tau_prior=0.56,
        m_upper_bound=200,
    )
    allocator.acquisition_seed = 29
    event_times = (torch.arange(n) % width + 1).to(torch.long)
    quantiles = torch.full((n, 1), width)

    np.random.seed(13)
    permutation = np.random.permutation(n)
    phase1_indices = permutation[:100]
    phase2_indices = permutation[100:]
    np.random.seed(13)
    result = allocator.allocate_budget(
        probability_est=torch.randn(n),
        x=torch.randn(n, 3),
        t=event_times,
        quantile_est=quantiles,
    )

    p = result.additional_metrics[
        "random_constant_continuation_probability"
    ]
    active_lengths = torch.minimum(
        event_times[phase2_indices],
        quantiles[phase2_indices, 0].to(torch.long),
    )
    raw_propensities = torch.pow(
        torch.tensor(p, dtype=torch.float64), active_lengths
    )
    expected_propensities = 0.005 + 0.995 * raw_propensities
    torch.testing.assert_close(
        result.C_probs[phase2_indices],
        expected_propensities,
    )
    torch.testing.assert_close(
        result.C_probs[phase1_indices],
        torch.ones(100, dtype=result.C_probs.dtype),
    )
    assert result.max_weight <= 200 + 1e-7


def test_stochastic_simulator_probability_zero_never_advances():
    n, width = 2, 4
    expected_remaining = torch.ones(n, width)
    endpoints = torch.full((n,), width)
    zero_uniforms = torch.zeros(n, width)

    censoring, _, cost = simulate_process_vectorized(
        expected_remaining,
        endpoints,
        endpoints,
        0.0,
        stochastic=True,
        uniforms=zero_uniforms,
        pi_func=lambda probability: probability * expected_remaining,
        terminal_pi_min=None,
    )

    assert cost == 0
    torch.testing.assert_close(censoring, torch.zeros(n, dtype=torch.long))
