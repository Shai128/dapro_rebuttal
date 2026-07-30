import torch

from src.safety_evaluation.budget_allocators.budget_allocator import (
    BudgetAllocator,
)
from src.safety_evaluation.budget_allocators.adaptive_optimized_allocator import (
    AdaptiveOptimizedBudgetAllocator,
    phase1_empirical_budget_limit,
)
from src.safety_evaluation.budget_allocators.vectorized_adaptive_allocator_patch import (
    expected_acquisition_cost,
    precompute_expected_remaining,
    simulate_process_vectorized,
)


def legacy_expected_remaining(grid, prior_q):
    n, t_max, _ = grid.shape
    rows = []
    for t_curr in range(t_max):
        belief = grid[:, t_curr, t_curr:]
        remaining = torch.clamp(
            prior_q - t_curr,
            min=0,
            max=belief.shape[1],
        ).float()
        mask = (
            torch.arange(1, belief.shape[1] + 1, device=grid.device)[None, :]
            <= remaining[:, None]
        )
        expected = (
            belief
            * mask
            * torch.arange(
                1, belief.shape[1] + 1,
                device=grid.device,
                dtype=grid.dtype,
            )[None, :]
        ).sum(dim=-1)
        expected += remaining * (belief * (~mask)).sum(dim=-1)
        rows.append(expected)
    return torch.stack(rows, dim=1)


def legacy_with_fixed_uniforms(
    expected_remaining,
    prior_q,
    true_t,
    lam,
    uniforms,
    reach_t_max_is_success=False,
):
    n, t_max = expected_remaining.shape
    sim_C = torch.zeros(n, dtype=torch.long)
    active = torch.ones(n, dtype=torch.bool)
    cumulative_all = torch.ones(n)
    policy_cumulative = torch.ones(n)
    total = 0.0

    for t_curr in range(t_max):
        alive = (true_t > t_curr) & (prior_q > t_curr)
        active = active & alive
        target_terminal_probability = torch.rsqrt(
            lam * expected_remaining[:, t_curr] + 1e-12
        ).clamp(max=1.0)
        pi = torch.minimum(
            target_terminal_probability / policy_cumulative.clamp_min(1e-12),
            torch.ones_like(target_terminal_probability),
        )
        policy_cumulative *= pi
        cumulative_all *= torch.where(alive, pi, torch.ones_like(pi))
        keep = (uniforms[:, t_curr] <= pi) & active
        total += keep.sum().item()
        sim_C[keep] += 1
        active &= keep

    succeeded = (sim_C >= prior_q) | (sim_C >= true_t)
    if reach_t_max_is_success:
        succeeded |= sim_C == t_max
    sim_C = torch.where(succeeded, prior_q, sim_C)
    return sim_C, cumulative_all, total


def test_vectorized_matches_legacy_logic():
    torch.manual_seed(123)
    n, t_max = 19, 13
    grid = torch.rand(n, t_max, t_max)
    prior_q = torch.randint(0, t_max, (n,))
    true_t = torch.randint(0, t_max, (n,))
    lam = 0.73
    uniforms = torch.rand(n, t_max)

    legacy_expected = legacy_expected_remaining(grid, prior_q)
    vector_expected = precompute_expected_remaining(
        grid, prior_q, sample_chunk_size=7
    )
    torch.testing.assert_close(
        vector_expected, legacy_expected, rtol=1e-6, atol=1e-6
    )

    expected_C, expected_prob, expected_cost = legacy_with_fixed_uniforms(
        legacy_expected, prior_q, true_t, lam, uniforms
    )
    actual_C, actual_prob, actual_cost = simulate_process_vectorized(
        vector_expected,
        prior_q,
        true_t,
        lam,
        stochastic=True,
        uniforms=uniforms,
    )
    torch.testing.assert_close(actual_C, expected_C)
    torch.testing.assert_close(
        actual_prob,
        expected_prob.to(torch.float64),
        rtol=1e-6,
        atol=1e-7,
    )
    assert actual_cost == expected_cost


def test_whole_grid_precompute_matches_split_precompute_bitwise():
    torch.manual_seed(321)
    n, t_max = 37, 11
    grid = torch.rand(n, t_max, t_max)
    prior_q = torch.randint(0, t_max + 1, (n,))
    permutation = torch.randperm(n)
    validation_indices = permutation[:9]
    test_indices = permutation[9:]

    whole = precompute_expected_remaining(
        grid,
        prior_q,
        sample_chunk_size=7,
    )
    validation_split = precompute_expected_remaining(
        grid[validation_indices],
        prior_q[validation_indices],
        sample_chunk_size=7,
    )
    test_split = precompute_expected_remaining(
        grid[test_indices],
        prior_q[test_indices],
        sample_chunk_size=7,
    )

    assert torch.equal(whole[validation_indices], validation_split)
    assert torch.equal(whole[test_indices], test_split)


def test_cost_only_path_matches_full_deterministic_simulator():
    torch.manual_seed(654)
    n, t_max = 23, 12
    expected_remaining = torch.rand(n, t_max) + 0.05
    prior_q = torch.randint(0, t_max + 1, (n,))
    true_t = torch.randint(0, t_max + 1, (n,))
    lam = 0.63

    for use_constant_policy in (False, True):
        pi_func = None
        if use_constant_policy:
            pi_func = lambda value: (
                value * torch.ones_like(expected_remaining)
            ).clamp(max=1.0)
        for floor, mode in (
            (None, "none"),
            (0.05, "hard"),
            (0.05, "mixture"),
        ):
            _, _, full_cost = simulate_process_vectorized(
                expected_remaining,
                prior_q,
                true_t,
                lam,
                stochastic=False,
                pi_func=pi_func,
                terminal_pi_min=floor,
                terminal_floor_mode=mode,
            )
            cost_only = expected_acquisition_cost(
                expected_remaining,
                prior_q,
                true_t,
                lam,
                pi_func=pi_func,
                terminal_pi_min=floor,
                terminal_floor_mode=mode,
            )
            assert cost_only.ndim == 0
            assert cost_only.item() == full_cost


def test_terminal_probability_is_early_endpoint_propensity():
    torch.manual_seed(456)
    n, t_max = 11, 9
    expected_remaining = torch.rand(n, t_max) + 0.1
    prior_q = torch.randint(0, t_max, (n,))
    true_t = torch.randint(0, t_max, (n,))
    lam = 1.4

    _, probability_a, _ = simulate_process_vectorized(
        expected_remaining,
        prior_q,
        true_t,
        lam,
        stochastic=True,
        uniforms=torch.rand(n, t_max),
    )
    _, probability_b, _ = simulate_process_vectorized(
        expected_remaining,
        prior_q,
        true_t,
        lam,
        stochastic=True,
        uniforms=torch.rand(n, t_max),
    )

    # Propensity must not depend on the realized acquisition path.
    torch.testing.assert_close(probability_a, probability_b)

    target_terminal = torch.rsqrt(
        lam * expected_remaining + 1e-12
    ).clamp(max=1.0)
    cumulative_terminal = torch.cummin(target_terminal, dim=1).values
    previous_terminal = torch.cat(
        [torch.ones(n, 1), cumulative_terminal[:, :-1]],
        dim=1,
    )
    pi = (cumulative_terminal / previous_terminal).clamp(max=1.0)
    time = torch.arange(t_max)[None, :]
    endpoint_mask = (
        (time < true_t[:, None])
        & (time < prior_q[:, None])
    )
    expected_probability = torch.where(
        endpoint_mask, pi, torch.ones_like(pi)
    ).prod(dim=1)
    torch.testing.assert_close(
        probability_a,
        expected_probability.to(torch.float64),
    )


def test_one_interaction_endpoint_has_one_probability_factor():
    expected_remaining = torch.ones(1, 1)
    prior_q = torch.tensor([1])
    true_t = torch.tensor([1])

    _, terminal_probability, expected_cost = simulate_process_vectorized(
        expected_remaining,
        prior_q,
        true_t,
        0.5,
        stochastic=False,
        pi_func=lambda _: torch.tensor([[0.5]]),
    )

    torch.testing.assert_close(
        terminal_probability,
        torch.tensor([0.5], dtype=torch.float64),
    )
    assert expected_cost == 0.5


def test_hard_terminal_floor_is_minimally_distortive():
    expected_remaining = torch.ones(1, 3)
    prior_q = torch.tensor([3])
    true_t = torch.tensor([3])
    raw_conditionals = torch.tensor([[0.8, 0.5, 0.01]])
    pi_func = lambda _: raw_conditionals

    _, raw_terminal, raw_cost = simulate_process_vectorized(
        expected_remaining,
        prior_q,
        true_t,
        1.0,
        stochastic=False,
        pi_func=pi_func,
        terminal_pi_min=None,
    )
    _, hard_terminal, hard_cost = simulate_process_vectorized(
        expected_remaining,
        prior_q,
        true_t,
        1.0,
        stochastic=False,
        pi_func=pi_func,
        terminal_pi_min=0.05,
        terminal_floor_mode="hard",
    )
    _, mixture_terminal, mixture_cost = simulate_process_vectorized(
        expected_remaining,
        prior_q,
        true_t,
        1.0,
        stochastic=False,
        pi_func=pi_func,
        terminal_pi_min=0.05,
        terminal_floor_mode="mixture",
    )

    torch.testing.assert_close(
        raw_terminal,
        torch.tensor([0.004], dtype=torch.float64),
    )
    torch.testing.assert_close(
        hard_terminal,
        torch.tensor([0.05], dtype=torch.float64),
    )
    torch.testing.assert_close(
        mixture_terminal,
        torch.tensor([0.0538], dtype=torch.float64),
    )
    assert raw_cost < hard_cost < mixture_cost


class _SeedProbeAllocator(BudgetAllocator):
    @property
    def name(self):
        return "seed_probe"

    def allocate_budget(self, probability_est, x, t, quantile_est):
        raise NotImplementedError


def test_explicit_acquisition_seed_is_reproducible_and_optional():
    allocator = _SeedProbeAllocator(
        1.0,
        torch.tensor([0.1]),
        0.1,
    )

    torch.manual_seed(91)
    before = torch.rand(4)
    allocator.reset_acquisition_rng()
    after_without_seed = torch.rand(4)
    torch.manual_seed(91)
    assert torch.equal(before, torch.rand(4))
    assert torch.equal(after_without_seed, torch.rand(4))

    allocator.acquisition_seed = 17
    allocator.reset_acquisition_rng()
    first = torch.rand(8)
    torch.manual_seed(999)
    allocator.reset_acquisition_rng()
    second = torch.rand(8)
    assert torch.equal(first, second)


def test_local_default_is_a_hard_terminal_floor():
    allocator = AdaptiveOptimizedBudgetAllocator(
        conditional_grid=torch.ones(200, 10, 10),
        budget_per_sample=6.0,
        taus_range=torch.tensor([0.56]),
        tau_prior=0.56,
        m_upper_bound=200,
    )
    assert allocator.name == "adaptive_optimized"
    assert allocator.terminal_floor_mode == "hard"
    assert allocator.min_pi == 0.005


def test_local_explicit_none_mode_disables_floor_diagnostics():
    allocator = AdaptiveOptimizedBudgetAllocator(
        conditional_grid=torch.ones(200, 10, 10),
        budget_per_sample=6.0,
        taus_range=torch.tensor([0.56]),
        tau_prior=0.56,
        m_upper_bound=200,
        terminal_pi_min=0.005,
        terminal_floor_mode="none",
    )
    assert allocator.name == "adaptive_optimized_no_terminal_floor"
    assert allocator.terminal_floor_mode == "none"
    assert allocator.min_pi is None


def test_crc_budget_limit_matches_appendix_d4_selector():
    target = 16.4
    n1 = 100
    n2 = 2900
    t_max = 200
    rho = (n1 + 1) / n2
    envelope = (1 + rho) * t_max
    limit = phase1_empirical_budget_limit(
        target,
        n1,
        t_max,
        "crc",
        phase2_sample_count=n2,
    )
    assert limit == ((n1 + 1) * target - envelope) / n1
    selector_left = (
        n1 / (n1 + 1) * limit + envelope / (n1 + 1)
    )
    assert abs(selector_left - target) <= 1e-12
    assert phase1_empirical_budget_limit(
        target,
        n1,
        t_max,
        "empirical",
    ) == target


def test_local_crc_selector_is_integrated_with_full_budget_metrics():
    n, width = 200, 10
    grid = torch.ones(n, width, width) / width
    allocator = AdaptiveOptimizedBudgetAllocator(
        conditional_grid=grid,
        budget_per_sample=6.0,
        taus_range=torch.tensor([0.56]),
        tau_prior=0.56,
        m_upper_bound=200,
        budget_control_mode="crc",
    )
    allocator.acquisition_seed = 19
    event_times = torch.full((n,), width)
    quantiles = torch.full((n, 1), width)

    result = allocator.allocate_budget(
        probability_est=torch.empty(n),
        x=torch.empty(n),
        t=event_times,
        quantile_est=quantiles,
    )
    metrics = result.additional_metrics
    expected_envelope = (1 + 101 / 100) * width
    expected_limit = ((101 * 2.0) - expected_envelope) / 100

    assert allocator.name == "adaptive_optimized_crc"
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


if __name__ == "__main__":
    test_vectorized_matches_legacy_logic()
    test_terminal_probability_is_early_endpoint_propensity()
    test_one_interaction_endpoint_has_one_probability_factor()
    test_hard_terminal_floor_is_minimally_distortive()
    test_explicit_acquisition_seed_is_reproducible_and_optional()
    test_local_default_is_a_hard_terminal_floor()
    test_local_explicit_none_mode_disables_floor_diagnostics()
    test_crc_budget_limit_matches_appendix_d4_selector()
    test_local_crc_selector_is_integrated_with_full_budget_metrics()
    print("All vectorized simulation tests passed.")
