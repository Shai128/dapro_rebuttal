import torch

from vectorized_adaptive_allocator_patch import (
    precompute_expected_remaining,
    simulate_process_vectorized,
)


def legacy_expected_remaining(grid, prior_q):
    n, t_max, _ = grid.shape
    rows = []
    for t_curr in range(t_max):
        belief = grid[:, t_curr, t_curr:]
        remaining = torch.clamp(
            prior_q - t_curr + 1,
            min=0,
            max=belief.shape[1],
        ).float()
        mask = (
            torch.arange(belief.shape[1], device=grid.device)[None, :]
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
    total = 0.0

    for t_curr in range(t_max):
        alive = (true_t >= t_curr) & (prior_q >= t_curr)
        active = active & alive
        pi = torch.rsqrt(
            lam * expected_remaining[:, t_curr] + 1e-12
        ).clamp(max=1.0)
        cumulative_all *= torch.where(alive, pi, torch.ones_like(pi))
        keep = (uniforms[:, t_curr] <= pi) & active
        total += keep.sum().item()
        sim_C[keep] += 1
        active &= keep

    succeeded = (sim_C > prior_q) | (sim_C > true_t)
    if reach_t_max_is_success:
        succeeded |= sim_C == t_max
    sim_C = torch.where(succeeded, prior_q + 1, sim_C)
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
    torch.testing.assert_close(actual_prob, expected_prob, rtol=1e-6, atol=1e-7)
    assert actual_cost == expected_cost


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

    pi = torch.rsqrt(lam * expected_remaining + 1e-12).clamp(max=1.0)
    time = torch.arange(t_max)[None, :]
    endpoint_mask = (
        (time <= true_t[:, None])
        & (time <= prior_q[:, None])
    )
    expected_probability = torch.where(
        endpoint_mask, pi, torch.ones_like(pi)
    ).prod(dim=1)
    torch.testing.assert_close(probability_a, expected_probability)


if __name__ == "__main__":
    test_vectorized_matches_legacy_logic()
    test_terminal_probability_is_early_endpoint_propensity()
    print("All vectorized simulation tests passed.")
