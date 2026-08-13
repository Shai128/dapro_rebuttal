import numpy as np
import torch

from src.predictive_bounds.budget_allocators.full_budget_oracle_allocator import (
    FullBudgetOracleAllocator,
    SplitFullBudgetOracleAllocator,
)
from src.predictive_bounds.budget_allocators.metric_optimal_allocator import (
    MetricOptimalPMFAllocator,
    MetricOptimalPMFCRCAllocator,
    MetricPrefixNeymanCRCAllocator,
    MetricOptimalPooledTimeAllocator,
    MetricOptimalPooledTimeCRCAllocator,
    antitonic_pav_bases,
    cumulative_paths_for_scale,
    cumulative_to_conditionals,
    initial_event_and_at_risk_probabilities,
    prefix_remaining_event_cost_index,
    row_horizon_cap_scales,
    solve_common_scale,
)
from src.predictive_bounds.budget_allocators.optimized_allocators import (
    OptimizedBudgetAllocator,
)
from src.predictive_bounds.budget_allocators.uniform_allocator import (
    UniformBudgetAllocator,
)


def _probability_tensor(n=40, width=4):
    probability = torch.zeros(n, width, width + 1, dtype=torch.float64)
    row = torch.arange(n, dtype=torch.float64)
    first = 0.05 + 0.20 * (row.remainder(5) / 4)
    second = 0.05 + 0.10 * (row.remainder(3) / 2)
    third = torch.full((n,), 0.08, dtype=torch.float64)
    fourth = torch.full((n,), 0.07, dtype=torch.float64)
    event = torch.stack([first, second, third, fourth], dim=1)
    probability[:, 0, :width] = event
    probability[:, 0, width] = 1 - event.sum(dim=1)
    # These rows represent predictions after future histories.  A causal
    # pre-run allocator must be invariant to all of them.
    probability[:, 1:, :] = 1 / (width + 1)
    return probability


def test_initial_pmf_uses_explicit_tail_and_only_current_time_zero():
    probability = _probability_tensor(n=3, width=4)
    event, at_risk = initial_event_and_at_risk_probabilities(probability, 4)

    np.testing.assert_allclose(event, probability[:, 0, :4].numpy())
    np.testing.assert_allclose(at_risk[:, 0], 1.0)
    np.testing.assert_allclose(at_risk[:, 1], 1.0 - event[:, 0])

    changed = probability.clone()
    changed[:, 1:, :] = torch.rand_like(changed[:, 1:, :]) * 100
    changed_event, changed_at_risk = initial_event_and_at_risk_probabilities(
        changed, 4
    )
    np.testing.assert_array_equal(changed_event, event)
    np.testing.assert_array_equal(changed_at_risk, at_risk)

    # Synthetic and legacy callers provide discrete hazards as a 2-D matrix.
    hazard = torch.full((3, 4), 0.1, dtype=torch.float64)
    hazard_event, hazard_at_risk = initial_event_and_at_risk_probabilities(
        hazard, 4
    )
    np.testing.assert_allclose(
        hazard_at_risk[0], [1.0, 0.9, 0.81, 0.729]
    )
    np.testing.assert_allclose(
        hazard_event[0], [0.1, 0.09, 0.081, 0.0729]
    )


def test_infinite_scale_keeps_exact_zero_event_bases_at_the_floor():
    bases = np.array([[0.5, 0.0, 0.0]], dtype=np.float64)
    cumulative = cumulative_paths_for_scale(bases, np.inf, floor=0.1)
    np.testing.assert_array_equal(cumulative, [[1.0, 0.1, 0.1]])
    assert np.isfinite(cumulative).all()


def test_prefix_neyman_index_uses_only_the_current_conditional_future():
    width = 3
    probability = torch.zeros(2, width, width + 1, dtype=torch.float64)
    probability[:, 0, :] = torch.tensor([0.2, 0.3, 0.1, 0.4])
    probability[:, 1, :] = torch.tensor([99.0, 0.3, 0.2, 0.5])
    probability[:, 2, :] = torch.tensor([99.0, 99.0, 0.4, 0.6])

    index = prefix_remaining_event_cost_index(probability, width)

    # At prefix one, stale mass on the impossible time-one outcome is ignored.
    expected_risk = 0.5
    expected_cost = 0.3 * 1 + 0.2 * 2 + 0.5 * 2
    np.testing.assert_allclose(
        index[:, 1],
        np.sqrt(expected_risk / expected_cost),
    )
    changed = probability.clone()
    changed[:, 2, :] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    changed_index = prefix_remaining_event_cost_index(changed, width)
    np.testing.assert_array_equal(changed_index[:, :2], index[:, :2])


def test_antitonic_pav_pools_an_increasing_hazard_block_exactly():
    numerator = np.array([[0.1, 0.9]], dtype=np.float64)
    denominator = np.ones_like(numerator)

    bases, blocks = antitonic_pav_bases(numerator, denominator)

    np.testing.assert_allclose(bases, np.sqrt(0.5))
    assert blocks.tolist() == [1]

    cumulative, _, boundary = solve_common_scale(
        bases,
        denominator,
        budget_per_sample=1.0,
        floor=0.01,
    )
    # With one row and two turns, monotonicity binds and the unit budget is
    # divided equally.  A coarse exhaustive grid has the same unique optimum.
    np.testing.assert_allclose(cumulative, [[0.5, 0.5]], atol=1e-8)
    assert boundary == "budget"
    candidates = []
    for first in np.linspace(0.01, 1.0, 100):
        for second in np.linspace(0.01, first, 100):
            if first + second <= 1.0 + 1e-12:
                candidates.append(0.1 / first + 0.9 / second)
    assert 0.1 / 0.5 + 0.9 / 0.5 <= min(candidates) + 2e-2


def test_cumulative_round_trip_and_row_cap_are_exact():
    bases = np.array([
        [1.0, 0.8, 0.4, 0.2],
        [0.7, 0.7, 0.3, 0.1],
    ])
    floor = 0.05
    cap = 1.6
    scales = row_horizon_cap_scales(bases, cap, floor)
    cumulative = np.clip(scales[:, None] * bases, floor, 1.0)
    assert np.all(np.diff(cumulative, axis=1) <= 1e-12)
    assert np.all(cumulative >= floor)
    np.testing.assert_allclose(cumulative.sum(axis=1), cap, atol=1e-8)

    conditionals = cumulative_to_conditionals(cumulative)
    np.testing.assert_allclose(np.cumprod(conditionals, axis=1), cumulative)


def test_plugin_allocator_is_causal_and_logs_exact_endpoint_propensity():
    n, width = 40, 4
    probability = _probability_tensor(n, width)
    changed = probability.clone()
    changed[:, 1:, :] = torch.rand_like(changed[:, 1:, :]) * 100
    event_times = (torch.arange(n) % (width + 1)) + 1
    quantiles = torch.full((n, 2), width, dtype=torch.float64)
    uniforms = np.random.default_rng(7).random((n, width))
    taus = torch.tensor([0.1, 0.5])

    first = MetricOptimalPMFAllocator(1.5, taus, 0.5, width)
    first.set_acquisition_randomness(seed=7, uniforms=uniforms)
    first_result = first.allocate_budget(
        probability, None, event_times, quantiles
    )
    second = MetricOptimalPMFAllocator(1.5, taus, 0.5, width)
    second.set_acquisition_randomness(seed=7, uniforms=uniforms)
    second_result = second.allocate_budget(
        changed, None, event_times, quantiles
    )

    np.testing.assert_array_equal(
        first.last_cumulative_probabilities,
        second.last_cumulative_probabilities,
    )
    assert torch.equal(first_result.C, second_result.C)
    assert torch.equal(first_result.C_probs, second_result.C_probs)
    lengths = torch.minimum(event_times, torch.tensor(width)).numpy()
    expected_endpoint = first.last_cumulative_probabilities[
        np.arange(n), lengths - 1
    ]
    np.testing.assert_allclose(first_result.C_probs.numpy(), expected_endpoint)
    assert first_result.additional_metrics[
        "metric_policy_uses_future_history"
    ] == 0


def test_plugin_allocator_horvitz_thompson_rate_is_monte_carlo_unbiased():
    n, width = 12000, 4
    base_probability = _probability_tensor(20, width)
    probability = base_probability.repeat((n // 20, 1, 1))
    event_times = (torch.arange(n) % (width + 1)) + 1
    quantiles = torch.full((n, 2), width, dtype=torch.float64)
    uniforms = np.random.default_rng(123).random((n, width))
    allocator = MetricOptimalPMFAllocator(
        1.5,
        torch.tensor([0.1, 0.5]),
        0.5,
        width,
    )
    allocator.set_acquisition_randomness(seed=123, uniforms=uniforms)
    result = allocator.allocate_budget(
        probability, None, event_times, quantiles
    )

    target = (event_times <= width).to(torch.float64)
    observed = (event_times <= result.C.reshape(-1)).to(torch.float64)
    estimate = (target * observed / result.C_probs).mean().item()
    truth = target.mean().item()
    conditional_variance = (
        target * (1 / result.C_probs - 1)
    ).sum().item() / n**2
    assert abs(estimate - truth) <= 5 * np.sqrt(conditional_variance)


def test_pooled_time_allocator_is_shared_causal_and_matches_model_budget():
    n, width = 40, 4
    probability = _probability_tensor(n, width)
    changed = probability.clone()
    changed[:, 1:, :] = torch.rand_like(changed[:, 1:, :]) * 100
    event_times = (torch.arange(n) % (width + 1)) + 1
    quantiles = torch.full((n, 2), width, dtype=torch.float64)
    uniforms = np.random.default_rng(31).random((n, width))
    taus = torch.tensor([0.1, 0.5])

    first = MetricOptimalPooledTimeAllocator(1.5, taus, 0.5, width)
    first.set_acquisition_randomness(seed=31, uniforms=uniforms)
    first_result = first.allocate_budget(
        probability, None, event_times, quantiles
    )
    second = MetricOptimalPooledTimeAllocator(1.5, taus, 0.5, width)
    second.set_acquisition_randomness(seed=31, uniforms=uniforms)
    second_result = second.allocate_budget(
        changed, None, event_times, quantiles
    )

    cumulative = first.last_cumulative_probabilities
    np.testing.assert_allclose(
        cumulative,
        np.broadcast_to(cumulative[[0]], cumulative.shape),
    )
    np.testing.assert_array_equal(
        cumulative,
        second.last_cumulative_probabilities,
    )
    assert torch.equal(first_result.C, second_result.C)
    assert torch.equal(first_result.C_probs, second_result.C_probs)
    assert np.all(np.diff(cumulative, axis=1) <= 1e-12)
    np.testing.assert_allclose(
        first_result.additional_metrics[
            "metric_policy_model_expected_cost_per_sample"
        ],
        1.5,
        atol=1e-8,
    )
    assert first_result.additional_metrics["metric_policy_pooled_across_rows"] == 1
    assert first_result.additional_metrics["metric_policy_uses_future_history"] == 0


def test_pooled_time_crc_has_shared_deployment_path_and_valid_selector():
    n, width = 80, 4
    probability = _probability_tensor(n, width)
    event_times = (torch.arange(n) % (width + 1)) + 1
    quantiles = torch.full((n, 2), width, dtype=torch.float64)
    uniforms = np.random.default_rng(41).random((n, width))
    allocator = MetricOptimalPooledTimeCRCAllocator(
        3.0,
        torch.tensor([0.1, 0.5]),
        0.5,
        width,
        control_size=5,
        candidate_count=101,
    )
    allocator.set_acquisition_randomness(seed=41, uniforms=uniforms)
    result = allocator.allocate_budget(
        probability, None, event_times, quantiles
    )

    cumulative = allocator.last_cumulative_probabilities
    np.testing.assert_array_equal(
        cumulative[allocator.last_control_indices],
        np.ones((5, width)),
    )
    np.testing.assert_allclose(
        cumulative[allocator.last_deployment_indices],
        np.broadcast_to(
            cumulative[[allocator.last_deployment_indices[0]]],
            (len(allocator.last_deployment_indices), width),
        ),
    )
    assert result.additional_metrics["metric_policy_crc_selector_valid"] == 1
    assert result.additional_metrics["metric_policy_budget_control_mode"] == "crc"
    assert result.additional_metrics["metric_policy_crc_row_cost_cap"] == width


def test_prefix_neyman_crc_is_causal_and_does_not_fit_deployment_labels():
    n, width = 80, 4
    probability = _probability_tensor(n, width)
    event_times = (torch.arange(n) % (width + 1)) + 1
    quantiles = torch.full((n, 2), width, dtype=torch.float64)
    uniforms = np.random.default_rng(53).random((n, width))
    kwargs = dict(
        budget_per_sample=3.0,
        taus_range=torch.tensor([0.1, 0.5]),
        tau_prior=0.5,
        m_upper_bound=width,
        control_size=5,
        candidate_count=101,
    )

    first = MetricPrefixNeymanCRCAllocator(probability, **kwargs)
    first.set_acquisition_randomness(seed=53, uniforms=uniforms)
    first.allocate_budget(probability, None, event_times, quantiles)

    changed_grid = probability.clone()
    deployment = first.last_deployment_indices
    changed_grid[deployment, 2:, :] = torch.rand_like(
        changed_grid[deployment, 2:, :]
    )
    changed_times = event_times.clone()
    changed_times[deployment] = torch.where(
        changed_times[deployment] == 1,
        torch.tensor(width + 1),
        torch.tensor(1),
    )
    second = MetricPrefixNeymanCRCAllocator(changed_grid, **kwargs)
    second.set_acquisition_randomness(seed=53, uniforms=uniforms)
    second.allocate_budget(probability, None, changed_times, quantiles)

    assert (
        first.last_selected_candidate_index
        == second.last_selected_candidate_index
    )
    np.testing.assert_array_equal(
        first.last_cumulative_probabilities[deployment, :2],
        second.last_cumulative_probabilities[deployment, :2],
    )
    assert first.last_prefix_index is not None


def test_crc_selection_uses_control_outcomes_but_not_deployment_outcomes():
    n, width = 40, 4
    probability = _probability_tensor(n, width)
    event_times = (torch.arange(n) % (width + 1)) + 1
    quantiles = torch.full((n, 2), width, dtype=torch.float64)
    uniforms = np.random.default_rng(11).random((n, width))
    taus = torch.tensor([0.1, 0.5])

    first = MetricOptimalPMFCRCAllocator(
        2.5,
        taus,
        0.5,
        width,
        control_size=5,
        candidate_count=101,
    )
    first.set_acquisition_randomness(seed=11, uniforms=uniforms)
    np.random.seed(11)
    first.allocate_budget(probability, None, event_times, quantiles)

    changed_times = event_times.clone()
    deployment = first.last_deployment_indices
    changed_times[deployment] = torch.where(
        changed_times[deployment] == 1,
        torch.tensor(width + 1),
        torch.tensor(1),
    )
    second = MetricOptimalPMFCRCAllocator(
        2.5,
        taus,
        0.5,
        width,
        control_size=5,
        candidate_count=101,
    )
    second.set_acquisition_randomness(seed=11, uniforms=uniforms)
    np.random.seed(11)
    second.allocate_budget(probability, None, changed_times, quantiles)

    np.testing.assert_array_equal(
        first.last_control_indices,
        second.last_control_indices,
    )
    assert (
        first.last_selected_candidate_index
        == second.last_selected_candidate_index
    )
    np.testing.assert_array_equal(
        first.last_cumulative_probabilities,
        second.last_cumulative_probabilities,
    )


def test_split_full_budget_oracle_is_distinct_from_fixed_truth_reference():
    taus = torch.tensor([0.1, 0.5])
    fixed = FullBudgetOracleAllocator(taus, 0.5, 4)
    split = SplitFullBudgetOracleAllocator(taus, 0.5, 4)
    assert fixed.uses_full_benchmark
    assert not split.uses_full_benchmark
    assert fixed.name == "oracle_full_budget"
    assert split.name == "oracle_split_full_budget"

    times = torch.tensor([1, 4, 5])
    quantiles = torch.full((3, 2), 4.0)
    result = split.allocate_budget(torch.empty(0), None, times, quantiles)
    assert torch.equal(result.C, torch.tensor([4, 4, 4]))
    assert torch.equal(result.C_probs, torch.ones(3, dtype=torch.float64))
    assert result.additional_metrics["is_split_full_budget_baseline"] == 1
    assert result.additional_metrics["reference_scope"] == "calibration_split"
