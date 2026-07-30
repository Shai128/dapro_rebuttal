import numpy as np
import pytest
import torch

from src.safety_evaluation.budget_allocators.adaptive_optimized_allocator import (
    AdaptiveOptimizedBudgetAllocator,
)
from src.safety_evaluation.budget_allocators.budget_allocator import (
    BudgetAllocator,
)
from src.safety_evaluation.budget_allocators.DAPRO import DAPRO
from src.safety_evaluation.budget_allocators.projected_optimization_utils import (
    adaptive_budget_allocation,
)
from src.safety_evaluation.budget_allocators.random_adaptive_optimized_allocator import (
    RandomAdaptiveOptimizedBudgetAllocator,
)
from src.safety_evaluation.construct_calibrated_bound import (
    _make_common_acquisition_uniforms,
)
from src.utils.utils import set_seeds


class _UniformProbeAllocator(BudgetAllocator):
    @property
    def name(self) -> str:
        return "uniform_probe"

    def allocate_budget(self, probability_est, x, t, quantile_est):
        raise NotImplementedError


def test_population_crns_are_mapped_by_original_sample_index():
    full = _make_common_acquisition_uniforms(17, 12, 5)
    first_indices = np.array([8, 1, 10, 4])
    second_indices = np.array([4, 11, 8])

    first = _make_common_acquisition_uniforms(
        17,
        12,
        5,
        selected_indices=first_indices,
    )
    second = _make_common_acquisition_uniforms(
        17,
        12,
        5,
        selected_indices=second_indices,
    )

    np.testing.assert_array_equal(first, full[first_indices])
    np.testing.assert_array_equal(second, full[second_indices])
    np.testing.assert_array_equal(first[0], second[2])
    np.testing.assert_array_equal(first[3], second[0])


def test_allocator_crns_do_not_depend_on_global_rng_or_device():
    source = np.random.default_rng(91).random((9, 7))
    allocator = _UniformProbeAllocator(
        budget_per_sample=2,
        taus_range=torch.tensor([0.5]),
        tau_prior=0.5,
    )
    allocator.set_acquisition_randomness(seed=91, uniforms=source)

    torch.manual_seed(1)
    _ = torch.rand(100)
    on_cpu = allocator.get_acquisition_uniforms(
        9,
        7,
        device="cpu",
        dtype=torch.float64,
    )
    torch.manual_seed(999)
    _ = torch.rand(31)
    repeated = allocator.get_acquisition_uniforms(
        9,
        7,
        device="cpu",
        dtype=torch.float64,
    )
    assert torch.equal(on_cpu, repeated)

    # Installing a table makes a private copy.
    source[0, 0] = 0.0
    assert on_cpu[0, 0].item() != 0.0

    if torch.cuda.is_available():
        on_cuda = allocator.get_acquisition_uniforms(
            9,
            7,
            device="cuda:0",
            dtype=torch.float64,
        )
        assert torch.equal(on_cpu, on_cuda.cpu())


def test_dapro_sampler_is_rng_and_device_invariant_with_explicit_crns():
    probabilities = torch.tensor(
        [
            [0.8, 0.7, 0.6, 0.5],
            [0.2, 0.9, 0.9, 0.9],
            [1.0, 0.4, 0.4, 0.4],
            [0.6, 0.6, 0.6, 0.6],
        ],
        dtype=torch.float64,
    )
    prior_q = torch.tensor([4, 3, 2, 4])
    event_times = torch.tensor([4, 4, 3, 2])
    uniforms = torch.as_tensor(
        np.random.default_rng(7).random((4, 4)),
        dtype=torch.float64,
    )

    torch.manual_seed(3)
    first = adaptive_budget_allocation(
        probabilities,
        prior_q,
        event_times,
        4,
        "cpu",
        uniforms=uniforms,
    )
    torch.manual_seed(888)
    _ = torch.rand(200)
    second = adaptive_budget_allocation(
        probabilities,
        prior_q,
        event_times,
        4,
        "cpu",
        uniforms=uniforms,
    )
    assert torch.equal(first[0], second[0])
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])

    if torch.cuda.is_available():
        cuda_result = adaptive_budget_allocation(
            probabilities.cuda(),
            prior_q.cuda(),
            event_times.cuda(),
            4,
            "cuda:0",
            uniforms=uniforms.cuda(),
        )
        assert torch.equal(first[0], cuda_result[0].cpu())
        assert first[1] == cuda_result[1]
        assert torch.equal(first[2], cuda_result[2].cpu())


def _allocation_inputs():
    n, t_max = 140, 4
    generator = torch.Generator().manual_seed(22)
    grid = torch.rand(
        n,
        t_max,
        t_max,
        generator=generator,
        dtype=torch.float32,
    )
    grid = grid / grid.sum(dim=-1, keepdim=True)
    taus = torch.tensor([0.5], dtype=torch.float32)
    quantiles = torch.full((n, 1), 2.0, dtype=torch.float32)
    event_times = torch.full((n,), 2.0, dtype=torch.float32)
    probability_est = torch.zeros((n, t_max), dtype=torch.float32)
    uniforms = _make_common_acquisition_uniforms(31, n, t_max)
    return grid, taus, quantiles, event_times, probability_est, uniforms


def _run_dynamic_allocators_in_order(order):
    grid, taus, quantiles, event_times, probability_est, uniforms = (
        _allocation_inputs()
    )
    results = {}
    for name in order:
        if name == "dapro":
            allocator = DAPRO(
                grid,
                1.8,
                taus,
                0.5,
                200,
                projection="direct_time",
                score="prob",
                n1=100,
            )
        else:
            allocator_type = {
                "local": AdaptiveOptimizedBudgetAllocator,
                "random": RandomAdaptiveOptimizedBudgetAllocator,
            }[name]
            allocator = allocator_type(
                grid,
                1.8,
                taus,
                0.5,
                200,
            )
        allocator.set_acquisition_randomness(seed=31, uniforms=uniforms)
        # This mirrors run_one_experiment: policy RNGs restart for every
        # method, while acquisition draws come from their independent table.
        set_seeds(44)
        result = allocator.allocate_budget(
            probability_est,
            None,
            event_times,
            quantiles,
        )
        results[name] = (
            result.C.detach().cpu().clone(),
            result.C_probs.detach().cpu().clone(),
            result.total_budget_used,
        )
        # Deliberately perturb both global streams between methods.
        np.random.random(73)
        torch.rand(73)
    return results


def test_dynamic_allocations_are_method_order_invariant():
    names = ["local", "random", "dapro"]
    forward = _run_dynamic_allocators_in_order(names)
    reverse = _run_dynamic_allocators_in_order(list(reversed(names)))
    for name in names:
        assert torch.equal(forward[name][0], reverse[name][0])
        assert torch.equal(forward[name][1], reverse[name][1])
        assert forward[name][2] == reverse[name][2]


def test_acquisition_table_shape_is_checked():
    allocator = _UniformProbeAllocator(
        budget_per_sample=2,
        taus_range=torch.tensor([0.5]),
        tau_prior=0.5,
    )
    allocator.set_acquisition_randomness(
        seed=2,
        uniforms=np.zeros((3, 4), dtype=np.float64),
    )
    with pytest.raises(ValueError, match="row count"):
        allocator.get_acquisition_uniforms(
            4,
            4,
            device="cpu",
            dtype=torch.float32,
        )
    with pytest.raises(ValueError, match="too narrow"):
        allocator.get_acquisition_uniforms(
            3,
            5,
            device="cpu",
            dtype=torch.float32,
        )
