import numpy as np
import torch

from src.predictive_bounds.budget_allocators.DAPRO import (
    InformationGainUPBDAPRO,
    SoftPrefixEndpointUPBDAPRO,
)
from src.predictive_bounds.calibration.survival_upb_calibration_with_known_weights import (
    SurvivalUPBCalibrationWithKnownWeights,
)


def _toy_grid(n: int, width: int = 2) -> torch.Tensor:
    grid = torch.zeros(n, width, width + 1, dtype=torch.float64)
    grid[:, 0, :] = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    grid[:, 1, :] = torch.tensor([0.0, 0.25, 0.75], dtype=torch.float64)
    return grid


def test_sequential_upb_aht_is_exactly_design_unbiased():
    # Acquisition states: stop immediately, observe turn one only, observe both.
    grid = _toy_grid(3)
    times = torch.full((3,), 201.0)
    candidates = torch.tensor(
        [[1.0, 2.0, 201.0]] * 3, dtype=torch.float64
    )
    horizons = torch.tensor([0.0, 1.0, 2.0])
    conditionals = torch.tensor(
        [[0.5, 0.4]] * 3, dtype=torch.float64
    )
    contributions = (
        SurvivalUPBCalibrationWithKnownWeights
        ._sequential_augmented_miscoverage_contributions(
            times, candidates, horizons, conditionals, grid
        )
    )
    state_probabilities = torch.tensor([0.5, 0.3, 0.2])
    expectation = (state_probabilities[:, None] * contributions).sum(dim=0)
    torch.testing.assert_close(
        expectation,
        torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64),
    )
    exact = (
        SurvivalUPBCalibrationWithKnownWeights
        ._sequential_augmented_path_variance(
            times[:1], candidates[:1, :2], conditionals[:1], grid[:1]
        )
    )
    empirical = (
        state_probabilities[:, None]
        * (contributions[:, :2] - expectation[:2]).square()
    ).sum(dim=0)
    torch.testing.assert_close(exact[0], empirical)


def test_block_path_sequential_aht_telescopes_to_terminal_residual():
    grid = _toy_grid(2)
    times = torch.full((2,), 201.0)
    candidates = torch.tensor([[2.0], [2.0]])
    horizons = torch.tensor([0.0, 2.0])
    conditionals = torch.tensor([[0.2, 1.0], [0.2, 1.0]])
    sequential = (
        SurvivalUPBCalibrationWithKnownWeights
        ._sequential_augmented_miscoverage_contributions(
            times, candidates, horizons, conditionals, grid
        )
    )
    model = torch.full((2, 1), 0.5, dtype=torch.float64)
    terminal = (
        SurvivalUPBCalibrationWithKnownWeights
        ._augmented_miscoverage_contributions(
            times,
            candidates,
            horizons,
            torch.full((2, 1), 0.2, dtype=torch.float64),
            torch.full((2,), 0.2, dtype=torch.float64),
            model,
        )
    )
    torch.testing.assert_close(sequential, terminal)


def test_history_value_score_is_prefix_causal():
    n, width = 6, 3
    grid = torch.zeros(n, width, width + 1, dtype=torch.float64)
    grid[:, 0, :] = torch.tensor([0.1, 0.2, 0.3, 0.4])
    grid[:, 1, :] = torch.tensor([0.0, 0.2, 0.3, 0.5])
    grid[:, 2, :] = torch.tensor([0.0, 0.0, 0.4, 0.6])
    taus = torch.tensor([0.5, 0.7, 0.9])
    quantiles = torch.tensor([[1.0, 2.0, 3.0]] * n)
    times = torch.full((n,), 201.0)
    allocator = InformationGainUPBDAPRO(
        grid, 2.0, taus, 0.9, width, n1=2, target_coverage=0.7
    )
    allocator._select_target_anchor(times, quantiles)
    before = allocator.policy_scores(quantiles)
    allocator.conditional_grid[:, 1:, :] = torch.rand_like(
        allocator.conditional_grid[:, 1:, :]
    )
    after = allocator.policy_scores(quantiles)
    torch.testing.assert_close(before[:, 0], after[:, 0])


def test_dynamic_upb_allocators_log_full_path_propensities_and_budget():
    n, width = 30, 4
    grid = torch.zeros(n, width, width + 1, dtype=torch.float64)
    for row in range(n):
        for step in range(width):
            hazard = 0.1 + 0.05 * ((row + step) % 3)
            grid[row, step, step] = hazard
            grid[row, step, -1] = 1.0 - hazard
    quantiles = torch.tensor([[2.0, 3.0, 201.0]] * n)
    times = torch.tensor(
        [1 + row % 4 if row % 5 else 201 for row in range(n)],
        dtype=torch.float64,
    )
    taus = torch.tensor([0.5, 0.7, 0.9])
    common_uniforms = torch.linspace(0.01, 0.99, n * width).reshape(n, width)
    for allocator_type in (SoftPrefixEndpointUPBDAPRO, InformationGainUPBDAPRO):
        np.random.seed(0)
        allocator = allocator_type(
            grid.clone(),
            2.5,
            taus,
            0.9,
            width,
            n1=8,
            target_coverage=0.7,
            projection_budget_margin=0.1,
        )
        allocator.acquisition_uniforms = common_uniforms
        result = allocator.allocate_budget(grid, None, times, quantiles)
        assert result.continuation_probabilities.shape == (n, width)
        assert torch.all(result.continuation_probabilities > 0)
        assert torch.all(result.continuation_probabilities <= 1)
        cumulative = result.continuation_probabilities.cumprod(dim=1)
        finite = quantiles < 201
        indices = quantiles.long().clamp(min=1, max=width) - 1
        expected_candidate_pi = cumulative.gather(1, indices)
        torch.testing.assert_close(
            result.candidate_C_probs[finite], expected_candidate_pi[finite]
        )
        assert result.additional_metrics["total_expected_budget_per_sample"] <= 2.5 + 1e-7


def test_soft_prefix_upb_uses_the_sequential_aht_estimator_contract():
    allocator = SoftPrefixEndpointUPBDAPRO(
        _toy_grid(12),
        1.5,
        torch.tensor([0.5, 0.7, 0.9]),
        0.9,
        2,
        n1=4,
        target_coverage=0.7,
        projection_budget_margin=0.0,
    )

    assert allocator.upb_estimator_kind == "sequential"
    assert "seq_estimator_v3_hazard_score" in allocator.name
