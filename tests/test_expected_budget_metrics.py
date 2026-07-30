import numpy as np
import pytest
import torch

from src.safety_evaluation.budget_allocators.basic_allocator import (
    BasicBudgetAllocator,
)
from src.safety_evaluation.budget_allocators.budget_allocator import (
    summarize_expected_budget,
)
from src.safety_evaluation.budget_allocators.DAPRO import (
    DAPRO,
    RandomAnchoredTargetAWeightedDAPRO,
    TargetAWeightedDAPRO,
)
from src.safety_evaluation.budget_allocators.naive_allocator import (
    NaiveBudgetAllocator,
)
from src.safety_evaluation.budget_allocators.optimized_allocators import (
    OptimizedBudgetAllocator,
)
from src.safety_evaluation.budget_allocators.trimmed_allocator import (
    TrimmedBudgetAllocator,
)
from src.safety_evaluation.budget_allocators.uniform_allocator import (
    UniformBudgetAllocator,
)
from src.safety_evaluation.calibration.survival_calibration_with_known_weights import (
    SurvivalCalibrationWithKnownWeights,
)
from src.train_model.models.utils import SurvivalModelPrediction
from src.utils.utils import set_seeds


def test_expected_budget_summary_uses_configured_total():
    metrics = summarize_expected_budget(
        19.5,
        10,
        2.0,
        cost_semantics="unit_test",
    )
    assert metrics["configured_total_budget"] == 20.0
    assert metrics["configured_budget_per_sample"] == 2.0
    assert metrics["total_expected_budget"] == 19.5
    assert metrics["total_expected_budget_per_sample"] == 1.95
    assert metrics["total_expected_budget_gap"] == -0.5
    np.testing.assert_allclose(
        metrics["total_expected_budget_gap_per_sample"],
        -0.05,
    )
    assert metrics["total_expected_budget_valid"] == 1
    assert metrics["expected_budget_cost_semantics"] == "unit_test"


def _static_inputs():
    taus = torch.tensor([0.5])
    quantiles = torch.tensor([[2.0], [4.0], [8.0]])
    event_times = torch.tensor([1.0, 3.0, 7.0])
    probability_est = torch.zeros((3, 2))
    return taus, quantiles, event_times, probability_est


@pytest.mark.parametrize(
    "allocator_factory, executed_horizon",
    [
        (
            lambda taus: BasicBudgetAllocator(2.0, taus, 0.5),
            torch.tensor([2.0, 4.0, 8.0]),
        ),
        (
            lambda taus: TrimmedBudgetAllocator(2.0, taus, 0.5, 5),
            torch.tensor([2.0, 4.0, 5.0]),
        ),
        (
            lambda taus: OptimizedBudgetAllocator(2.0, taus, 0.5, 5),
            torch.tensor([2.0, 4.0, 8.0]),
        ),
    ],
)
def test_static_expected_budget_matches_bernoulli_event_stopping(
        allocator_factory,
        executed_horizon,
):
    taus, quantiles, event_times, probability_est = _static_inputs()
    allocator = allocator_factory(taus)
    set_seeds(4)
    result = allocator.allocate_budget(
        probability_est,
        None,
        event_times,
        quantiles,
    )

    probabilities = result.C_probs.reshape(-1).to(torch.float64)
    expected = (
        probabilities
        * torch.minimum(
            event_times.to(torch.float64),
            executed_horizon.to(torch.float64),
        )
    ).sum().item()
    metrics = result.additional_metrics
    np.testing.assert_allclose(metrics["total_expected_budget"], expected)
    np.testing.assert_allclose(
        metrics["total_expected_budget_per_sample"],
        expected / len(event_times),
    )
    np.testing.assert_allclose(
        metrics["total_expected_budget_gap"],
        expected - 2.0 * len(event_times),
    )
    assert metrics["total_expected_budget_valid"] == int(
        expected <= 2.0 * len(event_times) + 1e-7 * len(event_times)
    )


def test_uniform_expected_budget_accounts_for_early_events():
    taus, quantiles, event_times, probability_est = _static_inputs()
    allocator = UniformBudgetAllocator(2.0, taus, 0.5, 5)
    set_seeds(8)
    result = allocator.allocate_budget(
        probability_est,
        None,
        event_times,
        quantiles,
    )

    expected = 0.4 * (1 + 3 + 5)
    metrics = result.additional_metrics
    np.testing.assert_allclose(metrics["total_expected_budget"], expected)
    np.testing.assert_allclose(
        metrics["static_expected_assigned_horizon_total"],
        0.4 * 5 * 3,
    )
    assert metrics["total_expected_budget_valid"] == 1


def test_naive_metrics_distinguish_assigned_and_event_stopped_cost():
    taus, quantiles, event_times, probability_est = _static_inputs()
    allocator = NaiveBudgetAllocator(2.0, taus, 0.5)
    set_seeds(9)
    result = allocator.allocate_budget(
        probability_est,
        None,
        event_times,
        quantiles,
    )
    probabilities = result.C_probs.reshape(-1).to(torch.float64)
    horizons = quantiles.reshape(-1).to(torch.float64)
    expected_assigned = (probabilities * horizons).sum().item()
    expected_stopped = (
        probabilities
        * torch.minimum(event_times.to(torch.float64), horizons)
    ).sum().item()

    metrics = result.additional_metrics
    np.testing.assert_allclose(
        metrics["total_expected_budget"],
        expected_assigned,
    )
    np.testing.assert_allclose(
        metrics["static_expected_event_stopped_cost_total"],
        expected_stopped,
    )
    assert metrics["expected_budget_cost_semantics"] == (
        "bernoulli_assigned_horizon_matching_legacy_budget_used"
    )


def test_calibration_metrics_propagate_allocator_expected_budget_fields():
    taus, quantiles, event_times, probability_est = _static_inputs()
    allocator = BasicBudgetAllocator(2.0, taus, 0.5)
    calibration = SurvivalCalibrationWithKnownWeights(
        allocator,
        taus,
        0.5,
    )
    prediction = SurvivalModelPrediction(
        quantile_est=quantiles,
        probability_est=probability_est,
    )
    set_seeds(13)
    calibration.calibrate(None, event_times, prediction)
    metrics = calibration.compute_metrics(
        prediction,
        torch.tensor([0.1]),
    )

    assert "total_expected_budget" in metrics
    assert "total_expected_budget_per_sample" in metrics
    assert "total_expected_budget_gap_per_sample" in metrics
    assert "total_expected_budget_valid" in metrics
    np.testing.assert_allclose(
        metrics["total_expected_budget"],
        calibration.allocation_result.additional_metrics[
            "total_expected_budget"
        ],
    )


def _dapro_inputs():
    n, t_max = 140, 4
    generator = torch.Generator().manual_seed(12)
    grid = torch.rand(n, t_max, t_max, generator=generator)
    grid = grid / grid.sum(dim=-1, keepdim=True)
    taus = torch.tensor([0.5])
    quantiles = torch.full((n, 1), 2.0)
    event_times = torch.full((n,), 2.0)
    probability_est = torch.zeros((n, t_max))
    return grid, taus, quantiles, event_times, probability_est


@pytest.mark.parametrize("target_a", [False, True])
def test_all_dapro_families_report_complete_expected_budget_identity(target_a):
    grid, taus, quantiles, event_times, probability_est = _dapro_inputs()
    common_kwargs = dict(
        conditional_grid=grid,
        budget_per_sample=1.8,
        taus_range=taus,
        tau_prior=0.5,
        m_upper_bound=200,
        projection="direct_time",
        score="prob",
        n1=100,
    )
    allocator = (
        TargetAWeightedDAPRO(
            **common_kwargs,
            anchor_kind="raw_alpha",
            target_alpha=0.10,
        )
        if target_a
        else DAPRO(**common_kwargs)
    )
    allocator.set_acquisition_randomness(
        seed=31,
        uniforms=np.random.default_rng(31).random((len(grid), grid.shape[1])),
    )
    set_seeds(44)
    metrics = allocator.allocate_budget(
        probability_est,
        None,
        event_times,
        quantiles,
    ).additional_metrics

    phase1_total = metrics["phase1_expected_cost_total"]
    phase2_total = metrics["phase2_expected_cost_total"]
    total = phase1_total + phase2_total
    n1 = metrics["phase1_sample_count"]
    n2 = metrics["phase2_sample_count"]
    n = n1 + n2
    np.testing.assert_allclose(
        phase2_total,
        metrics["phase2_expected_cost_per_sample"] * n2,
    )
    np.testing.assert_allclose(metrics["total_expected_budget"], total)
    np.testing.assert_allclose(
        metrics["total_expected_budget_per_sample"],
        total / n,
    )
    np.testing.assert_allclose(
        metrics["total_expected_budget_gap"],
        total - 1.8 * n,
    )
    np.testing.assert_allclose(
        metrics["total_expected_budget_gap_per_sample"],
        total / n - 1.8,
    )
    assert metrics["expected_budget_cost_semantics"] == (
        "phase1_fully_observed_plus_phase2_expected_interactions"
    )


def test_random_anchored_dapro_reports_valid_independent_crc_selector():
    grid, taus, quantiles, event_times, probability_est = _dapro_inputs()
    allocator = RandomAnchoredTargetAWeightedDAPRO(
        conditional_grid=grid,
        budget_per_sample=1.8,
        taus_range=taus,
        tau_prior=0.5,
        m_upper_bound=200,
        projection="direct_time",
        score="prob",
        n1=120,
        target_alpha=0.10,
        target_policy_fraction=0.50,
        budget_control_mode="crc",
        budget_control_size=20,
    )
    allocator.set_acquisition_randomness(
        seed=31,
        uniforms=np.random.default_rng(31).random(
            (len(grid), grid.shape[1])
        ),
    )
    set_seeds(44)
    metrics = allocator.allocate_budget(
        probability_est,
        None,
        event_times,
        quantiles,
    ).additional_metrics

    assert metrics["risk_budget_selector_valid"] == 1
    assert metrics["risk_budget_control_size"] == 20
    assert metrics["risk_budget_policy_fit_size"] == 100
    assert metrics["risk_budget_guarantee_kind"] == (
        "crc_marginal_expected_total_budget"
    )
    assert (
        "risk_budget_crc_worst_case_control_feasible_conditional_on_fit"
        in metrics
    )
    assert metrics["risk_budget_selector_left_side_per_sample"] <= (
        metrics["risk_budget_deployment_target_per_sample"] + 1e-12
    )
    total_count = (
        metrics["phase1_sample_count"] + metrics["phase2_sample_count"]
    )
    expected_shape_target = (
        1.8 * total_count
        - metrics["risk_budget_policy_fit_realized_cost_total"]
    ) / (
        total_count - metrics["risk_budget_policy_fit_size"]
    )
    np.testing.assert_allclose(
        metrics["risk_budget_policy_shape_target_per_sample"],
        expected_shape_target,
    )
    assert metrics["random_anchor_target_fraction"] == 0.5


def test_random_slack_fill_uses_available_fit_budget_without_lowering_reach():
    grid, taus, quantiles, event_times, probability_est = _dapro_inputs()
    allocator = RandomAnchoredTargetAWeightedDAPRO(
        conditional_grid=grid,
        budget_per_sample=1.8,
        taus_range=taus,
        tau_prior=0.5,
        m_upper_bound=200,
        projection="direct_time",
        score="prob",
        n1=120,
        target_alpha=0.10,
        target_policy_fraction=0.75,
        fill_random_slack=True,
        budget_control_mode="crc",
        budget_control_size=20,
    )
    allocator.set_acquisition_randomness(
        seed=31,
        uniforms=np.random.default_rng(31).random(
            (len(grid), grid.shape[1])
        ),
    )
    set_seeds(44)
    metrics = allocator.allocate_budget(
        probability_est,
        None,
        event_times,
        quantiles,
    ).additional_metrics

    assert metrics["random_anchor_slack_fill_enabled"] == 1
    assert (
        metrics["random_anchor_post_fill_expected_cost"]
        >= metrics["random_anchor_pre_fill_expected_cost"] - 1e-12
    )
    expected_post_fill = min(
        metrics["random_anchor_fit_budget_target"],
        metrics["random_anchor_upper_envelope_expected_cost"],
    )
    np.testing.assert_allclose(
        metrics["random_anchor_post_fill_expected_cost"],
        expected_post_fill,
        atol=1e-9,
    )


def test_random_anchored_base_policy_does_not_use_control_fold_outcomes():
    grid, taus, quantiles, event_times, probability_est = _dapro_inputs()
    n1 = 120
    policy_fit_size = 100
    permutation = np.random.RandomState(44).permutation(len(grid))
    changed_event_times = event_times.clone()
    control_indices = permutation[policy_fit_size:n1]
    changed_event_times[control_indices] = torch.clamp(
        changed_event_times[control_indices] + 1,
        max=grid.shape[1],
    )

    def run(current_event_times):
        allocator = RandomAnchoredTargetAWeightedDAPRO(
            conditional_grid=grid,
            budget_per_sample=1.8,
            taus_range=taus,
            tau_prior=0.5,
            m_upper_bound=200,
            projection="direct_time",
            score="prob",
            n1=n1,
            target_alpha=0.10,
            target_policy_fraction=0.50,
            fill_random_slack=True,
            budget_control_mode="crc",
            budget_control_size=n1 - policy_fit_size,
        )
        allocator.set_acquisition_randomness(
            seed=31,
            uniforms=np.random.default_rng(31).random(
                (len(grid), grid.shape[1])
            ),
        )
        set_seeds(44)
        return allocator.allocate_budget(
            probability_est,
            None,
            current_event_times,
            quantiles,
        ).additional_metrics

    original = run(event_times)
    perturbed = run(changed_event_times)
    frozen_policy_metrics = [
        "risk_budget_policy_shape_target_per_sample",
        "direct_time_base_policy_fit_expected_cost",
        "random_anchor_constant_probability",
        "random_anchor_reference_expected_cost",
        "random_anchor_pre_fill_expected_cost",
        "random_anchor_post_fill_expected_cost",
    ]
    for metric in frozen_policy_metrics:
        np.testing.assert_allclose(original[metric], perturbed[metric])
