import numpy as np
import torch

from src.predictive_bounds.budget_allocators.budget_allocator import (
    BudgetAllocationResult,
    BudgetAllocator,
)
from src.predictive_bounds.calibration.calibration_utils import (
    indexed_tensor_metrics,
)
from src.predictive_bounds.calibration.survival_calibration_with_known_weights import (
    SurvivalCalibrationWithKnownWeights,
)
from src.predictive_bounds.calibration.survival_upb_calibration_with_known_weights import (
    SurvivalUPBCalibrationWithKnownWeights,
)
from src.train_model.models.utils import SurvivalModelPrediction


class DummyAllocator(BudgetAllocator):
    @property
    def name(self):
        return "dummy"

    def allocate_budget(self, probability_est, x, t, quantile_est):
        raise NotImplementedError


class FixedAllocator(DummyAllocator):
    def __init__(self, result, taus):
        super().__init__(1.0, taus, tau_prior=float(taus[-1]))
        self.result = result

    def allocate_budget(self, probability_est, x, t, quantile_est):
        return self.result


def test_indexed_tensor_metrics_preserves_mixed_float_values():
    metrics = indexed_tensor_metrics({
        "float32": torch.tensor([0.1, 3.0], dtype=torch.float32),
        "float64": torch.tensor([1 / 3, 7.0], dtype=torch.float64),
    })

    assert metrics == {
        "float32_0": torch.tensor(0.1, dtype=torch.float32).item(),
        "float32_1": 3.0,
        "float64_0": 1 / 3,
        "float64_1": 7.0,
    }


def test_broadcast_calibration_weights_match_legacy_repeat_exactly():
    taus = torch.tensor([0.1, 0.2, 0.3])
    f = torch.tensor([
        [2.0, 3.0, 200.0],
        [1.0, 4.0, 5.0],
        [3.0, 6.0, 7.0],
        [2.0, 5.0, 8.0],
    ])
    t = torch.tensor([1.0, 5.0, 4.0, 9.0])
    c = torch.tensor([3.0, 4.0, 8.0, 5.0])
    probabilities = torch.tensor([0.5, 0.25, 1.0, 0.8])
    result = BudgetAllocationResult(
        f=f,
        C=c,
        C_probs=probabilities,
        total_budget_used=20,
        candidate_C_probs=probabilities[:, None].expand_as(f),
    )
    prediction = SurvivalModelPrediction(
        quantile_est=f,
        # Unit hazard gives zero model survivor probability, so augmented HT
        # reduces exactly to the historical unaugmented contribution tested
        # below.
        probability_est=torch.ones(4, 1),
    )

    lpb = SurvivalCalibrationWithKnownWeights(
        FixedAllocator(result, taus),
        taus,
        tau_prior=0.3,
    )
    lpb.calibrate(None, t, prediction)
    legacy_lpb = (1 / probabilities[:, None]).repeat(1, f.shape[1])
    legacy_lpb[t[:, None] >= f] = 0
    legacy_lpb[f > c[:, None]] = 0
    assert torch.equal(lpb.miscoverage, legacy_lpb.mean(dim=0))

    upb = SurvivalUPBCalibrationWithKnownWeights(
        FixedAllocator(result, taus),
        taus,
        tau_prior=0.3,
    )
    upb.calibrate(None, t, prediction)
    legacy_upb = (1 / probabilities[:, None]).repeat(1, f.shape[1])
    legacy_upb[t[:, None] <= f] = 0
    legacy_upb[f > c[:, None]] = 0
    legacy_upb[f == 200] = 0
    torch.testing.assert_close(
        upb.miscoverage,
        legacy_upb.mean(dim=0).to(torch.float64),
    )


def test_requested_inverse_probability_metrics_are_exact():
    taus = torch.tensor([0.1, 0.2, 0.3])
    allocator = DummyAllocator(1.0, taus, tau_prior=0.3)
    calibration = SurvivalCalibrationWithKnownWeights(
        allocator,
        taus,
        tau_prior=0.3,
    )
    f = torch.tensor(
        [
            [2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 4.0],
        ]
    )
    calibration.t_cal = torch.tensor([1.0, 2.0, 4.0])
    calibration.miscoverage = torch.tensor([0.0, 0.15, 0.25])
    calibration.allocation_result = BudgetAllocationResult(
        f=f,
        C=torch.tensor([4.0, 1.0, 4.0]),
        C_probs=torch.tensor([0.5, 0.25, 1.0]),
        total_budget_used=9,
    )
    prediction = SurvivalModelPrediction(
        quantile_est=f,
        probability_est=torch.zeros(3, 1),
    )

    metrics = calibration.compute_metrics(
        prediction,
        torch.tensor([0.1, 0.2]),
    )

    np.testing.assert_allclose(
        metrics["mean_inverse_probability_minus_one"],
        4 / 3,
    )
    np.testing.assert_allclose(
        metrics["mean_weight"] - 1,
        metrics["mean_inverse_probability_minus_one"],
    )
    np.testing.assert_allclose(metrics["variance_weight"], 14 / 9)
    np.testing.assert_allclose(
        metrics["mean_prior_a_weighted_inverse_probability_minus_one"],
        4 / 3,
    )
    np.testing.assert_allclose(
        metrics["mean_prior_a_weighted_inverse_probability"],
        2.0,
    )
    np.testing.assert_allclose(
        metrics["mean_tau_0p10_a_weighted_inverse_probability"],
        2 / 3,
    )
    np.testing.assert_allclose(
        metrics["mean_calibrated_a_weighted_inverse_probability_0"],
        metrics["mean_a_weighted_inverse_probability_0"],
    )
    assert metrics["budget_used"] == 9
    assert metrics["actual_event_stopped_budget_total"] == 6
    assert metrics["reported_budget_semantics"] == "sum_assigned_C_i"
    np.testing.assert_allclose(
        metrics["mean_a_weighted_inverse_probability_minus_one_0"],
        1 / 3,
    )
    np.testing.assert_allclose(
        metrics["mean_a_weighted_inverse_probability_0"],
        2 / 3,
    )
    np.testing.assert_allclose(
        metrics["variance_a_weighted_inverse_probability_0"],
        8 / 9,
    )
    np.testing.assert_allclose(
        metrics["conditional_variance_of_ht_mean_0"],
        1 / 9,
    )
    np.testing.assert_allclose(
        metrics["mean_a_weighted_inverse_probability_minus_one_1"],
        4 / 3,
    )
    np.testing.assert_allclose(
        metrics["mean_a_weighted_inverse_probability_1"],
        2.0,
    )
    np.testing.assert_allclose(
        metrics["variance_a_weighted_inverse_probability_1"],
        8 / 3,
    )


def test_conditional_variance_identity_by_monte_carlo():
    generator = torch.Generator().manual_seed(1234)
    probability = 0.2
    draws = (
        torch.rand(400_000, generator=generator) < probability
    ).to(torch.float64)
    v = draws / probability

    np.testing.assert_allclose(
        v.var(unbiased=False).item(),
        1 / probability - 1,
        rtol=0.015,
    )


def test_literal_selected_a_is_separate_from_estimability_gate():
    taus = torch.tensor([0.1, 0.2])
    allocator = DummyAllocator(1.0, taus, tau_prior=0.2)
    calibration = SurvivalCalibrationWithKnownWeights(
        allocator,
        taus,
        tau_prior=0.2,
    )
    # Row-specific selected bounds make an erroneous (N, 1) versus (N,)
    # broadcast observably different from the intended elementwise comparison.
    f = torch.tensor([[5.0, 3.0], [1.0, 3.0]])
    calibration.t_cal = torch.tensor([2.0, 6.0])
    calibration.miscoverage = torch.tensor([0.0, 0.3])
    calibration.allocation_result = BudgetAllocationResult(
        f=f,
        C=torch.tensor([3.0, 3.0]),
        C_probs=torch.tensor([0.5, 0.5]),
        total_budget_used=6,
    )
    prediction = SurvivalModelPrediction(
        quantile_est=f,
        probability_est=torch.zeros(2, 1),
    )

    metrics = calibration.compute_metrics(
        prediction,
        torch.tensor([0.1]),
    )

    np.testing.assert_allclose(
        metrics["mean_a_weighted_inverse_probability_minus_one_0"],
        0.5,
    )
    np.testing.assert_allclose(
        metrics[
            "mean_estimable_a_weighted_inverse_probability_minus_one_0"
        ],
        0.0,
    )
    assert metrics["all_observed_jailbreaks_0"] == 1
    assert metrics["all_f_lower_c_0"] == 1
    assert metrics["all_observed_both_0"] == 0


def test_upb_selected_a_weight_metrics_use_the_upper_tail_event():
    taus = torch.tensor([0.5, 0.7, 0.9])
    allocator = DummyAllocator(1.0, taus, tau_prior=0.9)
    calibration = SurvivalUPBCalibrationWithKnownWeights(
        allocator,
        taus,
        tau_prior=0.9,
    )
    f = torch.tensor([
        [2.0, 3.0, 4.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0],
    ])
    calibration.t_cal = torch.tensor([1.0, 5.0, 4.0])
    calibration.miscoverage = torch.tensor([0.4, 0.2, 0.1])
    calibration.coverage = 1.0 - calibration.miscoverage
    calibration.model_miscoverage = torch.zeros_like(f, dtype=torch.float64)
    calibration.allocation_result = BudgetAllocationResult(
        f=f,
        C=torch.tensor([4.0, 4.0, 4.0]),
        C_probs=torch.tensor([0.5, 0.25, 1.0]),
        total_budget_used=12,
        candidate_C_probs=torch.tensor(
            [[0.5], [0.25], [1.0]], dtype=torch.float64
        ).expand_as(f),
    )
    prediction = SurvivalModelPrediction(
        quantile_est=f,
        probability_est=torch.zeros(3, 1),
    )

    metrics = calibration.compute_metrics(
        prediction,
        torch.tensor([0.7]),
    )

    np.testing.assert_allclose(metrics["alpha_hat_per_tau_0"], 0.2)
    np.testing.assert_allclose(
        metrics["mean_a_weighted_inverse_probability_minus_one_0"],
        1.0,
    )
    np.testing.assert_allclose(
        metrics["mean_a_weighted_inverse_probability_0"],
        4 / 3,
    )
    np.testing.assert_allclose(
        metrics["mean_prior_a_weighted_inverse_probability"],
        4 / 3,
    )
    np.testing.assert_allclose(
        metrics["mean_tau_0p10_a_weighted_inverse_probability"],
        5 / 3,
    )
    np.testing.assert_allclose(
        metrics["mean_calibrated_a_weighted_inverse_probability_0"],
        4 / 3,
    )
    assert metrics["budget_used"] == 12
    assert metrics["actual_event_stopped_budget_total"] == 9
    np.testing.assert_allclose(
        metrics["variance_a_weighted_inverse_probability_0"],
        32 / 9,
    )
    assert metrics["all_observed_jailbreaks_0"] == 1
    assert metrics["all_observed_both_0"] == 1
    assert metrics["upb_calibration_estimator"] == (
        "terminal_residual_augmented_ht"
    )
