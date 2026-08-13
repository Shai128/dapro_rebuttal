import torch

from src.predictive_bounds.budget_allocators.DAPRO import (
    SoftTargetCRCUPBDAPRO,
    SoftTargetUPBDAPRO,
)
from src.predictive_bounds.budget_allocators.budget_allocator import (
    BudgetAllocationResult,
    BudgetAllocator,
)
from src.predictive_bounds.calibration.calibration_utils import (
    quantiles_to_interaction_counts,
    select_upb_calibration_positions,
)
from src.predictive_bounds.calibration.survival_upb_calibration_with_known_weights import (
    SurvivalUPBCalibrationWithKnownWeights,
)
from src.predictive_bounds.construct_calibrated_bound import (
    compute_metrics_bound,
)
from src.predictive_bounds.utils.get_calibration_methods_utils import (
    get_upb_calibrations,
)
from src.train_model.models.utils import SurvivalModelPrediction


class _FixedUPBAllocator(BudgetAllocator):
    def __init__(self, f, C, pi, taus):
        super().__init__(1.0, taus, 0.9)
        self._f = f
        self._C = C
        self._pi = pi
        self.seen_candidates = None

    def allocate_budget(self, probability_est, x, t, quantile_est):
        self.seen_candidates = quantile_est.clone()
        return BudgetAllocationResult(
            self._f.clone(), self._C.clone(), self._pi.clone(),
            total_budget_used=int(self._C.sum().item()),
        )

    @property
    def name(self):
        return "fixed_upb_test"


def test_upb_quantile_conversion_preserves_201_only_when_requested():
    raw = torch.tensor([198.0, 199.0, 200.0])
    torch.testing.assert_close(
        quantiles_to_interaction_counts(raw, width=200),
        torch.tensor([199.0, 200.0, 200.0]),
    )
    torch.testing.assert_close(
        quantiles_to_interaction_counts(
            raw, width=200, upper_bound=201,
            allow_no_event_sentinel=True,
        ),
        torch.tensor([199.0, 200.0, 201.0]),
    )


def test_upb_selector_uses_first_attained_coverage_and_largest_fallback():
    coverage = torch.tensor([0.40, 0.68, 0.82])
    target = torch.tensor([0.50, 0.80, 0.95])
    assert select_upb_calibration_positions(coverage, target).tolist() == [
        1, 2, 2,
    ]


def test_upb_ht_coverage_uses_event_propensity_and_deterministic_infinity():
    taus = torch.tensor([0.5, 0.7, 0.9])
    candidates = torch.tensor([
        [1.0, 200.0, 201.0],
        [2.0, 200.0, 201.0],
        [1.0, 200.0, 201.0],
    ])
    allocator = _FixedUPBAllocator(
        candidates,
        torch.tensor([2.0, 199.0, 200.0]),
        torch.tensor([0.5, 0.25, 0.2]),
        taus,
    )
    calibration = SurvivalUPBCalibrationWithKnownWeights(
        allocator, taus, 0.9
    )
    prediction = SurvivalModelPrediction(
        candidates,
        torch.zeros(3, 200, 201),
    )
    calibration.calibrate(
        None, torch.tensor([2.0, 200.0, 201.0]), prediction
    )

    torch.testing.assert_close(
        calibration.coverage,
        torch.tensor([0.0, 2.0 / 3.0, 1.0], dtype=torch.float64),
    )
    # Ordinary allocators see an executable acquisition horizon, while the
    # calibration object restores the semantic 201 UPB candidates.
    assert allocator.seen_candidates.max().item() == 200
    assert calibration.allocation_result.f.max().item() == 201
    test_bound = calibration.get_calibrated_upb(
        torch.tensor([0.6, 0.9]), None, prediction
    )
    assert test_bound[:, 0].tolist() == [200.0, 200.0, 200.0]
    assert test_bound[:, 1].tolist() == [201.0, 201.0, 201.0]


def test_upb_coverage_treats_200_as_real_and_201_as_infinity():
    times = torch.tensor([200.0, 201.0])
    bounds = torch.tensor([[200.0, 201.0], [200.0, 201.0]])
    coverage, size = compute_metrics_bound(bounds, times, "upb")
    torch.testing.assert_close(coverage, torch.tensor([0.5, 1.0]))
    torch.testing.assert_close(size, torch.tensor([200.0, 201.0]))


def test_soft_upb_dapro_uses_finite_lower_or_equal_target_and_no_pav_crc():
    n, width = 4, 200
    grid = torch.zeros(n, width, width + 1)
    step = torch.arange(width)
    grid[:, step, step] = 0.01
    grid[:, :, -1] = 0.99
    taus = torch.tensor([0.50, 0.70, 0.90])
    quantiles = torch.tensor([
        [1.0, 2.0, 5.0],
        [2.0, 201.0, 201.0],
        [2.0, 4.0, 5.0],
        [3.0, 201.0, 201.0],
    ])
    times = torch.tensor([1.0, 201.0, 4.0, 201.0])
    prior = quantiles[:, -1].clamp(max=width)
    allocator = SoftTargetUPBDAPRO(
        grid, 20.0, taus, 0.90, width, n1=2,
        target_coverage=0.70,
    )
    allocator.phase1_objective_weights(times, prior, quantiles)
    target = allocator.phase2_target_indicator(times, prior, quantiles)
    torch.testing.assert_close(
        target, torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float64)
    )
    masses = allocator.phase1_objective_masses(
        times, prior, quantiles, grid
    )
    assert masses[0].sum() > masses[1].sum()
    assert masses[2].sum() > masses[3].sum()
    assert "upb_coverage_0p70_phase1_anchor" in allocator.name

    crc = SoftTargetCRCUPBDAPRO(
        grid, 20.0, taus, 0.90, width,
        n1=2, budget_control_size=1,
    )
    assert "budget_crc_control_1" in crc.name
    assert "row_cap" not in crc.name
    assert "causal_shared_pav" not in crc.name
    assert crc.risk_candidate_row_cost_cap is None


def test_upb_registry_contains_only_soft_prefix_dapro_variants():
    methods = get_upb_calibrations(
        None, 20.0, torch.linspace(0.5, 0.95, 30), 0.97, 200,
        dapro_n1_values=(200,),
    )
    dapro_names = [method.name for method in methods if "dapro" in method.name]
    assert len(dapro_names) == 2
    assert all("soft_prefix" in name for name in dapro_names)
    assert any("projection_margin" in name for name in dapro_names)
    assert any("budget_crc_control_100" in name for name in dapro_names)
    assert all("causal_shared_pav" not in name for name in dapro_names)
