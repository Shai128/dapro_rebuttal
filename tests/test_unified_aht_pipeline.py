import torch

from src.predictive_bounds.calibration.sequential_aht import (
    metric_aht_contributions,
    sequential_lower_curve,
)
from src.predictive_bounds.utils.get_calibration_methods_utils import (
    get_metric_allocators,
    get_unified_bound_calibrations,
)


def _grid(n: int = 3) -> torch.Tensor:
    grid = torch.zeros(n, 2, 3, dtype=torch.float64)
    grid[:, 0, :] = torch.tensor([0.2, 0.3, 0.5])
    grid[:, 1, :] = torch.tensor([0.0, 0.25, 0.75])
    return grid


def test_metric_sequential_aht_is_design_unbiased_by_enumeration():
    grid = _grid()
    times = torch.full((3,), 3.0)
    acquired = torch.tensor([0.0, 1.0, 2.0])
    conditionals = torch.tensor([[0.5, 0.4]] * 3, dtype=torch.float64)
    contribution = metric_aht_contributions(
        times,
        acquired,
        conditionals.prod(dim=1),
        grid,
        2,
        continuation_probabilities=conditionals,
    )
    state_probability = torch.tensor([0.5, 0.3, 0.2])
    torch.testing.assert_close(
        (state_probability * contribution).sum(),
        torch.tensor(0.0, dtype=torch.float64),
    )


def test_lpb_sequential_curve_uses_strict_event_target():
    grid = _grid()
    # T=1 is positive for f=2; T=2 is not.
    times = torch.tensor([1.0, 2.0, 3.0])
    candidates = torch.full((3, 1), 2.0)
    conditionals = torch.ones(3, 2, dtype=torch.float64)
    estimate = sequential_lower_curve(
        times,
        candidates,
        torch.tensor([1.0, 2.0, 2.0]),
        conditionals,
        grid,
        strict=True,
    )
    torch.testing.assert_close(
        estimate.reshape(-1),
        torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
    )


def test_unified_metric_registry_has_only_requested_families():
    taus = torch.arange(0.01, 1.0, 0.01)
    names = [allocator.name for allocator in get_metric_allocators(
        _grid(20),
        2.0,
        2,
        taus,
        0.56,
        "cpu",
        dapro_n1=10,
        crc_control_size=5,
        method_suite="unified_aht",
    )]
    assert len(names) == 10
    assert names[0] == "optimized"
    assert names[-1] == "oracle_split_full_budget"
    assert sum("information_gain_sequential_aht" in name for name in names) == 2
    assert sum("residual_sequential_aht" in name for name in names) == 2
    assert sum("endpoint_block_terminal_residual_aht" in name for name in names) == 2
    raw_projection = [name for name in names if "projection_margin" in name]
    assert raw_projection and all("projection_margin_0p00" in name for name in raw_projection)


def test_unified_upb_registry_separates_three_policy_targets():
    taus = torch.arange(0.01, 1.0, 0.01)
    calibrations = get_unified_bound_calibrations(
        _grid(20),
        2.0,
        taus,
        0.98,
        2,
        bound_type="upb",
        dapro_n1_values=(10,),
        target_coverages=(0.70, 0.80, 0.90),
    )
    names = [calibration.name for calibration in calibrations]
    assert len(names) == len(set(names)) == 26
    assert any("coverage_0p70" in name for name in names)
    assert any("coverage_0p80" in name for name in names)
    assert any("coverage_0p90" in name for name in names)
    # UPB CRC intentionally uses the full 200-turn support rather than the
    # shared-PAV cap.
    assert not any("causal_shared_pav" in name for name in names)
