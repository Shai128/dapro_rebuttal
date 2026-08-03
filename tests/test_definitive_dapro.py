"""Tests for the single public DAPRO algorithm and its budget assumption."""

import numpy as np
import torch

from src.predictive_bounds.budget_allocators.DAPRO import (
    DAPRO,
    DefinitiveCRCDAPRO,
    DefinitiveCRCUPBDAPRO,
    DefinitiveDAPRO,
    LegacyMeanWeightDAPRO,
    TargetAWeightedDAPRO,
)
from src.utils.utils import set_seeds


def _inputs(n: int = 80, width: int = 4):
    generator = torch.Generator().manual_seed(91)
    grid = torch.rand(n, width, width, generator=generator)
    grid = grid / grid.sum(dim=-1, keepdim=True)
    taus = torch.tensor([0.05, 0.09, 0.50])
    quantiles = torch.tensor([1.0, 2.0, 3.0]).repeat(n, 1)
    event_times = torch.tensor(
        [1.0, 3.0, 5.0] * (n // 3) + [1.0] * (n % 3)
    )
    probability_est = torch.zeros((n, width))
    return grid, taus, quantiles, event_times, probability_est


def test_public_dapro_is_the_definitive_algorithm():
    assert DAPRO is DefinitiveCRCDAPRO
    assert DAPRO is not LegacyMeanWeightDAPRO
    assert DAPRO.DEFAULT_N1 == 200
    assert DAPRO.DEFAULT_BUDGET_CONTROL_SIZE == 100
    assert DAPRO.DEFAULT_ROW_COST_CAP_MULTIPLIER == 2.0


def test_definitive_dapro_configuration_is_fixed_and_auditable():
    grid, taus, _, _, _ = _inputs()
    allocator = DAPRO(
        grid,
        3.0,
        taus,
        0.50,
        4,
    )

    assert allocator.projection == "direct_bins_2"
    assert allocator.score == "prob"
    assert allocator.anchor_kind == "raw_alpha"
    assert allocator.target_alpha == 0.10
    assert allocator.global_regularization == 0.001
    assert allocator.terminal_pi_min == 0.005
    assert allocator.projection_budget_margin == 0.0
    assert allocator.budget_control_mode == "crc"
    assert allocator.budget_control_size == 100
    assert allocator.row_cost_cap_multiplier == 2.0
    assert allocator.objective_kind == (
        "definitive_regularized_target_a_variance_crc"
    )
    assert "bins_2" in allocator.name
    assert "budget_crc" in allocator.name
    assert "row_cap_2p00x_budget" in allocator.name
    assert allocator.name.endswith("n1_200")


def test_generic_crc_dapro_names_do_not_collide_with_uncontrolled_variants():
    grid, taus, _, _, _ = _inputs(n=220)
    common = {
        "conditional_grid": grid,
        "budget_per_sample": 3.0,
        "taus_range": taus,
        "tau_prior": 0.50,
        "m_upper_bound": 4,
        "projection": "direct_bins_2",
        "score": "prob",
        "n1": 200,
    }
    legacy = LegacyMeanWeightDAPRO(**common)
    legacy_crc = LegacyMeanWeightDAPRO(
        **common,
        budget_control_mode="crc",
        budget_control_size=100,
        risk_candidate_row_cost_cap=6.0,
    )
    target = TargetAWeightedDAPRO(
        **common,
        anchor_kind="raw_alpha",
        target_alpha=0.10,
    )
    target_crc = TargetAWeightedDAPRO(
        **common,
        anchor_kind="raw_alpha",
        target_alpha=0.10,
        budget_control_mode="crc",
        budget_control_size=100,
        risk_candidate_row_cost_cap=6.0,
    )

    assert legacy.name != legacy_crc.name
    assert target.name != target_crc.name
    assert "budget_crc_control_100_row_cap_2p00x_budget" in legacy_crc.name
    assert "budget_crc_control_100_row_cap_2p00x_budget" in target_crc.name


def test_projection_margin_implies_expected_total_budget_bound():
    grid, taus, quantiles, event_times, probability_est = _inputs()
    allocator = DefinitiveDAPRO(
        grid,
        3.0,
        taus,
        0.50,
        4,
        n1=20,
    )
    allocator.set_acquisition_randomness(
        seed=7,
        uniforms=np.random.default_rng(7).random(grid.shape[:2]),
    )
    set_seeds(7)
    metrics = allocator.allocate_budget(
        probability_est,
        None,
        event_times,
        quantiles,
    ).additional_metrics

    n_phase2 = metrics["phase2_sample_count"]
    implied_upper_bound = (
        metrics["phase1_expected_cost_total"]
        + n_phase2
        * (
            metrics["projection_corrected_phase1_expected_cost"]
            + metrics["projection_budget_margin_per_sample"]
        )
    )
    np.testing.assert_allclose(
        metrics[
            "projection_expected_total_budget_upper_bound_under_assumption"
        ],
        implied_upper_bound,
    )
    assert metrics[
        "projection_expected_budget_guarantee_valid_under_assumption"
    ] == 1
    assert implied_upper_bound <= 3.0 * len(grid) + 1e-7 * len(grid)

    transfer_error = (
        metrics["phase2_expected_cost_per_sample"]
        - metrics["projection_corrected_phase1_expected_cost"]
    )
    np.testing.assert_allclose(
        metrics["projection_transfer_cost_error_per_sample"],
        transfer_error,
    )
    assert metrics["projection_transfer_assumption_satisfied"] == int(
        transfer_error <= 1.0 + 1e-7
    )

    assert 0 <= metrics["phase2_focus_a_rate"] <= 1
    assert 0 <= metrics["phase2_focus_expected_query_share"] <= 1
    assert metrics["phase2_focus_expected_query_lift"] >= 0
    assert metrics["phase2_focus_mean_expected_queries"] >= 0
    assert metrics["phase2_nonfocus_mean_expected_queries"] >= 0
    assert 0 <= metrics["phase2_focus_mean_terminal_probability"] <= 1
    assert 0 <= metrics["phase2_nonfocus_mean_terminal_probability"] <= 1


def test_definitive_crc_dapro_uses_an_independent_control_fold():
    grid, taus, quantiles, event_times, probability_est = _inputs()
    allocator = DefinitiveCRCDAPRO(
        grid,
        3.0,
        taus,
        0.50,
        4,
        n1=30,
        budget_control_size=10,
    )
    allocator.set_acquisition_randomness(
        seed=7,
        uniforms=np.random.default_rng(7).random(grid.shape[:2]),
    )
    set_seeds(7)
    metrics = allocator.allocate_budget(
        probability_est,
        None,
        event_times,
        quantiles,
    ).additional_metrics

    assert metrics["risk_budget_policy_fit_size"] == 20
    assert metrics["risk_budget_control_size"] == 10
    assert metrics["risk_budget_selector_valid"] == 1
    assert metrics["expected_budget_guarantee_requires_projection_accuracy"] == 0
    assert metrics["expected_budget_guarantee_is_marginal_finite_sample"] == 1
    assert metrics["risk_budget_guarantee_kind"] == (
        "crc_marginal_expected_total_budget"
    )
    assert 0 <= metrics["risk_budget_selected_mixture_parameter"] <= 1


def test_crc_dapro_row_cap_reduces_the_distribution_free_envelope():
    grid, taus, quantiles, event_times, probability_est = _inputs()
    results = {}
    for label, multiplier in [("uncapped", None), ("capped", 1.0)]:
        allocator = DefinitiveCRCDAPRO(
            grid,
            3.0,
            taus,
            0.50,
            4,
            n1=30,
            budget_control_size=10,
            row_cost_cap_multiplier=multiplier,
        )
        allocator.set_acquisition_randomness(
            seed=7,
            uniforms=np.random.default_rng(7).random(grid.shape[:2]),
        )
        set_seeds(7)
        results[label] = allocator.allocate_budget(
            probability_est,
            None,
            event_times,
            quantiles,
        ).additional_metrics

    assert results["capped"]["risk_budget_row_cost_cap_enabled"] == 1
    assert results["capped"]["risk_budget_row_cost_cap_per_sample"] == 3.0
    assert (
        results["capped"]["risk_budget_maximum_candidate_cost_per_sample"]
        == 3.0
    )
    assert (
        results["capped"]["risk_budget_correction_per_sample"]
        < results["uncapped"]["risk_budget_correction_per_sample"]
    )


def test_definitive_upb_dapro_uses_the_upper_tail_target_event():
    grid, _, quantiles, event_times, _ = _inputs()
    taus = torch.tensor([0.50, 0.70, 0.90])
    allocator = DefinitiveCRCUPBDAPRO(
        grid,
        3.0,
        taus,
        0.90,
        4,
        n1=30,
        budget_control_size=10,
    )

    weights = allocator.phase1_objective_weights(
        event_times,
        quantiles[:, -1],
        quantiles,
    )
    expected_a = (event_times > quantiles[:, 1]).to(torch.float64)
    expected = (expected_a + 0.001) / 1.001

    torch.testing.assert_close(weights, expected)
    assert allocator.target_coverage == 0.70
    assert allocator.objective_kind == (
        "definitive_regularized_upb_target_a_variance_crc"
    )
    assert allocator.name.startswith("dapro_upb_variance_aligned")
    assert "alpha_0p70" in allocator.name
