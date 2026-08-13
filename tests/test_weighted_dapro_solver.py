import numpy as np
import pytest
import torch
from types import SimpleNamespace

from src.predictive_bounds.budget_allocators.DAPRO import (
    BandRegularizedTargetAWeightedDAPRO,
    DefinitiveDAPRO,
    RandomAnchoredTargetAWeightedDAPRO,
    RegularizedTargetAWeightedDAPRO,
    RobustTargetAWeightedDAPRO,
    SoftTargetDAPRO,
    TargetAWeightedDAPRO,
)
from src.predictive_bounds.budget_allocators.optimization_solver_utils import (
    solve_binned_deployable_policy,
    solve_exact_fast,
    solve_time_only_cumulative_policy,
)
from src.predictive_bounds.budget_allocators.projected_optimization_utils import (
    correct_projected_cumulative_probabilities_to_budget,
    correct_projected_probabilities_to_budget,
    enforce_terminal_probability_floor,
    expected_acquisition_cost,
)
from src.predictive_bounds.calibration.calibration_utils import (
    quantiles_to_interaction_counts,
    select_calibration_positions,
)
from src.predictive_bounds.construct_calibrated_bound import (
    get_baseline_calibrations,
)
from src.predictive_bounds.merge_bounds_results import (
    get_calibration_methods as get_merge_calibration_methods,
)
from src.predictive_bounds.survival_utils.compute_mean_time_given_pmf import (
    compute_quantiles_survival_time,
)
from src.predictive_bounds.utils.utils import (
    make_lpb_tau_grid,
    resolve_m_upper_bound,
)


def terminal_probabilities(probabilities, lengths):
    time = np.arange(probabilities.shape[1])[None, :]
    active = time < np.asarray(lengths)[:, None]
    return np.prod(np.where(active, probabilities, 1.0), axis=1)


def test_common_calibration_selector_uses_strict_initial_prefix():
    miscoverage = torch.tensor([0.01, 0.08, 0.10, 0.09, 0.20])
    targets = torch.tensor([0.10, 0.15])

    selected = select_calibration_positions(miscoverage, targets)

    # The 0.10 candidate itself is infeasible because selection is strict.
    # Once that candidate fails, a later finite-sample dip cannot re-enter.
    assert selected.tolist() == [1, 3]


def test_lpb_tau_grid_has_safe_zero_candidate_and_preserves_original_grid():
    taus = make_lpb_tau_grid()

    assert taus[0].item() == 0.0
    assert torch.all(taus[1:] > taus[:-1])
    assert torch.isclose(
        taus,
        torch.tensor(0.001, dtype=taus.dtype),
    ).any()
    assert len(taus) == 1257

    probabilities = torch.tensor([
        [[[0.0, 0.1, 0.2, 0.7]]],
        [[[0.00001, 0.2, 0.3, 0.49999]]],
    ]).squeeze(1)
    quantiles = compute_quantiles_survival_time(
        probabilities,
        taus[:2],
        tail_distribution="geometric",
    ).squeeze(1)
    interaction_counts = quantiles_to_interaction_counts(
        quantiles,
        width=probabilities.shape[-1] - 1,
    )
    assert interaction_counts[:, 0].tolist() == [1.0, 1.0]


def _make_target_allocator(anchor_kind, target_alpha):
    taus = torch.tensor([0.05, 0.09, 0.10, 0.20, 0.56])
    return TargetAWeightedDAPRO(
        conditional_grid=torch.ones(4, 5, 5),
        budget_per_sample=2,
        taus_range=taus,
        tau_prior=0.56,
        m_upper_bound=5,
        projection="platt",
        score="prob",
        anchor_kind=anchor_kind,
        target_alpha=target_alpha,
    )


def test_target_alpha_must_be_inside_prior_envelope():
    with pytest.raises(ValueError, match="strictly smaller"):
        _make_target_allocator("raw_alpha", 0.56)


def test_every_crc_capable_name_override_preserves_cap_provenance():
    common = dict(
        conditional_grid=torch.ones(100, 5, 5),
        budget_per_sample=2.0,
        taus_range=torch.tensor([0.05, 0.10, 0.56]),
        tau_prior=0.56,
        m_upper_bound=5,
        n1=100,
        budget_control_mode="crc",
        budget_control_size=50,
        risk_candidate_row_cost_cap=4.0,
        projection_budget_margin=0.0,
    )
    projected = dict(projection="direct_time", score="prob")
    allocators = [
        RandomAnchoredTargetAWeightedDAPRO(
            **common,
            **projected,
            target_alpha=0.10,
            target_policy_fraction=0.50,
        ),
        RobustTargetAWeightedDAPRO(
            **common,
            **projected,
            target_alpha=0.10,
            robustness_weight=0.10,
        ),
        RegularizedTargetAWeightedDAPRO(
            **common,
            **projected,
            target_alpha=0.10,
            global_regularization=0.001,
        ),
        BandRegularizedTargetAWeightedDAPRO(
            **common,
            **projected,
            target_alphas=(0.07, 0.13),
            global_regularization=0.001,
        ),
        DefinitiveDAPRO(**common),
        SoftTargetDAPRO(**common),
    ]
    for allocator in allocators:
        assert (
            "_budget_crc_control_50_row_cap_2p00x_budget_"
            "causal_shared_pav_v1"
        ) in allocator.name


def test_construct_registers_all_a_weighted_lpb_variants():
    conditional_grid = torch.ones(101, 1, 1)
    taus = torch.tensor([0.10, 0.56])
    prediction = SimpleNamespace(
        quantile_est=torch.ones(101, len(taus)),
    )

    calibrations = get_baseline_calibrations(
        conditional_grid=conditional_grid,
        budget_per_sample=2.0,
        taus_range=taus,
        tau_prior=0.56,
        m_upper_bound=5,
        cal_model_prediction=prediction,
        t_tilde_cal=torch.ones(101),
        bound_type="lpb",
        dapro_n1_values=(100,),
    )
    names = {calibration.name for calibration in calibrations}

    assert (
        "calibration_projected_optimization_platt_prob_a_weighted_allocation"
        in names
    )
    assert (
        "calibration_projected_optimization_platt_prob_a_target_raw_alpha_0p10_allocation"
        in names
    )
    assert (
        "calibration_projected_optimization_platt_prob_a_target_phase1_unweighted_alpha_0p10_allocation"
        in names
    )
    assert (
        "calibration_projected_optimization_cumulative_platt_prob_allocation"
        in names
    )
    assert (
        "calibration_projected_optimization_cumulative_platt_prob_a_target_phase1_unweighted_alpha_0p10_allocation"
        in names
    )
    assert (
        "calibration_projected_optimization_direct_time_prob_a_target_phase1_unweighted_alpha_0p10_allocation"
        in names
    )
    assert (
        "calibration_projected_optimization_direct_time_prob_a_target_phase1_robust_raw_0p10_alpha_0p10_allocation"
        in names
    )
    assert (
        "calibration_projected_optimization_direct_bins_2_prob_a_target_raw_alpha_0p10_allocation"
        in names
    )
    assert (
        "calibration_projected_optimization_direct_time_prob_a_target_raw_regularized_global_0p010_alpha_0p10_allocation"
        in names
    )
    assert (
        "calibration_projected_optimization_direct_time_prob_a_target_phase1_unweighted_regularized_global_0p010_alpha_0p10_allocation"
        in names
    )
    assert (
        "calibration_projected_optimization_direct_time_prob_a_band_0p07_0p13_global_0p010_allocation"
        in names
    )
    assert (
        "calibration_projected_optimization_direct_time_prob"
        "_a_target_raw_random_anchor_target_0p50_alpha_0p10"
        "_budget_crc_control_50_allocation"
        in names
    )
    assert (
        "calibration_dapro_soft_prefix_bins_2_lpb_alpha_0p10_"
        "global_0p001_projection_margin_1p00_n1_100_allocation"
        in names
    )
    assert (
        "calibration_dapro_soft_prefix_bins_2_lpb_alpha_0p10_"
        "global_0p001_budget_crc_control_50_row_cap_2p00x_budget_"
        "causal_shared_pav_v1_"
        "n1_100_allocation"
        in names
    )
    merge_names = {
        calibration.name
        for calibration in get_merge_calibration_methods(
            conditional_grid=None,
            budget_per_sample=2.0,
            taus_range=taus,
            tau_prior=0.56,
            m_upper_bound=5,
                allocations="none",
                device="cpu",
                bound_type="lpb",
                dapro_n1_values=(100,),
            )
    }
    assert names == merge_names
    ordinary_name = (
        "calibration_projected_optimization_platt_prob_allocation"
    )
    ordinary = next(
        calibration
        for calibration in calibrations
        if calibration.name == ordinary_name
    )
    assert not ordinary.budget_allocator.evaluate_projection
    diagnostic_calibrations = get_baseline_calibrations(
        conditional_grid=conditional_grid,
        budget_per_sample=2.0,
        taus_range=taus,
        tau_prior=0.56,
        m_upper_bound=5,
        cal_model_prediction=prediction,
        t_tilde_cal=torch.ones(101),
        bound_type="lpb",
        evaluate_dapro_projection=True,
        dapro_n1_values=(100,),
    )
    diagnostic_ordinary = next(
        calibration
        for calibration in diagnostic_calibrations
        if calibration.name == ordinary_name
    )
    assert diagnostic_ordinary.budget_allocator.evaluate_projection


def test_construct_registers_requested_dapro_phase1_sizes():
    conditional_grid = torch.ones(61, 1, 1)
    taus = torch.tensor([0.10, 0.56])
    prediction = SimpleNamespace(
        quantile_est=torch.ones(61, len(taus)),
    )

    calibrations = get_baseline_calibrations(
        conditional_grid=conditional_grid,
        budget_per_sample=2.0,
        taus_range=taus,
        tau_prior=0.56,
        m_upper_bound=5,
        cal_model_prediction=prediction,
        t_tilde_cal=torch.ones(61),
        bound_type="lpb",
        dapro_n1_values=(20, 40),
    )
    names = {calibration.name for calibration in calibrations}

    for n1 in (20, 40):
        expected_name = (
            "calibration_projected_optimization_direct_time_prob"
            "_a_target_raw_regularized_global_0p001_alpha_0p10"
            f"_n1_{n1}_allocation"
        )
        expected_phase1_name = (
            "calibration_projected_optimization_direct_time_prob"
            "_a_target_phase1_unweighted_regularized_global_0p001_alpha_0p10"
            f"_n1_{n1}_allocation"
        )
        assert expected_name in names
        assert expected_phase1_name in names
        merge_names = {
            calibration.name
            for calibration in get_merge_calibration_methods(
                conditional_grid=None,
                budget_per_sample=2.0,
                taus_range=taus,
                tau_prior=0.56,
                m_upper_bound=5,
                allocations="none",
                device="cpu",
                bound_type="lpb",
                dapro_n1_values=(20, 40),
            )
        }
        assert expected_name in merge_names
        assert expected_phase1_name in merge_names
    assert not any(
        "projected_optimization" in name and "_n1_100_" in name
        for name in names
    )


def test_upper_bound_resolution_honors_gamma_and_rejects_ambiguity():
    assert resolve_m_upper_bound(True, 20.0) == 200.0
    assert resolve_m_upper_bound(False, 20.0) == 20.0
    assert resolve_m_upper_bound(True, 10.0, gamma=10.0) == 100.0
    assert resolve_m_upper_bound(
        True,
        10.0,
        m_upper_bound=150.0,
    ) == 150.0
    with pytest.raises(ValueError, match="at most one"):
        resolve_m_upper_bound(
            True,
            20.0,
            gamma=10.0,
            m_upper_bound=200.0,
        )


def test_raw_alpha_target_weights_freeze_largest_strict_candidate():
    allocator = _make_target_allocator("raw_alpha", 0.10)
    quantiles = torch.tensor([
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
    ])
    event_times = torch.tensor([1, 2, 3, 4])
    prior_q = quantiles[:, -1]

    weights = allocator.phase1_objective_weights(
        event_times,
        prior_q,
        quantiles,
    )

    assert allocator._target_anchor_index == 1
    assert weights.tolist() == [1.0, 0.0, 0.0, 0.0]
    assert (
        allocator.objective_metadata()[
            "target_anchor_phase1_within_prior"
        ]
        == 1
    )


def test_phase1_unweighted_target_uses_same_strict_selector_and_is_frozen():
    allocator = _make_target_allocator("phase1_unweighted", 0.50)
    quantiles = torch.tensor([
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
    ])
    event_times = torch.tensor([1, 2, 3, 4])
    prior_q = quantiles[:, -1]
    phase1_weights = allocator.phase1_objective_weights(
        event_times,
        prior_q,
        quantiles,
    )
    phase2_quantiles = quantiles + torch.tensor([0, 0, 10, 10, 10])
    phase2_weights = allocator.phase2_objective_weights(
        event_times,
        phase2_quantiles[:, -1],
        phase2_quantiles,
    )

    # Miscoverage is [0, .25, .50, .75, 1], so strict alpha=.50
    # selects index 1. The Phase-II quantiles deliberately change later
    # candidates; the frozen index remains unchanged.
    assert allocator._target_anchor_index == 1
    assert phase1_weights.tolist() == [1.0, 0.0, 0.0, 0.0]
    assert phase2_weights.tolist() == [1.0, 0.0, 0.0, 0.0]


def test_robust_target_blends_frozen_phase1_and_raw_anchors():
    taus = torch.tensor([0.05, 0.09, 0.10, 0.20, 0.56])
    allocator = RobustTargetAWeightedDAPRO(
        conditional_grid=torch.ones(4, 5, 5),
        budget_per_sample=2,
        taus_range=taus,
        tau_prior=0.56,
        m_upper_bound=200,
        projection="direct_time",
        score="prob",
        target_alpha=0.10,
        robustness_weight=0.5,
    )
    quantiles = torch.tensor([
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
    ])
    event_times = torch.tensor([1, 2, 3, 4])
    prior_q = quantiles[:, -1]

    weights = allocator.phase1_objective_weights(
        event_times,
        prior_q,
        quantiles,
    )

    # Phase I selects index 0 because index 1 has empirical rate .25, whereas
    # raw alpha selects index 1.  The raw-only event receives gamma/(1+gamma).
    assert weights.tolist() == [1 / 3, 0.0, 0.0, 0.0]
    assert allocator._target_anchor_index == 0
    assert allocator._raw_anchor_index == 1


def test_global_regularization_keeps_all_objective_weights_positive():
    taus = torch.tensor([0.05, 0.09, 0.10, 0.20, 0.56])
    allocator = RegularizedTargetAWeightedDAPRO(
        conditional_grid=torch.ones(4, 5, 5),
        budget_per_sample=2,
        taus_range=taus,
        tau_prior=0.56,
        m_upper_bound=200,
        projection="direct_time",
        score="prob",
        target_alpha=0.10,
        global_regularization=0.01,
    )
    quantiles = torch.tensor([
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
    ])
    weights = allocator.phase1_objective_weights(
        torch.tensor([1, 2, 3, 4]),
        quantiles[:, -1],
        quantiles,
    )

    np.testing.assert_allclose(
        weights.numpy(),
        [1.0, 0.01 / 1.01, 0.01 / 1.01, 0.01 / 1.01],
        atol=1e-12,
    )


def test_global_regularization_supports_phase1_selected_target():
    taus = torch.tensor([0.05, 0.09, 0.10, 0.20, 0.56])
    allocator = RegularizedTargetAWeightedDAPRO(
        conditional_grid=torch.ones(4, 5, 5),
        budget_per_sample=2,
        taus_range=taus,
        tau_prior=0.56,
        m_upper_bound=200,
        projection="direct_time",
        score="prob",
        target_alpha=0.10,
        anchor_kind="phase1_unweighted",
        global_regularization=0.01,
    )
    quantiles = torch.tensor([
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
    ])
    weights = allocator.phase1_objective_weights(
        torch.tensor([1, 2, 3, 4]),
        quantiles[:, -1],
        quantiles,
    )

    assert (
        allocator.name
        == "projected_optimization_direct_time_prob"
        "_a_target_phase1_unweighted_regularized_global_0p010_alpha_0p10"
    )
    # The Phase-I empirical target selects the first candidate, for which no
    # row has T < q. Global regularization still keeps every weight positive.
    np.testing.assert_allclose(
        weights.numpy(),
        np.full(4, 0.01 / 1.01),
        atol=1e-12,
    )


def test_band_target_averages_candidate_indicators_and_regularizes():
    taus = torch.tensor([
        0.05,
        0.07,
        0.08,
        0.09,
        0.10,
        0.11,
        0.12,
        0.13,
        0.56,
    ])
    alphas = tuple(0.07 + 0.01 * offset for offset in range(7))
    allocator = BandRegularizedTargetAWeightedDAPRO(
        conditional_grid=torch.ones(4, 9, 9),
        budget_per_sample=2,
        taus_range=taus,
        tau_prior=0.56,
        m_upper_bound=200,
        projection="direct_time",
        score="prob",
        target_alphas=alphas,
        global_regularization=0.01,
    )
    quantiles = torch.arange(
        1,
        10,
        dtype=torch.float64,
    ).repeat(4, 1)
    event_times = torch.tensor([0, 1, 4, 8])

    weights = allocator.phase1_objective_weights(
        event_times,
        quantiles[:, -1],
        quantiles,
    )

    # Strict candidate selection maps .07,...,.13 to tau indices 0,...,6.
    raw_band = np.array([1.0, 6 / 7, 3 / 7, 0.0])
    np.testing.assert_allclose(
        weights.numpy(),
        (raw_band + 0.01) / 1.01,
        atol=1e-12,
    )
    metadata = allocator.objective_metadata()
    assert metadata["band_target_alpha_count"] == 7
    assert metadata["band_target_anchor_index_low"] == 0
    assert metadata["band_target_anchor_index_high"] == 6
    assert metadata["band_target_phase1_within_prior"] == 1


def test_binary_a_one_step_has_analytic_solution():
    scores = torch.tensor([[1.0], [0.0]])
    lengths = torch.tensor([1, 1])
    a = torch.tensor([1.0, 0.0])

    probabilities = solve_exact_fast(
        scores,
        lengths,
        0.4,
        objective_weights=a,
        max_outer=80,
        max_inner=30,
        verbose=False,
    )

    np.testing.assert_allclose(
        probabilities[:, 0],
        [0.8, np.exp(-700)],
        atol=2e-6,
    )
    assert probabilities.mean() <= 0.4 + 1e-8


def test_weighted_solver_improves_requested_proxy_at_same_budget():
    scores = torch.tensor([[1.0], [0.0]])
    lengths = torch.tensor([1, 1])
    a = np.array([1.0, 0.0])
    weighted = solve_exact_fast(
        scores,
        lengths,
        0.4,
        objective_weights=torch.tensor(a),
        verbose=False,
    )
    ordinary = solve_exact_fast(
        scores,
        lengths,
        0.4,
        objective_weights=torch.ones(2),
        verbose=False,
    )
    weighted_proxy = np.mean(a * (1 / weighted[:, 0] - 1))
    ordinary_proxy = np.mean(a * (1 / ordinary[:, 0] - 1))

    assert weighted_proxy < ordinary_proxy
    assert weighted.mean() <= 0.4 + 1e-8
    assert ordinary.mean() <= 0.4 + 1e-8


def test_score_monotonicity_can_pool_zero_weight_row():
    scores = torch.tensor([[0.0], [1.0]])
    lengths = torch.tensor([1, 1])
    probabilities = solve_exact_fast(
        scores,
        lengths,
        0.4,
        objective_weights=torch.tensor([1.0, 0.0]),
        verbose=False,
    )

    assert probabilities[0, 0] <= probabilities[1, 0] + 1e-9
    np.testing.assert_allclose(
        probabilities[:, 0],
        [0.4, 0.4],
        atol=2e-5,
    )


def test_sparse_a_saturation_prefers_objective_over_budget_tightness():
    scores = torch.arange(5, dtype=torch.float64).reshape(-1, 1)
    lengths = torch.ones(5, dtype=torch.long)
    a = torch.tensor([0, 0, 0, 0, 1], dtype=torch.float64)

    probabilities = solve_exact_fast(
        scores,
        lengths,
        0.5,
        objective_weights=a,
        max_outer=80,
        max_inner=30,
        verbose=False,
    )

    np.testing.assert_allclose(probabilities[-1, 0], 1.0, atol=1e-9)
    assert probabilities.mean() <= 0.5 + 1e-8
    assert np.mean(
        a.numpy() * (1 / probabilities[:, 0] - 1)
    ) <= 1e-9


def test_all_zero_objective_returns_numerical_minimum_cost_policy():
    scores = torch.tensor(
        [[0.1, 0.2, 0.3], [0.3, 0.4, 0.5], [0.2, 0.1, 0.4]]
    )
    lengths = torch.tensor([1, 2, 3])
    all_zero = solve_exact_fast(
        scores,
        lengths,
        0.8,
        objective_weights=torch.zeros(3),
        verbose=False,
    )
    terminal = terminal_probabilities(all_zero, lengths.numpy())

    np.testing.assert_allclose(terminal, np.exp(-700), rtol=1e-12)
    assert expected_acquisition_cost(
        torch.tensor(all_zero),
        lengths,
    ) <= 0.8


def test_direct_time_policy_contains_floor_matched_random():
    lengths = torch.tensor([1, 2, 3, 4, 4, 5])
    target_a = torch.tensor([1, 1, 0, 0, 0, 0], dtype=torch.float64)
    epsilon = 0.05
    budget = 1.25
    conditionals, cumulative, diagnostics = (
        solve_time_only_cumulative_policy(
            lengths,
            budget,
            target_a,
            width=5,
            terminal_pi_min=epsilon,
        )
    )
    assert np.all(cumulative[1:] <= cumulative[:-1] + 1e-12)
    assert np.all(cumulative >= epsilon - 1e-12)
    assert diagnostics["direct_time_expected_cost"] <= budget + 1e-9
    np.testing.assert_allclose(
        np.cumprod(conditionals),
        cumulative,
        atol=1e-12,
    )

    active = np.arange(5)[None, :] < lengths.numpy()[:, None]

    def random_cumulative(per_step):
        raw = per_step ** np.arange(1, 6)
        return epsilon + (1 - epsilon) * raw

    low, high = 0.0, 1.0
    for _ in range(100):
        mid = (low + high) / 2
        candidate = random_cumulative(mid)
        candidate_cost = np.mean((candidate * active).sum(axis=1))
        if candidate_cost <= budget:
            low = mid
        else:
            high = mid
    random_schedule = random_cumulative(low)
    endpoints = lengths.numpy() - 1
    direct_objective = np.mean(target_a.numpy() / cumulative[endpoints])
    random_objective = np.mean(
        target_a.numpy() / random_schedule[endpoints]
    )
    assert direct_objective <= random_objective + 1e-9


def test_direct_time_marks_sparse_objective_budget_plateau():
    _, cumulative, diagnostics = solve_time_only_cumulative_policy(
        lengths=torch.tensor([1, 4]),
        budget_per_sample=2.0,
        objective_weights=torch.tensor([1.0, 0.0]),
        width=4,
        terminal_pi_min=0.05,
    )

    assert diagnostics["direct_time_budget_boundary"] == "objective_plateau"
    assert diagnostics["direct_time_budget_slack"] > 0.8
    np.testing.assert_allclose(cumulative, [1.0, 0.05, 0.05, 0.05])


def test_direct_binned_policy_is_deployable_by_exact_lookup():
    validation_scores = torch.tensor(
        [
            [0.1, 0.2],
            [0.2, 0.3],
            [0.8, 0.7],
            [0.9, 0.8],
        ],
        dtype=torch.float64,
    )
    deployment_scores = torch.tensor(
        [[0.15, 0.25], [0.85, 0.75]],
        dtype=torch.float64,
    )
    lengths = torch.tensor([1, 2, 2, 2])
    weights = torch.tensor([0.0, 0.0, 1.0, 1.0])

    optimal, validation, deployment, diagnostics = (
        solve_binned_deployable_policy(
            validation_scores,
            deployment_scores,
            lengths,
            budget_per_sample=0.8,
            objective_weights=weights,
            n_bins=2,
        )
    )

    assert optimal.shape == (4, 2)
    assert validation.shape == (4, 2)
    assert deployment.shape == (2, 2)
    assert expected_acquisition_cost(validation, lengths) <= 0.8 + 1e-8
    assert diagnostics["direct_score_bin_count"] == 2
    assert (
        diagnostics[
            "direct_score_bin_max_within_bin_probability_spread"
        ]
        <= 1e-9
    )
    # Deployment rows occupy the low/high Phase-I bins respectively.
    np.testing.assert_allclose(
        deployment[0].numpy(),
        validation[0].numpy(),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        deployment[1].numpy(),
        validation[2].numpy(),
        atol=1e-10,
    )


def test_coupled_terminal_floor_is_explicitly_rejected():
    with pytest.raises(NotImplementedError, match="couples coordinates"):
        solve_exact_fast(
            torch.tensor([[0.0], [1.0]]),
            torch.tensor([1, 1]),
            0.4,
            objective_weights=torch.tensor([1.0, 0.0]),
            terminal_pi_min=0.1,
            verbose=False,
        )


def test_default_and_explicit_all_one_weights_agree():
    torch.manual_seed(7)
    scores = torch.rand(5, 3)
    lengths = torch.tensor([1, 2, 3, 2, 3])
    default = solve_exact_fast(
        scores,
        lengths,
        0.9,
        verbose=False,
    )
    explicit = solve_exact_fast(
        scores,
        lengths,
        0.9,
        objective_weights=torch.ones(5),
        verbose=False,
    )
    np.testing.assert_allclose(default, explicit, rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    "weights",
    [
        torch.tensor([-1.0, 1.0]),
        torch.tensor([float("nan"), 1.0]),
        torch.ones(3),
    ],
)
def test_invalid_objective_weights_are_rejected(weights):
    with pytest.raises(ValueError):
        solve_exact_fast(
            torch.tensor([[0.0], [1.0]]),
            torch.tensor([1, 1]),
            0.4,
            objective_weights=weights,
            verbose=False,
        )


def test_projection_budget_correction_includes_terminal_floor():
    validation_raw = torch.full((4, 3), 0.8)
    deployment_raw = torch.tensor(
        [[0.2, 0.4, 0.8], [0.9, 0.7, 0.5]]
    )
    validation_lengths = torch.tensor([1, 2, 3, 3])
    validation_q = torch.tensor([3, 3, 3, 3])
    deployment_q = torch.tensor([3, 2])

    validation, deployment, diagnostics = (
        correct_projected_probabilities_to_budget(
            validation_raw,
            deployment_raw,
            validation_lengths,
            validation_q,
            deployment_q,
            target_budget_per_sample=0.9,
            terminal_pi_min=0.05,
        )
    )

    assert expected_acquisition_cost(validation, validation_lengths) <= 0.9 + 1e-7
    validation_terminal = terminal_probabilities(
        validation.numpy(),
        validation_q.numpy(),
    )
    deployment_terminal = terminal_probabilities(
        deployment.numpy(),
        deployment_q.numpy(),
    )
    assert np.all(validation_terminal >= 0.05 - 1e-10)
    assert np.all(deployment_terminal >= 0.05 - 1e-10)
    np.testing.assert_allclose(
        diagnostics["projection_corrected_phase1_expected_cost"],
        0.9,
        atol=1e-7,
    )


def test_exploration_mixture_telescopes_and_preserves_structural_ones():
    raw = torch.tensor(
        [[0.2, 1.0, 0.4], [0.9, 0.3, 1.0]],
        dtype=torch.float64,
    )
    q = torch.tensor([3, 2])
    epsilon = 0.05

    mixed = enforce_terminal_probability_floor(raw, q, epsilon)
    time = torch.arange(raw.shape[1]).unsqueeze(0)
    active = time < q.unsqueeze(1)
    raw_prefix = torch.where(active, raw, torch.ones_like(raw)).cumprod(1)
    mixed_prefix = torch.where(
        active,
        mixed,
        torch.ones_like(mixed),
    ).cumprod(1)
    expected_prefix = epsilon + (1 - epsilon) * raw_prefix

    np.testing.assert_allclose(
        mixed_prefix[active].numpy(),
        expected_prefix[active].numpy(),
        atol=1e-12,
    )
    assert mixed[0, 1].item() == 1.0


def test_cumulative_projection_correction_hits_budget_and_telescopes():
    validation_cumulative = torch.tensor(
        [
            [0.90, 0.70, 0.50],
            [0.80, 0.60, 0.40],
            [0.95, 0.75, 0.55],
            [0.85, 0.65, 0.45],
        ],
        dtype=torch.float64,
    )
    deployment_cumulative = torch.tensor(
        [[0.70, 0.50, 0.30], [0.95, 0.80, 0.60]],
        dtype=torch.float64,
    )
    validation_lengths = torch.tensor([1, 2, 3, 3])
    validation_q = torch.tensor([3, 3, 3, 3])
    deployment_q = torch.tensor([3, 2])

    validation, deployment, diagnostics = (
        correct_projected_cumulative_probabilities_to_budget(
            validation_cumulative,
            deployment_cumulative,
            validation_lengths,
            validation_q,
            deployment_q,
            target_budget_per_sample=1.0,
            terminal_pi_min=0.05,
        )
    )

    np.testing.assert_allclose(
        expected_acquisition_cost(validation, validation_lengths),
        1.0,
        atol=1e-7,
    )
    validation_prefix = validation.cumprod(dim=1)
    deployment_prefix = deployment.cumprod(dim=1)
    assert torch.all(
        validation_prefix[:, 1:] <= validation_prefix[:, :-1] + 1e-12
    )
    assert torch.all(
        deployment_prefix[:, 1:] <= deployment_prefix[:, :-1] + 1e-12
    )
    assert np.all(
        terminal_probabilities(
            validation.numpy(),
            validation_q.numpy(),
        ) >= 0.05 - 1e-10
    )
    assert np.all(
        terminal_probabilities(
            deployment.numpy(),
            deployment_q.numpy(),
        ) >= 0.05 - 1e-10
    )
    assert diagnostics["projection_space"] == "cumulative_probability"
