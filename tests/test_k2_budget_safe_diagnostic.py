import numpy as np
import torch

from analysis.diagnostics.k2_budget_safe_experiment import (
    ONLINE_STATE,
    _direct_cost,
    all_high_correction,
    online_compensator_allocation,
    online_identity_correction,
    x0_high_correction,
)


def _toy_cumulative(seed=0):
    rng = np.random.default_rng(seed)
    n_fit, n_deploy, width = 40, 70, 12
    table = np.sort(rng.uniform(0.45, 0.98, size=(2, width)), axis=0)
    fit_bins = rng.integers(0, 2, size=(n_fit, width))
    deploy_bins = rng.integers(0, 2, size=(n_deploy, width))
    fit_p = table[fit_bins, np.arange(width)[None, :]]
    deploy_p = table[deploy_bins, np.arange(width)[None, :]]
    fit_q = torch.from_numpy(rng.integers(5, width + 1, size=n_fit))
    deploy_q = torch.from_numpy(rng.integers(5, width + 1, size=n_deploy))
    return (
        torch.from_numpy(fit_p.cumprod(1)),
        torch.from_numpy(deploy_p.cumprod(1)),
        fit_q,
        deploy_q,
    )


def test_static_path_envelopes_control_deployment_expected_cost():
    fit, deploy, fit_q, deploy_q = _toy_cumulative()
    for correction in (all_high_correction, x0_high_correction):
        _, p_deploy, metrics = correction(
            fit,
            deploy,
            fit_q,
            fit_q,
            deploy_q,
            target_budget_per_sample=4.0,
            terminal_pi_min=0.005,
        )
        actual = expected = _direct_cost(
            p_deploy.cumprod(1), deploy_q
        )
        assert expected <= 4.0 + 1e-8
        assert actual <= metrics[
            "k2_safe_deployment_upper_bound_cost_per_sample"
        ] + 1e-8
        assert metrics["k2_safe_valid"] == 1
        assert torch.all(p_deploy > 0)


def test_online_compensator_is_positive_and_never_exceeds_credit():
    fit, deploy, fit_q, deploy_q = _toy_cumulative(seed=1)
    _, deploy_p, _ = online_identity_correction(
        fit,
        deploy,
        fit_q,
        fit_q,
        deploy_q,
        target_budget_per_sample=4.0,
        terminal_pi_min=None,
    )
    event = torch.full_like(deploy_q, 12)
    uniforms = torch.from_numpy(
        np.random.default_rng(4).random(deploy_p.shape)
    )
    _, _, propensity = online_compensator_allocation(
        deploy_p,
        deploy_q,
        event,
        deploy_p.shape[1],
        torch.device("cpu"),
        uniforms=uniforms,
    )
    assert ONLINE_STATE.phase2_compensator <= ONLINE_STATE.phase2_budget + 1e-8
    assert torch.all(propensity > 0)


def test_cross_row_adaptation_keeps_ht_products_uncorrelated():
    """Exact enumeration of a two-row, two-turn shared-ledger policy."""
    p_first = (0.6, 0.7)
    moments = np.zeros(3, dtype=float)  # E W1, E W2, E W1 W2
    for keep1 in (0, 1):
        for keep2 in (0, 1):
            probability_first = (
                (p_first[0] if keep1 else 1 - p_first[0])
                * (p_first[1] if keep2 else 1 - p_first[1])
            )
            # Second-turn probabilities depend on the joint first-turn active
            # set, mimicking a common online ledger adjustment.
            if keep1 and keep2:
                p_second = (0.4, 0.8)
            elif keep1:
                p_second = (0.9, None)
            elif keep2:
                p_second = (None, 0.9)
            else:
                p_second = (None, None)
            outcomes1 = (0, 1) if keep1 else (0,)
            outcomes2 = (0, 1) if keep2 else (0,)
            for end1 in outcomes1:
                for end2 in outcomes2:
                    probability = probability_first
                    if keep1:
                        probability *= (
                            p_second[0] if end1 else 1 - p_second[0]
                        )
                    if keep2:
                        probability *= (
                            p_second[1] if end2 else 1 - p_second[1]
                        )
                    w1 = (
                        1 / (p_first[0] * p_second[0])
                        if keep1 and end1 else 0.0
                    )
                    w2 = (
                        1 / (p_first[1] * p_second[1])
                        if keep2 and end2 else 0.0
                    )
                    moments += probability * np.array([w1, w2, w1 * w2])
    assert np.allclose(moments, (1.0, 1.0, 1.0), atol=1e-12)


def test_exact_median_k2_kkt_counterexample_overruns_fresh_budget():
    """Exact six-case calculation used in the K2 budget theory section."""
    delta, floor, budget = 0.001, 0.005, 0.2
    fit_bin_mass = np.array([2 / 5, 3 / 5], dtype=float)
    target_counts = np.array(
        [(2, 3), (1, 3), (0, 3), (0, 2), (0, 1), (0, 0)],
        dtype=float,
    )
    median_means = np.array(
        [3 / 5, 13 / 25, 2 / 5, 3 / 20, 3 / 25, 1 / 10],
        dtype=float,
    )
    case_probabilities = np.array(
        [1024, 1280, 640, 160, 20, 1], dtype=float
    ) / 3125

    fresh_costs = []
    for counts, mean_median in zip(target_counts, median_means):
        # Cellwise regularized target-to-cost ratios for the exact one-step
        # inverse-probability KKT solution.
        ratios = (counts + np.array([2, 3]) * delta) / (
            np.array([2, 3]) * (1 + delta)
        )
        root = np.sqrt(ratios)
        raw = budget * root / np.dot(fit_bin_mass, root)

        # Production one-step cumulative-logit correction, solved to machine
        # precision so fitted cost remains exactly the configured budget.
        lo, hi = -60.0, 60.0
        for _ in range(200):
            shift = (lo + hi) / 2
            logits = np.log(raw) - np.log1p(-raw) + shift
            corrected = floor + (1 - floor) / (1 + np.exp(-logits))
            if np.dot(fit_bin_mass, corrected) < budget:
                lo = shift
            else:
                hi = shift
        corrected = floor + (1 - floor) / (
            1 + np.exp(-(np.log(raw) - np.log1p(-raw) + (lo + hi) / 2))
        )
        assert np.isclose(
            np.dot(fit_bin_mass, corrected), budget, atol=1e-13
        )
        fresh_costs.append(
            mean_median * corrected[0]
            + (1 - mean_median) * corrected[1]
        )

    fresh_expected_cost = float(
        np.dot(case_probabilities, np.asarray(fresh_costs))
    )
    assert np.isclose(fresh_expected_cost, 0.2012423238, atol=1e-10)
    assert fresh_expected_cost > budget
