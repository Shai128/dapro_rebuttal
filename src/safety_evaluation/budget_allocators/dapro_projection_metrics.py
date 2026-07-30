import json

import torch


def _masked_values(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = values[mask]
    if selected.numel() == 0:
        raise ValueError("DAPRO projection diagnostics have no active test steps.")
    return selected


def _quantile(values: torch.Tensor, q: float) -> float:
    return torch.quantile(values.float(), q).item()


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float()
    y = y.float()
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = torch.sqrt(
        x_centered.square().sum() * y_centered.square().sum()
    )
    if denominator <= 0:
        return float("nan")
    return ((x_centered * y_centered).sum() / denominator).item()


def _mean_over_active_time(
        values: torch.Tensor,
        mask: torch.Tensor,
) -> torch.Tensor:
    counts = mask.sum(dim=0)
    sums = torch.where(mask, values, torch.zeros_like(values)).sum(dim=0)
    return torch.where(
        counts > 0,
        sums / counts.clamp_min(1),
        torch.full_like(sums, float("nan")),
    )


def compute_dapro_projection_metrics(
        projected_probabilities: torch.Tensor,
        oracle_probabilities: torch.Tensor,
        prior_q: torch.Tensor,
        event_times: torch.Tensor,
        val_budget_used: float,
        target_budget_avg: float,
        realized_test_budget: float,
        total_sample_count: int,
        budget_per_sample: float,
        epsilon: float = 1e-8,
) -> dict:
    """
    Compare the learned DAPRO projection with the full-data oracle policy.

    ``solver`` budget uses the strict mask optimized by solve_exact_fast.
    ``runtime`` budget uses the inclusive mask in adaptive_budget_allocation.
    Keeping both makes any indexing/implementation gap visible rather than
    attributing it to projection error.
    """
    if projected_probabilities.shape != oracle_probabilities.shape:
        raise ValueError(
            "Projected and oracle probability tensors must have identical shapes."
        )
    if projected_probabilities.ndim != 2:
        raise ValueError("DAPRO probability tensors must have shape [N, T].")
    n_test, horizon = projected_probabilities.shape
    if n_test == 0:
        raise ValueError("DAPRO projection diagnostics require test samples.")
    if len(prior_q) != n_test or len(event_times) != n_test:
        raise ValueError("prior_q/event_times do not match the test probabilities.")

    projected = projected_probabilities.float().clamp(epsilon, 1.0)
    oracle = oracle_probabilities.float().clamp(epsilon, 1.0)
    time = torch.arange(horizon, device=projected.device).unsqueeze(0)
    max_steps = torch.minimum(prior_q, event_times).long().clamp(0, horizon)
    solver_mask = time < max_steps.unsqueeze(1)
    runtime_mask = time < max_steps.unsqueeze(1)

    probability_error = projected - oracle
    probability_abs_error = probability_error.abs()
    active_error = _masked_values(probability_error, solver_mask)
    active_abs_error = active_error.abs()
    active_projected = _masked_values(projected, solver_mask)
    active_oracle = _masked_values(oracle, solver_mask)
    active_log_error = torch.log(active_projected) - torch.log(active_oracle)
    projected_for_cosine = torch.where(
        solver_mask,
        projected,
        torch.zeros_like(projected),
    )
    oracle_for_cosine = torch.where(
        solver_mask,
        oracle,
        torch.zeros_like(oracle),
    )
    cosine_denominator = (
        projected_for_cosine.norm(dim=1)
        * oracle_for_cosine.norm(dim=1)
    ).clamp_min(epsilon)
    cosine_per_sample = (
        projected_for_cosine * oracle_for_cosine
    ).sum(dim=1) / cosine_denominator

    projected_cumprod = torch.cumprod(projected, dim=1)
    oracle_cumprod = torch.cumprod(oracle, dim=1)
    cumprod_error = projected_cumprod - oracle_cumprod
    active_cumprod_error = _masked_values(cumprod_error, solver_mask)
    active_cumprod_abs_error = active_cumprod_error.abs()

    projected_solver_cost = torch.where(
        solver_mask,
        projected_cumprod,
        torch.zeros_like(projected_cumprod),
    ).sum(dim=1)
    oracle_solver_cost = torch.where(
        solver_mask,
        oracle_cumprod,
        torch.zeros_like(oracle_cumprod),
    ).sum(dim=1)
    projected_runtime_cost = torch.where(
        runtime_mask,
        projected_cumprod,
        torch.zeros_like(projected_cumprod),
    ).sum(dim=1)
    oracle_runtime_cost = torch.where(
        runtime_mask,
        oracle_cumprod,
        torch.zeros_like(oracle_cumprod),
    ).sum(dim=1)

    solver_cost_error = projected_solver_cost - oracle_solver_cost
    runtime_cost_error = projected_runtime_cost - oracle_runtime_cost
    projected_solver_mean = projected_solver_cost.mean()
    oracle_solver_mean = oracle_solver_cost.mean()
    projected_runtime_mean = projected_runtime_cost.mean()
    oracle_runtime_mean = oracle_runtime_cost.mean()

    projected_inclusion_probability = torch.where(
        runtime_mask,
        projected,
        torch.ones_like(projected),
    ).prod(dim=1)
    oracle_inclusion_probability = torch.where(
        runtime_mask,
        oracle,
        torch.ones_like(oracle),
    ).prod(dim=1)
    inclusion_log_ratio = (
        torch.log(projected_inclusion_probability.clamp_min(epsilon))
        - torch.log(oracle_inclusion_probability.clamp_min(epsilon))
    )

    target_total_budget = float(budget_per_sample * total_sample_count)
    projected_runtime_test_total = projected_runtime_cost.sum().item()
    oracle_runtime_test_total = oracle_runtime_cost.sum().item()
    projected_expected_total = val_budget_used + projected_runtime_test_total
    oracle_expected_total = val_budget_used + oracle_runtime_test_total
    realized_total = val_budget_used + realized_test_budget
    projection_expected_gap = projected_expected_total - target_total_budget
    sampling_gap = realized_test_budget - projected_runtime_test_total
    realized_gap = realized_total - target_total_budget

    probability_mae_over_time = _mean_over_active_time(
        probability_abs_error,
        solver_mask,
    )
    cumprod_mae_over_time = _mean_over_active_time(
        cumprod_error.abs(),
        solver_mask,
    )
    expected_continuation_error_over_time = torch.where(
        solver_mask,
        cumprod_error,
        torch.zeros_like(cumprod_error),
    ).mean(dim=0)
    active_counts_over_time = solver_mask.sum(dim=0)
    finite_probability_time_error = probability_mae_over_time[
        torch.isfinite(probability_mae_over_time)
    ]
    probability_mse = active_error.square().mean()

    return {
        "projection_probability_mae": active_abs_error.mean().item(),
        "projection_probability_mse": probability_mse.item(),
        "projection_probability_rmse": probability_mse.sqrt().item(),
        "projection_probability_bias": active_error.mean().item(),
        "projection_probability_max_abs_error": active_abs_error.max().item(),
        "projection_probability_p50_abs_error": _quantile(active_abs_error, 0.50),
        "projection_probability_p90_abs_error": _quantile(active_abs_error, 0.90),
        "projection_probability_p95_abs_error": _quantile(active_abs_error, 0.95),
        "projection_probability_p99_abs_error": _quantile(active_abs_error, 0.99),
        "projection_probability_pearson": _pearson(
            active_projected,
            active_oracle,
        ),
        "projection_probability_cosine_similarity": (
            cosine_per_sample.mean().item()
        ),
        "projection_log_probability_mae": active_log_error.abs().mean().item(),
        "projection_log_probability_bias": active_log_error.mean().item(),
        "projection_cumprod_mae": active_cumprod_abs_error.mean().item(),
        "projection_cumprod_rmse": (
            active_cumprod_error.square().mean().sqrt().item()
        ),
        "projection_cumprod_bias": active_cumprod_error.mean().item(),
        "projection_cumprod_max_abs_error": active_cumprod_abs_error.max().item(),
        "projection_cumprod_p95_abs_error": _quantile(
            active_cumprod_abs_error,
            0.95,
        ),
        "expected_sum_over_A_max_abs": (
            expected_continuation_error_over_time.abs().max().item()
        ),
        "expected_sum_over_A_max": (
            expected_continuation_error_over_time.abs().max().item()
        ),
        "expected_sum_over_A_sum": (
            expected_continuation_error_over_time.sum().item()
        ),
        "expected_sum_over_A_l1": (
            expected_continuation_error_over_time.abs().sum().item()
        ),
        "oracle_solver_budget_per_test_sample": oracle_solver_mean.item(),
        "projected_solver_budget_per_test_sample": projected_solver_mean.item(),
        "projection_solver_budget_bias_per_test_sample": (
            projected_solver_mean - oracle_solver_mean
        ).item(),
        "projection_solver_budget_mae_per_sample": (
            solver_cost_error.abs().mean().item()
        ),
        "projection_solver_budget_rmse_per_sample": (
            solver_cost_error.square().mean().sqrt().item()
        ),
        "projection_solver_budget_relative_bias": (
            (projected_solver_mean - oracle_solver_mean)
            / oracle_solver_mean.abs().clamp_min(epsilon)
        ).item(),
        "projection_solver_budget_identity_residual": (
            expected_continuation_error_over_time.sum()
            - (projected_solver_mean - oracle_solver_mean)
        ).item(),
        "oracle_runtime_budget_per_test_sample": oracle_runtime_mean.item(),
        "projected_runtime_budget_per_test_sample": projected_runtime_mean.item(),
        "projection_runtime_budget_bias_per_test_sample": (
            projected_runtime_mean - oracle_runtime_mean
        ).item(),
        "projection_runtime_budget_mae_per_sample": (
            runtime_cost_error.abs().mean().item()
        ),
        "target_test_budget_per_sample": float(target_budget_avg),
        "solver_to_runtime_budget_gap_per_test_sample": (
            projected_runtime_mean - projected_solver_mean
        ).item(),
        "projected_expected_total_budget": projected_expected_total,
        "oracle_expected_total_budget": oracle_expected_total,
        "realized_total_budget": realized_total,
        "target_total_budget": target_total_budget,
        "projected_expected_budget_gap": projection_expected_gap,
        "projected_expected_budget_relative_gap": (
            projection_expected_gap / max(abs(target_total_budget), epsilon)
        ),
        "oracle_expected_budget_gap": oracle_expected_total - target_total_budget,
        "oracle_expected_budget_relative_gap": (
            (oracle_expected_total - target_total_budget)
            / max(abs(target_total_budget), epsilon)
        ),
        "realized_budget_gap": realized_gap,
        "realized_budget_relative_gap": (
            realized_gap / max(abs(target_total_budget), epsilon)
        ),
        "budget_sampling_gap": sampling_gap,
        "budget_sampling_relative_gap": (
            sampling_gap / max(abs(target_total_budget), epsilon)
        ),
        "budget_gap_decomposition_residual": (
            realized_gap - projection_expected_gap - sampling_gap
        ),
        "projection_inclusion_probability_mae": (
            projected_inclusion_probability - oracle_inclusion_probability
        ).abs().mean().item(),
        "projection_inclusion_log_ratio_mae": (
            inclusion_log_ratio.abs().mean().item()
        ),
        "projection_inclusion_log_ratio_p95": _quantile(
            inclusion_log_ratio.abs(),
            0.95,
        ),
        "projected_mean_ipcw": (
            1.0 / projected_inclusion_probability.clamp_min(epsilon)
        ).mean().item(),
        "oracle_mean_ipcw": (
            1.0 / oracle_inclusion_probability.clamp_min(epsilon)
        ).mean().item(),
        "projection_ipcw_ratio": (
            (
                1.0 / projected_inclusion_probability.clamp_min(epsilon)
            ).mean()
            / (
                1.0 / oracle_inclusion_probability.clamp_min(epsilon)
            ).mean().clamp_min(epsilon)
        ).item(),
        "projection_probability_mae_over_time": json.dumps(
            probability_mae_over_time.detach().cpu().tolist()
        ),
        "projection_cumprod_mae_over_time": json.dumps(
            cumprod_mae_over_time.detach().cpu().tolist()
        ),
        "expected_sum_over_A_over_time": json.dumps(
            expected_continuation_error_over_time.detach().cpu().tolist()
        ),
        "projection_active_count_over_time": json.dumps(
            active_counts_over_time.detach().cpu().tolist()
        ),
        # Backward-compatible names used by the original diagnostic snippet.
        "mae_test": active_abs_error.mean().item(),
        "mse_test": probability_mse.item(),
        "mean_cos_sim": cosine_per_sample.mean().item(),
        "mae_over_time_test_mean": (
            finite_probability_time_error.mean().item()
        ),
        "mae_over_time_test_max": (
            finite_probability_time_error.max().item()
        ),
        "mae_over_time_test_std": (
            finite_probability_time_error.std(unbiased=False).item()
        ),
    }
