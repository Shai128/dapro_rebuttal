"""Fifty-split audit for model-assisted Horvitz--Thompson metric estimation.

This is an analysis driver, not a production allocator.  It compares the
ordinary unsafe-event HT estimator with the design-unbiased augmented HT
estimator under initial-PMF schedules optimized for either raw event mass or
the augmented estimator's squared residual influence.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.diagnostics.dapro_binning_audit import load_setup
from src.predictive_bounds.budget_allocators.metric_optimal_allocator import (
    antitonic_pav_bases,
    initial_event_and_at_risk_probabilities,
    solve_common_scale,
)


def _conditionals_from_cumulative(cumulative: np.ndarray) -> np.ndarray:
    conditionals = cumulative.copy()
    conditionals[:, 1:] = np.divide(
        cumulative[:, 1:],
        cumulative[:, :-1],
        out=np.ones_like(cumulative[:, 1:]),
        where=cumulative[:, :-1] > 0,
    )
    return np.clip(conditionals, 0.0, 1.0)


def _simulate_endpoint_observation(
    cumulative: np.ndarray,
    lengths: np.ndarray,
    uniforms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    conditionals = _conditionals_from_cumulative(cumulative)
    acquired = np.cumprod(uniforms < conditionals, axis=1, dtype=np.int8)
    endpoint_observed = acquired[np.arange(len(lengths)), lengths - 1].astype(bool)
    active = np.arange(cumulative.shape[1])[None, :] < lengths[:, None]
    realized_cost = (acquired * active).sum(axis=1)
    return endpoint_observed, realized_cost, acquired


def _prediction_before_each_interaction(
    grid,
    width: int,
) -> np.ndarray:
    """Return model P(event by width | history before each interaction)."""
    prediction = np.empty((len(grid), width), dtype=np.float64)
    for step in range(width):
        pmf = grid[:, step, :]
        valid = pmf[:, step:].sum(dim=1).clamp_min(
            np.finfo(np.float32).tiny
        )
        event = pmf[:, step:width].sum(dim=1)
        prediction[:, step] = (event / valid).to(
            dtype=pmf.dtype
        ).cpu().numpy().astype(np.float64)
    return np.clip(prediction, 0.0, 1.0)


def _sequential_augmented_estimate_and_variance(
    *,
    target: np.ndarray,
    lengths: np.ndarray,
    prediction_before: np.ndarray,
    cumulative: np.ndarray,
    acquired: np.ndarray,
) -> tuple[float, float]:
    """Evaluate sequential AHT and its exact fixed-row design variance."""
    n, width = cumulative.shape
    active = np.arange(width)[None, :] < lengths[:, None]
    after = prediction_before.copy()
    after[:, :-1] = prediction_before[:, 1:]
    endpoint = np.arange(width)[None, :] == (lengths - 1)[:, None]
    after = np.where(endpoint, target[:, None], after)
    increments = (after - prediction_before) * active
    estimate_per_row = (
        prediction_before[:, 0]
        + np.sum(acquired * increments / cumulative, axis=1)
    )

    previous_reciprocal = np.ones_like(cumulative)
    previous_reciprocal[:, 1:] = 1.0 / cumulative[:, :-1]
    reciprocal_increment = 1.0 / cumulative - previous_reciprocal
    residual_before = target[:, None] - prediction_before
    row_variance = np.sum(
        reciprocal_increment * residual_before**2 * active,
        axis=1,
    )
    return float(estimate_per_row.mean()), float(row_variance.mean() / n)


def _objective_masses(
    kind: str,
    event: np.ndarray,
    model_event_probability: np.ndarray,
) -> np.ndarray:
    if kind == "event":
        return event
    if kind != "residual":
        raise ValueError(f"Unknown objective {kind!r}.")
    residual_event = event * (1.0 - model_event_probability[:, None]) ** 2
    safe_tail = np.clip(1.0 - event.sum(axis=1), 0.0, 1.0)
    # A safe outcome is only established after reaching the benchmark horizon.
    residual_event[:, -1] += safe_tail * model_event_probability**2
    return residual_event


def _initial_pmf_information_gain_masses(
    event: np.ndarray,
    at_risk: np.ndarray,
) -> np.ndarray:
    """Expected squared target-prediction update from observing each turn."""
    suffix_event = np.flip(
        np.cumsum(np.flip(event, axis=1), axis=1), axis=1
    )
    model_before = np.divide(
        suffix_event,
        at_risk,
        out=np.zeros_like(event),
        where=at_risk > 0,
    )
    survive_after = np.maximum(at_risk - event, 0.0)
    suffix_after = np.maximum(suffix_event - event, 0.0)
    model_after_survival = np.divide(
        suffix_after,
        survive_after,
        out=np.zeros_like(event),
        where=survive_after > 0,
    )
    return (
        event * (1.0 - model_before) ** 2
        + survive_after * (model_after_survival - model_before) ** 2
    )


def _common_scale_to_true_cost(
    bases: np.ndarray,
    lengths: np.ndarray,
    budget: float,
    floor: float,
) -> tuple[np.ndarray, float]:
    """Offline scalar diagnostic using latent event-stopped lengths."""
    def cumulative(scale: float) -> np.ndarray:
        return np.clip(scale * bases, floor, 1.0)

    def cost(scale: float) -> float:
        candidate = cumulative(scale)
        return float(np.mean([
            candidate[row, : lengths[row]].sum()
            for row in range(len(lengths))
        ]))

    low, high = 0.0, 1.0
    while cost(high) < budget and high < 1e12:
        high *= 2.0
    for _ in range(80):
        middle = 0.5 * (low + high)
        if cost(middle) <= budget:
            low = middle
        else:
            high = middle
    return cumulative(low), float(low)


def run_setup(
    setup_key: str,
    *,
    budget: float,
    width: int,
    seed_start: int,
    seed_end: int,
    cal_size: int,
    scale_mode: str,
) -> pd.DataFrame:
    grid, times_tensor, dataset, data_setup = load_setup(setup_key)
    times = times_tensor.numpy().astype(np.int64)
    event, at_risk = initial_event_and_at_risk_probabilities(grid, width)
    model_event_probability = event.sum(axis=1)
    prediction_before = _prediction_before_each_interaction(grid, width)
    target = (times <= width).astype(np.float64)
    lengths = np.minimum(times, width)
    full_truth = float(target.mean())
    floor = 1.0 / width

    bases = {}
    for objective in ("event", "residual", "information_gain"):
        masses = (
            _initial_pmf_information_gain_masses(event, at_risk)
            if objective == "information_gain"
            else _objective_masses(objective, event, model_event_probability)
        )
        bases[objective], _ = antitonic_pav_bases(masses, at_risk)

    rows: list[dict[str, float | int | str]] = []
    for seed in range(seed_start, seed_end):
        np.random.seed(seed)
        permutation = np.random.permutation(len(times))
        cal_idx = permutation[:cal_size]
        split_target = target[cal_idx]
        split_model = model_event_probability[cal_idx]
        split_lengths = lengths[cal_idx]
        uniforms = np.random.default_rng(seed).random((cal_size, width))

        for objective in ("event", "residual", "information_gain"):
            if scale_mode == "model":
                cumulative, scale, boundary = solve_common_scale(
                    bases[objective][cal_idx],
                    at_risk[cal_idx],
                    budget,
                    floor,
                )
            elif scale_mode == "true_cost_oracle":
                cumulative, scale = _common_scale_to_true_cost(
                    bases[objective][cal_idx],
                    split_lengths,
                    budget,
                    floor,
                )
                boundary = "offline_true_cost_oracle"
            else:
                raise ValueError(f"Unknown scale mode {scale_mode!r}.")
            propensity = cumulative[
                np.arange(cal_size), split_lengths - 1
            ]
            observed, realized_cost, acquired = _simulate_endpoint_observation(
                cumulative,
                split_lengths,
                uniforms,
            )
            ordinary = np.mean(observed * split_target / propensity)
            augmented = np.mean(
                split_model
                + observed * (split_target - split_model) / propensity
            )
            ordinary_conditional_variance = np.mean(
                split_target * (1.0 / propensity - 1.0)
            ) / cal_size
            augmented_conditional_variance = np.mean(
                (split_target - split_model) ** 2
                * (1.0 / propensity - 1.0)
            ) / cal_size
            sequential_augmented, sequential_augmented_variance = (
                _sequential_augmented_estimate_and_variance(
                    target=split_target,
                    lengths=split_lengths,
                    prediction_before=prediction_before[cal_idx],
                    cumulative=cumulative,
                    acquired=acquired,
                )
            )
            expected_cost = np.mean([
                cumulative[row, : split_lengths[row]].sum()
                for row in range(cal_size)
            ])
            predicted_cost = float(np.mean(
                np.sum(at_risk[cal_idx] * cumulative, axis=1)
            ))
            rows.append({
                "setup_key": setup_key,
                "dataset": dataset,
                "data_setup": data_setup,
                "seed": seed,
                "objective": objective,
                "budget": budget,
                "width": width,
                "cal_size": cal_size,
                "scale_mode": scale_mode,
                "full_union_truth_pct": 100.0 * full_truth,
                "split_full_metric_pct": 100.0 * float(split_target.mean()),
                "model_plugin_metric_pct": 100.0 * float(split_model.mean()),
                "ordinary_ht_metric_pct": 100.0 * float(ordinary),
                "augmented_ht_metric_pct": 100.0 * float(augmented),
                "sequential_augmented_ht_metric_pct": (
                    100.0 * sequential_augmented
                ),
                "ordinary_ht_conditional_variance_pp2": (
                    10_000.0 * ordinary_conditional_variance
                ),
                "augmented_ht_conditional_variance_pp2": (
                    10_000.0 * augmented_conditional_variance
                ),
                "sequential_augmented_ht_conditional_variance_pp2": (
                    10_000.0 * sequential_augmented_variance
                ),
                "predicted_expected_cost_per_sample": predicted_cost,
                "true_expected_cost_per_sample": float(expected_cost),
                "realized_cost_per_sample": float(realized_cost.mean()),
                "common_scale": float(scale),
                "budget_boundary": boundary,
                "mean_endpoint_propensity": float(propensity.mean()),
                "minimum_endpoint_propensity": float(propensity.min()),
                "observed_endpoint_fraction": float(observed.mean()),
            })
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (setup, objective, scale_mode), group in frame.groupby(
        ["setup_key", "objective", "scale_mode"]
    ):
        for estimator, column, variance_column in [
            (
                "ordinary_ht",
                "ordinary_ht_metric_pct",
                "ordinary_ht_conditional_variance_pp2",
            ),
            (
                "augmented_ht",
                "augmented_ht_metric_pct",
                "augmented_ht_conditional_variance_pp2",
            ),
            (
                "sequential_augmented_ht",
                "sequential_augmented_ht_metric_pct",
                "sequential_augmented_ht_conditional_variance_pp2",
            ),
        ]:
            values = group[column].to_numpy(dtype=np.float64)
            truth = group["full_union_truth_pct"].iloc[0]
            rows.append({
                "setup_key": setup,
                "objective": objective,
                "scale_mode": scale_mode,
                "estimator": estimator,
                "n_splits": len(group),
                "mean_metric_pct": float(values.mean()),
                "mean_error_from_full_truth_pp": float(values.mean() - truth),
                "variance_across_splits_pp2": float(values.var(ddof=1)),
                "mean_exact_conditional_variance_pp2": float(
                    group[variance_column].mean()
                ),
                "split_full_variance_pp2": float(
                    group["split_full_metric_pct"].var(ddof=1)
                ),
                "law_total_expected_variance_pp2": float(
                    group[variance_column].mean()
                    + group["split_full_metric_pct"].var(ddof=1)
                ),
                "true_expected_cost_per_sample": float(
                    group["true_expected_cost_per_sample"].mean()
                ),
                "realized_cost_per_sample": float(
                    group["realized_cost_per_sample"].mean()
                ),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--setups", nargs="+", default=["toxicity_qwen", "red_qwen"]
    )
    parser.add_argument("--budget", type=float, default=20.0)
    parser.add_argument("--width", type=int, default=200)
    parser.add_argument("--cal-size", type=int, default=3000)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=50)
    parser.add_argument(
        "--scale-mode",
        choices=["model", "true_cost_oracle"],
        default="model",
        help=(
            "Use model-predicted cost (deployable assumption) or an offline "
            "latent-cost scalar diagnostic."
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/dapro_binning_audit"),
    )
    args = parser.parse_args()
    frames = [
        run_setup(
            setup,
            budget=args.budget,
            width=args.width,
            seed_start=args.seed_start,
            seed_end=args.seed_end,
            cal_size=args.cal_size,
            scale_mode=args.scale_mode,
        )
        for setup in args.setups
    ]
    result = pd.concat(frames, ignore_index=True)
    summary = summarize(result)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "metric_augmented_ht_50split.csv", index=False)
    summary.to_csv(
        args.output_dir / "metric_augmented_ht_50split_summary.csv", index=False
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
