"""Construct bounds after corrupting only calibration judge outcomes."""

from __future__ import annotations

import argparse
import torch

from src.predictive_bounds.experiments.common.bounds import (
    make_bound_grid,
    run_bound_split,
)
from src.predictive_bounds.experiments.common.results import stable_experiment_name
from src.predictive_bounds.experiments.judge_noise.config import (
    DEFAULT_DATASET_NAME,
    DEFAULT_DATASET_SETUP,
    DEFAULT_NOISE_LEVELS,
)
from src.predictive_bounds.experiments.judge_noise.noise import corrupt_event_times
from src.predictive_bounds.utils.utils import setup_experiment_data, split_data


def noise_configurations(args) -> tuple[tuple[str, float, float], ...]:
    levels = tuple(args.noise_levels)
    configs = [("clean", 0.0, 0.0)] if args.include_clean else []
    if args.noise_mode in {"false_negative", "all"}:
        configs.extend(("false_negative", level, 0.0) for level in levels)
    if args.noise_mode in {"false_positive", "all"}:
        configs.extend(("false_positive", 0.0, level) for level in levels)
    if args.noise_mode in {"both", "all"}:
        configs.extend(("both", level, level) for level in levels)
    return tuple(configs)


def metadata(args, noise_type, fn_rate, fp_rate) -> dict:
    return {
        "experiment_type": "judge_noise",
        "dataset_name": args.dataset_name,
        "dataset_setup": args.dataset_setup,
        "bound_type": args.bound_type,
        "noise_type": noise_type,
        "false_negative_rate": fn_rate,
        "false_positive_rate": fp_rate,
        "cal_size": args.cal_size,
        "budget_per_sample": args.budget_per_sample,
        "tau_prior": args.tau_prior,
        "target_coverage": args.target_coverage,
        "m_upper_bound": args.m_upper_bound,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset-setup", default=DEFAULT_DATASET_SETUP)
    parser.add_argument("--bound-type", choices=["lpb", "upb"], default="lpb")
    parser.add_argument("--noise-mode", choices=["false_negative", "false_positive", "both", "all"], default="all")
    parser.add_argument("--noise-levels", type=float, nargs="+", default=list(DEFAULT_NOISE_LEVELS))
    parser.add_argument("--include-clean", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cal-size", type=int, default=3000)
    parser.add_argument("--budget-per-sample", type=float, default=20.0)
    parser.add_argument("--tau-prior", type=float, default=0.56)
    parser.add_argument("--target-coverage", type=float, default=0.90)
    parser.add_argument("--m-upper-bound", type=float, default=200.0)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-skip-existing", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.seed_end <= args.seed_start:
        raise ValueError("seed-end must be greater than seed-start.")
    if len(set(args.noise_levels)) != len(args.noise_levels):
        raise ValueError("noise-levels must be distinct.")
    device = args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu"
    grid = make_bound_grid(args.bound_type, device, target_coverage=args.target_coverage)
    (
        max_time, times, quantiles, probabilities, conditional_grid, test_size,
    ) = setup_experiment_data(
        args.cal_size, True, device, args.dataset_name, args.dataset_setup,
        grid.taus, args.m_upper_bound,
    )
    for seed in range(args.seed_start, args.seed_end):
        split = split_data(
            seed, args.cal_size, test_size, None, times, probabilities, quantiles
        )
        (_, _, t_cal, p_cal, q_cal, t_test, q_test, p_test, cal_idx, _) = split
        for noise_type, fn_rate, fp_rate in noise_configurations(args):
            result = corrupt_event_times(
                t_cal,
                max_time,
                false_negative_rate=fn_rate,
                false_positive_rate=fp_rate,
                seed=10_000_019 * seed + int(fn_rate * 10_000) * 101 + int(fp_rate * 10_000),
            )
            fn_eligible = t_cal.reshape(-1) <= max_time
            # Before FN corruption, rows with T=1 have no observed negative
            # turn available for a stand-alone false positive.
            fp_eligible_before_fn = t_cal.reshape(-1) > 1
            current_metadata = {
                **metadata(args, noise_type, fn_rate, fp_rate),
                "realized_false_negative_fraction": float(result.false_negative_rows.float().mean()),
                "realized_false_positive_fraction": float(result.false_positive_rows.float().mean()),
                "realized_false_negative_rate_among_eligible": (
                    float(result.false_negative_rows[fn_eligible].float().mean())
                    if bool(fn_eligible.any()) else 0.0
                ),
                "realized_false_positive_rate_among_initially_eligible": (
                    float(
                        result.false_positive_rows[
                            fp_eligible_before_fn
                        ].float().mean()
                    )
                    if bool(fp_eligible_before_fn.any()) else 0.0
                ),
                "test_labels_perturbed": 0,
            }
            name = stable_experiment_name("judge_noise", metadata(args, noise_type, fn_rate, fp_rate))
            run_bound_split(
                experiment_name=name,
                seed=seed,
                grid=grid,
                t_cal=result.event_times,
                quantile_cal=q_cal,
                probability_cal=p_cal,
                conditional_grid_cal=conditional_grid[cal_idx],
                t_test=t_test,
                quantile_test=q_test,
                probability_test=p_test,
                budget_per_sample=args.budget_per_sample,
                tau_prior=args.tau_prior,
                m_upper_bound=args.m_upper_bound,
                device=device,
                metadata=current_metadata,
                skip_existing=not args.no_skip_existing,
            )


if __name__ == "__main__":
    main()
