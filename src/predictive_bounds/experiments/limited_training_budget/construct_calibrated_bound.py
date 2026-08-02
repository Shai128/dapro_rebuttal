"""Construct bounds from a prediction cache trained with limited labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.predictive_bounds.experiments.common.bounds import make_bound_grid, run_bound_split
from src.predictive_bounds.experiments.common.results import stable_experiment_name
from src.predictive_bounds.utils.utils import setup_experiment_data, split_data


def metadata(args):
    return {
        "experiment_type": "limited_training_budget",
        "dataset_name": args.dataset_name,
        "dataset_setup": args.dataset_setup,
        "training_budget_label": args.training_budget_label,
        "training_budget_fraction": args.training_budget_fraction,
        "prediction_cache": Path(args.prediction_cache).as_posix(),
        "bound_type": args.bound_type,
        "cal_size": args.cal_size,
        "budget_per_sample": args.budget_per_sample,
        "tau_prior": args.tau_prior,
        "target_coverage": args.target_coverage,
        "m_upper_bound": args.m_upper_bound,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default="dataset_toxicity")
    parser.add_argument("--dataset-setup", default=("attack_toxic_attack_qwen25_14b_instruct_lm_target_"
                                                     "qwen25_14b_instruct_judge_detoxify"))
    parser.add_argument("--prediction-cache", required=True)
    parser.add_argument("--training-budget-label", default="10% uniform training budget")
    parser.add_argument("--training-budget-fraction", type=float, default=0.10)
    parser.add_argument("--bound-type", choices=["lpb", "upb"], default="lpb")
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
    cache = Path(args.prediction_cache)
    if not cache.exists():
        raise FileNotFoundError(f"Limited-budget prediction cache is missing: {cache.resolve()}")
    device = args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu"
    grid = make_bound_grid(args.bound_type, device, target_coverage=args.target_coverage)
    data = setup_experiment_data(
        args.cal_size, True, device, args.dataset_name, args.dataset_setup,
        grid.taus, args.m_upper_bound, prediction_cache_path=str(cache),
    )
    _, times, quantiles, probabilities, conditional_grid, test_size = data
    current = metadata(args)
    name = stable_experiment_name("limited_training_budget", current)
    for seed in range(args.seed_start, args.seed_end):
        split = split_data(
            seed, args.cal_size, test_size, None, times, probabilities, quantiles
        )
        (_, _, t_cal, p_cal, q_cal, t_test, q_test, p_test, cal_idx, _) = split
        run_bound_split(
            experiment_name=name, seed=seed, grid=grid,
            t_cal=t_cal, quantile_cal=q_cal, probability_cal=p_cal,
            conditional_grid_cal=conditional_grid[cal_idx],
            t_test=t_test, quantile_test=q_test, probability_test=p_test,
            budget_per_sample=args.budget_per_sample, tau_prior=args.tau_prior,
            m_upper_bound=args.m_upper_bound, device=device, metadata=current,
            skip_existing=not args.no_skip_existing,
        )
    print(name)


if __name__ == "__main__":
    main()
