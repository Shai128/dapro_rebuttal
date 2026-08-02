"""Construct bounds under training-domain or attacker shift.

``train_calibration_test_shift`` trains the survival model on a source domain,
then calibrates and tests it on disjoint rows from one different target domain;
a domain is the pair (dataset name, dataset setup).
``attacker_shift`` trains and calibrates on the source attacker and tests on a
different attacker while holding target model and judge fixed.
"""

from __future__ import annotations

import argparse

import torch

from src.predictive_bounds.experiments.common.bounds import (
    make_bound_grid,
    run_bound_split,
)
from src.predictive_bounds.experiments.common.results import stable_experiment_name
from src.predictive_bounds.experiments.distribution_shift.config import (
    DEFAULT_BUDGET,
    DEFAULT_CALIBRATION_SETUP,
    DEFAULT_CAL_SIZE,
    DEFAULT_DATASET_NAME,
    DEFAULT_HORIZON,
    DEFAULT_MODEL_SETUP,
    DEFAULT_TARGET_COVERAGE,
    DEFAULT_TAU_PRIOR,
    DEFAULT_TEST_SETUP,
    DEFAULT_TEST_SIZE,
)
from src.predictive_bounds.experiments.distribution_shift.utils import (
    load_prediction_pool,
    select_shift_indices,
)


def experiment_metadata(args) -> dict:
    return {
        "experiment_type": args.shift_type,
        # Keep dataset_name for compatibility with the common plotting code.
        "dataset_name": args.test_dataset_name,
        "model_dataset_name": args.model_dataset_name,
        "calibration_dataset_name": args.calibration_dataset_name,
        "test_dataset_name": args.test_dataset_name,
        "model_dataset_setup": args.model_dataset_setup,
        "calibration_dataset_setup": args.calibration_dataset_setup,
        "test_dataset_setup": args.test_dataset_setup,
        "bound_type": args.bound_type,
        "cal_size": args.cal_size,
        "test_size": args.test_size,
        "budget_per_sample": args.budget_per_sample,
        "tau_prior": args.tau_prior,
        "target_coverage": args.target_coverage,
        "m_upper_bound": args.m_upper_bound,
    }


def validate_attacker_only_shift(source_setup: str, test_setup: str) -> None:
    """Require identical target/judge suffixes and distinct attacker prefixes."""
    marker = "_lm_target_"
    if not source_setup.startswith("attack_") or marker not in source_setup:
        raise ValueError(f"Cannot parse source attacker setup: {source_setup!r}")
    if not test_setup.startswith("attack_") or marker not in test_setup:
        raise ValueError(f"Cannot parse test attacker setup: {test_setup!r}")
    source_attacker, source_suffix = source_setup[len("attack_"):].split(marker, 1)
    test_attacker, test_suffix = test_setup[len("attack_"):].split(marker, 1)
    if source_attacker == test_attacker:
        raise ValueError("attacker_shift requires two distinct attackers.")
    if source_suffix != test_suffix:
        raise ValueError(
            "attacker_shift may change only the attacker; target model and "
            "judge suffixes must be identical."
        )


def validate_shift_design(args) -> None:
    """Validate which dataset/setup components each shift is allowed to change."""
    model_domain = (args.model_dataset_name, args.model_dataset_setup)
    calibration_domain = (
        args.calibration_dataset_name,
        args.calibration_dataset_setup,
    )
    test_domain = (args.test_dataset_name, args.test_dataset_setup)
    if args.shift_type == "train_calibration_test_shift":
        if model_domain == calibration_domain:
            raise ValueError(
                "Training-domain shift requires the model-training domain and "
                "calibration/test domain to differ."
            )
        if calibration_domain != test_domain:
            raise ValueError(
                "Training-domain shift uses one target domain for both calibration "
                "and test; their dataset names and setups must match."
            )
        return

    if not (
        args.model_dataset_name
        == args.calibration_dataset_name
        == args.test_dataset_name
    ):
        raise ValueError(
            "Attacker shift changes only the attacker setup, not the dataset."
        )
    if model_domain != calibration_domain:
        raise ValueError(
            "Attacker shift trains and calibrates on the same source setup."
        )
    if calibration_domain == test_domain:
        raise ValueError("Attacker shift requires a distinct test attacker.")
    validate_attacker_only_shift(
        args.calibration_dataset_setup, args.test_dataset_setup
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shift-type",
        choices=["train_calibration_test_shift", "attacker_shift"],
        default="train_calibration_test_shift",
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="Legacy shared dataset default; overridden by the three dataset-specific options.",
    )
    parser.add_argument("--model-dataset-name")
    parser.add_argument("--calibration-dataset-name")
    parser.add_argument("--test-dataset-name")
    parser.add_argument("--model-dataset-setup", default=DEFAULT_MODEL_SETUP)
    parser.add_argument(
        "--calibration-dataset-setup", default=DEFAULT_CALIBRATION_SETUP
    )
    parser.add_argument("--test-dataset-setup", default=DEFAULT_TEST_SETUP)
    parser.add_argument("--bound-type", choices=["lpb", "upb"], default="lpb")
    parser.add_argument("--cal-size", type=int, default=DEFAULT_CAL_SIZE)
    parser.add_argument("--test-size", type=int, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--budget-per-sample", type=float, default=DEFAULT_BUDGET)
    parser.add_argument("--tau-prior", type=float, default=DEFAULT_TAU_PRIOR)
    parser.add_argument(
        "--target-coverage", type=float, default=DEFAULT_TARGET_COVERAGE
    )
    parser.add_argument("--m-upper-bound", type=float, default=DEFAULT_HORIZON)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args(argv)
    args.model_dataset_name = args.model_dataset_name or args.dataset_name
    args.calibration_dataset_name = (
        args.calibration_dataset_name or args.dataset_name
    )
    args.test_dataset_name = args.test_dataset_name or args.dataset_name
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.seed_end <= args.seed_start:
        raise ValueError("seed-end must be greater than seed-start.")
    validate_shift_design(args)
    device = args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu"
    grid = make_bound_grid(
        args.bound_type, device, target_coverage=args.target_coverage
    )
    calibration_pool = load_prediction_pool(
        model_dataset_name=args.model_dataset_name,
        evaluation_dataset_name=args.calibration_dataset_name,
        model_dataset_setup=args.model_dataset_setup,
        evaluation_dataset_setup=args.calibration_dataset_setup,
        device=device,
        taus_range=grid.taus,
        m_upper_bound=args.m_upper_bound,
    )
    test_pool = (
        calibration_pool
        if (
            args.test_dataset_name == args.calibration_dataset_name
            and args.test_dataset_setup == args.calibration_dataset_setup
        )
        else load_prediction_pool(
            model_dataset_name=args.model_dataset_name,
            evaluation_dataset_name=args.test_dataset_name,
            model_dataset_setup=args.model_dataset_setup,
            evaluation_dataset_setup=args.test_dataset_setup,
            device=device,
            taus_range=grid.taus,
            m_upper_bound=args.m_upper_bound,
        )
    )
    _, cal_times, cal_quantiles, cal_probabilities, cal_grid, _ = calibration_pool
    _, test_times, test_quantiles, test_probabilities, _, _ = test_pool
    metadata = experiment_metadata(args)
    name = stable_experiment_name(args.shift_type, metadata)
    for seed in range(args.seed_start, args.seed_end):
        if calibration_pool is test_pool:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            permutation = torch.randperm(len(cal_times), generator=generator)
            if args.cal_size >= len(permutation):
                raise ValueError("cal-size must leave at least one disjoint test row.")
            cal_idx = permutation[:args.cal_size]
            remaining = permutation[args.cal_size:]
            test_count = len(remaining) if args.test_size == 0 else args.test_size
            if test_count > len(remaining):
                raise ValueError("test-size exceeds rows remaining after calibration.")
            test_idx = remaining[:test_count]
        else:
            cal_idx = select_shift_indices(len(cal_times), args.cal_size, seed)
            test_idx = select_shift_indices(
                len(test_times), args.test_size, 1_000_003 + seed
            )
        run_bound_split(
            experiment_name=name,
            seed=seed,
            grid=grid,
            t_cal=cal_times[cal_idx],
            quantile_cal=cal_quantiles[cal_idx],
            probability_cal=cal_probabilities[cal_idx],
            conditional_grid_cal=cal_grid[cal_idx],
            t_test=test_times[test_idx],
            quantile_test=test_quantiles[test_idx],
            probability_test=test_probabilities[test_idx],
            budget_per_sample=args.budget_per_sample,
            tau_prior=args.tau_prior,
            m_upper_bound=args.m_upper_bound,
            device=device,
            metadata=metadata,
            skip_existing=not args.no_skip_existing,
        )
    print(name)


if __name__ == "__main__":
    main()
