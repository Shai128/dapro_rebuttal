"""Run selected ablations over the manuscript dataset/target-model matrix.

The matrix is imported from ``full_bounds.config`` so budgets, coverages,
judges, and display configurations cannot drift between the main comparison
and the ablations.  Use ``--available-only`` for a local cache subset and
``--dry-run`` to inspect commands without executing them.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from src.predictive_bounds.experiments.full_bounds.config import (
    TARGET_MODELS,
    select_configs,
)
from src.predictive_bounds.experiments.full_bounds.summarize import ROOT


def _run(command: list[str], dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def _base(config, args) -> list[str]:
    return [
        "--dataset-name", config.dataset_name,
        "--dataset-setup", config.dataset_setup,
        "--bound-type", config.bound_type,
        "--cal-size", str(config.cal_size),
        "--budget-per-sample", str(config.budget_per_sample),
        "--tau-prior", str(config.tau_prior),
        "--target-coverage", str(config.target_coverage),
        "--m-upper-bound", str(config.m_upper_bound),
        "--seed-start", str(args.seed_start),
        "--seed-end", str(args.seed_end),
        "--device", args.device,
    ]


def _limited_cache(config, root: Path) -> Path:
    return (
        root / "alg_playground_model" / "limited_training_budget"
        / config.dataset_name / config.dataset_setup
        / "probability_est_cal_test_fraction_0p10.pt"
    )


def commands_for_config(config, args) -> list[list[str]]:
    python = sys.executable
    common = _base(config, args)
    output = Path("figures/ablations") / args.experiment / config.key
    if args.experiment == "autoif_cross_class":
        if config.dataset_name != "dataset_autoif":
            return []
        return [[
            python, "-m",
            "src.predictive_bounds.experiments.autoif_cross_class.run_all",
            "--dataset-name", config.dataset_name,
            "--dataset-setup", config.dataset_setup,
            "--bound-type", config.bound_type,
            "--autoif-data-path", args.autoif_data_path,
            "--classifications-path", args.classifications_path,
            "--calibration-class", args.calibration_class,
            "--test-class", args.test_class,
            "--cal-size", str(args.cross_class_cal_size),
            "--test-size", str(args.cross_class_test_size),
            "--budget-per-sample", str(config.budget_per_sample),
            "--tau-prior", str(config.tau_prior),
            "--m-upper-bound", str(config.m_upper_bound),
            "--allocations", "one", "--max-workers", "1",
            "--paper-methods-only",
            "--seed-start", str(args.seed_start),
            "--seed-end", str(args.seed_end),
            "--device", args.device,
            "--output-dir", str(output),
        ]]
    if args.experiment == "allocation_focus":
        return [[
            python, "-m", "src.predictive_bounds.experiments.allocation_focus.run_all",
            *common, "--output-dir", str(output),
        ]]
    if args.experiment == "judge_noise":
        return [[
            python, "-m", "src.predictive_bounds.experiments.judge_noise.run_all",
            *common, "--noise-mode", "all", "--noise-levels",
            *[str(level) for level in args.noise_levels],
            "--output-dir", str(output),
        ]]
    if args.experiment == "variance_components":
        if config.bound_type != "lpb":
            return []
        # The variance CLI is LPB-specific and does not accept bound/coverage flags.
        filtered = []
        skip_next = False
        for index, value in enumerate(common):
            if skip_next:
                skip_next = False
                continue
            if value in {"--bound-type", "--target-coverage", "--seed-start", "--seed-end"}:
                skip_next = True
                continue
            filtered.append(value)
        variance_args = [
            *filtered,
            "--n1-values", *[str(value) for value in args.n1_values],
            "--replicates", str(args.variance_replicates),
            "--crossed-groups", str(args.crossed_groups),
            "--suffix-prefix", f"variance_components_{config.key}",
            "--output-dir", str(output),
        ]
        return [[
            python, "-m", f"src.predictive_bounds.experiments.variance_components.{module}",
            *variance_args,
        ] for module in ("run", "merge_results", "summarize")]
    if args.experiment == "limited_training_budget":
        limited = _limited_cache(config, ROOT)
        full = config.model_cache_dir(ROOT) / "probability_est_cal_test.pt"
        train = [
            python, "-m", "src.train_model.train_model",
            "--dataset-name", config.dataset_name,
            "--dataset-setup", config.dataset_setup,
            "--acquisition-strategy", "naive",
            "--uniform-training-budget-fraction", "0.10",
            "--full-budget-per-sample", str(config.m_upper_bound),
            "--prediction-cache-output", str(limited),
            "--total-budget", "0", "--seed-start", "0", "--seed-end", "1",
            "--device", args.device,
        ]
        compare = [
            python, "-m", "src.predictive_bounds.experiments.limited_training_budget.run_comparison",
            "--dataset-name", config.dataset_name,
            "--dataset-setup", config.dataset_setup,
            "--bound-type", config.bound_type,
            "--limited-prediction-cache", str(limited),
            "--full-prediction-cache", str(full),
            "--cal-size", str(config.cal_size),
            "--budget-per-sample", str(config.budget_per_sample),
            "--tau-prior", str(config.tau_prior),
            "--target-coverage", str(config.target_coverage),
            "--m-upper-bound", str(config.m_upper_bound),
            "--seed-start", str(args.seed_start), "--seed-end", str(args.seed_end),
            "--device", args.device, "--output-dir", str(output),
        ]
        return ([train] if args.train_limited_models else []) + [compare]
    raise ValueError(f"Unknown experiment {args.experiment!r}.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=[
            "autoif_cross_class",
            "allocation_focus",
            "judge_noise",
            "variance_components",
            "limited_training_budget",
        ],
        required=True,
    )
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--available-only", action="store_true")
    parser.add_argument("--config", action="append", dest="configs")
    parser.add_argument("--target-model", action="append", choices=[model.key for model in TARGET_MODELS])
    parser.add_argument("--noise-levels", type=float, nargs="+", default=[0.01, 0.05, 0.10, 0.20])
    parser.add_argument(
        "--autoif-data-path",
        default="src/multi_turn_data_generation/data/autoif_helper_dataset.csv",
    )
    parser.add_argument(
        "--classifications-path",
        default="src/multi_turn_data_generation/data/classified_instructions.csv",
    )
    parser.add_argument(
        "--calibration-class", default="Programming & Technology"
    )
    parser.add_argument("--test-class", default="Marketing & Social Media")
    parser.add_argument("--cross-class-cal-size", type=int, default=800)
    parser.add_argument("--cross-class-test-size", type=int, default=100)
    parser.add_argument("--n1-values", type=int, nargs="+", default=[100, 200, 400, 800])
    parser.add_argument("--variance-replicates", type=int, default=50)
    parser.add_argument("--crossed-groups", type=int, default=10)
    parser.add_argument("--train-limited-models", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    configs = select_configs(
        ROOT,
        keys=set(args.configs or []),
        target_models=set(args.target_model or []),
        available_only=args.available_only,
    )
    # LPB and UPB AutoIF share one model cache. Avoid retraining it twice while
    # still running both bound comparisons.
    trained: set[tuple[str, str]] = set()
    for config in configs:
        commands = commands_for_config(config, args)
        for command in commands:
            is_train = (
                len(command) > 2
                and command[2] == "src.train_model.train_model"
            )
            cache_key = (config.dataset_name, config.dataset_setup)
            if is_train and cache_key in trained:
                continue
            _run(command, args.dry_run)
            if is_train:
                trained.add(cache_key)


if __name__ == "__main__":
    main()
