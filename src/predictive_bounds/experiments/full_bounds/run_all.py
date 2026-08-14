"""Run construct, merge, figures, and tables for the full paper matrix."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

from src.predictive_bounds.experiments.full_bounds.config import (
    TARGET_MODELS,
    select_configs,
)
from src.predictive_bounds.experiments.full_bounds.make_tables import (
    render_latex_tables,
)
from src.predictive_bounds.experiments.full_bounds.summarize import (
    DEFAULT_OUTPUT_DIR,
    ROOT,
    generate_all_figures,
    load_comparison_data,
)


STAGES = ("construct", "merge", "figures", "tables")


def _run(command: list[str], *, dry_run: bool) -> None:
    print(shlex.join(command), flush=True)
    if dry_run:
        return
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def _base_command(config, args) -> list[str]:
    target_coverages = (
        ["0.90"] if config.bound_type == "lpb"
        else ["0.70", "0.80", "0.90"]
    )
    return [
        "--bound-type", config.bound_type,
        "--data-type", "real",
        "--dataset-name", config.dataset_name,
        "--dataset-setup", config.dataset_setup,
        "--cal-size", str(config.cal_size),
        "--tau-prior", str(config.tau_prior),
        "--budget-per-sample", str(config.budget_per_sample),
        "--m-upper-bound", str(config.m_upper_bound),
        "--seed-start", str(args.seed_start),
        "--seed-end", str(args.seed_end),
        "--device", args.device,
        "--experiment-suffix", args.suffix,
        "--method-suite", "unified_aht",
        "--dapro-n1-values", *[str(value) for value in args.dapro_n1_values],
        "--target-coverages", *target_coverages,
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dapro-n1-values", type=int, nargs="+", default=[200, 100, 50]
    )
    parser.add_argument("--suffix", default="full_bounds_v5_soft_upb_aht")
    parser.add_argument(
        "--quality", choices=["high", "low"], default="low"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument(
        "--table-output", type=Path,
        default=DEFAULT_OUTPUT_DIR / "full_bounds_tables.tex",
    )
    parser.add_argument(
        "--stage", action="append", choices=STAGES,
        help="Run only selected stages; repeat for multiple stages.",
    )
    parser.add_argument("--config", action="append", dest="configs")
    parser.add_argument(
        "--target-model",
        action="append",
        choices=[model.key for model in TARGET_MODELS],
    )
    parser.add_argument(
        "--bound-type",
        action="append",
        choices=["lpb", "upb"],
        help=(
            "Run only the requested bound type; repeat to request both. "
            "The dedicated UPB shell launcher passes --bound-type upb."
        ),
    )
    parser.add_argument(
        "--available-only",
        action="store_true",
        help="Run only configurations whose prediction cache exists locally.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.seed_end <= args.seed_start:
        raise SystemExit("seed-end must be greater than seed-start.")
    stages = tuple(args.stage or STAGES)
    configs = select_configs(
        ROOT,
        keys=set(args.configs or []),
        target_models=set(args.target_model or []),
        bound_types=set(args.bound_type or []),
        available_only=args.available_only,
    )
    if not configs:
        raise SystemExit("No configurations matched the requested filters.")

    completed_experiments = set()
    failed_experiments = []
    for config in configs:
        experiment_key = (
            config.dataset_name,
            config.dataset_setup,
            config.bound_type,
            config.budget_per_sample,
            config.tau_prior,
        )
        if experiment_key in completed_experiments:
            continue
        completed_experiments.add(experiment_key)
        common = _base_command(config, args)
        construct_ok = True
        if "construct" in stages:
            try:
                _run([
                    sys.executable,
                    "-m",
                    "src.predictive_bounds.construct_calibrated_bound",
                    *common,
                ], dry_run=args.dry_run)
            except subprocess.CalledProcessError as error:
                construct_ok = False
                failed_experiments.append((config.key, "construct", error.returncode))
                print(
                    f"WARNING: {config.key} construction failed; continuing.",
                    file=sys.stderr,
                )
        if "merge" in stages and construct_ok:
            try:
                _run([
                    sys.executable,
                    "-m",
                    "src.predictive_bounds.merge_bounds_results",
                    *common,
                ], dry_run=args.dry_run)
            except subprocess.CalledProcessError as error:
                failed_experiments.append((config.key, "merge", error.returncode))
                print(
                    f"WARNING: {config.key} merge failed; continuing.",
                    file=sys.stderr,
                )

    if args.dry_run:
        return
    if "figures" in stages or "tables" in stages:
        frame = load_comparison_data(
            configs, args.suffix, allow_missing=bool(failed_experiments)
        )
        if "figures" in stages:
            paths = generate_all_figures(
                frame,
                args.output_dir,
                args.quality,
            )
            for path in paths:
                print(
                    f"{path}: {path.stat().st_size / 1024:.1f} KiB",
                    flush=True,
                )
        if "tables" in stages:
            args.table_output.parent.mkdir(parents=True, exist_ok=True)
            args.table_output.write_text(
                render_latex_tables(frame), encoding="utf-8"
            )
            print(args.table_output.resolve(), flush=True)
    if failed_experiments:
        print("Completed with failed configurations:", file=sys.stderr)
        for key, stage, code in failed_experiments:
            print(f"  - {key}: {stage} exited {code}", file=sys.stderr)


if __name__ == "__main__":
    main()
