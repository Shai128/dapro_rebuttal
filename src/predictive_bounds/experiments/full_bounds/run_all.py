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
    calibration_names,
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
        "--dapro-n1-values", "200",
        "--definitive-dapro-margins", "1",
        "--calibration-names", ",".join(
            calibration_names(config.bound_type)
        ),
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--suffix", default="full_bounds_v1")
    parser.add_argument(
        "--quality", choices=["high", "low"], default="high"
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
        available_only=args.available_only,
    )
    if not configs:
        raise SystemExit("No configurations matched the requested filters.")

    for config in configs:
        common = _base_command(config, args)
        if "construct" in stages:
            _run([
                sys.executable,
                "-m",
                "src.predictive_bounds.construct_calibrated_bound",
                *common,
            ], dry_run=args.dry_run)
        if "merge" in stages:
            _run([
                sys.executable,
                "-m",
                "src.predictive_bounds.merge_bounds_results",
                *common,
            ], dry_run=args.dry_run)

    if args.dry_run:
        return
    if "figures" in stages or "tables" in stages:
        frame = load_comparison_data(configs, args.suffix)
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


if __name__ == "__main__":
    main()
