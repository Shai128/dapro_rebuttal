"""Validate and merge the sharded AutoIF cross-class LPB results."""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.predictive_bounds.experiments.autoif_cross_class.utils import (
    DEFAULT_AUTOIF_DATASET_SETUP,
    get_autoif_cross_class_experiment_name,
    get_autoif_cross_class_metadata,
)


def merge_autoif_cross_class_results(
        experiments_name,
        seeds,
        bound_type,
        dataset_name,
        dataset_setup,
        calibration_class,
        test_class,
        cal_size,
        test_size,
):
    temporary_kind = (
        "tmp_calibration_results"
        if bound_type == "lpb"
        else "tmp_upb_calibration_results"
    )
    merged_kind = (
        "merged_calibration_dfs"
        if bound_type == "lpb"
        else "merged_upb_calibration_dfs"
    )
    temporary_root = Path("results") / temporary_kind / experiments_name
    manifest_root = temporary_root / "_manifests"
    if not temporary_root.exists():
        raise FileNotFoundError(
            f"AutoIF cross-class result directory does not exist: "
            f"{temporary_root.resolve()}"
        )

    expected_metadata = get_autoif_cross_class_metadata(
        dataset_name,
        dataset_setup,
        calibration_class,
        test_class,
    )
    expected_manifest_metadata = {
        **expected_metadata,
        "bound_type": bound_type,
        "cal_size": int(cal_size),
    }
    if test_size is not None:
        expected_manifest_metadata["test_size"] = int(test_size)

    all_frames = []
    for seed in range(seeds[0], seeds[1]):
        manifest_path = manifest_root / f"seed={seed}.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Missing completion manifest for seed {seed}: "
                f"{manifest_path.resolve()}. Rerun "
                "src.predictive_bounds.experiments.autoif_cross_class."
                "construct_calibrated_bound for this seed."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatches = {
            key: (manifest.get(key), expected)
            for key, expected in expected_manifest_metadata.items()
            if manifest.get(key) != expected
        }
        if mismatches:
            raise ValueError(
                f"Manifest metadata mismatch in {manifest_path}: {mismatches}"
            )
        if int(manifest.get("test_size", 0)) <= 0:
            raise ValueError(
                f"Manifest contains an invalid test size: {manifest_path.resolve()}"
            )

        calibration_names = manifest.get("calibration_names", [])
        if not calibration_names:
            raise ValueError(f"No calibration methods recorded in {manifest_path}.")
        for calibration_name in calibration_names:
            csv_path = temporary_root / calibration_name / f"seed={seed}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(
                    f"Manifest lists {calibration_name} for seed {seed}, but "
                    f"{csv_path.resolve()} does not exist."
                )
            frame = pd.read_csv(csv_path)
            frame = frame.drop(
                columns=[
                    column
                    for column in frame.columns
                    if column.startswith("Unnamed:")
                ],
                errors="ignore",
            )
            if frame.empty:
                raise ValueError(f"Result file is empty: {csv_path.resolve()}")
            if not frame["seed"].eq(seed).all():
                raise ValueError(
                    f"Result file contains the wrong seed: {csv_path.resolve()}"
                )
            if not frame["calibration_name"].eq(calibration_name).all():
                raise ValueError(
                    "Result file contains the wrong calibration name: "
                    f"{csv_path.resolve()}"
                )
            for key, expected in expected_metadata.items():
                if key not in frame.columns or not frame[key].eq(expected).all():
                    raise ValueError(
                        f"Result metadata {key!r} does not match in "
                        f"{csv_path.resolve()}."
                    )
            all_frames.append(frame)

    if not all_frames:
        raise ValueError("No AutoIF cross-class bound results were found to merge.")
    all_df = pd.concat(all_frames, ignore_index=True)
    sort_columns = [
        column
        for column in ["calibration_name", "seed", "target_coverage"]
        if column in all_df.columns
    ]
    if sort_columns:
        all_df = all_df.sort_values(sort_columns).reset_index(drop=True)

    merged_dir = Path("results") / merged_kind / experiments_name
    merged_dir.mkdir(parents=True, exist_ok=True)
    results_path = merged_dir / "all_df.csv"
    all_df.to_csv(results_path, index=False)
    print(
        f"Merged {len(all_frames)} method/seed files ({len(all_df)} rows) into "
        f"{results_path.resolve()}"
    )
    return results_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge completed AutoIF cross-class calibrated-bound results."
    )
    parser.add_argument("--bound-type", choices=["lpb", "upb"], default="lpb")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=50)
    parser.add_argument("--dataset-name", default="dataset_autoif")
    parser.add_argument("--dataset-setup", default=DEFAULT_AUTOIF_DATASET_SETUP)
    parser.add_argument(
        "--calibration-class",
        default="Programming & Technology",
    )
    parser.add_argument(
        "--test-class",
        default="Marketing & Social Media",
    )
    parser.add_argument("--budget-per-sample", type=float, default=20.0)
    parser.add_argument("--cal-size", type=int, default=800)
    parser.add_argument(
        "--test-size",
        type=int,
        default=100,
        help="Must match construction; use 0 when construction used all test rows.",
    )
    parser.add_argument("--tau-prior", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--m-upper-bound", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed_end <= args.seed_start:
        raise ValueError("seed-end must be greater than seed-start.")
    if args.cal_size <= 0:
        raise ValueError("cal-size must be positive.")
    if args.test_size < 0:
        raise ValueError("test-size cannot be negative.")
    if args.budget_per_sample <= 0:
        raise ValueError("budget-per-sample must be positive.")
    if args.calibration_class == args.test_class:
        raise ValueError("Calibration and test classes must be different.")
    if args.gamma is not None and args.m_upper_bound is not None:
        raise ValueError("Specify at most one of --gamma and --m-upper-bound.")

    if args.m_upper_bound is not None:
        m_upper_bound = args.m_upper_bound
    elif args.gamma is not None:
        m_upper_bound = args.gamma * args.budget_per_sample
    else:
        m_upper_bound = 200.0
    if m_upper_bound <= 0:
        raise ValueError("The upper bound must be positive.")

    if args.tau_prior is None:
        tau_prior = 0.56 if args.bound_type == "lpb" else 0.98
    else:
        tau_prior = args.tau_prior
    gamma = m_upper_bound / args.budget_per_sample
    test_size = None if args.test_size == 0 else args.test_size
    experiments_name = get_autoif_cross_class_experiment_name(
        args.dataset_setup,
        args.calibration_class,
        args.test_class,
        args.budget_per_sample,
        args.cal_size,
        test_size,
        tau_prior,
        gamma,
    )
    print(f"Merging AutoIF cross-class experiment: {experiments_name}")
    merge_autoif_cross_class_results(
        experiments_name=experiments_name,
        seeds=(args.seed_start, args.seed_end),
        bound_type=args.bound_type,
        dataset_name=args.dataset_name,
        dataset_setup=args.dataset_setup,
        calibration_class=args.calibration_class,
        test_class=args.test_class,
        cal_size=args.cal_size,
        test_size=test_size,
    )


if __name__ == "__main__":
    main()
