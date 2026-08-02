"""Validate and merge DAPRO projection-evaluation shards."""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.predictive_bounds.experiments.dapro_projection.utils import (
    get_dapro_projection_experiment_name,
    get_dapro_projection_metadata,
)


def merge_dapro_projection_results(
        experiments_name,
        seeds,
        dataset_name,
        dataset_setup,
        budget_per_sample,
        cal_size,
        tau_prior,
        m_upper_bound,
):
    temporary_root = (
        Path("results")
        / "tmp_dapro_projection_evaluation"
        / experiments_name
    )
    manifest_root = temporary_root / "_manifests"
    if not temporary_root.exists():
        raise FileNotFoundError(
            f"DAPRO projection result directory does not exist: "
            f"{temporary_root.resolve()}"
        )
    expected_metadata = get_dapro_projection_metadata(
        dataset_name,
        dataset_setup,
        budget_per_sample,
        cal_size,
        tau_prior,
        m_upper_bound,
    )

    all_frames = []
    expected_allocator_names = None
    for seed in range(seeds[0], seeds[1]):
        manifest_path = manifest_root / f"seed={seed}.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Missing completion manifest for seed {seed}: "
                f"{manifest_path.resolve()}."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatches = {
            key: (manifest.get(key), expected)
            for key, expected in expected_metadata.items()
            if manifest.get(key) != expected
        }
        if mismatches:
            raise ValueError(
                f"Manifest metadata mismatch in {manifest_path}: {mismatches}"
            )
        allocator_names = manifest.get("allocator_names", [])
        if not allocator_names:
            raise ValueError(f"No DAPRO configurations recorded in {manifest_path}.")
        if len(set(allocator_names)) != len(allocator_names):
            raise ValueError(
                f"Duplicate allocator names recorded in {manifest_path}."
            )
        if expected_allocator_names is None:
            expected_allocator_names = set(allocator_names)
        elif set(allocator_names) != expected_allocator_names:
            raise ValueError(
                f"Seed {seed} has a different DAPRO configuration set."
            )

        for allocator_name in allocator_names:
            csv_path = temporary_root / allocator_name / f"seed={seed}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(
                    f"Manifest lists {allocator_name} for seed {seed}, but "
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
            if len(frame) != 1:
                raise ValueError(
                    f"Expected one row in {csv_path.resolve()}, found {len(frame)}."
                )
            if frame.loc[0, "seed"] != seed:
                raise ValueError(f"Wrong seed in {csv_path.resolve()}.")
            if frame.loc[0, "allocator_name"] != allocator_name:
                raise ValueError(
                    f"Wrong allocator name in {csv_path.resolve()}."
                )
            for key, expected in expected_metadata.items():
                if key not in frame.columns or frame.loc[0, key] != expected:
                    raise ValueError(
                        f"Result metadata {key!r} does not match in "
                        f"{csv_path.resolve()}."
                    )
            if frame.loc[0, "projection_evaluation_enabled"] != 1:
                raise ValueError(
                    f"Projection evaluation was not enabled in "
                    f"{csv_path.resolve()}."
                )
            all_frames.append(frame)

    if not all_frames:
        raise ValueError("No DAPRO projection results were found to merge.")
    all_df = pd.concat(all_frames, ignore_index=True)
    all_df = all_df.sort_values(
        ["projection", "score", "n1", "seed"]
    ).reset_index(drop=True)
    output_dir = (
        Path("results")
        / "merged_dapro_projection_evaluation"
        / experiments_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "all_df.csv"
    all_df.to_csv(output_path, index=False)
    print(
        f"Merged {len(all_frames)} DAPRO projection rows into "
        f"{output_path.resolve()}"
    )
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge completed DAPRO projection-evaluation results."
    )
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=50)
    parser.add_argument("--data-type", choices=["real", "synthetic"], default="real")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-setup", required=True)
    parser.add_argument("--budget-per-sample", type=float, default=20.0)
    parser.add_argument("--cal-size", type=int, default=3000)
    parser.add_argument("--tau-prior", type=float, default=0.56)
    parser.add_argument("--m-upper-bound", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed_end <= args.seed_start:
        raise ValueError("seed-end must be greater than seed-start.")
    if args.cal_size <= 1:
        raise ValueError("cal-size must be greater than 1.")
    if args.budget_per_sample <= 0:
        raise ValueError("budget-per-sample must be positive.")
    if not 0 < args.tau_prior < 1:
        raise ValueError("tau-prior must be between 0 and 1.")
    is_real = args.data_type == "real"
    m_upper_bound = args.m_upper_bound
    if m_upper_bound is None:
        m_upper_bound = 200.0 if is_real else 20.0
    if m_upper_bound <= 0:
        raise ValueError("m-upper-bound must be positive.")
    experiments_name = get_dapro_projection_experiment_name(
        args.dataset_name,
        args.dataset_setup,
        args.budget_per_sample,
        args.cal_size,
        args.tau_prior,
        m_upper_bound,
    )
    print(f"Merging DAPRO projection experiment: {experiments_name}")
    merge_dapro_projection_results(
        experiments_name=experiments_name,
        seeds=(args.seed_start, args.seed_end),
        dataset_name=args.dataset_name,
        dataset_setup=args.dataset_setup,
        budget_per_sample=args.budget_per_sample,
        cal_size=args.cal_size,
        tau_prior=args.tau_prior,
        m_upper_bound=m_upper_bound,
    )


if __name__ == "__main__":
    main()
