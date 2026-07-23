import argparse
import json
from pathlib import Path

import pandas as pd

from src.safety_evaluation.cross_setup_utils import get_cross_setup_experiment_name


def merge_cross_setup_results(
        experiments_name: str,
        seeds,
        bound_type: str,
        dataset_name: str,
        model_dataset_setup: str,
        evaluation_dataset_setup: str,
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
            f"Cross-setup result directory does not exist: {temporary_root.resolve()}"
        )

    all_frames = []
    for seed in range(seeds[0], seeds[1]):
        manifest_path = manifest_root / f"seed={seed}.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Missing completion manifest for seed {seed}: {manifest_path.resolve()}. "
                "Rerun construct_cross_setup_calibrated_bound for this seed."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_metadata = {
            "experiment_type": "cross_setup",
            "dataset_name": dataset_name,
            "model_dataset_setup": model_dataset_setup,
            "evaluation_dataset_setup": evaluation_dataset_setup,
            "bound_type": bound_type,
        }
        mismatches = {
            key: (manifest.get(key), expected)
            for key, expected in expected_metadata.items()
            if manifest.get(key) != expected
        }
        if mismatches:
            raise ValueError(
                f"Manifest metadata mismatch in {manifest_path}: {mismatches}"
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
                columns=[column for column in frame.columns if column.startswith("Unnamed:")],
                errors="ignore",
            )
            if frame.empty:
                raise ValueError(f"Result file is empty: {csv_path.resolve()}")
            if not frame["seed"].eq(seed).all():
                raise ValueError(f"Result file contains the wrong seed: {csv_path.resolve()}")
            if not frame["calibration_name"].eq(calibration_name).all():
                raise ValueError(
                    f"Result file contains the wrong calibration name: {csv_path.resolve()}"
                )
            for key, expected in expected_metadata.items():
                if key == "bound_type":
                    continue
                if key not in frame.columns or not frame[key].eq(expected).all():
                    raise ValueError(
                        f"Result metadata {key!r} does not match in {csv_path.resolve()}."
                    )
            all_frames.append(frame)

    if not all_frames:
        raise ValueError("No cross-setup bound results were found to merge.")

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
        description="Merge completed cross-setup calibrated-bound results."
    )
    parser.add_argument("--bound-type", choices=["lpb", "upb"], default="lpb")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=50)
    parser.add_argument("--data-type", choices=["real", "synthetic"], default="real")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--model-dataset-setup", required=True)
    parser.add_argument("--evaluation-dataset-setup", required=True)
    parser.add_argument("--budget-per-sample", type=float, default=20.0)
    parser.add_argument("--cal-size", type=int, default=3000)
    parser.add_argument("--tau-prior", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--m-upper-bound", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed_end <= args.seed_start:
        raise ValueError("seed-end must be greater than seed-start.")
    if args.budget_per_sample <= 0:
        raise ValueError("budget-per-sample must be positive.")
    if args.model_dataset_setup == args.evaluation_dataset_setup:
        raise ValueError(
            "model-dataset-setup and evaluation-dataset-setup must be different."
        )
    if args.gamma is not None and args.m_upper_bound is not None:
        raise ValueError("Specify at most one of --gamma and --m-upper-bound.")

    is_real = args.data_type == "real"
    default_upper_bound = 200.0 if is_real else 20.0
    if args.m_upper_bound is not None:
        m_upper_bound = args.m_upper_bound
    elif args.gamma is not None:
        m_upper_bound = args.gamma * args.budget_per_sample
    else:
        m_upper_bound = default_upper_bound
    gamma = m_upper_bound / args.budget_per_sample

    tau_prior = args.tau_prior
    if tau_prior is None:
        tau_prior = 0.56 if args.bound_type == "lpb" else 0.98
    experiments_name = get_cross_setup_experiment_name(
        args.dataset_name,
        args.model_dataset_setup,
        args.evaluation_dataset_setup,
        args.budget_per_sample,
        args.cal_size,
        tau_prior,
        gamma,
    )
    print(f"Merging cross-setup experiment: {experiments_name}")
    merge_cross_setup_results(
        experiments_name=experiments_name,
        seeds=(args.seed_start, args.seed_end),
        bound_type=args.bound_type,
        dataset_name=args.dataset_name,
        model_dataset_setup=args.model_dataset_setup,
        evaluation_dataset_setup=args.evaluation_dataset_setup,
    )


if __name__ == "__main__":
    main()
