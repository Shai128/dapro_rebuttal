"""Merge and validate per-seed supplementary metric-estimation outputs."""

import os
import argparse
import traceback
import pandas as pd
import numpy as np
import torch
import tqdm
from concurrent.futures import as_completed, ThreadPoolExecutor

from src.predictive_bounds.utils.get_calibration_methods_utils import get_metric_allocators
from src.predictive_bounds.utils.utils import get_tmp_metric_calibration_result_path, get_merged_metric_calibration_result_path
from src.evaluation.estimate import REQUIRED_RESULT_COLUMNS, metric_experiment_name


def process_calibration(calibration, seed, experiments_name):
    """Process a single calibration and return all computed rows."""
    try:
        dir_path = get_tmp_metric_calibration_result_path(
            experiments_name, calibration.name)
        csv_path = os.path.join(dir_path, f"seed={seed}.csv")
        abs_path = os.path.abspath(csv_path)

        # Handle Windows long paths
        if os.name == 'nt' and not abs_path.startswith('\\\\?\\'):
            abs_path = f"\\\\?\\{abs_path}"

        if not os.path.exists(abs_path):
            # Print warning but do not crash the ThreadPool
            print(
                f"Warning: File {abs_path} does not exist for {calibration.name} at seed {seed}")
            return None

        df = pd.read_csv(abs_path)
        if "Unnamed: 0" in df.columns:
            df.drop("Unnamed: 0", axis=1, inplace=True)

        missing = REQUIRED_RESULT_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"{abs_path} is stale or incomplete; missing {sorted(missing)}"
            )
        if len(df) != 1:
            raise ValueError(
                f"Expected one result row in {abs_path}; got {len(df)}")
        if int(df.iloc[0]["seed"]) != seed:
            raise ValueError(
                f"Seed column does not match filename in {abs_path}")
        if str(df.iloc[0]["allocator_name"]) != calibration.name:
            raise ValueError(
                f"Allocator column does not match directory in {abs_path}")
        return df

    except Exception:
        print(f"Loading calibration {calibration.name} failed at seed {seed}")
        traceback.print_exc()
        return None


def merge_results(
        experiments_name,
        seeds,
        budget_per_sample,
        taus_range,
        tau_prior,
        m_upper_bound,
        device,
        dapro_n1=200,
        crc_control_size=100,
):
    all_dfs = []
    num_cpus = os.cpu_count()

    # Pass None for conditional_grid to avoid loading data just for names
    all_calibrations = get_metric_allocators(
        conditional_grid=None,
        budget_per_sample=budget_per_sample,
        taus_range=taus_range,
        tau_prior=tau_prior,
        m_upper_bound=m_upper_bound,
        device=device,
        dapro_n1=dapro_n1,
        crc_control_size=crc_control_size,
    )

    print(f"Starting merge for experiment: {experiments_name}")
    print(f"Discovered {len(all_calibrations)} unique calibration methods.")

    with ThreadPoolExecutor(max_workers=num_cpus) as executor:
        futures = []

        for seed in range(seeds[0], seeds[1]):
            for calibration in all_calibrations:
                futures.append(
                    executor.submit(process_calibration,
                                    calibration, seed, experiments_name)
                )

        # Process with tqdm progress bar
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Merging CSVs"):
            try:
                result_df = future.result()
                if result_df is not None and not result_df.empty:
                    all_dfs.append(result_df)
            except Exception as e:
                print(f"Error resolving future: {e}")

    if len(all_dfs) == 0:
        raise Exception(
            "No calibrations found. Check your file paths and seed ranges.")

    all_df = pd.concat(all_dfs, ignore_index=True)
    expected_rows = (seeds[1] - seeds[0]) * len(all_calibrations)
    if len(all_df) != expected_rows:
        raise RuntimeError(
            f"Incomplete merge: expected {expected_rows} rows but loaded "
            f"{len(all_df)}. Re-run estimation for the missing methods/seeds."
        )
    duplicates = all_df.duplicated(["seed", "allocator_name"], keep=False)
    if duplicates.any():
        raise RuntimeError("Duplicate seed/allocator rows found during merge.")
    all_df = all_df.sort_values(
        ["seed", "allocator_name"]).reset_index(drop=True)

    results_dir = get_merged_metric_calibration_result_path(experiments_name)
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "all_df.csv")

    all_df.to_csv(results_path, index=False)
    print(
        f"\nSuccessfully stored merged dataset at: {os.path.abspath(results_path)}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge dynamic calibration results.")
    parser.add_argument('--seed-start', type=int, default=0)
    parser.add_argument('--seed-end', type=int, default=50)
    parser.add_argument('--data-type', type=str, default='real')
    parser.add_argument('--dataset-name', type=str, default='')
    parser.add_argument('--dataset-setup', type=str, default='')
    parser.add_argument('--budget-per-sample', type=float, default=40)
    parser.add_argument('--cal-size', type=int, default=4000)
    parser.add_argument('--tau-prior', type=float, default=0.56)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dapro-n1', type=int, default=200)
    parser.add_argument('--crc-control-size', type=int, default=100)
    parser.add_argument('--experiment-suffix', type=str, default='')
    args = parser.parse_args()

    # Reconstruct variables required for allocator initialization
    device = 'cuda:0' if torch.cuda.is_available() and 'cuda' in args.device else 'cpu'

    taus_range = torch.tensor(np.arange(0.01, 1.0, 0.01)).to(device)

    is_real = True if args.data_type.lower() == 'real' else False
    m_upper_bound = 200 if is_real else 20

    experiments_name = metric_experiment_name(
        args.dataset_name,
        args.dataset_setup,
        args.budget_per_sample,
        args.dapro_n1,
        args.crc_control_size,
        args.experiment_suffix,
    )

    merge_results(
        experiments_name=experiments_name,
        seeds=(args.seed_start, args.seed_end),
        budget_per_sample=args.budget_per_sample,
        taus_range=taus_range,
        tau_prior=args.tau_prior,
        m_upper_bound=m_upper_bound,
        device=device,
        dapro_n1=args.dapro_n1,
        crc_control_size=args.crc_control_size,
    )


if __name__ == '__main__':
    main()
