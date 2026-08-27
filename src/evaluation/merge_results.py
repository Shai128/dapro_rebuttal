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
        dapro_configs=None,
        exclude_legacy_dapro=False,
        exclude_locally_adaptive=False,
        allocator_names=None,
        method_suite="legacy",
        dapro_ablation_kind="score_noise",
        score_noise_lambdas=(0.0, 0.1, 0.25, 0.5, 0.75, 1.0),
        score_noise_seed=314159,
):
    all_dfs = []
    num_cpus = os.cpu_count()

    configurations = (
        [(dapro_n1, crc_control_size)]
        if dapro_configs is None
        else list(dapro_configs)
    )
    # Baselines are repeated by each registry call. Deduplicate by canonical
    # allocator name while retaining every N1/CRC-specific method.
    calibrations_by_name = {}
    for config_n1, config_crc in configurations:
        calibrations = get_metric_allocators(
            conditional_grid=None,
            budget_per_sample=budget_per_sample,
            taus_range=taus_range,
            tau_prior=tau_prior,
            m_upper_bound=m_upper_bound,
            device=device,
            dapro_n1=config_n1,
            crc_control_size=config_crc,
            include_legacy_dapro=not exclude_legacy_dapro,
            include_locally_adaptive=not exclude_locally_adaptive,
            method_suite=method_suite,
            dapro_ablation_kind=dapro_ablation_kind,
            score_noise_lambdas=score_noise_lambdas,
            score_noise_seed=score_noise_seed,
        )
        for calibration in calibrations:
            entry = calibrations_by_name.setdefault(
                calibration.name,
                {"calibration": calibration, "configs": []},
            )
            entry["configs"].append((config_n1, config_crc))
    calibration_entries = list(calibrations_by_name.values())
    if allocator_names is not None:
        requested = set(allocator_names)
        missing = requested - set(calibrations_by_name)
        if missing:
            raise ValueError(
                "Unknown metric allocator name(s): "
                + ", ".join(sorted(missing))
            )
        calibration_entries = [
            calibrations_by_name[name]
            for name in calibrations_by_name
            if name in requested
        ]

    print(f"Starting merge for experiment: {experiments_name}")
    print(f"Discovered {len(calibration_entries)} unique calibration methods.")

    with ThreadPoolExecutor(max_workers=num_cpus) as executor:
        futures = {}

        for seed in range(seeds[0], seeds[1]):
            for entry in calibration_entries:
                future = executor.submit(
                    process_calibration,
                    entry["calibration"], seed, experiments_name,
                )
                futures[future] = entry["configs"]

        # Process with tqdm progress bar
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Merging CSVs"):
            try:
                result_df = future.result()
                if result_df is not None and not result_df.empty:
                    configs = futures[future]
                    configured = result_df.copy()
                    configured["applicable_dapro_configs"] = "|".join(
                        f"{config_n1}:{config_crc}"
                        for config_n1, config_crc in configs
                    )
                    if len(configs) == 1:
                        config_n1, config_crc = configs[0]
                        configured["configuration_scope"] = "specific"
                        configured["configured_dapro_n1"] = config_n1
                        configured["configured_crc_control_size"] = config_crc
                    else:
                        # A shared baseline is one experimental observation,
                        # not one observation per registry configuration.
                        configured["configuration_scope"] = "shared"
                        configured["configured_dapro_n1"] = pd.NA
                        configured["configured_crc_control_size"] = pd.NA
                    all_dfs.append(configured)
            except Exception as e:
                print(f"Error resolving future: {e}")

    if len(all_dfs) == 0:
        raise Exception(
            "No calibrations found. Check your file paths and seed ranges.")

    all_df = pd.concat(all_dfs, ignore_index=True)
    expected_rows = (seeds[1] - seeds[0]) * len(calibration_entries)
    if len(all_df) != expected_rows:
        raise RuntimeError(
            f"Incomplete merge: expected {expected_rows} rows but loaded "
            f"{len(all_df)}. Re-run estimation for the missing methods/seeds."
        )
    identity_columns = ["seed", "allocator_name"]
    duplicates = all_df.duplicated(identity_columns, keep=False)
    if duplicates.any():
        raise RuntimeError("Duplicate seed/allocator rows found during merge.")
    all_df = all_df.sort_values(identity_columns).reset_index(drop=True)

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
    parser.add_argument(
        '--dapro-config',
        action='append',
        default=[],
        metavar='N1:CRC',
        help=(
            "N1/CRC pair to include in a consolidated merge. Repeat the "
            "option for multiple configurations."
        ),
    )
    parser.add_argument('--experiment-suffix', type=str, default='')
    parser.add_argument(
        '--method-suite',
        choices=['legacy', 'unified_aht', 'dapro_ablation'],
        default='legacy',
    )
    parser.add_argument(
        '--dapro-ablation-kind',
        choices=[
            'n1', 'score_noise', 'budget', 'hard_soft',
            'representation', 'score', 'cmax', 'optimization',
        ],
        default='score_noise',
    )
    parser.add_argument(
        '--score-noise-lambdas',
        type=float,
        nargs='+',
        default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
    )
    parser.add_argument('--score-noise-seed', type=int, default=314159)
    parser.add_argument('--exclude-legacy-dapro', action='store_true')
    parser.add_argument('--exclude-locally-adaptive', action='store_true')
    parser.add_argument(
        '--allocator-name',
        action='append',
        dest='allocator_names',
        help=(
            'Merge only the named allocator; repeat the option for multiple '
            'methods.'
        ),
    )
    args = parser.parse_args()

    dapro_configs = []
    for value in args.dapro_config:
        try:
            n1_text, crc_text = value.split(':', 1)
            config = (int(n1_text), int(crc_text))
        except (TypeError, ValueError):
            parser.error(f"Invalid --dapro-config {value!r}; expected N1:CRC.")
        if not 0 < config[1] < config[0]:
            parser.error(f"Invalid --dapro-config {value!r}; require 0 < CRC < N1.")
        dapro_configs.append(config)

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
        dapro_configs=dapro_configs or None,
        exclude_legacy_dapro=args.exclude_legacy_dapro,
        exclude_locally_adaptive=args.exclude_locally_adaptive,
        allocator_names=args.allocator_names,
        method_suite=args.method_suite,
        dapro_ablation_kind=args.dapro_ablation_kind,
        score_noise_lambdas=args.score_noise_lambdas,
        score_noise_seed=args.score_noise_seed,
    )


if __name__ == '__main__':
    main()
