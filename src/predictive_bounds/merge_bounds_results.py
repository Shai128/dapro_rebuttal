import os
import pandas as pd
import torch
import tqdm
import argparse
import numpy as np
from concurrent.futures import as_completed, ThreadPoolExecutor

from src.predictive_bounds.calibration.oracle_survival_calibration import (
    OracleSurvivalCalibration,
    OracleSurvivalUPBCalibration,
)
from src.predictive_bounds.calibration.survival_calibration_with_known_weights import get_gamma, SurvivalCalibrationWithKnownWeights
from src.predictive_bounds.utils.get_calibration_methods_utils import (
    get_baseline_calibrations,
    get_new_allocation_algorithms,
    get_upb_calibrations,
    get_unified_bound_calibrations,
)
from src.predictive_bounds.utils.utils import (
    get_tmp_calibration_result_path,
    get_merged_calibration_result_path,
    get_tmp_upb_calibration_result_path,
    get_merged_upb_calibration_result_path,
    get_calibration_experiment_name,
    make_lpb_tau_grid,
    make_upb_tau_grid,
    resolve_m_upper_bound,
)
from src.utils.utils import set_seeds
from src.predictive_bounds.construct_calibrated_bound import (
    REQUIRED_BOUND_RESULT_COLUMNS,
)

# Attempt to import UPB wrapper if it exists in your repository structure
try:
    from src.predictive_bounds.calibration.survival_upb_calibration_with_known_weights import SurvivalUPBCalibrationWithKnownWeights
except ImportError:
    # Fallback to the known weights if the specific UPB class is missing from imports
    SurvivalUPBCalibrationWithKnownWeights = SurvivalCalibrationWithKnownWeights


def absolute_io_path(path):
    """Return an absolute path that supports long Windows experiment names."""
    abs_path = os.path.abspath(path)
    if os.name == 'nt' and not abs_path.startswith('\\\\?\\'):
        return f"\\\\?\\{abs_path}"
    return abs_path


def process_calibration(calibration, seed, experiments_name, bound_type):
    """Process a single calibration and return all computed rows."""
    try:
        if bound_type == 'lpb':
            dir_path = get_tmp_calibration_result_path(experiments_name, calibration.name)
        else:
            dir_path = get_tmp_upb_calibration_result_path(experiments_name, calibration.name)

        csv_path = f"{dir_path}/seed={seed}.csv"
        abs_path = absolute_io_path(csv_path)
        if not os.path.exists(abs_path):
            raise Exception(f"Warning, method does not exist: {calibration.name} at {abs_path}")

        df = pd.read_csv(abs_path)
        if "Unnamed: 0" in df.columns:
            df.drop("Unnamed: 0", axis=1, inplace=True)
        missing = sorted(REQUIRED_BOUND_RESULT_COLUMNS - set(df.columns))
        if missing:
            raise ValueError(
                f"Stale or incomplete result {abs_path}; missing columns: "
                f"{missing}"
            )
        return df
    except Exception as error:
        raise RuntimeError(
            f"Loading calibration {calibration.name} for seed {seed} failed."
        ) from error


def get_calibration_methods(conditional_grid, budget_per_sample, taus_range, tau_prior, m_upper_bound, allocations: str,
                            device, bound_type, dapro_n1_values=(200,),
                            definitive_dapro_margins=(1.0,),
                            method_suite="legacy",
                            target_coverages=(0.90,)):
    if method_suite == "unified_aht":
        return get_unified_bound_calibrations(
            conditional_grid,
            budget_per_sample,
            taus_range,
            tau_prior,
            m_upper_bound,
            bound_type=bound_type,
            dapro_n1_values=dapro_n1_values,
            target_coverages=target_coverages,
        )
    if bound_type == "upb":
        return get_upb_calibrations(
            conditional_grid,
            budget_per_sample,
            taus_range,
            tau_prior,
            m_upper_bound,
            dapro_n1_values=dapro_n1_values,
            projection_budget_margin=float(definitive_dapro_margins[0]),
            target_coverage=0.70,
        )
    baseline_calibrations = get_baseline_calibrations(
        conditional_grid,
        budget_per_sample,
        taus_range,
        tau_prior,
        m_upper_bound,
        include_a_weighted=(bound_type == 'lpb'),
        dapro_n1_values=dapro_n1_values,
        definitive_dapro_margins=definitive_dapro_margins,
    )
    new_allocations = get_new_allocation_algorithms(conditional_grid, budget_per_sample, taus_range, tau_prior,
                                                    m_upper_bound,
                                                    allocations, device=device)
    new_allocations = []
    all_allocations = new_allocations

    if bound_type == 'lpb':
        all_calibrations = (
            baseline_calibrations
            + [OracleSurvivalCalibration(taus_range, tau_prior)]
            + [
                SurvivalCalibrationWithKnownWeights(allocation, taus_range, tau_prior)
                for allocation in all_allocations
            ]
        )
    return all_calibrations


def merge_results(experiments_name, seeds, budget_per_sample, taus_range, tau_prior, m_upper_bound, target_taus_list,
                  allocations, device, bound_type, calibration_names=None,
                  dapro_n1_values=(200,), definitive_dapro_margins=(1.0,),
                  method_suite="legacy", target_coverages=(0.90,)):
    all_dfs = []
    errors = []
    for seed in tqdm.tqdm(range(seeds[0], seeds[1]), desc="merging csvs"):

        all_calibrations = get_calibration_methods(None, budget_per_sample, taus_range, tau_prior, m_upper_bound,
                                                   allocations, device=device, bound_type=bound_type,
                                                   dapro_n1_values=dapro_n1_values,
                                                   definitive_dapro_margins=definitive_dapro_margins,
                                                   method_suite=method_suite,
                                                   target_coverages=target_coverages)
        if calibration_names:
            available = {calibration.name for calibration in all_calibrations}
            missing = sorted(set(calibration_names) - available)
            if missing:
                raise ValueError(
                    "Unknown calibration names: "
                    f"{missing}. Available names: {sorted(available)}"
                )
            selected_names = set(calibration_names)
            all_calibrations = [
                calibration
                for calibration in all_calibrations
                if calibration.name in selected_names
            ]

        num_cpus = os.cpu_count()

        with ThreadPoolExecutor(max_workers=num_cpus) as executor:
            futures = [executor.submit(process_calibration, calibration, seed, experiments_name, bound_type) for
                       calibration in all_calibrations]

            for future in as_completed(futures):
                try:
                    result_df = future.result()
                    all_dfs.append(result_df)
                except Exception as error:
                    errors.append(error)

    if errors:
        details = "\n".join(f"- {error}" for error in errors)
        raise RuntimeError(
            "Refusing to write a partial merged result because expected "
            f"method/seed files are missing or invalid:\n{details}"
        )

    if len(all_dfs) == 0:
        raise Exception("No calibrations found")

    all_df = pd.concat(all_dfs)

    if bound_type == 'lpb':
        results_dir = get_merged_calibration_result_path(experiments_name)
    else:
        results_dir = get_merged_upb_calibration_result_path(experiments_name)

    results_dir_io = absolute_io_path(results_dir)
    os.makedirs(results_dir_io, exist_ok=True)
    results_path = os.path.join(results_dir_io, "all_df.csv")
    all_df.to_csv(results_path, index=False)
    display_path = os.path.abspath(os.path.join(results_dir, "all_df.csv"))
    print(f"stored successfully at {display_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge Calibrated LPB or UPB Results")
    parser.add_argument('--bound-type', type=str, choices=['lpb', 'upb'], default='lpb',
                        help="Whether to merge LPB or UPB results")
    parser.add_argument('--seed-start', type=int, default=0)
    parser.add_argument('--seed-end', type=int, default=1)
    parser.add_argument('--allocations', type=str, default='all')
    parser.add_argument('--data-type', type=str, default='real')
    parser.add_argument('--dataset-name', type=str, default='')
    parser.add_argument('--dataset-setup', type=str, default='')
    parser.add_argument('--budget-per-sample', type=float, default=1)
    parser.add_argument('--cal-size', type=int, default=4000)
    parser.add_argument('--tau-prior', type=float, default=None, help="Prior for tau (defaults depend on bound-type)")
    parser.add_argument(
        '--gamma',
        type=float,
        default=None,
        help="Optional ratio m_upper_bound / budget_per_sample.",
    )
    parser.add_argument(
        '--m-upper-bound',
        type=float,
        default=None,
        help="Optional explicit interaction horizon.",
    )
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument(
        '--experiment-suffix',
        type=str,
        default='',
        help="Optional suffix used to isolate this run from existing results.",
    )
    parser.add_argument(
        '--method-suite', choices=['legacy', 'unified_aht'], default='legacy'
    )
    parser.add_argument('--target-coverages', type=float, nargs='+', default=None)
    parser.add_argument(
        '--calibration-names',
        type=str,
        default='',
        help=(
            "Optional comma-separated exact calibration names. When supplied, "
            "only those method/seed files are required and merged."
        ),
    )
    parser.add_argument(
        '--dapro-n1-values',
        type=int,
        nargs='+',
        default=[200],
        help=(
            "Phase-I sample sizes used to reconstruct DAPRO-family method "
            "names. Must match the corresponding construction command."
        ),
    )
    parser.add_argument(
        '--definitive-dapro-margins',
        type=float,
        nargs='+',
        default=[1.0],
        help=(
            "Projection-error reserves used to reconstruct assumption-based "
            "projection-DAPRO ablation names. Must match the construction "
            "command."
        ),
    )

    args = parser.parse_args()
    args.is_real = True if args.data_type.lower() == 'real' else False
    bound_type = args.bound_type

    seed_start = args.seed_start
    seed_end = args.seed_end
    dataset_name = args.dataset_name
    data_setup = args.dataset_setup
    is_real = args.is_real
    cal_size = args.cal_size
    if (
        len(set(args.dapro_n1_values)) != len(args.dapro_n1_values)
        or any(
            n1 <= 0 or n1 >= cal_size
            for n1 in args.dapro_n1_values
        )
    ):
        parser.error(
            "--dapro-n1-values must contain distinct integers between 1 "
            "and cal-size - 1."
        )
    if (
            len(set(args.definitive_dapro_margins))
            != len(args.definitive_dapro_margins)
            or any(
                not np.isfinite(margin) or margin < 0
                for margin in args.definitive_dapro_margins
            )
    ):
        parser.error(
            "--definitive-dapro-margins must contain distinct finite "
            "nonnegative values."
        )

    device = (
        args.device
        if torch.cuda.is_available() and 'cuda' in args.device
        else 'cpu'
    )
    set_seeds(0)

    seeds = (seed_start, seed_end)

    # Contextual initialization parameters dependent on bound-type choice
    if bound_type == 'lpb':
        tau_prior = args.tau_prior if args.tau_prior is not None else 0.56
        target_taus_list = np.arange(0.01, 0.5, 0.01)
        taus_range = make_lpb_tau_grid(device=device)
    else:
        tau_prior = args.tau_prior if args.tau_prior is not None else 0.98
        target_taus_list = 1 - np.arange(0.01, 0.5, 0.01)
        taus_range = make_upb_tau_grid(device=device)
    target_coverages = tuple(
        args.target_coverages
        if args.target_coverages is not None
        else ([0.90] if bound_type == 'lpb' else [0.70, 0.80, 0.90])
    )
    if any(not 0 < value < 1 for value in target_coverages):
        parser.error('--target-coverages values must lie in (0, 1).')

    budget_per_sample = args.budget_per_sample

    try:
        m_upper_bound = resolve_m_upper_bound(
            is_real,
            budget_per_sample,
            gamma=args.gamma,
            m_upper_bound=args.m_upper_bound,
        )
    except ValueError as error:
        parser.error(str(error))

    gamma = get_gamma(m_upper_bound, budget_per_sample)
    allocations = args.allocations
    calibration_names = [
        name.strip()
        for name in args.calibration_names.split(',')
        if name.strip()
    ]

    global global_calibration_set_size
    global global_budget_per_sample
    global_calibration_set_size = cal_size
    global_budget_per_sample = budget_per_sample

    print(f"Executing for Bound Type: {bound_type.upper()}")
    print(f"budget_per_sample: {budget_per_sample}, gamma: {gamma}, m_upper_bound: {m_upper_bound}")

    try:
        experiments_name = get_calibration_experiment_name(
            dataset_name,
            data_setup,
            budget_per_sample,
            cal_size,
            tau_prior,
            gamma,
            args.experiment_suffix,
        )
    except ValueError as error:
        parser.error(str(error))

    merge_results(experiments_name, seeds, budget_per_sample, taus_range, tau_prior, m_upper_bound,
                  target_taus_list, allocations=allocations, device=device, bound_type=bound_type,
                  calibration_names=calibration_names,
                  dapro_n1_values=tuple(args.dapro_n1_values),
                  definitive_dapro_margins=tuple(
                      args.definitive_dapro_margins
                  ),
                  method_suite=args.method_suite,
                  target_coverages=target_coverages)

    print("Finished")


if __name__ == '__main__':
    main()
