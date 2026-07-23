import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import tqdm

from src.safety_evaluation.autoif_cross_class_utils import (
    DEFAULT_AUTOIF_CLASSIFICATIONS_PATH,
    DEFAULT_AUTOIF_DATA_PATH,
    DEFAULT_AUTOIF_DATASET_SETUP,
    get_autoif_candidate_classes,
    get_autoif_cross_class_experiment_name,
    get_autoif_cross_class_metadata,
    load_autoif_classes_in_dataset_order,
    select_autoif_cross_class_indices,
)
from src.safety_evaluation.calibration.survival_calibration_with_known_weights import (
    get_gamma,
)
from src.safety_evaluation.construct_calibrated_bound import (
    get_calibration_methods,
    run_one_experiment,
)
from src.safety_evaluation.utils.utils import setup_experiment_data
from src.train_model.models.utils import SurvivalModelPrediction
from src.utils.utils import set_seeds


def _write_seed_manifest(
        experiments_name,
        seed,
        bound_type,
        calibration_names,
        experiment_metadata,
        cal_size,
        test_size,
):
    result_kind = (
        "tmp_calibration_results"
        if bound_type == "lpb"
        else "tmp_upb_calibration_results"
    )
    manifest_dir = Path("results") / result_kind / experiments_name / "_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"seed={seed}.json"
    temporary_path = manifest_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "bound_type": bound_type,
                "calibration_names": list(calibration_names),
                "cal_size": int(cal_size),
                "test_size": int(test_size),
                **experiment_metadata,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary_path.replace(manifest_path)


def run_autoif_cross_class_experiments(
        dataset_name,
        dataset_setup,
        autoif_data_path,
        classifications_path,
        calibration_class,
        test_class,
        cal_size,
        test_size,
        device,
        experiments_name,
        seeds,
        taus_range,
        budget_per_sample,
        tau_prior,
        m_upper_bound,
        target_taus_list,
        skip_existing,
        allocations,
        bound_type,
        max_workers,
):
    classes_in_dataset_order = load_autoif_classes_in_dataset_order(
        autoif_data_path,
        classifications_path,
    )
    candidate_classes = get_autoif_candidate_classes(classes_in_dataset_order)
    known_classes = set(classes_in_dataset_order)
    for role, requested_class in [
        ("calibration", calibration_class),
        ("test", test_class),
    ]:
        if requested_class not in known_classes:
            available = ", ".join(sorted(known_classes))
            raise ValueError(
                f"Unknown {role} class {requested_class!r}. Available classes: "
                f"{available}"
            )

    (
        max_time,
        t_tilde_cal_test,
        quantile_est_cal_test,
        probability_est,
        conditional_grid,
        _,
    ) = setup_experiment_data(
        cal_size=cal_size,
        is_real=True,
        device=device,
        dataset_name=dataset_name,
        data_setup=dataset_setup,
        taus_range=taus_range,
        m_upper_bound=m_upper_bound,
    )
    expected_candidate_count = len(t_tilde_cal_test)
    if len(candidate_classes) != expected_candidate_count:
        raise ValueError(
            "AutoIF class labels do not align with the survival tensors. The CSV "
            f"split implies {len(candidate_classes)} calibration/test rows, but "
            f"the loaded tensors contain {expected_candidate_count}. Ensure attack "
            "logs/tensors were generated from this exact autoif_helper_dataset.csv."
        )
    if (
            len(probability_est) != expected_candidate_count
            or len(quantile_est_cal_test) != expected_candidate_count
            or len(conditional_grid) != expected_candidate_count
    ):
        raise ValueError(
            "Cached AutoIF model predictions do not match the loaded "
            "calibration/test tensors. Remove the prediction cache and rerun."
        )

    experiment_metadata = get_autoif_cross_class_metadata(
        dataset_name,
        dataset_setup,
        calibration_class,
        test_class,
    )
    print(
        f"Eligible rows: calibration class={calibration_class!r}: "
        f"{int((candidate_classes == calibration_class).sum())}; "
        f"test class={test_class!r}: "
        f"{int((candidate_classes == test_class).sum())}"
    )
    taus_range = taus_range.detach()

    for seed in tqdm.tqdm(
            range(seeds[0], seeds[1]),
            desc="running AutoIF cross-class calibration algorithms",
    ):
        cal_idx, test_idx = select_autoif_cross_class_indices(
            candidate_classes,
            calibration_class,
            test_class,
            cal_size,
            test_size,
            seed,
        )
        t_tilde_cal = t_tilde_cal_test[cal_idx].detach()
        t_tilde_test = t_tilde_cal_test[test_idx].detach()
        probability_est_cal = probability_est[cal_idx].detach()
        probability_est_test = probability_est[test_idx].detach()
        quantile_est_cal = quantile_est_cal_test[cal_idx].detach()
        quantile_est_test = quantile_est_cal_test[test_idx].detach()
        curr_conditional_grid = conditional_grid[cal_idx]
        if bound_type == "lpb":
            quantile_est_cal = quantile_est_cal.clip(max=max_time)

        cal_model_prediction = SurvivalModelPrediction(
            quantile_est_cal,
            probability_est_cal,
        )
        test_model_prediction = SurvivalModelPrediction(
            quantile_est_test,
            probability_est_test,
        )
        all_calibrations = get_calibration_methods(
            curr_conditional_grid,
            budget_per_sample,
            taus_range,
            tau_prior,
            m_upper_bound,
            allocations,
            cal_model_prediction,
            t_tilde_cal,
            device,
            bound_type,
        )

        def run_calibration(calibration):
            return run_one_experiment(
                experiments_name,
                seed,
                calibration,
                None,
                t_tilde_cal,
                cal_model_prediction,
                None,
                t_tilde_test,
                test_model_prediction,
                target_taus_list,
                bound_type,
                skip_existing,
                experiment_metadata=experiment_metadata,
            )

        if max_workers == 1:
            for calibration in all_calibrations:
                run_calibration(calibration)
        else:
            errors = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_name = {
                    executor.submit(run_calibration, calibration): calibration.name
                    for calibration in all_calibrations
                }
                for future in as_completed(future_to_name):
                    try:
                        future.result()
                    except Exception as exc:
                        errors.append((future_to_name[future], exc))
            if errors:
                details = "; ".join(f"{name}: {exc}" for name, exc in errors)
                raise RuntimeError(
                    f"{len(errors)} AutoIF cross-class calibrations failed for "
                    f"seed {seed}: {details}"
                )

        _write_seed_manifest(
            experiments_name,
            seed,
            bound_type,
            [calibration.name for calibration in all_calibrations],
            experiment_metadata,
            cal_size,
            len(test_idx),
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Construct calibrated bounds using one AutoIF task class for "
            "calibration and a different task class for testing."
        )
    )
    parser.add_argument("--bound-type", choices=["lpb", "upb"], default="lpb")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=50)
    parser.add_argument("--allocations", choices=["none", "one", "all"], default="one")
    parser.add_argument("--dataset-name", default="dataset_autoif")
    parser.add_argument("--dataset-setup", default=DEFAULT_AUTOIF_DATASET_SETUP)
    parser.add_argument(
        "--autoif-data-path",
        default=str(DEFAULT_AUTOIF_DATA_PATH),
    )
    parser.add_argument(
        "--classifications-path",
        default=str(DEFAULT_AUTOIF_CLASSIFICATIONS_PATH),
    )
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
        help="Number of requested-class test rows per seed; use 0 for all.",
    )
    parser.add_argument("--tau-prior", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--m-upper-bound", type=float, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Use 1 for deterministic calibration RNG; larger values enable threads.",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Recompute result files that already exist.",
    )
    parser.set_defaults(skip_existing=True)
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
    if args.max_workers < 1:
        raise ValueError("max-workers must be at least 1.")
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
    if args.bound_type == "lpb":
        target_taus_list = np.arange(0.01, 0.5, 0.01)
        taus_range = torch.tensor(np.logspace(-3, -0.01, 1000))
    else:
        target_taus_list = 1 - np.arange(0.01, 0.5, 0.01)
        taus_range = torch.tensor(np.linspace(0.5, 0.95, 3000))

    device = (
        args.device
        if torch.cuda.is_available() and "cuda" in args.device
        else "cpu"
    )
    taus_range = taus_range.to(device)
    gamma = get_gamma(m_upper_bound, args.budget_per_sample)
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

    set_seeds(0)
    print(f"AutoIF cross-class experiment: {experiments_name}")
    print(f"Calibration class: {args.calibration_class}")
    print(f"Test class: {args.test_class}")
    print(
        f"Bound: {args.bound_type.upper()} | budget/sample: "
        f"{args.budget_per_sample} | gamma: {gamma} | upper bound: "
        f"{m_upper_bound} | device: {device}"
    )
    run_autoif_cross_class_experiments(
        dataset_name=args.dataset_name,
        dataset_setup=args.dataset_setup,
        autoif_data_path=args.autoif_data_path,
        classifications_path=args.classifications_path,
        calibration_class=args.calibration_class,
        test_class=args.test_class,
        cal_size=args.cal_size,
        test_size=test_size,
        device=device,
        experiments_name=experiments_name,
        seeds=(args.seed_start, args.seed_end),
        taus_range=taus_range,
        budget_per_sample=args.budget_per_sample,
        tau_prior=tau_prior,
        m_upper_bound=m_upper_bound,
        target_taus_list=target_taus_list,
        skip_existing=args.skip_existing,
        allocations=args.allocations,
        bound_type=args.bound_type,
        max_workers=args.max_workers,
    )
    print("Finished AutoIF cross-class calibrated-bound construction.")


if __name__ == "__main__":
    main()
