import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import tqdm

from src.safety_evaluation.calibration.survival_calibration_with_known_weights import get_gamma
from src.safety_evaluation.construct_calibrated_bound import (
    get_calibration_methods,
    run_one_experiment,
)
from src.safety_evaluation.cross_setup_utils import (
    get_cross_setup_experiment_name,
    get_cross_setup_metadata,
    setup_cross_setup_experiment_data,
)
from src.safety_evaluation.utils.utils import split_data
from src.train_model.models.utils import SurvivalModelPrediction
from src.utils.utils import set_seeds


def _write_seed_manifest(
        experiments_name: str,
        seed: int,
        bound_type: str,
        calibration_names,
        experiment_metadata: dict,
):
    result_kind = "tmp_calibration_results" if bound_type == "lpb" else "tmp_upb_calibration_results"
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
                **experiment_metadata,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary_path.replace(manifest_path)


def run_cross_setup_experiments(
        cal_size,
        is_real,
        device,
        dataset_name,
        model_dataset_setup,
        evaluation_dataset_setup,
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
    (
        max_time,
        t_tilde_cal_test,
        quantile_est_cal_test,
        probability_est,
        conditional_grid,
        test_size,
    ) = setup_cross_setup_experiment_data(
        cal_size=cal_size,
        is_real=is_real,
        device=device,
        dataset_name=dataset_name,
        model_dataset_setup=model_dataset_setup,
        evaluation_dataset_setup=evaluation_dataset_setup,
        taus_range=taus_range,
        m_upper_bound=m_upper_bound,
    )
    taus_range = taus_range.detach()
    experiment_metadata = get_cross_setup_metadata(
        dataset_name,
        model_dataset_setup,
        evaluation_dataset_setup,
    )

    for seed in tqdm.tqdm(
            range(seeds[0], seeds[1]),
            desc="running cross-setup calibration algorithms",
    ):
        (
            x_cal,
            x_test,
            t_tilde_cal,
            probability_est_cal,
            quantile_est_cal,
            t_tilde_test,
            quantile_est_test,
            probability_est_test,
            cal_idx,
            test_idx,
        ) = split_data(
            seed,
            cal_size,
            test_size,
            None,
            t_tilde_cal_test,
            probability_est,
            quantile_est_cal_test,
        )

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
                x_cal,
                t_tilde_cal,
                cal_model_prediction,
                x_test,
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
                    f"{len(errors)} cross-setup calibrations failed for seed {seed}: {details}"
                )

        _write_seed_manifest(
            experiments_name,
            seed,
            bound_type,
            [calibration.name for calibration in all_calibrations],
            experiment_metadata,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Construct calibrated bounds using a survival model trained on one setup "
            "and calibration/test data from another setup."
        )
    )
    parser.add_argument("--bound-type", choices=["lpb", "upb"], default="lpb")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=50)
    parser.add_argument("--allocations", choices=["none", "one", "all"], default="one")
    parser.add_argument("--data-type", choices=["real", "synthetic"], default="real")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--model-dataset-setup", required=True)
    parser.add_argument("--evaluation-dataset-setup", required=True)
    parser.add_argument("--budget-per-sample", type=float, default=20.0)
    parser.add_argument("--cal-size", type=int, default=3000)
    parser.add_argument("--tau-prior", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--m-upper-bound", type=float, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Use 1 for deterministic calibration RNG; larger values enable thread parallelism.",
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
    if args.budget_per_sample <= 0:
        raise ValueError("budget-per-sample must be positive.")
    if args.max_workers < 1:
        raise ValueError("max-workers must be at least 1.")
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
    if m_upper_bound <= 0:
        raise ValueError("The upper bound must be positive.")

    bound_type = args.bound_type
    if args.tau_prior is None:
        tau_prior = 0.56 if bound_type == "lpb" else 0.98
    else:
        tau_prior = args.tau_prior

    if bound_type == "lpb":
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
    experiments_name = get_cross_setup_experiment_name(
        args.dataset_name,
        args.model_dataset_setup,
        args.evaluation_dataset_setup,
        args.budget_per_sample,
        args.cal_size,
        tau_prior,
        gamma,
    )

    set_seeds(0)
    print(f"Cross-setup experiment: {experiments_name}")
    print(f"Model setup: {args.model_dataset_setup}")
    print(f"Evaluation setup: {args.evaluation_dataset_setup}")
    print(
        f"Bound: {bound_type.upper()} | budget/sample: {args.budget_per_sample} | "
        f"gamma: {gamma} | upper bound: {m_upper_bound} | device: {device}"
    )

    run_cross_setup_experiments(
        cal_size=args.cal_size,
        is_real=is_real,
        device=device,
        dataset_name=args.dataset_name,
        model_dataset_setup=args.model_dataset_setup,
        evaluation_dataset_setup=args.evaluation_dataset_setup,
        experiments_name=experiments_name,
        seeds=(args.seed_start, args.seed_end),
        taus_range=taus_range,
        budget_per_sample=args.budget_per_sample,
        tau_prior=tau_prior,
        m_upper_bound=m_upper_bound,
        target_taus_list=target_taus_list,
        skip_existing=args.skip_existing,
        allocations=args.allocations,
        bound_type=bound_type,
        max_workers=args.max_workers,
    )
    print("Finished cross-setup calibrated-bound construction.")


if __name__ == "__main__":
    main()
