import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tqdm

from src.safety_evaluation.budget_allocators.DAPRO import DAPRO
from src.safety_evaluation.dapro_projection_evaluation_utils import (
    get_dapro_projection_experiment_name,
    get_dapro_projection_metadata,
)
from src.safety_evaluation.utils.utils import setup_experiment_data, split_data
from src.train_model.models.utils import SurvivalModelPrediction
from src.utils.utils import set_seeds


def _write_seed_manifest(
        experiments_name,
        seed,
        allocator_names,
        metadata,
):
    manifest_dir = (
        Path("results")
        / "tmp_dapro_projection_evaluation"
        / experiments_name
        / "_manifests"
    )
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"seed={seed}.json"
    temporary_path = manifest_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(
            {
                "seed": int(seed),
                "allocator_names": list(allocator_names),
                **metadata,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary_path.replace(manifest_path)


def _result_path(experiments_name, allocator_name, seed):
    return (
        Path("results")
        / "tmp_dapro_projection_evaluation"
        / experiments_name
        / allocator_name
        / f"seed={seed}.csv"
    )


def run_dapro_projection_evaluation(
        cal_size,
        is_real,
        device,
        dataset_name,
        dataset_setup,
        experiments_name,
        seeds,
        budget_per_sample,
        tau_prior,
        m_upper_bound,
        projections,
        scores,
        n1_values,
        skip_existing,
):
    taus_range = torch.tensor(
        np.logspace(-3, -0.01, 1000),
        device=device,
    )
    (
        max_time,
        t_tilde_cal_test,
        quantile_est_cal_test,
        probability_est,
        conditional_grid,
        test_size,
    ) = setup_experiment_data(
        cal_size,
        is_real,
        device,
        dataset_name,
        dataset_setup,
        taus_range,
        m_upper_bound,
    )
    metadata = get_dapro_projection_metadata(
        dataset_name,
        dataset_setup,
        budget_per_sample,
        cal_size,
        tau_prior,
        m_upper_bound,
    )

    configurations = [
        (projection, score, n1)
        for n1 in n1_values
        for projection in projections
        for score in scores
    ]
    if not configurations:
        raise ValueError("At least one DAPRO configuration is required.")
    if len(set(configurations)) != len(configurations):
        raise ValueError("DAPRO projection configurations must be unique.")

    for seed in tqdm.tqdm(
            range(seeds[0], seeds[1]),
            desc="evaluating DAPRO projections",
    ):
        (
            x_cal,
            _,
            t_tilde_cal,
            probability_est_cal,
            quantile_est_cal,
            _,
            _,
            _,
            cal_idx,
            _,
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
        quantile_est_cal = quantile_est_cal.clip(max=max_time)
        cal_model_prediction = SurvivalModelPrediction(
            quantile_est_cal,
            probability_est_cal,
        )
        completed_names = []
        for projection, score, n1 in configurations:
            allocator = DAPRO(
                curr_conditional_grid,
                budget_per_sample,
                taus_range,
                tau_prior,
                m_upper_bound,
                projection=projection,
                score=score,
                n1=n1,
                evaluate_projection=True,
            )
            save_path = _result_path(
                experiments_name,
                allocator.name,
                seed,
            )
            if save_path.exists() and skip_existing:
                completed_names.append(allocator.name)
                continue

            # Reset before every configuration so comparisons use the same
            # validation split and continuation-sampling random stream.
            set_seeds(seed)
            with torch.no_grad():
                allocation = allocator.allocate_budget(
                    cal_model_prediction.probability_est,
                    x_cal,
                    t_tilde_cal,
                    cal_model_prediction.quantile_est,
                )
            if not allocation.additional_metrics:
                raise RuntimeError(
                    f"{allocator.name} returned no projection diagnostics."
                )
            row = {
                "seed": seed,
                "allocator_name": allocator.name,
                "projection": projection,
                "score": score,
                "n1": n1,
                "calibration_sample_count": len(t_tilde_cal),
                "total_budget_used": allocation.total_budget_used,
                "realized_budget_per_sample": (
                    float(allocation.total_budget_used) / len(t_tilde_cal)
                ),
                **metadata,
                **allocation.additional_metrics,
            }
            save_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = save_path.with_suffix(".csv.tmp")
            pd.DataFrame([row]).to_csv(temporary_path, index=False)
            temporary_path.replace(save_path)
            completed_names.append(allocator.name)

        if len(completed_names) != len(configurations):
            raise RuntimeError(
                f"Seed {seed} completed {len(completed_names)} of "
                f"{len(configurations)} DAPRO configurations."
            )
        _write_seed_manifest(
            experiments_name,
            seed,
            completed_names,
            metadata,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate DAPRO projected continuation probabilities against the "
            "full-calibration oracle policy and decompose budget-control error."
        )
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
    parser.add_argument(
        "--projections",
        nargs="+",
        choices=["ir", "platt", "beta"],
        default=["platt", "beta"],
    )
    parser.add_argument(
        "--scores",
        nargs="+",
        choices=["prob", "quantile"],
        default=["prob", "quantile"],
    )
    parser.add_argument(
        "--n1-values",
        nargs="+",
        type=int,
        default=[100],
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
    )
    parser.set_defaults(skip_existing=True)
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
    if any(n1 <= 0 or n1 >= args.cal_size for n1 in args.n1_values):
        raise ValueError("Every n1 value must be between 1 and cal-size - 1.")

    is_real = args.data_type == "real"
    m_upper_bound = args.m_upper_bound
    if m_upper_bound is None:
        m_upper_bound = 200.0 if is_real else 20.0
    if m_upper_bound <= 0:
        raise ValueError("m-upper-bound must be positive.")
    device = (
        args.device
        if torch.cuda.is_available() and "cuda" in args.device
        else "cpu"
    )
    experiments_name = get_dapro_projection_experiment_name(
        args.dataset_name,
        args.dataset_setup,
        args.budget_per_sample,
        args.cal_size,
        args.tau_prior,
        m_upper_bound,
    )
    print(f"DAPRO projection experiment: {experiments_name}")
    print(
        f"Configurations: projections={args.projections}, scores={args.scores}, "
        f"n1={args.n1_values}"
    )
    run_dapro_projection_evaluation(
        cal_size=args.cal_size,
        is_real=is_real,
        device=device,
        dataset_name=args.dataset_name,
        dataset_setup=args.dataset_setup,
        experiments_name=experiments_name,
        seeds=(args.seed_start, args.seed_end),
        budget_per_sample=args.budget_per_sample,
        tau_prior=args.tau_prior,
        m_upper_bound=m_upper_bound,
        projections=args.projections,
        scores=args.scores,
        n1_values=args.n1_values,
        skip_existing=args.skip_existing,
    )
    print("Finished DAPRO projection evaluation.")


if __name__ == "__main__":
    main()
