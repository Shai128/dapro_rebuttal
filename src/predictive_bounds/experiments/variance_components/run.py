"""Run or print every DAPRO variance-decomposition construction job."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import torch

from src.predictive_bounds.budget_allocators.DAPRO import DefinitiveCRCDAPRO
from src.predictive_bounds.experiments.full_bounds.config import CONSTANT
from src.predictive_bounds.experiments.variance_components.design import variance_jobs


def dapro_names(n1_values, budget, tau_prior, horizon):
    taus = torch.tensor([0.1])
    return tuple(
        "calibration_"
        + DefinitiveCRCDAPRO(
            None, budget, taus, tau_prior, horizon,
            n1=n1, budget_control_size=min(100, n1 // 2),
            row_cost_cap_multiplier=2.0,
        ).name
        + "_allocation"
        for n1 in n1_values
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default="dataset_toxicity")
    parser.add_argument("--dataset-setup", default=("attack_toxic_attack_qwen25_14b_instruct_lm_target_"
                                                     "qwen25_14b_instruct_judge_detoxify"))
    parser.add_argument("--cal-size", type=int, default=3000)
    parser.add_argument("--budget-per-sample", type=float, default=20.0)
    parser.add_argument("--tau-prior", type=float, default=0.56)
    parser.add_argument("--m-upper-bound", type=float, default=200.0)
    parser.add_argument("--n1-values", type=int, nargs="+", default=[100, 200, 400, 800])
    parser.add_argument("--replicates", type=int, default=50)
    parser.add_argument("--crossed-groups", type=int, default=10)
    parser.add_argument("--suffix-prefix", default="variance_components_v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("figures/ablations/variance_components"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def command_for_job(args, job, module):
    names = (CONSTANT, *dapro_names(
        args.n1_values, args.budget_per_sample, args.tau_prior, args.m_upper_bound
    ))
    command = [
        sys.executable, "-m", module,
        "--bound-type", "lpb", "--data-type", "real",
        "--dataset-name", args.dataset_name,
        "--dataset-setup", args.dataset_setup,
        "--cal-size", str(args.cal_size),
        "--budget-per-sample", str(args.budget_per_sample),
        "--tau-prior", str(args.tau_prior),
        "--m-upper-bound", str(args.m_upper_bound),
        "--seed-start", str(job.seed_start), "--seed-end", str(job.seed_end),
        "--device", args.device,
        "--experiment-suffix", job.suffix,
        "--dapro-n1-values", *[str(value) for value in args.n1_values],
        "--calibration-names", ",".join(names),
    ]
    for flag, value in (
        ("--fixed-data-seed", job.fixed_data_seed),
        ("--fixed-policy-seed", job.fixed_policy_seed),
        ("--fixed-acquisition-seed", job.fixed_acquisition_seed),
    ):
        if value is not None:
            command.extend([flag, str(value)])
    return command


def main(argv=None):
    args = parse_args(argv)
    if any(n1 < 100 or n1 >= args.cal_size for n1 in args.n1_values):
        raise ValueError("CRC-DAPRO N1 values must be in [100, cal_size).")
    for job in variance_jobs(
        replicates=args.replicates,
        crossed_groups=args.crossed_groups,
        suffix_prefix=args.suffix_prefix,
    ):
        command = command_for_job(
            args, job, "src.predictive_bounds.construct_calibrated_bound"
        )
        print(" ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
