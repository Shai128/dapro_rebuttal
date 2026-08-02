"""Declarative one-factor and crossed variance-decomposition designs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VarianceJob:
    design: str
    suffix: str
    seed_start: int
    seed_end: int
    fixed_data_seed: int | None = None
    fixed_policy_seed: int | None = None
    fixed_acquisition_seed: int | None = None


def variance_jobs(
        *,
        replicates: int,
        crossed_groups: int,
        suffix_prefix: str,
) -> tuple[VarianceJob, ...]:
    if replicates < 2 or crossed_groups < 2:
        raise ValueError("Variance designs require at least two replicates/groups.")
    jobs = [
        VarianceJob("all_random", f"{suffix_prefix}_all", 0, replicates),
        VarianceJob("acquisition_only", f"{suffix_prefix}_acq", 0, replicates, 0, 0, None),
        VarianceJob("policy_only", f"{suffix_prefix}_policy", 0, replicates, 0, None, 0),
        VarianceJob("data_split_only", f"{suffix_prefix}_data", 0, replicates, None, 0, 0),
    ]
    for group in range(crossed_groups):
        jobs.extend([
            VarianceJob(
                "policy_x_acquisition",
                f"{suffix_prefix}_policy-x-acq_g{group}",
                0, replicates, 0, group, None,
            ),
            VarianceJob(
                "data_x_acquisition",
                f"{suffix_prefix}_data-x-acq_g{group}",
                0, replicates, group, 0, None,
            ),
            VarianceJob(
                "data_x_policy",
                f"{suffix_prefix}_data-x-policy_g{group}",
                0, replicates, group, None, 0,
            ),
        ])
    return tuple(jobs)

