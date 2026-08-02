"""Merge every variance-decomposition construction job."""

from __future__ import annotations

import subprocess

from src.predictive_bounds.experiments.variance_components.design import variance_jobs
from src.predictive_bounds.experiments.variance_components.run import (
    command_for_job,
    parse_args,
)


def main(argv=None):
    args = parse_args(argv)
    for job in variance_jobs(
        replicates=args.replicates,
        crossed_groups=args.crossed_groups,
        suffix_prefix=args.suffix_prefix,
    ):
        command = command_for_job(
            args, job, "src.predictive_bounds.merge_bounds_results"
        )
        # Merge does not accept fixed-randomness flags; they are provenance in CSVs.
        for flag in ("--fixed-data-seed", "--fixed-policy-seed", "--fixed-acquisition-seed"):
            while flag in command:
                index = command.index(flag)
                del command[index:index + 2]
        print(" ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

