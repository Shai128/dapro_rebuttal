"""Merge one completed distribution-shift experiment."""

from __future__ import annotations

from src.predictive_bounds.experiments.common.results import (
    merge_sharded_bounds,
    stable_experiment_name,
)
from src.predictive_bounds.experiments.distribution_shift.construct_calibrated_bound import (
    experiment_metadata,
    parse_args,
)


def main(argv=None):
    args = parse_args(argv)
    metadata = experiment_metadata(args)
    name = stable_experiment_name(args.shift_type, metadata)
    path = merge_sharded_bounds(
        name,
        (args.seed_start, args.seed_end),
        args.bound_type,
        expected_metadata={"experiment_type": args.shift_type},
    )
    print(path.resolve())


if __name__ == "__main__":
    main()

