"""Merge limited-training-budget bound shards."""

from src.predictive_bounds.experiments.common.results import merge_sharded_bounds, stable_experiment_name
from src.predictive_bounds.experiments.limited_training_budget.construct_calibrated_bound import metadata, parse_args


def main(argv=None):
    args = parse_args(argv)
    current = metadata(args)
    name = stable_experiment_name("limited_training_budget", current)
    path = merge_sharded_bounds(
        name, (args.seed_start, args.seed_end), args.bound_type,
        expected_metadata={"experiment_type": "limited_training_budget"},
    )
    print(path.resolve())


if __name__ == "__main__":
    main()

