"""Merge per-sample allocation-focus shards."""

from src.predictive_bounds.experiments.allocation_focus.construct import metadata, parse_args
from src.predictive_bounds.experiments.common.results import merge_table_shards, stable_experiment_name


def main(argv=None):
    args = parse_args(argv)
    current = metadata(args)
    name = stable_experiment_name("allocation_focus", current)
    path = merge_table_shards(
        "allocation_focus", name, (args.seed_start, args.seed_end),
        expected_metadata={"experiment_type": "allocation_focus"},
        unique_keys=("seed", "sample_id", "method"),
    )
    print(path.resolve())


if __name__ == "__main__":
    main()

