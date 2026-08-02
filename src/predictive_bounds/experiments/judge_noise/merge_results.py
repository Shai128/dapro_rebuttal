"""Merge every requested judge-noise configuration."""

from __future__ import annotations

from src.predictive_bounds.experiments.common.results import merge_sharded_bounds, stable_experiment_name
from src.predictive_bounds.experiments.judge_noise.construct_calibrated_bound import (
    metadata,
    noise_configurations,
    parse_args,
)


def main(argv=None):
    args = parse_args(argv)
    for noise_type, fn_rate, fp_rate in noise_configurations(args):
        current = metadata(args, noise_type, fn_rate, fp_rate)
        name = stable_experiment_name("judge_noise", current)
        path = merge_sharded_bounds(
            name, (args.seed_start, args.seed_end), args.bound_type,
            expected_metadata={
                "experiment_type": "judge_noise",
                "noise_type": noise_type,
                "false_negative_rate": fn_rate,
                "false_positive_rate": fp_rate,
            },
        )
        print(path.resolve())


if __name__ == "__main__":
    main()

