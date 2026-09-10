"""Generate AutoIF UPB quantiles at every possible current time.

The ordinary bound pipeline only materializes predictions at current time zero.
This script expands an existing conditional event-time probability cache into a
tensor with layout ``(calibration + test, current_time, candidate_quantile)``.
It never trains or modifies a model and is independent of the split-model
experiment in :mod:`train_independent_split_models`.

The production output is large (6000 * 200 * 3000 float32 values, about
14.4 GB).  A temporary file-backed tensor keeps peak RAM proportional to the
configured sample chunk rather than to the complete output.  Final output is a
normal ``torch.save`` file and can be loaded with ``torch.load(..., mmap=True)``.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import shutil
import uuid
from pathlib import Path

import torch

from src.predictive_bounds.calibration.calibration_utils import (
    quantiles_to_interaction_counts,
)
from src.predictive_bounds.survival_utils.compute_mean_time_given_pmf import (
    compute_quantiles_survival_time,
)


AUTOIF_SETUP = (
    "attack_autoif_helper_qwen25_14b_instruct_lm_target_"
    "qwen25_14b_instruct_judge_autoif"
)
DEFAULT_PROBABILITY_CACHE = Path(
    "alg_playground_model"
) / (
    "is_real_True_dataset_dataset_autoif_dataset_" + AUTOIF_SETUP
) / "probability_est_cal_test.pt"
DEFAULT_OUTPUT = DEFAULT_PROBABILITY_CACHE.with_name(
    "all_current_time_upb_quantiles.pt"
)


def make_candidate_quantile_levels(
        count: int = 3000,
        minimum: float = 0.5,
        maximum: float = 0.95,
) -> torch.Tensor:
    """Return the historical 3,000-candidate paper UPB grid."""
    if count <= 0:
        raise ValueError("candidate quantile count must be positive")
    if not 0.0 <= minimum <= maximum <= 1.0:
        raise ValueError("quantile endpoints must satisfy 0 <= min <= max <= 1")
    return torch.linspace(minimum, maximum, count, dtype=torch.float64)


def _extract_probability_tensor(payload, key: str) -> torch.Tensor:
    if torch.is_tensor(payload):
        probabilities = payload
    elif isinstance(payload, dict):
        if key not in payload:
            available = ", ".join(sorted(str(item) for item in payload))
            raise KeyError(
                f"probability key {key!r} is absent; available keys: {available}"
            )
        probabilities = payload[key]
    else:
        raise TypeError(
            "probability cache must contain a tensor or a dictionary of tensors"
        )
    if not torch.is_tensor(probabilities) or probabilities.ndim != 3:
        raise ValueError(
            "probabilities must have shape (samples, current_time, outcomes)"
        )
    return probabilities


def load_probability_cache(path: Path, key: str = "probability_est") -> torch.Tensor:
    """Load a tensor cache without eagerly reading its storage when possible."""
    try:
        payload = torch.load(
            path, map_location="cpu", weights_only=True, mmap=True
        )
    except (RuntimeError, ValueError) as error:
        # mmap is supported only by the modern zip serialization.  Old project
        # caches remain valid, albeit without lazy storage loading.
        if "mmap" not in str(error).lower():
            raise
        payload = torch.load(path, map_location="cpu", weights_only=True)
    return _extract_probability_tensor(payload, key)


def validate_probability_tensor(
        probabilities: torch.Tensor,
        expected_horizon: int | None = 200,
) -> None:
    """Validate the conditional-PMF contract used by the survival model."""
    samples, current_times, outcomes = probabilities.shape
    if samples <= 0:
        raise ValueError("probability cache contains no samples")
    if outcomes != current_times + 1:
        raise ValueError(
            "the outcome axis must contain one event class per current time "
            "plus the terminal no-event class"
        )
    if expected_horizon is not None and current_times != expected_horizon:
        raise ValueError(
            f"expected horizon {expected_horizon}, found {current_times}"
        )
    # A bounded slice catches corrupt schemas cheaply.  Every computation chunk
    # is checked again before use, so validation remains complete without a
    # full eager pass over a memory-mapped production cache.
    preview = probabilities[:min(samples, 32)]
    if not torch.isfinite(preview).all():
        raise ValueError("probability cache contains non-finite values")
    if bool((preview < -1e-6).any()):
        raise ValueError("probability cache contains negative values")
    row_sums = preview.sum(dim=-1)
    if not torch.allclose(
            row_sums,
            torch.ones_like(row_sums),
            rtol=1e-4,
            atol=1e-5,
    ):
        raise ValueError("conditional probability rows do not sum to one")


def compute_current_time_quantiles(
        probabilities: torch.Tensor,
        quantile_levels: torch.Tensor,
        *,
        quantile_chunk_size: int = 128,
        output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compute one-based UPB quantiles for all current times in a tensor chunk."""
    validate_probability_tensor(probabilities, expected_horizon=None)
    quantile_levels = torch.as_tensor(
        quantile_levels, dtype=probabilities.dtype, device=probabilities.device
    ).reshape(-1)
    zero_based = compute_quantiles_survival_time(
        probabilities,
        quantile_levels,
        tail_distribution="geometric",
        quantile_chunk_size=quantile_chunk_size,
    )
    one_based = quantiles_to_interaction_counts(
        zero_based,
        width=probabilities.shape[1],
        upper_bound=probabilities.shape[1] + 1,
        allow_no_event_sentinel=True,
    )
    return one_based.to(dtype=output_dtype)


def _dtype_from_name(name: str) -> torch.dtype:
    return {"float16": torch.float16, "float32": torch.float32}[name]


def _required_storage_bytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    return math.prod(shape) * torch.empty((), dtype=dtype).element_size()


def _check_free_space(
        output_directory: Path,
        temporary_directory: Path,
        storage_bytes: int,
) -> None:
    """Require room for both the raw mmap and serialized artifact."""
    reserve = 64 * 1024 * 1024
    same_storage_device = (
        os.stat(output_directory).st_dev == os.stat(temporary_directory).st_dev
    )
    if same_storage_device:
        required = 2 * storage_bytes + reserve
        free = shutil.disk_usage(output_directory).free
        if free < required:
            raise OSError(
                f"insufficient free space: need about {required / 1e9:.2f} GB "
                f"in {output_directory}, have {free / 1e9:.2f} GB"
            )
        return
    requirements = (
        (output_directory, storage_bytes + reserve),
        (temporary_directory, storage_bytes + reserve),
    )
    for directory, required in requirements:
        free = shutil.disk_usage(directory).free
        if free < required:
            raise OSError(
                f"insufficient free space: need about {required / 1e9:.2f} GB "
                f"in {directory}, have {free / 1e9:.2f} GB"
            )


def generate_artifact(
        probability_cache: Path,
        output_path: Path,
        *,
        probability_key: str = "probability_est",
        candidate_count: int = 3000,
        quantile_min: float = 0.5,
        quantile_max: float = 0.95,
        sample_chunk_size: int = 8,
        quantile_chunk_size: int = 128,
        output_dtype: torch.dtype = torch.float32,
        expected_horizon: int | None = 200,
        temporary_directory: Path | None = None,
        overwrite: bool = False,
) -> Path:
    """Generate and atomically save the complete current-time artifact."""
    probability_cache = Path(probability_cache)
    output_path = Path(output_path)
    if not probability_cache.is_file():
        raise FileNotFoundError(
            f"probability cache does not exist: {probability_cache.resolve()}"
        )
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {output_path.resolve()}; pass --overwrite"
        )
    if sample_chunk_size <= 0:
        raise ValueError("sample chunk size must be positive")
    if quantile_chunk_size <= 0:
        raise ValueError("quantile chunk size must be positive")

    probabilities = load_probability_cache(probability_cache, probability_key)
    validate_probability_tensor(probabilities, expected_horizon)
    levels = make_candidate_quantile_levels(
        candidate_count, quantile_min, quantile_max
    )
    shape = (
        int(probabilities.shape[0]),
        int(probabilities.shape[1]),
        int(levels.numel()),
    )
    storage_bytes = _required_storage_bytes(shape, output_dtype)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = (
        output_path.parent
        if temporary_directory is None
        else Path(temporary_directory)
    )
    temp_dir.mkdir(parents=True, exist_ok=True)
    _check_free_space(output_path.parent, temp_dir, storage_bytes)

    token = uuid.uuid4().hex
    raw_path = temp_dir / f".{output_path.name}.{token}.raw"
    serialized_path = output_path.parent / f".{output_path.name}.{token}.tmp"
    output = None
    payload = None
    try:
        output = torch.from_file(
            str(raw_path),
            shared=True,
            size=math.prod(shape),
            dtype=output_dtype,
        ).reshape(shape)
        for start in range(0, shape[0], sample_chunk_size):
            stop = min(start + sample_chunk_size, shape[0])
            probability_chunk = probabilities[start:stop]
            if not torch.isfinite(probability_chunk).all():
                raise ValueError(
                    f"probability rows {start}:{stop} contain non-finite values"
                )
            if bool((probability_chunk < -1e-6).any()):
                raise ValueError(
                    f"probability rows {start}:{stop} contain negative values"
                )
            quantile_chunk = compute_current_time_quantiles(
                probability_chunk,
                levels,
                quantile_chunk_size=quantile_chunk_size,
                output_dtype=output_dtype,
            )
            output[start:stop].copy_(quantile_chunk)
            del probability_chunk, quantile_chunk

        payload = {
            "estimated_quantiles": output,
            "quantile_levels": levels,
            "shape": shape,
            "layout": (
                "calibration_plus_test_sample,current_time,candidate_quantile"
            ),
            "time_convention": (
                "one-based interaction counts; horizon+1 is no event in horizon"
            ),
            "tail_distribution": "geometric",
            "dataset_name": "dataset_autoif",
            "dataset_setup": AUTOIF_SETUP,
            "probability_source": str(probability_cache),
            "calibration_rows": 4000 if shape[0] == 6000 else None,
            "test_rows": 2000 if shape[0] == 6000 else None,
        }
        torch.save(payload, serialized_path)
        os.replace(serialized_path, output_path)
    finally:
        payload = None
        output = None
        gc.collect()
        raw_path.unlink(missing_ok=True)
        serialized_path.unlink(missing_ok=True)
    return output_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probability-cache", type=Path, default=DEFAULT_PROBABILITY_CACHE
    )
    parser.add_argument("--probability-key", default="probability_est")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidate-count", type=int, default=3000)
    parser.add_argument("--quantile-min", type=float, default=0.5)
    parser.add_argument("--quantile-max", type=float, default=0.95)
    parser.add_argument("--sample-chunk-size", type=int, default=8)
    parser.add_argument("--quantile-chunk-size", type=int, default=128)
    parser.add_argument(
        "--output-dtype", choices=["float16", "float32"], default="float32"
    )
    parser.add_argument("--expected-horizon", type=int, default=200)
    parser.add_argument("--temporary-directory", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    path = generate_artifact(
        args.probability_cache,
        args.output,
        probability_key=args.probability_key,
        candidate_count=args.candidate_count,
        quantile_min=args.quantile_min,
        quantile_max=args.quantile_max,
        sample_chunk_size=args.sample_chunk_size,
        quantile_chunk_size=args.quantile_chunk_size,
        output_dtype=_dtype_from_name(args.output_dtype),
        expected_horizon=args.expected_horizon,
        temporary_directory=args.temporary_directory,
        overwrite=args.overwrite,
    )
    print(f"Saved current-time quantiles to {path.resolve()}")


if __name__ == "__main__":
    main()
