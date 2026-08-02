"""Strict manifest-backed merge utilities shared by every experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd


def stable_experiment_name(prefix: str, metadata: Mapping[str, object]) -> str:
    """Return a short path-safe name whose hash covers every configuration field."""
    encoded = json.dumps(
        dict(metadata), sort_keys=True, separators=(",", ":"), default=str
    )
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:14]
    safe_prefix = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in prefix
    ).strip("-")
    return f"{safe_prefix}__{digest}"


def result_roots(bound_type: str, experiment_name: str) -> tuple[Path, Path]:
    if bound_type == "lpb":
        temporary_kind = "tmp_calibration_results"
        merged_kind = "merged_calibration_dfs"
    elif bound_type == "upb":
        temporary_kind = "tmp_upb_calibration_results"
        merged_kind = "merged_upb_calibration_dfs"
    else:
        raise ValueError("bound_type must be 'lpb' or 'upb'.")
    return (
        Path("results") / temporary_kind / experiment_name,
        Path("results") / merged_kind / experiment_name,
    )


def write_seed_manifest(
        experiment_name: str,
        seed: int,
        bound_type: str,
        calibration_names: Iterable[str],
        metadata: Mapping[str, object],
) -> Path:
    temporary_root, _ = result_roots(bound_type, experiment_name)
    manifest_dir = temporary_root / "_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_dir / f"seed={seed}.json"
    payload = {
        "seed": int(seed),
        "bound_type": bound_type,
        "calibration_names": list(calibration_names),
        **dict(metadata),
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)
    return path


def merge_sharded_bounds(
        experiment_name: str,
        seeds: tuple[int, int],
        bound_type: str,
        *,
        expected_metadata: Mapping[str, object] | None = None,
) -> Path:
    """Validate every manifest and method shard before writing one CSV."""
    if seeds[1] <= seeds[0]:
        raise ValueError("The seed interval must be nonempty.")
    temporary_root, merged_root = result_roots(bound_type, experiment_name)
    frames: list[pd.DataFrame] = []
    reference_methods: tuple[str, ...] | None = None
    for seed in range(*seeds):
        manifest_path = temporary_root / "_manifests" / f"seed={seed}.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing seed manifest: {manifest_path.resolve()}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("seed") != seed or manifest.get("bound_type") != bound_type:
            raise ValueError(f"Invalid seed/bound metadata in {manifest_path}.")
        for key, expected in (expected_metadata or {}).items():
            if manifest.get(key) != expected:
                raise ValueError(
                    f"Manifest {key!r} mismatch in {manifest_path}: "
                    f"{manifest.get(key)!r} != {expected!r}."
                )
        methods = tuple(manifest.get("calibration_names", ()))
        if not methods or len(set(methods)) != len(methods):
            raise ValueError(f"Manifest has an invalid method list: {manifest_path}")
        if reference_methods is None:
            reference_methods = methods
        elif methods != reference_methods:
            raise ValueError("All seed manifests must list the same ordered methods.")
        for method in methods:
            path = temporary_root / method / f"seed={seed}.csv"
            if not path.exists():
                raise FileNotFoundError(f"Missing method shard: {path.resolve()}")
            frame = pd.read_csv(path)
            frame = frame.drop(
                columns=[
                    column for column in frame.columns
                    if str(column).startswith("Unnamed:")
                ],
                errors="ignore",
            )
            if frame.empty:
                raise ValueError(f"Empty method shard: {path.resolve()}")
            if not frame["seed"].eq(seed).all():
                raise ValueError(f"Wrong seed in {path.resolve()}")
            if not frame["calibration_name"].eq(method).all():
                raise ValueError(f"Wrong calibration name in {path.resolve()}")
            frames.append(frame)
    merged_root.mkdir(parents=True, exist_ok=True)
    merged = pd.concat(frames, ignore_index=True)
    sort_columns = [
        column for column in ("calibration_name", "seed", "target_coverage")
        if column in merged
    ]
    merged = merged.sort_values(sort_columns).reset_index(drop=True)
    output = merged_root / "all_df.csv"
    merged.to_csv(output, index=False)
    return output


def write_table_shard(
        experiment_group: str,
        experiment_name: str,
        seed: int,
        frame: pd.DataFrame,
        metadata: Mapping[str, object],
) -> Path:
    """Atomically write one non-bound experimental table and its manifest."""
    if frame.empty:
        raise ValueError("A table shard cannot be empty.")
    root = Path("results") / "experiments" / experiment_group / experiment_name
    shard_dir = root / "shards"
    manifest_dir = root / "manifests"
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    output = shard_dir / f"seed={seed}.csv"
    temporary = output.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(output)
    manifest = manifest_dir / f"seed={seed}.json"
    manifest_tmp = manifest.with_suffix(".json.tmp")
    manifest_tmp.write_text(json.dumps({
        "seed": int(seed),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        **dict(metadata),
    }, indent=2), encoding="utf-8")
    manifest_tmp.replace(manifest)
    return output


def merge_table_shards(
        experiment_group: str,
        experiment_name: str,
        seeds: tuple[int, int],
        *,
        expected_metadata: Mapping[str, object] | None = None,
        unique_keys: Iterable[str] = (),
) -> Path:
    """Validate and merge generic per-seed CSV shards."""
    if seeds[1] <= seeds[0]:
        raise ValueError("The seed interval must be nonempty.")
    root = Path("results") / "experiments" / experiment_group / experiment_name
    frames = []
    reference_columns = None
    for seed in range(*seeds):
        manifest_path = root / "manifests" / f"seed={seed}.json"
        shard_path = root / "shards" / f"seed={seed}.csv"
        if not manifest_path.exists() or not shard_path.exists():
            raise FileNotFoundError(
                f"Incomplete table shard for seed {seed} under {root.resolve()}."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("seed") != seed:
            raise ValueError(f"Wrong seed in {manifest_path}.")
        for key, expected in (expected_metadata or {}).items():
            if manifest.get(key) != expected:
                raise ValueError(f"Manifest field {key!r} mismatches in {manifest_path}.")
        frame = pd.read_csv(shard_path)
        if len(frame) != manifest.get("rows"):
            raise ValueError(f"Row count mismatch in {shard_path}.")
        columns = tuple(frame.columns)
        if reference_columns is None:
            reference_columns = columns
        elif columns != reference_columns:
            raise ValueError("Table shard schemas do not match.")
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True)
    keys = list(unique_keys)
    if keys and merged.duplicated(keys).any():
        raise ValueError(f"Duplicate rows found for unique keys {keys}.")
    output = root / "all_df.csv"
    merged.to_csv(output, index=False)
    return output
