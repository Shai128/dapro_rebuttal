"""Run a JSON-defined matrix of distribution-shift configurations.

Each JSON list entry is one explicit source/calibration/test domain triple.
The runner invokes the ordinary construct/merge/report lifecycle in a separate
process, keeping failures isolated and placing figures in a named subdirectory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys


SHIFT_TYPES = {"train_calibration_test_shift", "attacker_shift"}
ROOT = Path(__file__).resolve().parents[4]
PASSTHROUGH_FIELDS = (
    "dataset_name",
    "model_dataset_name",
    "calibration_dataset_name",
    "test_dataset_name",
    "model_dataset_setup",
    "calibration_dataset_setup",
    "test_dataset_setup",
    "bound_type",
    "cal_size",
    "test_size",
    "budget_per_sample",
    "tau_prior",
    "target_coverage",
    "m_upper_bound",
)


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    return value or "shift"


def load_matrix(path: Path, selected_shift_type: str | None = None) -> list[dict]:
    """Load and validate a list of explicit shift configurations."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("The shift matrix must be a nonempty JSON list.")
    selected = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Shift entry {index} must be a JSON object.")
        shift_type = item.get("shift_type")
        if shift_type not in SHIFT_TYPES:
            raise ValueError(
                f"Shift entry {index} has invalid shift_type={shift_type!r}."
            )
        missing = [
            field for field in (
                "model_dataset_setup",
                "calibration_dataset_setup",
                "test_dataset_setup",
            )
            if not item.get(field)
        ]
        if missing:
            raise ValueError(f"Shift entry {index} is missing {missing}.")
        if selected_shift_type is None or shift_type == selected_shift_type:
            selected.append(item)
    return selected


def command_for_configuration(item: dict, index: int, args) -> list[str]:
    """Translate one validated JSON entry into the standard runner CLI."""
    command = [
        sys.executable,
        "-m",
        "src.predictive_bounds.experiments.distribution_shift.run_all",
        "--shift-type",
        str(item["shift_type"]),
        "--seed-start",
        str(args.seed_start),
        "--seed-end",
        str(args.seed_end),
        "--device",
        args.device,
    ]
    for field in PASSTHROUGH_FIELDS:
        if field in item and item[field] is not None:
            command.extend(["--" + field.replace("_", "-"), str(item[field])])
    label = _slug(str(item.get("name", f"shift-{index:02d}")))
    command.extend(["--output-dir", str(args.output_dir / label)])
    return command


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--shift-type", choices=sorted(SHIFT_TYPES))
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/ablations/distribution_shift"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.seed_end <= args.seed_start:
        raise ValueError("seed-end must be greater than seed-start.")
    items = load_matrix(args.config_file, args.shift_type)
    if not items:
        raise ValueError("No shift-matrix entries match the requested shift type.")
    for index, item in enumerate(items):
        command = command_for_configuration(item, index, args)
        print(" ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
