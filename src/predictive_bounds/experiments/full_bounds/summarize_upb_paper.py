"""Generate final-paper UPB figures and the combined AutoIF main panels."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.paper_figures.bounds import (  # noqa: E402
    generate_autoif_main_figures,
    generate_bound_appendix_figures,
)
from src.paper_figures.data import load_bound_paper_data  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=_ROOT / "results" / "merged_upb_calibration_dfs",
    )
    parser.add_argument(
        "--lpb-input-dir",
        type=Path,
        default=_ROOT / "results" / "merged_calibration_dfs",
    )
    parser.add_argument(
        "--figures-root", type=Path, default=_ROOT / "figures"
    )
    parser.add_argument("--quality", choices=("low", "high"), default="low")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any requested method or diagnostic is absent.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    frame, inventory, gaps = load_bound_paper_data(args.input_dir, "upb")
    lpb_frame, _, lpb_gaps = load_bound_paper_data(
        args.lpb_input_dir, "lpb"
    )
    if args.strict and (not gaps.empty or not lpb_gaps.empty):
        combined = pd.concat([gaps, lpb_gaps], ignore_index=True)
        raise ValueError(
            "UPB/AutoIF paper matrix is incomplete:\n"
            + combined.to_string(index=False)
        )
    data_dir = args.figures_root / "paper" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(data_dir / "upb_seed_level_plot_data.csv", index=False)
    inventory.to_csv(data_dir / "upb_result_inventory.csv", index=False)
    gaps.to_csv(data_dir / "upb_schema_gaps.csv", index=False)
    appendix = generate_bound_appendix_figures(
        frame,
        task="upb",
        output_dir=args.figures_root / "full",
        quality=args.quality,
    )
    main_manifest = generate_autoif_main_figures(
        lpb_frame,
        frame,
        output_dir=args.figures_root / "paper" / "main",
        quality=args.quality,
    )
    manifest = pd.concat([appendix, main_manifest], ignore_index=True)
    manifest.to_csv(data_dir / "upb_figure_manifest.csv", index=False)
    print(
        f"UPB: generated {int(manifest['generated'].sum())}/"
        f"{len(manifest)} requested figures."
    )
    if not gaps.empty:
        print(
            f"UPB: {len(gaps)} source gaps were recorded in "
            f"{data_dir / 'upb_schema_gaps.csv'}."
        )


if __name__ == "__main__":
    main()
