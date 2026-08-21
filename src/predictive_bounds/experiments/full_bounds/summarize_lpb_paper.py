"""Generate final-paper LPB figures at the recommended dataset settings."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.paper_figures.bounds import (  # noqa: E402
    generate_bound_appendix_figures,
    generate_lpb_main_figures,
)
from src.paper_figures.data import load_bound_paper_data  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
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
    frame, inventory, gaps = load_bound_paper_data(args.input_dir, "lpb")
    if args.strict and not gaps.empty:
        raise ValueError(
            "LPB paper matrix is incomplete:\n" + gaps.to_string(index=False)
        )
    data_dir = args.figures_root / "paper" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(data_dir / "lpb_seed_level_plot_data.csv", index=False)
    inventory.to_csv(data_dir / "lpb_result_inventory.csv", index=False)
    gaps.to_csv(data_dir / "lpb_schema_gaps.csv", index=False)
    appendix = generate_bound_appendix_figures(
        frame,
        task="lpb",
        output_dir=args.figures_root / "full",
        quality=args.quality,
    )
    main_manifest = generate_lpb_main_figures(
        frame,
        output_dir=args.figures_root / "paper" / "main",
        quality=args.quality,
    )
    manifest = pd.concat([appendix, main_manifest], ignore_index=True)
    manifest.to_csv(data_dir / "lpb_figure_manifest.csv", index=False)
    print(
        f"LPB: generated {int(manifest['generated'].sum())}/"
        f"{len(manifest)} requested figures."
    )
    if not gaps.empty:
        print(
            f"LPB: {len(gaps)} source gaps were recorded in "
            f"{data_dir / 'lpb_schema_gaps.csv'}."
        )


if __name__ == "__main__":
    main()
