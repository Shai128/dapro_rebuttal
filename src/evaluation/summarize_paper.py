"""Generate final-paper metric-estimation figures at recommended settings."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.paper_figures.data import load_metric_paper_data  # noqa: E402
from src.paper_figures.metrics import (  # noqa: E402
    generate_metric_appendix_figures,
    generate_metric_main_figures,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=_ROOT / "results" / "merged_metric_calibration_dfs",
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
    frame, inventory, gaps = load_metric_paper_data(args.input_dir)
    if args.strict and not gaps.empty:
        raise ValueError(
            "Metric paper matrix is incomplete:\n" + gaps.to_string(index=False)
        )
    data_dir = args.figures_root / "paper" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(data_dir / "metric_seed_level_plot_data.csv", index=False)
    inventory.to_csv(data_dir / "metric_result_inventory.csv", index=False)
    gaps.to_csv(data_dir / "metric_schema_gaps.csv", index=False)
    appendix = generate_metric_appendix_figures(
        frame,
        output_dir=args.figures_root / "metrics",
        quality=args.quality,
    )
    main_manifest = generate_metric_main_figures(
        frame,
        output_dir=args.figures_root / "paper" / "main",
        quality=args.quality,
    )
    manifest = pd.concat([appendix, main_manifest], ignore_index=True)
    manifest.to_csv(data_dir / "metric_figure_manifest.csv", index=False)
    print(
        f"Metrics: generated {int(manifest['generated'].sum())}/"
        f"{len(manifest)} requested figures."
    )
    if not gaps.empty:
        print(
            f"Metrics: {len(gaps)} source gaps were recorded in "
            f"{data_dir / 'metric_schema_gaps.csv'}."
        )


if __name__ == "__main__":
    main()
