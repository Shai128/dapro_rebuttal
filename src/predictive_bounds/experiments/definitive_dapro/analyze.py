"""Recreate the historical projection-DAPRO tables and figures.

The script consumes immutable summary CSVs from the final 50-seed run and the
earlier 100-seed candidate audit. It never mixes their seed-level estimates.
The purpose is reproducible reporting, not hyperparameter fitting.  This
package is an ablation: the public DAPRO alias now denotes independent
row-capped CRC-DAPRO.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
FINAL_AUDIT = ROOT / "outputs" / "dapro_final_audit_2026-08-01"
CANDIDATE_AUDIT = (
    ROOT / "outputs" / "risk_hybrid_exact_crc_100_all_v5_audit"
)
OUTPUT = ROOT / "outputs" / "dapro_definitive_report"

DISPLAY_DATASET = {
    "toxicity": "Toxicity",
    "autoif": "AutoIF",
    "hallucination": "Hallucination",
    "redteam_llamaguard": "Red-team / Llama-Guard",
    "redteam_qwen": "Red-team / Qwen judge",
}

TARGET_BUDGET = {
    "toxicity": 20.0,
    "autoif": 20.0,
    "hallucination": 10.0,
    "redteam_llamaguard": 10.0,
    "redteam_qwen": 20.0,
}

FOCUS_RESULT_DIR = {
    "toxicity": "dataset_toxicity_attack_toxic_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_detoxify_20.0_3000_0.56_10.0__definitive_focus_v2",
    "autoif": "dataset_autoif_attack_autoif_helper_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_autoif_20.0_3000_0.56_10.0__definitive_focus_v2",
    "hallucination": "dataset_hallucination3_attack_hallucination_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_llm-judge_qwen25_14b_instruct_10.0_3000_0.56_20.0__definitive_focus_v2",
    "redteam_llamaguard": "dataset_red_team_attack_default_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_llama_guard_10.0_3000_0.56_20.0__definitive_focus_v2",
    "redteam_qwen": "dataset_red_team_attack_default_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_llm-judge_qwen25_14b_instruct_20.0_3000_0.56_10.0__definitive_focus_v2",
}

DEFINITIVE_NAME = (
    "calibration_dapro_variance_aligned_bins_2_alpha_0p10_global_0p001_"
    "projection_margin_1p00_n1_200_allocation"
)
RANDOM_NAME = "calibration_random_adaptive_optimized_allocation"


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 220,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
    })


def build_final_summary() -> pd.DataFrame:
    source = FINAL_AUDIT / "definitive_cross_dataset_results_50seeds.csv"
    frame = pd.read_csv(source)
    frame["dataset_display"] = frame["dataset"].map(DISPLAY_DATASET)
    frame["target_budget"] = frame["dataset"].map(TARGET_BUDGET)
    frame["budget_utilization"] = (
        frame["expected_budget"] / frame["target_budget"]
    )

    wide = frame.pivot(index="dataset", columns="method")
    rows = []
    for dataset in DISPLAY_DATASET:
        definitive_variance = wide.loc[
            dataset, ("coverage_variance_pp2", "Definitive DAPRO")
        ]
        random_variance = wide.loc[
            dataset, ("coverage_variance_pp2", "Random")
        ]
        rows.append({
            "dataset": dataset,
            "dataset_display": DISPLAY_DATASET[dataset],
            "n_seeds": int(wide.loc[
                dataset, ("n_seeds", "Definitive DAPRO")
            ]),
            "definitive_coverage_pct": wide.loc[
                dataset, ("coverage_pct", "Definitive DAPRO")
            ],
            "random_coverage_pct": wide.loc[
                dataset, ("coverage_pct", "Random")
            ],
            "definitive_variance_pp2": definitive_variance,
            "random_variance_pp2": random_variance,
            "variance_ratio": definitive_variance / random_variance,
            "variance_reduction_pct": 100 * (
                1 - definitive_variance / random_variance
            ),
            "definitive_expected_budget": wide.loc[
                dataset, ("expected_budget", "Definitive DAPRO")
            ],
            "random_expected_budget": wide.loc[
                dataset, ("expected_budget", "Random")
            ],
            "target_budget": TARGET_BUDGET[dataset],
            "definitive_mean_weight": wide.loc[
                dataset, ("mean_weight", "Definitive DAPRO")
            ],
            "random_mean_weight": wide.loc[
                dataset, ("mean_weight", "Random")
            ],
            "definitive_mean_A_over_pi": wide.loc[
                dataset, ("mean_A_over_pi", "Definitive DAPRO")
            ],
            "random_mean_A_over_pi": wide.loc[
                dataset, ("mean_A_over_pi", "Random")
            ],
            "phase1_objective": wide.loc[
                dataset, ("phase1_objective", "Definitive DAPRO")
            ],
            "phase2_objective": wide.loc[
                dataset, ("phase2_objective", "Definitive DAPRO")
            ],
            "objective_transfer_gap": wide.loc[
                dataset, ("phase2_phase1_gap", "Definitive DAPRO")
            ],
            "projection_cost_error_mean": wide.loc[
                dataset, ("transfer_mean", "Definitive DAPRO")
            ],
            "projection_cost_error_q95": wide.loc[
                dataset, ("transfer_q95", "Definitive DAPRO")
            ],
            "projection_assumption_rate": wide.loc[
                dataset, ("assumption_rate", "Definitive DAPRO")
            ],
        })
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT / "final_comparison.csv", index=False)
    return result


def build_candidate_screen() -> pd.DataFrame:
    source = CANDIDATE_AUDIT / "method_summary.csv"
    frame = pd.read_csv(source)
    selected = {
        "DAPRO (direct time)": "Legacy mean-weight DAPRO",
        "Direct raw-target + global 0.001 (N1=200)": (
            "Direct raw-target (unreserved)"
        ),
        "Random (hard pi>=0.005, CRC)": "Constant probability (CRC)",
        "Random-anchored target-A (target=0.75, CRC, control=100, N1=200)": (
            "Random-anchored hybrid"
        ),
        "Local (hard pi>=0.005, CRC)": "Locally adaptive (CRC)",
    }
    keep = frame[frame["method"].isin(selected)].copy()
    keep["candidate"] = keep["method"].map(selected)
    columns = [
        "dataset",
        "candidate",
        "n_seeds",
        "coverage_variance_pp2",
        "expected_budget_mean_per_sample",
        "mean_weight_all_calibration",
        "mean_target_a_over_pi_all_calibration",
        "mean_target_variance_proxy",
        "runtime_mean_seconds",
    ]
    keep = keep[columns].sort_values(["candidate", "dataset"])
    keep.to_csv(OUTPUT / "candidate_screen_100seeds.csv", index=False)
    return keep


def build_projection_sensitivity() -> pd.DataFrame:
    toxicity = pd.read_csv(
        FINAL_AUDIT / "definitive_margin_screen_toxicity.csv"
    )
    toxicity = toxicity[toxicity["margin"].notna()].copy()
    toxicity["scope"] = "Toxicity, 10 seeds"
    toxicity["dataset"] = "toxicity"

    margin_175 = pd.read_csv(
        FINAL_AUDIT / "definitive_margin175_cross_dataset_results.csv"
    )
    margin_175 = margin_175[
        margin_175["method"] == "Definitive DAPRO"
    ].copy()
    margin_175["N1"] = 200
    margin_175["margin"] = 1.75
    margin_175["scope"] = "All datasets, 20 seeds"
    margin_175 = margin_175.rename(columns={
        "coverage_variance_pp2": "variance_pp2",
    })

    columns = [
        "scope", "dataset", "N1", "margin", "variance_pp2",
        "expected_budget", "budget_gap", "assumption_rate",
        "transfer_mean", "transfer_q95", "mean_weight", "mean_A_over_pi",
    ]
    for column in columns:
        if column not in margin_175:
            if column == "budget_gap":
                margin_175[column] = margin_175["budget_gap_mean"]
            else:
                margin_175[column] = np.nan
    combined = pd.concat([toxicity[columns], margin_175[columns]])
    combined.to_csv(OUTPUT / "projection_sensitivity.csv", index=False)
    return combined


def _long_path(path: Path) -> str:
    absolute = str(path.resolve())
    if not absolute.startswith("\\\\?\\") and len(absolute) >= 240:
        return f"\\\\?\\{absolute}"
    return absolute


def build_allocation_focus() -> pd.DataFrame:
    columns = [
        "phase2_focus_a_rate",
        "phase2_focus_expected_query_share",
        "phase2_focus_expected_query_lift",
        "phase2_focus_mean_expected_queries",
        "phase2_nonfocus_mean_expected_queries",
        "phase2_focus_mean_terminal_probability",
        "phase2_nonfocus_mean_terminal_probability",
        "phase2_focus_terminal_probability_correlation",
    ]
    rows = []
    for dataset, directory in FOCUS_RESULT_DIR.items():
        source = (
            ROOT / "results" / "merged_calibration_dfs" / directory
            / "all_df.csv"
        )
        frame = pd.read_csv(_long_path(source))
        per_seed = frame[
            ["calibration_name", "seed", *columns]
        ].drop_duplicates(["calibration_name", "seed"])
        for method_name, display in (
            (DEFINITIVE_NAME, "Definitive DAPRO"),
            (RANDOM_NAME, "Constant"),
        ):
            method = per_seed[per_seed["calibration_name"] == method_name]
            if len(method) != 5:
                raise RuntimeError(
                    f"Expected five focus seeds for {dataset}/{display}; "
                    f"found {len(method)}."
                )
            row = {
                "dataset": dataset,
                "dataset_display": DISPLAY_DATASET[dataset],
                "method": display,
                "n_seeds": len(method),
            }
            row.update(method[columns].mean().to_dict())
            rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT / "allocation_focus_5seeds.csv", index=False)
    return result


def plot_final_comparison(summary: pd.DataFrame) -> None:
    labels = summary["dataset_display"].tolist()
    y = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8))

    ratios = summary["variance_ratio"].to_numpy()
    axes[0].barh(y, ratios, color="#247a4b")
    axes[0].axvline(1, color="#222222", linewidth=1)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("variance ratio (DAPRO / constant)")
    axes[0].set_title("Coverage variance")
    for index, value in enumerate(ratios):
        axes[0].text(value + 0.008, index, f"{value:.2f}", va="center")

    width = 0.36
    axes[1].barh(
        y - width / 2,
        summary["definitive_expected_budget"],
        height=width,
        label="Definitive DAPRO",
        color="#247a4b",
    )
    axes[1].barh(
        y + width / 2,
        summary["random_expected_budget"],
        height=width,
        label="Constant",
        color="#8e9aaf",
    )
    for index, target in enumerate(summary["target_budget"]):
        axes[1].plot(target, index, marker="|", color="#b11f2e", ms=16)
    axes[1].set_yticks(y, [])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("expected interactions per row")
    axes[1].set_title("Expected budget (red mark = target)")
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].barh(
        y - width / 2,
        summary["definitive_mean_A_over_pi"],
        height=width,
        color="#247a4b",
        label="Definitive DAPRO",
    )
    axes[2].barh(
        y + width / 2,
        summary["random_mean_A_over_pi"],
        height=width,
        color="#8e9aaf",
        label="Constant",
    )
    axes[2].set_yticks(y, [])
    axes[2].invert_yaxis()
    axes[2].set_xlabel(r"mean $A_i/\pi_i$")
    axes[2].set_title("Variance-aligned diagnostic")

    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT / f"final_comparison.{suffix}", bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity(summary: pd.DataFrame, sensitivity: pd.DataFrame) -> None:
    toxicity = sensitivity[
        sensitivity["scope"] == "Toxicity, 10 seeds"
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5))
    for n1, group in toxicity.groupby("N1"):
        group = group.sort_values("margin")
        axes[0].plot(
            group["margin"], group["variance_pp2"], marker="o", label=f"N1={int(n1)}"
        )
    axes[0].set_xlabel("projection reserve (interactions / row)")
    axes[0].set_ylabel(r"coverage variance (pp$^2$)")
    axes[0].set_title("Toxicity reserve and Phase-I sensitivity")
    axes[0].legend(frameon=False)

    x = np.arange(len(summary))
    axes[1].bar(x - 0.18, summary["phase1_objective"], 0.36, label="Phase I")
    axes[1].bar(x + 0.18, summary["phase2_objective"], 0.36, label="Phase II")
    axes[1].set_xticks(x, [DISPLAY_DATASET[d] for d in summary["dataset"]], rotation=30, ha="right")
    axes[1].set_ylabel(r"mean $(A_i+\delta)/(1+\delta)\,\pi_i^{-1}$")
    axes[1].set_title("Objective transfer")
    axes[1].legend(frameon=False)

    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT / f"sensitivity_and_transfer.{suffix}", bbox_inches="tight")
    plt.close(fig)


def plot_allocation_focus(focus: pd.DataFrame) -> None:
    wide = focus.pivot(index="dataset", columns="method")
    datasets = list(DISPLAY_DATASET)
    labels = [DISPLAY_DATASET[dataset] for dataset in datasets]
    y = np.arange(len(datasets))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7))

    for offset, method, color in (
        (-width / 2, "Definitive DAPRO", "#247a4b"),
        (width / 2, "Constant", "#8e9aaf"),
    ):
        target_pi = [
            wide.loc[
                dataset,
                ("phase2_focus_mean_terminal_probability", method),
            ]
            for dataset in datasets
        ]
        nonfocus_queries = [
            wide.loc[
                dataset,
                ("phase2_nonfocus_mean_expected_queries", method),
            ]
            for dataset in datasets
        ]
        axes[0].barh(
            y + offset,
            target_pi,
            height=width,
            label=method,
            color=color,
        )
        axes[1].barh(
            y + offset,
            nonfocus_queries,
            height=width,
            label=method,
            color=color,
        )

    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 1.04)
    axes[0].set_xlabel(r"mean target-event $\pi_i$")
    axes[0].set_title("Preserve target-event observability")
    axes[1].set_yticks(y, [])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("expected queries for non-target rows")
    axes[1].set_title("Avoid low-value late queries")

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT / f"allocation_focus.{suffix}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _style()
    summary = build_final_summary()
    build_candidate_screen()
    sensitivity = build_projection_sensitivity()
    focus = build_allocation_focus()
    plot_final_comparison(summary)
    plot_sensitivity(summary, sensitivity)
    plot_allocation_focus(focus)

    geometric_ratio = float(np.exp(np.log(summary["variance_ratio"]).mean()))
    print(f"Wrote final artifacts to {OUTPUT}")
    print(f"Geometric-mean variance ratio: {geometric_ratio:.6f}")
    print(f"Geometric-mean variance reduction: {100 * (1 - geometric_ratio):.2f}%")


if __name__ == "__main__":
    main()
