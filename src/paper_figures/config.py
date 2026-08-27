"""Fixed display and experiment choices for the final paper figures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RecommendedConfiguration:
    budget_per_sample: float
    n1: int
    target_coverage: float | None = None

    @property
    def crc_control_size(self) -> int:
        return self.n1 // 2


DATASET_ORDER = (
    "toxicity",
    "red_team_qwen_judge",
    "red_team_llama_guard",
    "autoif",
    "hallucination3",
)

DATASET_FILE_STEMS = {
    "toxicity": "toxicity",
    "red_team_qwen_judge": "red_team_qwen",
    "red_team_llama_guard": "red_team_llama_guard",
    "autoif": "autoif",
    "hallucination3": "hallucination",
}

DATASET_DISPLAY = {
    "toxicity": "Toxicity",
    "red_team_qwen_judge": "Red Team (Qwen judge)",
    "red_team_llama_guard": "Red Team (Llama-Guard)",
    "autoif": "AutoIF",
    "hallucination3": "Hallucination",
}

TARGET_MODEL_ORDER = (
    "Llama3.1",
    "Phi-4 Mini",
    "Qwen2.5",
    "Gemma3",
)

# Compact labels are used only in the narrow, main-text panels.  The appendix
# retains the fully qualified display names above.
MAIN_TARGET_MODEL_LABELS = {
    "Llama3.1": "Llama",
    "Phi-4 Mini": "Phi-4",
    "Qwen2.5": "Qwen",
    "Gemma3": "Gemma",
}
MAIN_TARGET_MODEL_ORDER = tuple(MAIN_TARGET_MODEL_LABELS.values())

METHOD_ORDER = (
    "Uncalibrated",
    "Static",
    "DAPRO",
    "DAPRO w/o CRC",
    "Oracle",
)

MAIN_METHOD_ORDER = (
    "Uncalibrated",
    "Static",
    "DAPRO",
)

METRIC_MAIN_METHOD_ORDER = (
    "Uncalibrated",
    "Static",
    "DAPRO",
)

# The red/blue/green family follows the manuscript references supplied by the
# user.  The two DAPRO variants deliberately remain visually related.
METHOD_COLORS = {
    "Uncalibrated": "#B64A46",
    "Static": "#4C78A8",
    "DAPRO": "#3A923A",
    "DAPRO w/o CRC": "#8BC77B",
    "Oracle": "#555555",
}

# Paper cells are selected with task-level hyperparameters rather than
# dataset-specific N1 tuning.  Within each task, every dataset at the same
# budget uses the same N1; both Red-Team judge variants also share one budget.
# UPB uses one target coverage for the entire benchmark matrix.
RECOMMENDED_CONFIGURATIONS = {
    "lpb": {
        "autoif": RecommendedConfiguration(10.0, 50, 0.90),
        "hallucination3": RecommendedConfiguration(10.0, 50, 0.90),
        "red_team_qwen_judge": RecommendedConfiguration(10.0, 50, 0.90),
        "red_team_llama_guard": RecommendedConfiguration(10.0, 50, 0.90),
        "toxicity": RecommendedConfiguration(20.0, 50, 0.90),
    },
    "upb": {
        "autoif": RecommendedConfiguration(20.0, 200, 0.80),
        "hallucination3": RecommendedConfiguration(20.0, 200, 0.80),
        "red_team_qwen_judge": RecommendedConfiguration(20.0, 200, 0.80),
        "red_team_llama_guard": RecommendedConfiguration(20.0, 200, 0.80),
        "toxicity": RecommendedConfiguration(20.0, 200, 0.80),
    },
    "metrics": {
        "autoif": RecommendedConfiguration(20.0, 100),
        "hallucination3": RecommendedConfiguration(20.0, 100),
        "red_team_qwen_judge": RecommendedConfiguration(20.0, 100),
        "red_team_llama_guard": RecommendedConfiguration(20.0, 100),
        "toxicity": RecommendedConfiguration(20.0, 100),
    },
}


def _configuration_line(
        task: str, dataset_key: str,
        recommendation: RecommendedConfiguration) -> str:
    coverage = (
        f", target coverage={100 * recommendation.target_coverage:g}%"
        if recommendation.target_coverage is not None else ""
    )
    return (
        f"- {task.upper()} / {DATASET_DISPLAY[dataset_key]}: "
        f"budget/sample={recommendation.budget_per_sample:g}, "
        f"total adaptive-calibration size="
        f"|I_cal1|+|I_crc|={recommendation.n1} "
        f"({recommendation.n1 - recommendation.crc_control_size}+"
        f"{recommendation.crc_control_size} with CRC){coverage}"
    )


def write_paper_configuration_summary(path: Path) -> None:
    """Write the authoritative configuration key beside the paper figures."""
    lines = [
        "PAPER EXPERIMENT CONFIGURATIONS",
        "===============================",
        "",
        "Notation: the configured code value `n1` is the total adaptive-",
        "calibration size |I_cal1|+|I_crc| for DAPRO with CRC. The standard",
        "split is even. DAPRO w/o CRC uses the full configured size as",
        "|I_cal1| and has |I_crc|=0. Static, Uncalibrated, and Oracle do not",
        "use this DAPRO split.",
        "",
        "MAIN-TEXT FIGURES",
        "-----------------",
        _configuration_line(
            "LPB", "toxicity", RECOMMENDED_CONFIGURATIONS["lpb"]["toxicity"]
        ),
        _configuration_line(
            "LPB", "autoif", RECOMMENDED_CONFIGURATIONS["lpb"]["autoif"]
        ),
        _configuration_line(
            "UPB", "autoif", RECOMMENDED_CONFIGURATIONS["upb"]["autoif"]
        ),
        _configuration_line(
            "Metrics", "red_team_llama_guard",
            RECOMMENDED_CONFIGURATIONS["metrics"]["red_team_llama_guard"],
        ),
        "",
        "FULL PAPER FIGURES",
        "------------------",
    ]
    for task in ("lpb", "upb", "metrics"):
        for dataset_key in DATASET_ORDER:
            lines.append(_configuration_line(
                task, dataset_key,
                RECOMMENDED_CONFIGURATIONS[task][dataset_key],
            ))
    lines.extend([
        "",
        "DAPRO ABLATIONS (Toxicity, Qwen2.5 attacker and target)",
        "-------------------------------------------------------",
        "- LPB target coverage=90%; metric task estimates event rate.",
        "- Default cell: budget/sample=20, "
        "|I_cal1|+|I_crc|=50 (25+25 with CRC).",
        "- Phase-I sample-size: total sizes 50, 100, 200, 300, 400; "
        "CRC receives half of each total.",
        "- Score noise, optimization objective, representation, score "
        "function, row-cost cap, and optimization-process studies: "
        "budget/sample=20, total size=50 (25+25 with CRC).",
        "- Budget study pairs budget/sample -> total size as "
        "5->25, 10->50, 20->50, 30->100, 40->150, 50->200.",
        "- CRC row-cost-cap multipliers: 0.1, 0.5, 1, 2, 5, and 10 "
        "times the target budget (the final value equals t_max=200).",
        "- Attacker shift (LPB only): budget/sample=10, total size=50; "
        "Qwen2.5 -> Gemma3 for both Red Team and Toxicity.",
        "- Uniform+CRC optimization control: all 50 adaptive-calibration "
        "rows are CRC rows; no policy-fitting fold is used.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
