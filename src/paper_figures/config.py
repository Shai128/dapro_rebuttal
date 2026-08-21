"""Fixed display and experiment choices for the final paper figures."""

from __future__ import annotations

from dataclasses import dataclass


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

# The red/blue/green family follows the manuscript references supplied by the
# user.  The two DAPRO variants deliberately remain visually related.
METHOD_COLORS = {
    "Uncalibrated": "#B64A46",
    "Static": "#4C78A8",
    "DAPRO": "#3A923A",
    "DAPRO w/o CRC": "#8BC77B",
    "Oracle": "#555555",
}

RECOMMENDED_CONFIGURATIONS = {
    "lpb": {
        "autoif": RecommendedConfiguration(10.0, 50, 0.90),
        "hallucination3": RecommendedConfiguration(10.0, 50, 0.90),
        "red_team_qwen_judge": RecommendedConfiguration(20.0, 50, 0.90),
        "red_team_llama_guard": RecommendedConfiguration(20.0, 50, 0.90),
        "toxicity": RecommendedConfiguration(20.0, 50, 0.90),
    },
    "upb": {
        "autoif": RecommendedConfiguration(20.0, 50, 0.80),
        "hallucination3": RecommendedConfiguration(20.0, 100, 0.80),
        "red_team_qwen_judge": RecommendedConfiguration(20.0, 100, 0.80),
        "red_team_llama_guard": RecommendedConfiguration(20.0, 100, 0.80),
        "toxicity": RecommendedConfiguration(20.0, 100, 0.80),
    },
    "metrics": {
        "autoif": RecommendedConfiguration(20.0, 50),
        "hallucination3": RecommendedConfiguration(20.0, 100),
        "red_team_qwen_judge": RecommendedConfiguration(20.0, 50),
        "red_team_llama_guard": RecommendedConfiguration(20.0, 50),
        "toxicity": RecommendedConfiguration(20.0, 50),
    },
}

