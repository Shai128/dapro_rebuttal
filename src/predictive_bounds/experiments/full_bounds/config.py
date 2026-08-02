"""Immutable experiment and display configuration for the full comparison.

The matrix mirrors the manuscript: every LPB uses 90% target coverage;
toxicity, AutoIF, and Qwen-judged red-team use budget 20; hallucination and
Llama-Guard red-team use budget 10.  AutoIF UPB uses 70% target coverage and
budget 30.  Every configuration uses calibration size 3,000 and horizon 200.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class TargetModel:
    key: str
    model_id: str
    display_name: str


@dataclass(frozen=True)
class ExperimentConfig:
    key: str
    dataset_name: str
    dataset_setup: str
    target_model: TargetModel
    bound_type: str
    budget_per_sample: float
    tau_prior: float
    target_coverage: float
    figure_dataset_name: str
    display_dataset_name: str
    cal_size: int = 3000
    m_upper_bound: float = 200.0

    def model_cache_dir(self, root: Path) -> Path:
        return root / "alg_playground_model" / (
            f"is_real_True_dataset_{self.dataset_name}_dataset_"
            f"{self.dataset_setup}"
        )


TARGET_MODELS = (
    TargetModel("qwen", "qwen25_14b_instruct", "Qwen2.5"),
    TargetModel("llama", "llama_31_8B_instruct", "Llama3.1"),
    TargetModel("phi", "mini_phi_4_instruct", "Phi-4 Mini"),
    TargetModel("gemma", "gemma3_4b_it", "Gemma3"),
)


DATASET_SPECS = (
    {
        "key": "toxicity",
        "dataset_name": "dataset_toxicity",
        "setup": (
            "attack_toxic_attack_qwen25_14b_instruct_lm_target_{target}_"
            "judge_detoxify"
        ),
        "bound_type": "lpb",
        "budget": 20.0,
        "tau_prior": 0.56,
        "target_coverage": 0.90,
        "figure_name": "toxicity",
        "display_name": "Toxicity",
    },
    {
        "key": "red_team_qwen",
        "dataset_name": "dataset_red_team",
        "setup": (
            "attack_default_attack_qwen25_14b_instruct_lm_target_{target}_"
            "judge_llm-judge_qwen25_14b_instruct"
        ),
        "bound_type": "lpb",
        "budget": 20.0,
        "tau_prior": 0.56,
        "target_coverage": 0.90,
        "figure_name": "red_team_qwen",
        "display_name": "Red Team (Qwen judge)",
    },
    {
        "key": "red_team_llama_guard",
        "dataset_name": "dataset_red_team",
        "setup": (
            "attack_default_attack_qwen25_14b_instruct_lm_target_{target}_"
            "judge_llama_guard"
        ),
        "bound_type": "lpb",
        "budget": 10.0,
        "tau_prior": 0.56,
        "target_coverage": 0.90,
        "figure_name": "red_team_llama_guard",
        "display_name": "Red Team (Llama-Guard)",
    },
    {
        "key": "hallucination3",
        "dataset_name": "dataset_hallucination3",
        "setup": (
            "attack_hallucination_attack_qwen25_14b_instruct_lm_target_"
            "{target}_judge_llm-judge_qwen25_14b_instruct"
        ),
        "bound_type": "lpb",
        "budget": 10.0,
        "tau_prior": 0.56,
        "target_coverage": 0.90,
        "figure_name": "hallucination3",
        "display_name": "Hallucination",
    },
    {
        "key": "autoif",
        "dataset_name": "dataset_autoif",
        "setup": (
            "attack_autoif_helper_qwen25_14b_instruct_lm_target_{target}_"
            "judge_autoif"
        ),
        "bound_type": "lpb",
        "budget": 20.0,
        "tau_prior": 0.56,
        "target_coverage": 0.90,
        "figure_name": "autoif",
        "display_name": "AutoIF (LPB)",
    },
    {
        "key": "autoif_upb",
        "dataset_name": "dataset_autoif",
        "setup": (
            "attack_autoif_helper_qwen25_14b_instruct_lm_target_{target}_"
            "judge_autoif"
        ),
        "bound_type": "upb",
        "budget": 30.0,
        "tau_prior": 0.97,
        "target_coverage": 0.70,
        "figure_name": "autoif_upb",
        "display_name": "AutoIF (UPB)",
    },
)


UNCALIBRATED = "uncalibrated"
LPB_ORACLE = "oracle_survival_calibration"
UPB_ORACLE = "oracle_survival_upb_calibration"
STATIC = "calibration_optimized_allocation"
LOCALLY_ADAPTIVE = "calibration_adaptive_optimized_allocation"
CONSTANT = "calibration_random_adaptive_optimized_allocation"
POWER_REACH = "calibration_random_schedule_power_reach_alpha_2_crc_allocation"
LPB_DAPRO = (
    "calibration_dapro_variance_aligned_bins_2_alpha_0p10_global_0p001_"
    "budget_crc_control_100_row_cap_2p00x_budget_n1_200_allocation"
)
UPB_DAPRO = (
    "calibration_dapro_upb_variance_aligned_bins_2_alpha_0p70_global_0p001_"
    "budget_crc_control_100_row_cap_2p00x_budget_n1_200_allocation"
)

METHOD_ORDER = (
    "Uncalibrated",
    "Static Optimized",
    "Constant",
    "Power-Reach Constant",
    "Locally Adaptive",
    "DAPRO (CRC)",
    "Infinite-Budget Oracle",
)

METHOD_DISPLAY = {
    UNCALIBRATED: "Uncalibrated",
    LPB_ORACLE: "Infinite-Budget Oracle",
    UPB_ORACLE: "Infinite-Budget Oracle",
    STATIC: "Static Optimized",
    CONSTANT: "Constant",
    POWER_REACH: "Power-Reach Constant",
    LOCALLY_ADAPTIVE: "Locally Adaptive",
    LPB_DAPRO: "DAPRO (CRC)",
    UPB_DAPRO: "DAPRO (CRC)",
}

METHOD_COLORS = {
    "Uncalibrated": "#d62728",
    "Static Optimized": "#1f77b4",
    "Constant": "#9467bd",
    "Power-Reach Constant": "#8c564b",
    "Locally Adaptive": "#bcbd22",
    "DAPRO (CRC)": "#2ca02c",
    "Infinite-Budget Oracle": "#4d4d4d",
}


def calibration_names(bound_type: str) -> tuple[str, ...]:
    if bound_type not in {"lpb", "upb"}:
        raise ValueError(f"Unknown bound type: {bound_type!r}.")
    dapro = LPB_DAPRO if bound_type == "lpb" else UPB_DAPRO
    oracle = LPB_ORACLE if bound_type == "lpb" else UPB_ORACLE
    return (
        UNCALIBRATED,
        STATIC,
        CONSTANT,
        POWER_REACH,
        LOCALLY_ADAPTIVE,
        dapro,
        oracle,
    )


def all_experiment_configs() -> tuple[ExperimentConfig, ...]:
    configs = []
    for spec in DATASET_SPECS:
        for target in TARGET_MODELS:
            configs.append(ExperimentConfig(
                key=f"{spec['key']}__{target.key}",
                dataset_name=str(spec["dataset_name"]),
                dataset_setup=str(spec["setup"]).format(
                    target=target.model_id
                ),
                target_model=target,
                bound_type=str(spec["bound_type"]),
                budget_per_sample=float(spec["budget"]),
                tau_prior=float(spec["tau_prior"]),
                target_coverage=float(spec["target_coverage"]),
                figure_dataset_name=str(spec["figure_name"]),
                display_dataset_name=str(spec["display_name"]),
            ))
    return tuple(configs)


def select_configs(
        root: Path,
        *,
        keys: set[str] | None = None,
        target_models: set[str] | None = None,
        available_only: bool = False,
) -> tuple[ExperimentConfig, ...]:
    configs = all_experiment_configs()
    if keys:
        configs = tuple(config for config in configs if config.key in keys)
    if target_models:
        configs = tuple(
            config
            for config in configs
            if config.target_model.key in target_models
        )
    if available_only:
        configs = tuple(
            config
            for config in configs
            if _cache_exists(
                config.model_cache_dir(root) / "probability_est_cal_test.pt"
            )
        )
    return configs


def _cache_exists(path: Path) -> bool:
    absolute = str(path.resolve())
    if os.name == "nt" and not absolute.startswith("\\\\?\\"):
        absolute = f"\\\\?\\{absolute}"
    return os.path.exists(absolute)
