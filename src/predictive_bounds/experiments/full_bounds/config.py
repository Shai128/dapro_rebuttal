"""Immutable experiment and display configuration for the full comparison.

The matrix contains matching LPB and UPB experiments for every benchmark and
target model. LPBs use 90% coverage and UPBs use 70%, 80%, and 90%. UPB value
201 denotes infinity/no event through the 200-turn acquisition horizon.
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
        "key": "toxicity_upb",
        "dataset_name": "dataset_toxicity",
        "setup": (
            "attack_toxic_attack_qwen25_14b_instruct_lm_target_{target}_"
            "judge_detoxify"
        ),
        "bound_type": "upb",
        "budget": 20.0,
        "tau_prior": 0.97,
        "target_coverage": 0.70,
        "figure_name": "toxicity_upb",
        "display_name": "Toxicity (UPB)",
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
        "key": "red_team_qwen_upb",
        "dataset_name": "dataset_red_team",
        "setup": (
            "attack_default_attack_qwen25_14b_instruct_lm_target_{target}_"
            "judge_llm-judge_qwen25_14b_instruct"
        ),
        "bound_type": "upb",
        "budget": 20.0,
        "tau_prior": 0.97,
        "target_coverage": 0.70,
        "figure_name": "red_team_qwen_upb",
        "display_name": "Red Team (Qwen judge, UPB)",
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
        "key": "red_team_llama_guard_upb",
        "dataset_name": "dataset_red_team",
        "setup": (
            "attack_default_attack_qwen25_14b_instruct_lm_target_{target}_"
            "judge_llama_guard"
        ),
        "bound_type": "upb",
        "budget": 10.0,
        "tau_prior": 0.97,
        "target_coverage": 0.70,
        "figure_name": "red_team_llama_guard_upb",
        "display_name": "Red Team (Llama-Guard, UPB)",
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
        "key": "hallucination3_upb",
        "dataset_name": "dataset_hallucination3",
        "setup": (
            "attack_hallucination_attack_qwen25_14b_instruct_lm_target_"
            "{target}_judge_llm-judge_qwen25_14b_instruct"
        ),
        "bound_type": "upb",
        "budget": 10.0,
        "tau_prior": 0.97,
        "target_coverage": 0.70,
        "figure_name": "hallucination3_upb",
        "display_name": "Hallucination (UPB)",
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
LOCALLY_ADAPTIVE = "calibration_adaptive_optimized_crc_allocation"
CONSTANT = (
    "calibration_random_adaptive_optimized_mixture_terminal_floor_0p005_crc_"
    "allocation"
)
POWER_REACH = "calibration_random_schedule_power_reach_alpha_2_crc_allocation"
LEGACY_DAPRO = (
    "calibration_projected_optimization_direct_bins_2_prob_n1_200_allocation"
)
TARGET_A_DAPRO = (
    "calibration_projected_optimization_direct_bins_2_prob_a_target_raw_"
    "alpha_0p10_n1_200_allocation"
)
PROJECTION_DAPRO = (
    "calibration_dapro_variance_aligned_bins_2_alpha_0p10_global_0p001_"
    "projection_margin_1p00_n1_200_allocation"
)
LEGACY_CRC_DAPRO = (
    "calibration_projected_optimization_direct_bins_2_prob_budget_crc_"
    "control_100_row_cap_2p00x_budget_causal_shared_pav_v1_n1_200_"
    "allocation"
)
TARGET_A_CRC_DAPRO = (
    "calibration_projected_optimization_direct_bins_2_prob_a_target_raw_"
    "alpha_0p10_budget_crc_control_100_row_cap_2p00x_budget_"
    "causal_shared_pav_v1_n1_200_allocation"
)
LPB_DAPRO = (
    "calibration_dapro_variance_aligned_bins_2_alpha_0p10_global_0p001_"
    "budget_crc_control_100_row_cap_2p00x_budget_causal_shared_pav_v1_"
    "n1_200_allocation"
)
GENERALIZED_LPB_DAPRO = (
    "calibration_dapro_soft_prefix_bins_2_lpb_alpha_0p10_global_0p001_"
    "projection_margin_1p00_n1_200_allocation"
)
GENERALIZED_LPB_CRC_DAPRO = (
    "calibration_dapro_soft_prefix_bins_2_lpb_alpha_0p10_global_0p001_"
    "budget_crc_control_100_row_cap_2p00x_budget_causal_shared_pav_v1_"
    "n1_200_allocation"
)
UPB_DAPRO = (
    "calibration_dapro_soft_prefix_bins_2_upb_residual_aht_coverage_0p70_"
    "model_anchor_global_0p001_budget_crc_control_100_allocation"
)
GENERALIZED_UPB_DAPRO = (
    "calibration_dapro_soft_prefix_bins_2_upb_residual_aht_coverage_0p70_"
    "model_anchor_global_0p001_projection_margin_1p00_allocation"
)
GENERALIZED_UPB_CRC_DAPRO = (
    "calibration_dapro_soft_prefix_bins_2_upb_residual_aht_coverage_0p70_"
    "model_anchor_global_0p001_budget_crc_control_100_allocation"
)
SPLIT_DAPRO_ORACLE = (
    "calibration_oracle_target_a_dapro_alpha_0p10_n1_200_allocation"
)
CRC_DAPRO_ORACLE = (
    "calibration_oracle_target_a_dapro_alpha_0p10_crc_control_100_"
    "n1_200_allocation"
)
GLOBAL_DAPRO_ORACLE = (
    "calibration_oracle_target_a_dapro_no_split_alpha_0p10_allocation"
)

METHOD_ORDER = (
    "Static",
    "Soft-prefix DAPRO",
    "Soft-prefix DAPRO + CRC",
    "Information-gain + sequential AHT",
    "Information-gain + sequential AHT + CRC",
    "Residual + sequential AHT",
    "Residual + sequential AHT + CRC",
    "Endpoint/block + terminal residual AHT",
    "Endpoint/block + terminal residual AHT + CRC",
    "Oracle",
)

METHOD_DISPLAY = {
    UNCALIBRATED: "Raw",
    LPB_ORACLE: "Oracle",
    UPB_ORACLE: "Oracle",
    STATIC: "Static",
    CONSTANT: "Constant + CRC",
    POWER_REACH: "Power schedule + CRC",
    LOCALLY_ADAPTIVE: "Local + CRC",
    LEGACY_DAPRO: "Legacy DAPRO",
    TARGET_A_DAPRO: "Target-A DAPRO",
    PROJECTION_DAPRO: "DAPRO (projection)",
    LEGACY_CRC_DAPRO: "Legacy DAPRO + CRC",
    TARGET_A_CRC_DAPRO: "Target-A DAPRO + CRC",
    LPB_DAPRO: "DAPRO + CRC",
    GENERALIZED_LPB_DAPRO: "Generalized DAPRO (soft LPB)",
    GENERALIZED_LPB_CRC_DAPRO: "Generalized DAPRO (soft LPB) + CRC",
    GENERALIZED_UPB_DAPRO: "Generalized DAPRO (soft UPB)",
    GENERALIZED_UPB_CRC_DAPRO: "Generalized DAPRO (soft UPB) + CRC",
    SPLIT_DAPRO_ORACLE: "DAPRO Oracle (split)",
    CRC_DAPRO_ORACLE: "DAPRO Oracle + CRC",
    GLOBAL_DAPRO_ORACLE: "DAPRO Oracle (global)",
}

METHOD_COLORS = {
    "Static": "#1f77b4",
    "Soft-prefix DAPRO": "#665191",
    "Soft-prefix DAPRO + CRC": "#d45087",
    "Information-gain + sequential AHT": "#2ca02c",
    "Information-gain + sequential AHT + CRC": "#006d2c",
    "Residual + sequential AHT": "#ff9f40",
    "Residual + sequential AHT + CRC": "#c45a00",
    "Endpoint/block + terminal residual AHT": "#17a2b8",
    "Endpoint/block + terminal residual AHT + CRC": "#007f8b",
    "Oracle": "#4d4d4d",
}


def method_display_name(calibration_name: str) -> str | None:
    """Map versioned N1/coverage method names to stable paper labels."""
    name = str(calibration_name)
    if name in {LPB_ORACLE, UPB_ORACLE}:
        return "Oracle"
    if name == STATIC:
        return "Static"
    crc = "budget_crc" in name or "_crc_control_" in name
    if "endpoint_block_terminal_residual_aht" in name or (
            "upb_residual_aht_coverage" in name and "model_anchor" in name
    ):
        base = "Endpoint/block + terminal residual AHT"
    elif "dapro_information_gain" in name:
        base = "Information-gain + sequential AHT"
    elif "dapro_residual_sequential_aht" in name:
        base = "Residual + sequential AHT"
    elif "dapro_soft_prefix_bins_2" in name and (
            "lpb_alpha" in name or "upb_endpoint_dynamic_aht" in name
    ):
        base = "Soft-prefix DAPRO"
    else:
        return METHOD_DISPLAY.get(name)
    return f"{base} + CRC" if crc else base


def calibration_names(bound_type: str) -> tuple[str, ...]:
    if bound_type not in {"lpb", "upb"}:
        raise ValueError(f"Unknown bound type: {bound_type!r}.")
    oracle = LPB_ORACLE if bound_type == "lpb" else UPB_ORACLE
    methods = [
        UNCALIBRATED,
        STATIC,
        CONSTANT,
        POWER_REACH,
        oracle,
    ]
    if bound_type == "lpb":
        methods[-1:-1] = [
            GENERALIZED_LPB_DAPRO,
            GENERALIZED_LPB_CRC_DAPRO,
        ]
    else:
        methods[-1:-1] = [
            GENERALIZED_UPB_DAPRO,
            GENERALIZED_UPB_CRC_DAPRO,
        ]
    return tuple(methods)


def all_experiment_configs() -> tuple[ExperimentConfig, ...]:
    configs = []
    for spec in DATASET_SPECS:
        coverages = (
            (0.70, 0.80, 0.90)
            if spec["bound_type"] == "upb"
            else (float(spec["target_coverage"]),)
        )
        for target in TARGET_MODELS:
          for coverage in coverages:
            suffix = (
                f"__c{int(round(100 * coverage))}"
                if spec["bound_type"] == "upb"
                else ""
            )
            configs.append(ExperimentConfig(
                key=f"{spec['key']}__{target.key}{suffix}",
                dataset_name=str(spec["dataset_name"]),
                dataset_setup=str(spec["setup"]).format(
                    target=target.model_id
                ),
                target_model=target,
                bound_type=str(spec["bound_type"]),
                budget_per_sample=float(spec["budget"]),
                tau_prior=float(spec["tau_prior"]),
                target_coverage=float(coverage),
                figure_dataset_name=str(spec["figure_name"]),
                display_dataset_name=str(spec["display_name"]),
            ))
    return tuple(configs)


def select_configs(
        root: Path,
        *,
        keys: set[str] | None = None,
        target_models: set[str] | None = None,
        bound_types: set[str] | None = None,
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
    if bound_types:
        unknown = set(bound_types) - {"lpb", "upb"}
        if unknown:
            raise ValueError(f"Unknown bound types: {sorted(unknown)}")
        configs = tuple(
            config for config in configs
            if config.bound_type in bound_types
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
