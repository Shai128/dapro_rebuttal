"""Editable paper defaults for calibration-only judge noise."""

DEFAULT_DATASET_NAME = "dataset_red_team"
DEFAULT_DATASET_SETUP = (
    "attack_default_attack_qwen25_14b_instruct_lm_target_"
    "qwen25_14b_instruct_judge_llm-judge_qwen25_14b_instruct"
)
DEFAULT_NOISE_LEVELS = (0.01, 0.05, 0.10, 0.20)

