"""Editable defaults for the two distribution-shift experiment families.

Dataset-specific source/calibration/test names are CLI overrides of the shared
default below, which keeps legacy same-dataset invocations working.
"""

DEFAULT_DATASET_NAME = "dataset_red_team"
DEFAULT_MODEL_SETUP = (
    "attack_default_attack_qwen25_14b_instruct_lm_target_"
    "qwen25_14b_instruct_judge_llm-judge_qwen25_14b_instruct"
)
DEFAULT_CALIBRATION_SETUP = DEFAULT_MODEL_SETUP
DEFAULT_TEST_SETUP = (
    "attack_toxic_attack_qwen25_14b_instruct_lm_target_"
    "qwen25_14b_instruct_judge_llm-judge_qwen25_14b_instruct"
)

# Override these values in the shell scripts (environment variables) or CLI.
DEFAULT_CAL_SIZE = 3000
DEFAULT_TEST_SIZE = 0  # zero means every available row in the test domain
DEFAULT_BUDGET = 20.0
DEFAULT_TAU_PRIOR = 0.56
DEFAULT_TARGET_COVERAGE = 0.90
DEFAULT_HORIZON = 200.0
