"""Shared discovery metadata for LPB and metric-estimation result matrices."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


_SETUP_RE = re.compile(
    r"^dataset_(?P<dataset>.+?)_attack_.*_lm_target_"
    r"(?P<target>.+?)_judge_(?P<judge>.+)$"
)
_LPB_SUFFIX_RE = re.compile(
    r"^lpb_all_methods_v1_n1_(?P<n1>\d+)_crc_(?P<crc>\d+)_"
    r"budget_(?P<budget>\d+(?:\.\d+)?)$"
)
_METRIC_RE = re.compile(
    r"^(?P<setup>dataset_.+)_"
    r"(?P<budget>\d+(?:\.\d+)?)_metric_estimation_"
    r"n1_(?P<n1>\d+)_crc_(?P<crc>\d+)(?:__.+)?$"
)


@dataclass(frozen=True)
class MatrixResult:
    path: Path
    dataset: str
    judge: str
    target_model: str
    budget_per_sample: float
    dapro_n1: int
    crc_control_size: int

    @property
    def dataset_key(self) -> str:
        if self.dataset == "red_team":
            if self.judge == "llama_guard":
                return "red_team_llama_guard"
            return "red_team_qwen_judge"
        return self.dataset

    @property
    def dataset_display(self) -> str:
        labels = {
            "toxicity": "Toxicity",
            "autoif": "AutoIF",
            "hallucination3": "Hallucination",
            "red_team_llama_guard": "Red Team (Llama-Guard)",
            "red_team_qwen_judge": "Red Team (Qwen judge)",
        }
        return labels.get(self.dataset_key, self.dataset_key.replace("_", " ").title())

    @property
    def target_model_display(self) -> str:
        labels = {
            "qwen25_14b_instruct": "Qwen2.5",
            "llama_31_8B_instruct": "Llama3.1",
            "mini_phi_4_instruct": "Phi-4 Mini",
            "gemma3_4b_it": "Gemma3",
        }
        return labels.get(
            self.target_model,
            self.target_model.replace("_", " ").title(),
        )


def _parse_setup(setup: str) -> tuple[str, str, str] | None:
    match = _SETUP_RE.match(setup)
    if match is None:
        return None
    return match.group("dataset"), match.group("target"), match.group("judge")


def parse_lpb_result(path: Path) -> MatrixResult | None:
    """Parse a suffixed all-method LPB result path."""
    name = path.parent.name
    if "__" not in name:
        return None
    base, suffix = name.rsplit("__", 1)
    suffix_match = _LPB_SUFFIX_RE.match(suffix)
    if suffix_match is None:
        return None
    pieces = base.rsplit("_", 4)
    if len(pieces) != 5:
        return None
    setup, base_budget, _cal_size, _tau_prior, _gamma = pieces
    parsed_setup = _parse_setup(setup)
    if parsed_setup is None:
        return None
    dataset, target, judge = parsed_setup
    budget = float(suffix_match.group("budget"))
    if abs(float(base_budget) - budget) > 1e-9:
        raise ValueError(
            f"LPB budget mismatch in {path}: {base_budget} versus {budget}."
        )
    return MatrixResult(
        path=path,
        dataset=dataset,
        judge=judge,
        target_model=target,
        budget_per_sample=budget,
        dapro_n1=int(suffix_match.group("n1")),
        crc_control_size=int(suffix_match.group("crc")),
    )


def parse_metric_result(path: Path) -> MatrixResult | None:
    """Parse a metric-estimation result path."""
    match = _METRIC_RE.match(path.parent.name)
    if match is None:
        return None
    parsed_setup = _parse_setup(match.group("setup"))
    if parsed_setup is None:
        return None
    dataset, target, judge = parsed_setup
    return MatrixResult(
        path=path,
        dataset=dataset,
        judge=judge,
        target_model=target,
        budget_per_sample=float(match.group("budget")),
        dapro_n1=int(match.group("n1")),
        crc_control_size=int(match.group("crc")),
    )


def numeric_label(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def method_display_name(name: str) -> str:
    """Map N1-dependent canonical names to stable plot labels."""
    if name == "uncalibrated":
        return "Raw"
    if name in {"oracle_survival_calibration", "oracle_survival_upb_calibration"}:
        return "Oracle"
    if name == "oracle_full_budget":
        return "Full-budget oracle"
    if "UnweightedUniformBudgetAllocator" in name:
        return "Uniform (unweighted)"
    if "UniformBudgetAllocator" in name:
        return "Uniform + reweighting"
    if name == "calibration_optimized_allocation" or name == "optimized":
        return "Static"
    if (
        "random_adaptive_optimized" in name
        and "crc" in name
        and "random_schedule" not in name
    ):
        return "Constant + CRC"
    if "adaptive_optimized_crc" in name:
        return "Local + CRC"
    if "oracle_target_a_dapro_no_split" in name:
        return "DAPRO Oracle (global)"
    if "oracle_target_a_dapro" in name and "crc_control" in name:
        return "DAPRO Oracle + CRC"
    if "oracle_target_a_dapro" in name:
        return "DAPRO Oracle (split)"
    if "dapro_variance_aligned" in name and "budget_crc" in name:
        return "DAPRO + CRC"
    if "dapro_variance_aligned" in name:
        return "DAPRO (projection)"
    if "a_target_raw" in name and "budget_crc" in name:
        return "Target-A DAPRO + CRC"
    if "a_target_raw" in name:
        return "Target-A DAPRO"
    if "projected_optimization_direct_bins_2_prob" in name and "budget_crc" in name:
        return "Legacy DAPRO + CRC"
    if "projected_optimization_direct_bins_2_prob" in name:
        return "Legacy DAPRO"
    cleaned = name
    for token in ("calibration_", "_allocation"):
        cleaned = cleaned.replace(token, "")
    return cleaned.replace("_", " ").strip().title()


METHOD_ORDER = (
    "Raw",
    "Uniform (unweighted)",
    "Uniform + reweighting",
    "Static",
    "Constant + CRC",
    "Local + CRC",
    "Legacy DAPRO",
    "Target-A DAPRO",
    "DAPRO (projection)",
    "Legacy DAPRO + CRC",
    "Target-A DAPRO + CRC",
    "DAPRO + CRC",
    "DAPRO Oracle (split)",
    "DAPRO Oracle + CRC",
    "DAPRO Oracle (global)",
    "Oracle",
    "Full-budget oracle",
)

DAPRO_ORACLE_METHODS = frozenset({
    "DAPRO Oracle (split)",
    "DAPRO Oracle + CRC",
    "DAPRO Oracle (global)",
})


METHOD_COLORS = {
    "Raw": "#d62728",
    "Uniform (unweighted)": "#7f7f7f",
    "Uniform + reweighting": "#4c78a8",
    "Static": "#1f77b4",
    "Constant + CRC": "#9467bd",
    "Local + CRC": "#bcbd22",
    "Legacy DAPRO": "#ff9f40",
    "Target-A DAPRO": "#17a2b8",
    "DAPRO (projection)": "#2ca02c",
    "Legacy DAPRO + CRC": "#c45a00",
    "Target-A DAPRO + CRC": "#007f8b",
    "DAPRO + CRC": "#006d2c",
    "DAPRO Oracle (split)": "#e377c2",
    "DAPRO Oracle + CRC": "#7f3c8d",
    "DAPRO Oracle (global)": "#11a579",
    "Oracle": "#4d4d4d",
    "Full-budget oracle": "#4d4d4d",
}
