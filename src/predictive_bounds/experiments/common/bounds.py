"""Reusable primitives for running calibrated bounds on an explicit split.

Experiment packages should prepare calibration/test tensors and metadata, then
delegate method construction, common acquisition randomness, result writing,
and completion manifests to this module.  This keeps experimental perturbations
separate from the production calibration implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from src.predictive_bounds.construct_calibrated_bound import (
    _make_common_acquisition_uniforms,
    get_calibration_methods,
    run_one_experiment,
)
from src.predictive_bounds.experiments.common.results import (
    write_seed_manifest,
)
from src.predictive_bounds.experiments.full_bounds.config import (
    calibration_names as paper_calibration_names,
)
from src.train_model.models.utils import SurvivalModelPrediction


@dataclass(frozen=True)
class BoundGrid:
    bound_type: str
    taus: torch.Tensor
    target_levels: np.ndarray
    target_coverage: float


def make_bound_grid(
        bound_type: str,
        device: str | torch.device,
        *,
        target_coverage: float | None = None,
) -> BoundGrid:
    """Construct the repository's canonical candidate and reporting grids."""
    if bound_type == "lpb":
        targets = np.arange(0.01, 0.5, 0.01)
        taus = torch.tensor(np.logspace(-3, -0.01, 1000), device=device)
        default_coverage = 0.90
    elif bound_type == "upb":
        targets = 1 - np.arange(0.01, 0.5, 0.01)
        taus = torch.tensor(np.linspace(0.5, 0.95, 3000), device=device)
        default_coverage = 0.70
    else:
        raise ValueError("bound_type must be 'lpb' or 'upb'.")
    requested = default_coverage if target_coverage is None else target_coverage
    if not 0 < requested < 1:
        raise ValueError("target_coverage must lie in (0, 1).")
    if not np.any(np.isclose(
            1 - targets if bound_type == "lpb" else targets,
            requested,
            atol=5e-7,
    )):
        raise ValueError(
            f"Target coverage {requested} is not represented by the reporting grid."
        )
    return BoundGrid(bound_type, taus, targets, float(requested))


def select_paper_calibrations(
        conditional_grid: torch.Tensor,
        t_cal: torch.Tensor,
        quantile_cal: torch.Tensor,
        probability_cal: torch.Tensor,
        grid: BoundGrid,
        *,
        budget_per_sample: float,
        tau_prior: float,
        m_upper_bound: float,
        device: str | torch.device,
        calibration_names: Iterable[str] | None = None,
        dapro_n1_values: tuple[int, ...] = (200,),
):
    """Build only the requested paper methods, including the infinite oracle."""
    prediction = SurvivalModelPrediction(quantile_cal, probability_cal)
    methods = get_calibration_methods(
        conditional_grid,
        budget_per_sample,
        grid.taus,
        tau_prior,
        m_upper_bound,
        "one",
        prediction,
        t_cal,
        device,
        grid.bound_type,
        dapro_n1_values=dapro_n1_values,
        definitive_dapro_margins=(1.0,),
    )
    requested = tuple(
        calibration_names
        if calibration_names is not None
        else paper_calibration_names(grid.bound_type)
    )
    available = {method.name: method for method in methods}
    missing = sorted(set(requested) - set(available))
    if missing:
        raise ValueError(
            f"Unknown calibration methods {missing}; available={sorted(available)}"
        )
    return [available[name] for name in requested]


def run_bound_split(
        *,
        experiment_name: str,
        seed: int,
        grid: BoundGrid,
        t_cal: torch.Tensor,
        quantile_cal: torch.Tensor,
        probability_cal: torch.Tensor,
        conditional_grid_cal: torch.Tensor,
        t_test: torch.Tensor,
        quantile_test: torch.Tensor,
        probability_test: torch.Tensor,
        budget_per_sample: float,
        tau_prior: float,
        m_upper_bound: float,
        device: str | torch.device,
        metadata: dict,
        calibration_names: Iterable[str] | None = None,
        dapro_n1_values: tuple[int, ...] = (200,),
        policy_seed: int | None = None,
        acquisition_seed: int | None = None,
        skip_existing: bool = True,
) -> tuple[str, ...]:
    """Run a complete method set on caller-provided calibration/test rows."""
    if len(t_cal) != len(quantile_cal) or len(t_cal) != len(probability_cal):
        raise ValueError("Calibration tensors must have the same row count.")
    if len(t_test) != len(quantile_test) or len(t_test) != len(probability_test):
        raise ValueError("Test tensors must have the same row count.")
    methods = select_paper_calibrations(
        conditional_grid_cal,
        t_cal,
        quantile_cal,
        probability_cal,
        grid,
        budget_per_sample=budget_per_sample,
        tau_prior=tau_prior,
        m_upper_bound=m_upper_bound,
        device=device,
        calibration_names=calibration_names,
        dapro_n1_values=dapro_n1_values,
    )
    acquisition_seed = seed if acquisition_seed is None else acquisition_seed
    uniforms = _make_common_acquisition_uniforms(
        acquisition_seed,
        len(t_cal),
        int(conditional_grid_cal.shape[1]),
    )
    cal_prediction = SurvivalModelPrediction(quantile_cal, probability_cal)
    test_prediction = SurvivalModelPrediction(quantile_test, probability_test)
    complete_metadata = {
        "experiment_name": experiment_name,
        "configured_cal_size": int(len(t_cal)),
        "configured_test_size": int(len(t_test)),
        "configured_budget_per_sample": float(budget_per_sample),
        "configured_tau_prior": float(tau_prior),
        "configured_m_upper_bound": float(m_upper_bound),
        "target_coverage": float(grid.target_coverage),
        **metadata,
    }
    for calibration in methods:
        run_one_experiment(
            experiment_name,
            seed,
            calibration,
            None,
            t_cal,
            cal_prediction,
            None,
            t_test,
            test_prediction,
            grid.target_levels,
            grid.bound_type,
            skip_existing,
            experiment_metadata=complete_metadata,
            policy_seed=policy_seed,
            acquisition_seed=acquisition_seed,
            acquisition_uniforms=uniforms,
        )
    names = tuple(method.name for method in methods)
    write_seed_manifest(
        experiment_name,
        seed,
        grid.bound_type,
        names,
        complete_metadata,
    )
    return names
