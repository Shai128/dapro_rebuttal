"""Train two AutoIF survival models on disjoint halves of the training rows.

The first model is used only to produce current-time-zero UPB quantiles for the
calibration+test population.  The second model is used only to produce prompt
difficulty scores on that population.  No parameters, predictions, validation
rows, or outputs are combined between the two models.

Each result is a separate, self-contained ``.pt`` dictionary containing its
own model state, original training-row indices, and role-specific estimates.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.dataset_utils.data_utils import get_data
from src.dataset_utils.datasets import PartialSequenceDataset
from src.dataset_utils.real_data import generate_real_data
from src.predictive_bounds.calibration.calibration_utils import (
    quantiles_to_interaction_counts,
)
from src.predictive_bounds.experiments.autoif_cross_class.utils import (
    DEFAULT_AUTOIF_CLASSIFICATIONS_PATH,
    DEFAULT_AUTOIF_DATA_PATH,
    _normalize_target,
    get_autoif_candidate_classes,
    get_autoif_candidate_original_indices,
    load_autoif_classes_in_dataset_order,
)
from src.predictive_bounds.survival_utils.compute_mean_time_given_pmf import (
    compute_mean_survival_time,
    compute_quantiles_survival_time,
)
from src.train_model.acquisition_strategies.dummy_acquisition import (
    DummyAcquisition,
)
from src.train_model.active_learning import ActiveLearner
from src.train_model.models.transformer_survival_model import (
    DiscreteSurvivalLoss,
    TransformerSurvivalModel,
)
from src.utils.utils import set_seeds


AUTOIF_SETUP = (
    "attack_autoif_helper_qwen25_14b_instruct_lm_target_"
    "qwen25_14b_instruct_judge_autoif"
)
DEFAULT_OUTPUT_DIR = Path(
    "alg_playground_model/autoif_independent_split_models"
)


@dataclass(frozen=True)
class DisjointTrainingSplit:
    upb_indices: np.ndarray
    difficulty_indices: np.ndarray


@dataclass(frozen=True)
class FitMetadata:
    fit_indices: np.ndarray
    validation_indices: np.ndarray
    training_losses: tuple[float, ...]
    validation_losses: tuple[float, ...]


@dataclass(frozen=True)
class AutoIFCandidateMetadata:
    """Class and prompt identities aligned to calibration+test tensor rows."""

    class_labels: np.ndarray
    class_indices: np.ndarray
    class_names: tuple[str, ...]
    original_autoif_row_indices: np.ndarray
    prompt_sha256: tuple[str, ...]


def _candidate_event_times_from_stored_splits(
        dataset_name: str,
        dataset_setup: str,
        loader_seed: int,
) -> np.ndarray:
    """Independently replay only the event-time part of ``get_data``."""
    data = generate_real_data(dataset_name, dataset_setup, load_x=False)
    t_train, t_cal, t_test = data[9:12]
    loaded_times = np.concatenate([t_train, t_cal, t_test]).copy()
    rng = np.random.RandomState(loader_seed)
    rng.shuffle(loaded_times)
    return loaded_times[len(t_train):]


def load_and_validate_autoif_candidate_metadata(
        *,
        autoif_data_path: Path,
        classifications_path: Path,
        dataset_name: str,
        dataset_setup: str,
        loader_seed: int,
        evaluation_times: torch.Tensor,
        calibration_rows: int,
        test_rows: int,
        expected_class_count: int = 10,
) -> AutoIFCandidateMetadata:
    """Attach classes only after independently validating row correspondence.

    Validation has three separate layers:

    1. ``load_autoif_classes_in_dataset_order`` performs a normalized, unique,
       one-to-one prompt-text match between the helper and classification CSVs.
    2. Original CSV row indices are carried through the exact 40/40/20 split
       and the same seeded permutation used by ``get_data``.
    3. Stored event-time splits are replayed independently and must exactly
       equal the calibration+test event-time tensor passed by the model run.
    """
    autoif_data_path = Path(autoif_data_path)
    classifications_path = Path(classifications_path)
    classes_in_dataset_order = load_autoif_classes_in_dataset_order(
        autoif_data_path, classifications_path
    )
    original_indices = get_autoif_candidate_original_indices(
        len(classes_in_dataset_order), loader_seed=loader_seed
    )
    class_labels = get_autoif_candidate_classes(
        classes_in_dataset_order, loader_seed=loader_seed
    )
    expected_rows = int(calibration_rows) + int(test_rows)
    if len(class_labels) != expected_rows or len(original_indices) != expected_rows:
        raise ValueError(
            "AutoIF class mapping and evaluation tensors have different row "
            f"counts: labels={len(class_labels)}, original_indices="
            f"{len(original_indices)}, evaluation={expected_rows}."
        )
    if not np.array_equal(
            class_labels,
            np.asarray(classes_in_dataset_order)[original_indices],
    ):
        raise AssertionError(
            "class labels disagree with their reconstructed original CSV rows"
        )
    if len(np.unique(original_indices)) != len(original_indices):
        raise ValueError("an original AutoIF row was assigned more than once")

    class_names = tuple(sorted(np.unique(class_labels).tolist()))
    if len(class_names) != expected_class_count:
        raise ValueError(
            f"expected {expected_class_count} AutoIF classes, found "
            f"{len(class_names)}: {class_names}"
        )
    class_to_index = {name: index for index, name in enumerate(class_names)}
    class_indices = np.asarray(
        [class_to_index[label] for label in class_labels], dtype=np.int64
    )

    autoif_df = pd.read_csv(autoif_data_path, usecols=["target"])
    if len(autoif_df) != len(classes_in_dataset_order):
        raise ValueError(
            "AutoIF helper CSV changed between class matching and prompt audit"
        )
    normalized_prompts = autoif_df["target"].map(_normalize_target).to_numpy()
    candidate_prompts = normalized_prompts[original_indices]
    prompt_hashes = tuple(
        hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        for prompt in candidate_prompts
    )
    if len(set(prompt_hashes)) != len(prompt_hashes):
        raise ValueError("candidate AutoIF prompts are not uniquely identifiable")

    expected_times = _candidate_event_times_from_stored_splits(
        dataset_name, dataset_setup, loader_seed
    )
    actual_times = torch.as_tensor(evaluation_times).detach().cpu().numpy()
    shapes_match = expected_times.shape == actual_times.shape
    values_match = shapes_match and np.array_equal(expected_times, actual_times)
    if not values_match:
        preview = (
            np.flatnonzero(expected_times != actual_times)[:5].tolist()
            if shapes_match
            else []
        )
        raise ValueError(
            "AutoIF class rows do not align with calibration/test event times; "
            f"first mismatching evaluation rows: {preview}. The tensors and "
            "CSV files must come from the same dataset build and data seed."
        )
    return AutoIFCandidateMetadata(
        class_labels=np.asarray(class_labels, dtype=str),
        class_indices=class_indices,
        class_names=class_names,
        original_autoif_row_indices=original_indices.astype(np.int64),
        prompt_sha256=prompt_hashes,
    )


def make_disjoint_training_split(
        sample_count: int,
        seed: int,
) -> DisjointTrainingSplit:
    """Randomly partition every training row into two non-overlapping halves."""
    if sample_count < 4:
        raise ValueError("at least four training samples are required")
    if sample_count % 2:
        raise ValueError(
            "an even number of training rows is required for equal halves"
        )
    permutation = np.random.default_rng(int(seed)).permutation(sample_count)
    midpoint = sample_count // 2
    split = DisjointTrainingSplit(
        upb_indices=permutation[:midpoint].copy(),
        difficulty_indices=permutation[midpoint:].copy(),
    )
    if np.intersect1d(split.upb_indices, split.difficulty_indices).size:
        raise AssertionError("training halves overlap")
    if not np.array_equal(
            np.sort(np.concatenate([
                split.upb_indices, split.difficulty_indices
            ])),
            np.arange(sample_count),
    ):
        raise AssertionError("training halves do not partition all rows")
    return split


def _index_rows(tensor: torch.Tensor, indices: np.ndarray) -> torch.Tensor:
    index = torch.as_tensor(indices, dtype=torch.long, device=tensor.device)
    return tensor[index]


def fit_survival_model_on_rows(
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        event_times: torch.Tensor,
        original_indices: np.ndarray,
        *,
        device: torch.device,
        model_seed: int,
        validation_fraction: float = 0.1,
        max_epochs: int = 500,
        patience: int = 50,
        batch_size: int = 64,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-3,
        temperature_max_iter: int = 50,
) -> tuple[TransformerSurvivalModel, FitMetadata]:
    """Fit and temperature-calibrate one model using only ``original_indices``."""
    original_indices = np.asarray(original_indices, dtype=np.int64)
    if original_indices.ndim != 1 or len(original_indices) < 2:
        raise ValueError("each model needs at least two training rows")
    if len(np.unique(original_indices)) != len(original_indices):
        raise ValueError("training indices must be unique")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation fraction must lie in (0, 1)")
    if max_epochs <= 0 or patience <= 0 or batch_size <= 0:
        raise ValueError("epochs, patience, and batch size must be positive")

    set_seeds(model_seed)
    x_subset = _index_rows(x_train, original_indices)
    y_subset = _index_rows(y_train, original_indices)
    t_subset = _index_rows(event_times, original_indices)
    horizon = int(x_subset.shape[1])

    local_permutation = np.random.default_rng(model_seed).permutation(
        len(original_indices)
    )
    validation_count = max(1, int(round(
        validation_fraction * len(original_indices)
    )))
    validation_count = min(validation_count, len(original_indices) - 1)
    val_local = local_permutation[:validation_count]
    fit_local = local_permutation[validation_count:]

    dataset = PartialSequenceDataset(
        x_subset,
        y_subset,
        t_subset,
        dataset_name=AUTOIF_SETUP,
        initial_obs_len=1,
    )
    dataset.set_fully_observed(np.arange(len(dataset)))
    loss = DiscreteSurvivalLoss(censored_mode="full_survival")

    def loss_fn(model_output, batch):
        _, labels, _ = batch
        return loss(
            model_output,
            labels["censor_time"].to(device),
            labels["is_event"].long().to(device),
        )

    model_class = lambda: TransformerSurvivalModel(
        x_subset.shape[-1], horizon, dropout=0.2
    )
    learner = ActiveLearner(
        model_class=model_class,
        loss_fn=loss_fn,
        dataset=dataset,
        seed_indices=fit_local,
        val_indices=val_local,
        pool_indices=[],
        acquisition=DummyAcquisition(),
        device=device,
        retrain_from_scratch=True,
        verbose=False,
    )
    model = model_class()
    model, training_losses, validation_losses = learner._fit_model(
        model,
        fit_local.tolist(),
        val_local.tolist(),
        lr=learning_rate,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        weight_decay=weight_decay,
        checkpoint_path=None,
    )
    model.eval()
    model = model.to(device)
    x_validation = _index_rows(x_subset, val_local)
    t_validation = _index_rows(t_subset, val_local)
    model.calibrate(
        x_validation,
        t_validation,
        (t_validation <= horizon).long(),
        max_iter=temperature_max_iter,
        device=device,
    )
    model.eval()
    metadata = FitMetadata(
        fit_indices=original_indices[fit_local].copy(),
        validation_indices=original_indices[val_local].copy(),
        training_losses=tuple(float(value) for value in training_losses),
        validation_losses=tuple(float(value) for value in validation_losses),
    )
    return model, metadata


def make_candidate_quantile_levels(count: int = 3000) -> torch.Tensor:
    """Return the exact paper UPB grid without depending on the other script."""
    if count <= 0:
        raise ValueError("candidate quantile count must be positive")
    return torch.linspace(0.5, 0.95, count, dtype=torch.float64)


def estimate_upb_quantiles_in_batches(
        model,
        evaluation_tensors: tuple[torch.Tensor, ...],
        quantile_levels: torch.Tensor,
        *,
        device: torch.device,
        batch_size: int = 64,
        quantile_chunk_size: int = 128,
) -> torch.Tensor:
    """Use one model alone to estimate current-time-zero UPB quantiles."""
    if batch_size <= 0 or quantile_chunk_size <= 0:
        raise ValueError("batch and quantile chunk sizes must be positive")
    outputs = []
    model.eval()
    with torch.no_grad():
        for tensor in evaluation_tensors:
            for start in range(0, len(tensor), batch_size):
                batch = tensor[start:start + batch_size].float().to(device)
                probabilities = model.predict_proba(batch)
                zero_based = compute_quantiles_survival_time(
                    probabilities[:, :1],
                    quantile_levels,
                    tail_distribution="geometric",
                    quantile_chunk_size=quantile_chunk_size,
                ).squeeze(1)
                outputs.append(quantiles_to_interaction_counts(
                    zero_based,
                    width=probabilities.shape[1],
                    upper_bound=probabilities.shape[1] + 1,
                    allow_no_event_sentinel=True,
                ).to(torch.float32).cpu())
    if not outputs:
        raise ValueError("at least one calibration or test row is required")
    return torch.cat(outputs, dim=0)


def estimate_prompt_difficulty_in_batches(
        model,
        evaluation_tensors: tuple[torch.Tensor, ...],
        true_times: torch.Tensor,
        *,
        device: torch.device,
        batch_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ``abs(E[T | initial prompt] - true T)`` for every row."""
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    estimates = []
    model.eval()
    with torch.no_grad():
        for tensor in evaluation_tensors:
            for start in range(0, len(tensor), batch_size):
                batch = tensor[start:start + batch_size].float().to(device)
                probabilities = model.predict_proba(batch)
                # Repository utilities return zero-based coordinates.  Scores
                # compare one-based estimates against one-based true times.
                mean_time = compute_mean_survival_time(
                    probabilities[:, :1], tail_distribution="geometric"
                ).squeeze(1) + 1
                estimates.append(mean_time.to(torch.float32).cpu())
    if not estimates:
        raise ValueError("at least one calibration or test row is required")
    estimated_mean = torch.cat(estimates, dim=0)
    true_times = torch.as_tensor(true_times).detach().cpu().to(torch.float32)
    if estimated_mean.shape != true_times.shape:
        raise ValueError(
            "true times must contain one entry per calibration+test row"
        )
    return estimated_mean, torch.abs(estimated_mean - true_times)


def summarize_difficulty_by_class(
        *,
        class_labels,
        difficulty_scores,
        estimated_mean_times,
        true_times,
        event_observed,
        evaluation_splits,
        horizon: int,
) -> pd.DataFrame:
    """Build one sortable row of difficulty and event-time metrics per class."""
    labels = np.asarray(class_labels, dtype=str)
    absolute_error = torch.as_tensor(
        difficulty_scores
    ).detach().cpu().numpy().astype(np.float64)
    estimated = torch.as_tensor(
        estimated_mean_times
    ).detach().cpu().numpy().astype(np.float64)
    observed_time = torch.as_tensor(
        true_times
    ).detach().cpu().numpy().astype(np.float64)
    successful = torch.as_tensor(
        event_observed
    ).detach().cpu().numpy().astype(bool)
    splits = np.asarray(evaluation_splits, dtype=str)
    arrays = (labels, absolute_error, estimated, observed_time, successful, splits)
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("class-summary inputs must all be one-dimensional")
    if len({len(array) for array in arrays}) != 1 or len(labels) == 0:
        raise ValueError("class-summary inputs must have one shared nonzero length")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not set(splits).issubset({"calibration", "test"}):
        raise ValueError("evaluation splits must be 'calibration' or 'test'")
    if np.any(labels == ""):
        raise ValueError("class labels cannot be empty")
    if not all(np.isfinite(array).all() for array in (
            absolute_error, estimated, observed_time
    )):
        raise ValueError("difficulty and time values must be finite")
    if not np.allclose(
            absolute_error,
            np.abs(estimated - observed_time),
            rtol=1e-5,
            atol=1e-5,
    ):
        raise ValueError("difficulty scores do not equal the recorded absolute errors")
    if not np.array_equal(successful, observed_time <= horizon):
        raise ValueError("event indicators disagree with one-based event times")

    class_names = sorted(np.unique(labels).tolist())
    rows = []
    for class_index, class_name in enumerate(class_names):
        mask = labels == class_name
        errors = absolute_error[mask]
        estimates = estimated[mask]
        times = observed_time[mask]
        events = successful[mask]
        successful_errors = errors[events]
        unsuccessful_errors = errors[~events]
        successful_times = times[events]
        restricted_times = np.minimum(times, float(horizon))

        def conditional_stat(values, function):
            return float(function(values)) if len(values) else np.nan

        rows.append({
            "class_index": class_index,
            "class_name": class_name,
            "n_samples": int(mask.sum()),
            "n_calibration": int(np.sum(mask & (splits == "calibration"))),
            "n_test": int(np.sum(mask & (splits == "test"))),
            "n_successful": int(events.sum()),
            "n_unsuccessful": int((~events).sum()),
            "success_rate": float(events.mean()),
            "absolute_error_mean": float(errors.mean()),
            "absolute_error_median": float(np.median(errors)),
            "absolute_error_std": float(errors.std(ddof=0)),
            "absolute_error_q25": float(np.quantile(errors, 0.25)),
            "absolute_error_q75": float(np.quantile(errors, 0.75)),
            "absolute_error_p90": float(np.quantile(errors, 0.90)),
            "absolute_error_p95": float(np.quantile(errors, 0.95)),
            "absolute_error_max": float(errors.max()),
            "root_mean_squared_error": float(np.sqrt(np.mean(errors ** 2))),
            "signed_error_mean": float((estimates - times).mean()),
            "estimated_time_to_event_mean": float(estimates.mean()),
            "estimated_time_to_event_median": float(np.median(estimates)),
            # This preserves the repository's raw one-based representation;
            # no-event rows contribute horizon+1 (201 for AutoIF).
            "time_to_event_or_sentinel_mean": float(times.mean()),
            "time_to_event_or_sentinel_median": float(np.median(times)),
            # All tasks contribute here; no-event rows contribute the horizon.
            "restricted_time_to_event_mean": float(restricted_times.mean()),
            "restricted_time_to_event_median": float(
                np.median(restricted_times)
            ),
            # These are conventional time-to-event summaries conditional on
            # the event actually occurring inside the observed horizon.
            "successful_time_to_event_mean": conditional_stat(
                successful_times, np.mean
            ),
            "successful_time_to_event_median": conditional_stat(
                successful_times, np.median
            ),
            "successful_absolute_error_mean": conditional_stat(
                successful_errors, np.mean
            ),
            "successful_absolute_error_median": conditional_stat(
                successful_errors, np.median
            ),
            "unsuccessful_absolute_error_mean": conditional_stat(
                unsuccessful_errors, np.mean
            ),
            "unsuccessful_absolute_error_median": conditional_stat(
                unsuccessful_errors, np.median
            ),
        })
    summary = pd.DataFrame(rows)
    summary["difficulty_mean_rank_desc"] = summary[
        "absolute_error_mean"
    ].rank(method="min", ascending=False).astype(int)
    summary["difficulty_median_rank_desc"] = summary[
        "absolute_error_median"
    ].rank(method="min", ascending=False).astype(int)
    return summary


def _cpu_state_dict(model) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _save_result(payload: dict, path: Path, overwrite: bool) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {path.resolve()}; pass --overwrite"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _save_class_summary(
        summary: pd.DataFrame,
        path: Path,
        overwrite: bool,
) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {path.resolve()}; pass --overwrite"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        summary.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _base_payload(
        *,
        role: str,
        model,
        training_indices: np.ndarray,
        fit_metadata: FitMetadata,
        x_train: torch.Tensor,
        calibration_rows: int,
        test_rows: int,
        args,
) -> dict:
    return {
        "role": role,
        "dataset_name": args.dataset_name,
        "dataset_setup": args.dataset_setup,
        "data_seed": int(args.data_seed),
        "split_seed": int(args.split_seed),
        "model_seed": int(
            args.upb_model_seed
            if role == "upb_quantiles"
            else args.difficulty_model_seed
        ),
        "model_class": "TransformerSurvivalModel",
        "model_config": {
            "input_size": int(x_train.shape[-1]),
            "max_time": int(x_train.shape[1]),
            "dropout": 0.2,
        },
        "model_state_dict": _cpu_state_dict(model),
        "training_split_indices": torch.as_tensor(
            training_indices, dtype=torch.int64
        ),
        "fit_indices": torch.as_tensor(
            fit_metadata.fit_indices, dtype=torch.int64
        ),
        "temperature_validation_indices": torch.as_tensor(
            fit_metadata.validation_indices, dtype=torch.int64
        ),
        "training_losses": torch.tensor(fit_metadata.training_losses),
        "validation_losses": torch.tensor(fit_metadata.validation_losses),
        "calibration_rows": int(calibration_rows),
        "test_rows": int(test_rows),
        "evaluation_row_order": "calibration rows followed by test rows",
        "independence_contract": (
            "this file contains one model and only that model's estimates"
        ),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default="dataset_autoif")
    parser.add_argument("--dataset-setup", default=AUTOIF_SETUP)
    parser.add_argument(
        "--autoif-data-path", type=Path, default=DEFAULT_AUTOIF_DATA_PATH
    )
    parser.add_argument(
        "--classifications-path",
        type=Path,
        default=DEFAULT_AUTOIF_CLASSIFICATIONS_PATH,
    )
    parser.add_argument("--expected-class-count", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=1729)
    parser.add_argument("--upb-model-seed", type=int, default=1730)
    parser.add_argument("--difficulty-model-seed", type=int, default=1731)
    parser.add_argument("--candidate-count", type=int, default=3000)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--prediction-batch-size", type=int, default=64)
    parser.add_argument("--quantile-chunk-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--temperature-max-iter", type=int, default=50)
    parser.add_argument(
        "--upb-output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "upb_model_quantiles.pt",
    )
    parser.add_argument(
        "--difficulty-output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "difficulty_model_scores.pt",
    )
    parser.add_argument(
        "--class-summary-output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "difficulty_by_class.csv",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _fit_kwargs(args, device):
    return {
        "device": device,
        "validation_fraction": args.validation_fraction,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "temperature_max_iter": args.temperature_max_iter,
    }


def main(argv=None):
    args = parse_args(argv)
    output_paths = [
        Path(args.upb_output),
        Path(args.difficulty_output),
        Path(args.class_summary_output),
    ]
    if len({path.resolve() for path in output_paths}) != len(output_paths):
        raise ValueError("UPB, difficulty, and summary outputs must be different files")
    if not args.overwrite:
        for path in output_paths:
            if path.exists():
                raise FileExistsError(
                    f"output already exists: {path.resolve()}; "
                    "pass --overwrite"
                )
    device = torch.device(
        args.device
        if torch.cuda.is_available() and "cuda" in args.device
        else "cpu"
    )
    data = get_data(
        True,
        device,
        args.dataset_name,
        args.dataset_setup,
        load_x=True,
        seed=args.data_seed,
    )
    (
        _, _, _, x_train, x_cal, x_test, y_train, _, _, t_train,
        t_cal, t_test, _, e_cal, e_test, *_
    ) = data
    if x_train is None or x_cal is None or x_test is None:
        raise RuntimeError("AutoIF embeddings are required to train split models")
    split = make_disjoint_training_split(len(x_train), args.split_seed)
    levels = make_candidate_quantile_levels(args.candidate_count)
    evaluation = (x_cal, x_test)
    true_times = torch.cat([t_cal, t_test]).detach().cpu()
    event_observed = torch.cat([e_cal, e_test]).detach().cpu().to(torch.bool)
    expected_event_observed = true_times <= x_train.shape[1]
    if not torch.equal(event_observed, expected_event_observed):
        raise ValueError("loaded AutoIF event indicators disagree with event times")
    candidate_metadata = load_and_validate_autoif_candidate_metadata(
        autoif_data_path=args.autoif_data_path,
        classifications_path=args.classifications_path,
        dataset_name=args.dataset_name,
        dataset_setup=args.dataset_setup,
        loader_seed=args.data_seed,
        evaluation_times=true_times,
        calibration_rows=len(x_cal),
        test_rows=len(x_test),
        expected_class_count=args.expected_class_count,
    )

    # Model 1 has access only to the UPB half.  Save its artifact before model
    # 2 is created, preventing accidental parameter or prediction reuse.
    upb_model, upb_fit = fit_survival_model_on_rows(
        x_train,
        y_train,
        t_train,
        split.upb_indices,
        model_seed=args.upb_model_seed,
        **_fit_kwargs(args, device),
    )
    upb_quantiles = estimate_upb_quantiles_in_batches(
        upb_model,
        evaluation,
        levels,
        device=device,
        batch_size=args.prediction_batch_size,
        quantile_chunk_size=args.quantile_chunk_size,
    )
    upb_payload = _base_payload(
        role="upb_quantiles",
        model=upb_model,
        training_indices=split.upb_indices,
        fit_metadata=upb_fit,
        x_train=x_train,
        calibration_rows=len(x_cal),
        test_rows=len(x_test),
        args=args,
    )
    upb_payload.update({
        "estimated_quantiles": upb_quantiles,
        "quantile_levels": levels,
        "current_time_index": 0,
        "time_convention": (
            "one-based interaction counts; horizon+1 is no event in horizon"
        ),
        "tail_distribution": "geometric",
    })
    _save_result(upb_payload, args.upb_output, args.overwrite)
    del upb_payload, upb_quantiles, upb_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Model 2 starts from a fresh random initialization and receives only the
    # complementary training half.  It never reads the first result file.
    difficulty_model, difficulty_fit = fit_survival_model_on_rows(
        x_train,
        y_train,
        t_train,
        split.difficulty_indices,
        model_seed=args.difficulty_model_seed,
        **_fit_kwargs(args, device),
    )
    estimated_mean, difficulty_scores = estimate_prompt_difficulty_in_batches(
        difficulty_model,
        evaluation,
        true_times,
        device=device,
        batch_size=args.prediction_batch_size,
    )
    evaluation_splits = np.concatenate([
        np.repeat("calibration", len(x_cal)),
        np.repeat("test", len(x_test)),
    ])
    class_summary = summarize_difficulty_by_class(
        class_labels=candidate_metadata.class_labels,
        difficulty_scores=difficulty_scores,
        estimated_mean_times=estimated_mean,
        true_times=true_times,
        event_observed=event_observed,
        evaluation_splits=evaluation_splits,
        horizon=int(x_train.shape[1]),
    )
    difficulty_payload = _base_payload(
        role="prompt_difficulty",
        model=difficulty_model,
        training_indices=split.difficulty_indices,
        fit_metadata=difficulty_fit,
        x_train=x_train,
        calibration_rows=len(x_cal),
        test_rows=len(x_test),
        args=args,
    )
    difficulty_payload.update({
        "difficulty_scores": difficulty_scores,
        "score_definition": "abs(estimated_mean_time - true_time)",
        "estimated_mean_time_to_event": estimated_mean,
        "true_time_to_event": true_times.to(torch.float32),
        "event_observed": event_observed,
        "class_labels": candidate_metadata.class_labels.tolist(),
        "class_indices": torch.as_tensor(
            candidate_metadata.class_indices, dtype=torch.int64
        ),
        "class_names": list(candidate_metadata.class_names),
        "original_autoif_row_indices": torch.as_tensor(
            candidate_metadata.original_autoif_row_indices,
            dtype=torch.int64,
        ),
        "prompt_sha256": list(candidate_metadata.prompt_sha256),
        "evaluation_split_indices": torch.cat([
            torch.zeros(len(x_cal), dtype=torch.int8),
            torch.ones(len(x_test), dtype=torch.int8),
        ]),
        "evaluation_split_names": ["calibration", "test"],
        "autoif_data_path": str(args.autoif_data_path),
        "classifications_path": str(args.classifications_path),
        "class_summary_path": str(args.class_summary_output),
        "class_alignment_validation": (
            "unique normalized prompt match + original split indices + "
            "independent event-time permutation replay"
        ),
        "current_time_index": 0,
        "time_convention": (
            "one-based interaction counts; censored rows have true time horizon+1"
        ),
        "tail_distribution": "geometric",
    })
    _save_result(difficulty_payload, args.difficulty_output, args.overwrite)
    _save_class_summary(
        class_summary, args.class_summary_output, args.overwrite
    )
    print(f"Saved UPB model quantiles to {Path(args.upb_output).resolve()}")
    print(
        "Saved independent difficulty model scores to "
        f"{Path(args.difficulty_output).resolve()}"
    )
    print(
        "Saved sortable class difficulty summary to "
        f"{Path(args.class_summary_output).resolve()}"
    )


if __name__ == "__main__":
    main()
