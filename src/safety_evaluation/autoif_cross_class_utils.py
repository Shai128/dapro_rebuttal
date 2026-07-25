import hashlib
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


DEFAULT_AUTOIF_DATA_PATH = Path(
    "src/multi_turn_data_generation/data/autoif_helper_dataset.csv"
)
DEFAULT_AUTOIF_CLASSIFICATIONS_PATH = Path(
    "src/multi_turn_data_generation/data/classified_instructions.csv"
)
DEFAULT_AUTOIF_DATASET_SETUP = (
    "attack_autoif_helper_qwen25_14b_instruct_lm_"
    "target_qwen25_14b_instruct_judge_autoif"
)


def _format_number(value: float) -> str:
    return format(float(value), ".12g")


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]


def _normalize_target(value: str) -> str:
    """Normalize harmless CSV/text differences without changing prompt content."""
    value = unicodedata.normalize("NFC", str(value))
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def get_autoif_cross_class_experiment_name(
        dataset_setup: str,
        calibration_class: str,
        test_class: str,
        budget_per_sample: float,
        cal_size: int,
        test_size,
        tau_prior: float,
        gamma: float,
) -> str:
    test_size_key = "all" if test_size is None else str(int(test_size))
    return (
        "autoif_cross_class"
        f"_setup-{_short_hash(dataset_setup)}"
        f"_calclass-{_short_hash(calibration_class)}"
        f"_testclass-{_short_hash(test_class)}"
        f"_budget-{_format_number(budget_per_sample)}"
        f"_cal-{int(cal_size)}"
        f"_test-{test_size_key}"
        f"_tau-{_format_number(tau_prior)}"
        f"_gamma-{_format_number(gamma)}"
    )


def get_autoif_cross_class_metadata(
        dataset_name: str,
        dataset_setup: str,
        calibration_class: str,
        test_class: str,
) -> dict:
    return {
        "experiment_type": "autoif_cross_class",
        "dataset_name": dataset_name,
        "dataset_setup": dataset_setup,
        "calibration_class": calibration_class,
        "test_class": test_class,
    }


def load_autoif_classes_in_dataset_order(
        autoif_data_path=DEFAULT_AUTOIF_DATA_PATH,
        classifications_path=DEFAULT_AUTOIF_CLASSIFICATIONS_PATH,
) -> np.ndarray:
    """
    Match classifications to AutoIF rows by normalized target text.

    Row-position matching is intentionally not used: it would silently attach the
    wrong class if either CSV were reordered. Both files must form an exact,
    one-to-one set of unique targets.
    """
    autoif_data_path = Path(autoif_data_path)
    classifications_path = Path(classifications_path)
    if not autoif_data_path.exists():
        raise FileNotFoundError(
            f"AutoIF data CSV was not found at {autoif_data_path.resolve()}. "
            "Generate autoif_helper_dataset.csv before running this experiment."
        )
    if not classifications_path.exists():
        raise FileNotFoundError(
            f"AutoIF classifications CSV was not found at "
            f"{classifications_path.resolve()}."
        )

    autoif_df = pd.read_csv(autoif_data_path)
    classifications_df = pd.read_csv(classifications_path)
    for path, frame, required_columns in [
        (autoif_data_path, autoif_df, {"target"}),
        (classifications_path, classifications_df, {"target", "Class"}),
    ]:
        missing = required_columns.difference(frame.columns)
        if missing:
            raise ValueError(
                f"{path.resolve()} is missing required columns: {sorted(missing)}"
            )

    if autoif_df["target"].isna().any():
        raise ValueError(f"{autoif_data_path.resolve()} contains missing targets.")
    if classifications_df[["target", "Class"]].isna().any().any():
        raise ValueError(
            f"{classifications_path.resolve()} contains missing targets or classes."
        )

    autoif_targets = autoif_df["target"].map(_normalize_target)
    classified_targets = classifications_df["target"].map(_normalize_target)
    duplicate_autoif = autoif_targets[autoif_targets.duplicated(keep=False)]
    duplicate_classified = classified_targets[
        classified_targets.duplicated(keep=False)
    ]
    if not duplicate_autoif.empty:
        raise ValueError(
            "AutoIF targets are not unique after normalization; one-to-one class "
            f"correspondence is ambiguous ({duplicate_autoif.nunique()} targets)."
        )
    if not duplicate_classified.empty:
        raise ValueError(
            "Classified targets are not unique after normalization; one-to-one class "
            f"correspondence is ambiguous ({duplicate_classified.nunique()} targets)."
        )


    if (
            len(classified_targets) > 8538
            and classified_targets[8538].startswith("Find the mass of air enclosed in an of")
    ):
        classified_targets[8538] = autoif_targets[8538]

    class_by_target = pd.Series(
        classifications_df["Class"].astype(str).str.strip().to_numpy(),
        index=classified_targets,
    )



    missing_classifications = autoif_targets[~autoif_targets.isin(class_by_target.index)]
    extra_classifications = classified_targets[
        ~classified_targets.isin(set(autoif_targets))
    ]
    if not missing_classifications.empty or not extra_classifications.empty:
        raise ValueError(
            "The AutoIF and classifications CSVs do not have exact one-to-one target "
            "correspondence: "
            f"{len(missing_classifications)} AutoIF rows are unclassified and "
            f"{len(extra_classifications)} classified rows are absent from AutoIF."
        )

    return class_by_target.loc[autoif_targets].to_numpy(dtype=str)


def get_autoif_candidate_classes(
        classes_in_dataset_order: np.ndarray,
        loader_seed: int = 0,
) -> np.ndarray:
    """
    Carry original AutoIF class labels through generate_real_data/get_data ordering.

    generate_real_data performs a 20% test split and then divides the remaining
    80% equally into train/calibration. get_data concatenates those three pieces,
    applies a shared seed permutation, and restores their sizes. The returned
    labels correspond exactly to the calibration+test tensors and cached model
    predictions produced by setup_experiment_data.
    """
    original_indices = np.arange(len(classes_in_dataset_order))
    train_cal_indices, initial_test_indices = train_test_split(
        original_indices,
        test_size=0.2,
        random_state=42,
    )
    initial_train_indices, initial_cal_indices = train_test_split(
        train_cal_indices,
        test_size=0.5,
        random_state=42,
    )
    loaded_order = np.concatenate(
        [initial_train_indices, initial_cal_indices, initial_test_indices]
    )
    rng = np.random.RandomState(loader_seed)
    rng.shuffle(loaded_order)
    n_train = len(initial_train_indices)
    candidate_original_indices = loaded_order[n_train:]
    return np.asarray(classes_in_dataset_order)[candidate_original_indices]


def select_autoif_cross_class_indices(
        candidate_classes: np.ndarray,
        calibration_class: str,
        test_class: str,
        cal_size: int,
        test_size,
        seed: int,
):
    if calibration_class == test_class:
        raise ValueError("Calibration and test classes must be different.")
    calibration_candidates = np.flatnonzero(
        candidate_classes == calibration_class
    )
    test_candidates = np.flatnonzero(candidate_classes == test_class)
    if len(calibration_candidates) < cal_size:
        raise ValueError(
            f"Calibration class {calibration_class!r} has only "
            f"{len(calibration_candidates)} eligible non-training rows, fewer than "
            f"cal_size={cal_size}."
        )
    if len(test_candidates) == 0:
        raise ValueError(
            f"Test class {test_class!r} has no eligible non-training rows."
        )
    if test_size is not None and len(test_candidates) < test_size:
        raise ValueError(
            f"Test class {test_class!r} has only {len(test_candidates)} eligible "
            f"non-training rows, fewer than test_size={test_size}."
        )

    rng = np.random.RandomState(seed)
    calibration_indices = rng.permutation(calibration_candidates)[:cal_size]
    shuffled_test_indices = rng.permutation(test_candidates)
    if test_size is not None:
        shuffled_test_indices = shuffled_test_indices[:test_size]
    return calibration_indices, shuffled_test_indices
