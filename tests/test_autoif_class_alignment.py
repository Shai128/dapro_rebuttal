import hashlib

import numpy as np
import pandas as pd
import pytest
import torch

from src.predictive_bounds.experiments.autoif_cross_class.utils import (
    DEFAULT_AUTOIF_CLASSIFICATIONS_PATH,
    DEFAULT_AUTOIF_DATA_PATH,
    _normalize_target,
    get_autoif_candidate_classes,
    get_autoif_candidate_original_indices,
    load_autoif_classes_in_dataset_order,
)
from src.predictive_bounds.experiments.autoif_independent_models import (
    train_independent_split_models as script,
)


EXPECTED_CANDIDATE_CLASS_COUNTS = {
    "Academic & Educational": 94,
    "Arts & Entertainment": 71,
    "Business & Career": 135,
    "Communication": 596,
    "Creative Writing": 163,
    "Health & Lifestyle": 32,
    "Marketing & Social Media": 122,
    "Miscellaneous Tasks": 3793,
    "Programming & Technology": 929,
    "Text Processing": 65,
}


def test_real_autoif_candidate_classes_have_exact_expected_counts():
    classes = load_autoif_classes_in_dataset_order()
    candidate_classes = get_autoif_candidate_classes(classes, loader_seed=0)
    names, counts = np.unique(candidate_classes, return_counts=True)

    assert len(classes) == 10_000
    assert len(candidate_classes) == 6_000
    assert dict(zip(names, counts)) == EXPECTED_CANDIDATE_CLASS_COUNTS


def test_every_real_class_has_a_candidate_prompt_with_matching_classification():
    classes = load_autoif_classes_in_dataset_order()
    original_indices = get_autoif_candidate_original_indices(10_000, loader_seed=0)
    candidate_classes = classes[original_indices]
    helper = pd.read_csv(DEFAULT_AUTOIF_DATA_PATH, usecols=["target"])
    classified = pd.read_csv(
        DEFAULT_AUTOIF_CLASSIFICATIONS_PATH, usecols=["target", "Class"]
    )
    classified_targets = classified["target"].map(_normalize_target).copy()
    # The source classification file contains one known truncated target at
    # this row; the production matcher explicitly verifies and repairs it.
    classified_targets.iloc[8538] = _normalize_target(helper.iloc[8538]["target"])
    class_by_prompt = dict(zip(
        classified_targets,
        classified["Class"].astype(str).str.strip(),
    ))

    for class_name in sorted(np.unique(candidate_classes)):
        evaluation_row = int(np.flatnonzero(candidate_classes == class_name)[0])
        original_row = int(original_indices[evaluation_row])
        prompt = _normalize_target(helper.iloc[original_row]["target"])
        assert class_by_prompt[prompt] == class_name


def test_candidate_original_indices_replay_known_loader_boundaries():
    indices = get_autoif_candidate_original_indices(10_000, loader_seed=0)

    assert indices[:5].tolist() == [3293, 4286, 3461, 8239, 8413]
    assert indices[3998:4003].tolist() == [5930, 6422, 3965, 5071, 9445]
    assert indices[-5:].tolist() == [625, 2902, 4839, 9959, 2431]
    assert len(np.unique(indices)) == 6_000


def test_prompt_matching_is_not_dependent_on_classification_csv_row_order(tmp_path):
    helper = pd.DataFrame({
        "target": ["First prompt", "Second prompt\r\n", "Third prompt"]
    })
    classified = pd.DataFrame({
        "target": [" Third prompt ", "First prompt", "Second prompt\n"],
        "Class": ["C", "A", "B"],
    })
    helper_path = tmp_path / "helper.csv"
    classified_path = tmp_path / "classified.csv"
    helper.to_csv(helper_path, index=False)
    classified.to_csv(classified_path, index=False)

    actual = load_autoif_classes_in_dataset_order(helper_path, classified_path)

    assert actual.tolist() == ["A", "B", "C"]


def test_metadata_validation_checks_prompt_identity_and_event_time_order(
        tmp_path, monkeypatch
):
    helper = pd.DataFrame({"target": [f"Prompt {index}" for index in range(10)]})
    classified = pd.DataFrame({
        "target": helper["target"].sample(frac=1, random_state=3).to_numpy(),
    })
    class_lookup = {
        target: f"Class {int(target.split()[-1]) % 2}"
        for target in helper["target"]
    }
    classified["Class"] = classified["target"].map(class_lookup)
    helper_path = tmp_path / "helper.csv"
    classified_path = tmp_path / "classified.csv"
    helper.to_csv(helper_path, index=False)
    classified.to_csv(classified_path, index=False)
    expected_times = np.asarray([1, 2, 3, 4, 5, 6], dtype=np.float32)
    monkeypatch.setattr(
        script,
        "_candidate_event_times_from_stored_splits",
        lambda *args: expected_times.copy(),
    )

    metadata = script.load_and_validate_autoif_candidate_metadata(
        autoif_data_path=helper_path,
        classifications_path=classified_path,
        dataset_name="dataset_autoif",
        dataset_setup="toy",
        loader_seed=0,
        evaluation_times=torch.from_numpy(expected_times),
        calibration_rows=4,
        test_rows=2,
        expected_class_count=2,
    )
    original_prompts = helper["target"].to_numpy()[
        metadata.original_autoif_row_indices
    ]
    assert metadata.class_labels.tolist() == [
        class_lookup[prompt] for prompt in original_prompts
    ]
    assert metadata.prompt_sha256 == tuple(
        hashlib.sha256(_normalize_target(prompt).encode("utf-8")).hexdigest()
        for prompt in original_prompts
    )

    corrupted_times = torch.from_numpy(expected_times.copy())
    corrupted_times[2] += 1
    with pytest.raises(ValueError, match="do not align"):
        script.load_and_validate_autoif_candidate_metadata(
            autoif_data_path=helper_path,
            classifications_path=classified_path,
            dataset_name="dataset_autoif",
            dataset_setup="toy",
            loader_seed=0,
            evaluation_times=corrupted_times,
            calibration_rows=4,
            test_rows=2,
            expected_class_count=2,
        )


def test_class_summary_metrics_are_exact_and_sortable():
    true_times = torch.tensor([1.0, 3.0, 5.0, 2.0, 4.0, 5.0])
    estimated = torch.tensor([2.0, 1.0, 4.0, 3.0, 6.0, 5.0])
    errors = torch.abs(estimated - true_times)
    summary = script.summarize_difficulty_by_class(
        class_labels=["A", "A", "A", "B", "B", "B"],
        difficulty_scores=errors,
        estimated_mean_times=estimated,
        true_times=true_times,
        event_observed=true_times <= 4,
        evaluation_splits=[
            "calibration", "calibration", "test",
            "calibration", "test", "test",
        ],
        horizon=4,
    ).set_index("class_name")

    assert summary.loc["A", "n_samples"] == 3
    assert summary.loc["A", "n_calibration"] == 2
    assert summary.loc["A", "success_rate"] == pytest.approx(2 / 3)
    assert summary.loc["A", "absolute_error_mean"] == pytest.approx(4 / 3)
    assert summary.loc["A", "absolute_error_median"] == 1
    assert summary.loc["A", "time_to_event_or_sentinel_mean"] == 3
    assert summary.loc["A", "restricted_time_to_event_mean"] == pytest.approx(8 / 3)
    assert summary.loc["A", "successful_time_to_event_mean"] == 2
    assert summary.loc["A", "difficulty_mean_rank_desc"] == 1
    assert summary.loc["B", "difficulty_mean_rank_desc"] == 2
    assert summary.sort_values(
        "absolute_error_mean", ascending=False
    ).index.tolist() == ["A", "B"]
