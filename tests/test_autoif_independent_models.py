import numpy as np
import pytest
import torch

from src.predictive_bounds.experiments.autoif_independent_models.generate_current_time_quantiles import (
    compute_current_time_quantiles,
    generate_artifact,
    make_candidate_quantile_levels,
)
from src.predictive_bounds.experiments.autoif_independent_models.train_independent_split_models import (
    AutoIFCandidateMetadata,
    FitMetadata,
    estimate_prompt_difficulty_in_batches,
    estimate_upb_quantiles_in_batches,
    make_disjoint_training_split,
)
from src.predictive_bounds.survival_utils.compute_mean_time_given_pmf import (
    compute_mean_survival_time,
    compute_quantile_survival_time,
)


def _conditional_probabilities():
    # Three event bins and one terminal no-event class.  Invalid past classes
    # are exactly zero and every current-time row is normalized.
    return torch.tensor([
        [
            [0.20, 0.30, 0.10, 0.40],
            [0.00, 0.40, 0.20, 0.40],
            [0.00, 0.00, 0.60, 0.40],
        ],
        [
            [0.60, 0.10, 0.10, 0.20],
            [0.00, 0.25, 0.25, 0.50],
            [0.00, 0.00, 0.75, 0.25],
        ],
    ], dtype=torch.float32)


def test_all_current_time_quantiles_match_scalar_reference_and_conventions():
    probabilities = _conditional_probabilities()
    levels = torch.tensor([0.50, 0.70, 0.90], dtype=torch.float64)
    actual = compute_current_time_quantiles(
        probabilities, levels, quantile_chunk_size=2
    )
    expected = torch.stack([
        compute_quantile_survival_time(
            probabilities,
            quantile=float(level),
            tail_distribution="geometric",
        ) + 1
        for level in levels
    ], dim=-1).clamp(max=4)

    assert actual.shape == (2, 3, 3)
    torch.testing.assert_close(actual, expected)
    assert torch.all(actual[:, 1] >= 2)
    assert torch.all(actual[:, 2] >= 3)
    assert torch.all(actual.diff(dim=-1) >= 0)


def test_streamed_artifact_is_a_normal_mmap_loadable_pt_file(tmp_path):
    probabilities = _conditional_probabilities()
    source = tmp_path / "probabilities.pt"
    output = tmp_path / "quantiles.pt"
    torch.save({"probability_est": probabilities}, source)

    generate_artifact(
        source,
        output,
        candidate_count=5,
        sample_chunk_size=1,
        quantile_chunk_size=2,
        expected_horizon=3,
    )
    artifact = torch.load(
        output, map_location="cpu", weights_only=True, mmap=True
    )

    assert artifact["estimated_quantiles"].shape == (2, 3, 5)
    assert artifact["layout"] == (
        "calibration_plus_test_sample,current_time,candidate_quantile"
    )
    torch.testing.assert_close(
        artifact["quantile_levels"],
        make_candidate_quantile_levels(5),
    )
    assert not list(tmp_path.glob("*.raw"))


def test_training_split_is_equal_disjoint_exhaustive_and_reproducible():
    first = make_disjoint_training_split(4000, seed=19)
    second = make_disjoint_training_split(4000, seed=19)

    assert len(first.upb_indices) == len(first.difficulty_indices) == 2000
    assert np.array_equal(first.upb_indices, second.upb_indices)
    assert np.intersect1d(
        first.upb_indices, first.difficulty_indices
    ).size == 0
    assert np.array_equal(
        np.sort(np.concatenate([
            first.upb_indices, first.difficulty_indices
        ])),
        np.arange(4000),
    )
    with pytest.raises(ValueError, match="even"):
        make_disjoint_training_split(5, seed=0)


class _FixedPredictionModel:
    def __init__(self, probabilities):
        self.probabilities = probabilities
        self.offset = 0

    def eval(self):
        return self

    def predict_proba(self, batch):
        stop = self.offset + len(batch)
        result = self.probabilities[self.offset:stop].to(batch.device)
        self.offset = stop
        return result


def test_role_specific_estimators_use_only_their_supplied_model_outputs():
    evaluation = (torch.zeros(1, 3, 2), torch.zeros(1, 3, 2))
    probabilities = _conditional_probabilities()
    levels = torch.tensor([0.5, 0.9])

    quantile_model = _FixedPredictionModel(probabilities)
    quantiles = estimate_upb_quantiles_in_batches(
        quantile_model,
        evaluation,
        levels,
        device=torch.device("cpu"),
        batch_size=1,
        quantile_chunk_size=1,
    )
    expected_quantiles = torch.stack([
        compute_quantile_survival_time(
            probabilities[:, :1],
            quantile=float(level),
            tail_distribution="geometric",
        ).squeeze(1) + 1
        for level in levels
    ], dim=-1).clamp(max=4)
    torch.testing.assert_close(quantiles, expected_quantiles)

    difficulty_probabilities = probabilities.flip(0)
    difficulty_model = _FixedPredictionModel(difficulty_probabilities)
    true_times = torch.tensor([2.0, 4.0])
    estimated, scores = estimate_prompt_difficulty_in_batches(
        difficulty_model,
        evaluation,
        true_times,
        device=torch.device("cpu"),
        batch_size=1,
    )
    expected_mean = compute_mean_survival_time(
        difficulty_probabilities[:, :1], tail_distribution="geometric"
    ).squeeze(1) + 1
    torch.testing.assert_close(estimated, expected_mean)
    torch.testing.assert_close(scores, torch.abs(expected_mean - true_times))
    # Swapping model outputs really changes the difficulty estimate; there is
    # no hidden reuse of the UPB model's prediction.
    assert not torch.equal(
        estimated,
        compute_mean_survival_time(
            probabilities[:, :1], tail_distribution="geometric"
        ).squeeze(1) + 1,
    )


def test_second_script_rejects_one_shared_output_path():
    from src.predictive_bounds.experiments.autoif_independent_models import (
        train_independent_split_models as script,
    )

    with pytest.raises(ValueError, match="different files"):
        script.main([
            "--upb-output", "same.pt",
            "--difficulty-output", "same.pt",
        ])


def test_second_script_keeps_models_and_outputs_separate(tmp_path, monkeypatch):
    from src.predictive_bounds.experiments.autoif_independent_models import (
        train_independent_split_models as script,
    )

    n_train, n_cal, n_test, horizon = 8, 3, 2, 3
    x_train = torch.zeros(n_train, horizon, 2)
    x_cal = torch.zeros(n_cal, horizon, 2)
    x_test = torch.zeros(n_test, horizon, 2)
    y_train = torch.zeros(n_train, horizon, dtype=torch.bool)
    t_train = torch.full((n_train,), horizon + 1)
    t_cal = torch.tensor([1, 2, horizon + 1])
    t_test = torch.tensor([3, horizon + 1])
    e_train = t_train <= horizon
    e_cal = t_cal <= horizon
    e_test = t_test <= horizon
    unused = torch.empty(0)
    fake_data = (
        unused, unused, unused, x_train, x_cal, x_test,
        y_train, unused, unused, t_train, t_cal, t_test,
        e_train, e_cal, e_test, unused, unused, unused,
        unused, unused, unused,
    )
    monkeypatch.setattr(script, "get_data", lambda *args, **kwargs: fake_data)
    labels = np.asarray(["Class A", "Class B", "Class A", "Class B", "Class A"])
    metadata = AutoIFCandidateMetadata(
        class_labels=labels,
        class_indices=np.asarray([0, 1, 0, 1, 0]),
        class_names=("Class A", "Class B"),
        original_autoif_row_indices=np.arange(n_cal + n_test),
        prompt_sha256=tuple(f"hash-{index}" for index in range(n_cal + n_test)),
    )
    monkeypatch.setattr(
        script,
        "load_and_validate_autoif_candidate_metadata",
        lambda **kwargs: metadata,
    )

    class FakeModel:
        def __init__(self, tag):
            self.tag = tag

        def state_dict(self):
            return {"tag": torch.tensor(self.tag)}

    fit_calls = []

    def fake_fit(_x, _y, _t, indices, *, model_seed, **kwargs):
        del kwargs
        fit_calls.append(indices.copy())
        metadata = FitMetadata(
            fit_indices=indices[:-1].copy(),
            validation_indices=indices[-1:].copy(),
            training_losses=(float(model_seed),),
            validation_losses=(float(model_seed + 1),),
        )
        return FakeModel(len(fit_calls)), metadata

    def fake_quantiles(model, evaluation, quantile_levels, **kwargs):
        del evaluation, kwargs
        assert model.tag == 1
        return torch.full((n_cal + n_test, len(quantile_levels)), 11.0)

    def fake_difficulty(model, *args, **kwargs):
        del args, kwargs
        assert model.tag == 2
        mean = torch.arange(n_cal + n_test, dtype=torch.float32)
        true = torch.cat([t_cal, t_test]).to(torch.float32)
        return mean, torch.abs(mean - true)

    monkeypatch.setattr(script, "fit_survival_model_on_rows", fake_fit)
    monkeypatch.setattr(script, "estimate_upb_quantiles_in_batches", fake_quantiles)
    monkeypatch.setattr(
        script, "estimate_prompt_difficulty_in_batches", fake_difficulty
    )
    upb_path = tmp_path / "upb.pt"
    difficulty_path = tmp_path / "difficulty.pt"
    summary_path = tmp_path / "summary.csv"
    script.main([
        "--candidate-count", "4",
        "--upb-output", str(upb_path),
        "--difficulty-output", str(difficulty_path),
        "--class-summary-output", str(summary_path),
    ])

    upb = torch.load(upb_path, weights_only=True)
    difficulty = torch.load(difficulty_path, weights_only=True)
    assert upb["role"] == "upb_quantiles"
    assert difficulty["role"] == "prompt_difficulty"
    assert upb["model_state_dict"]["tag"].item() == 1
    assert difficulty["model_state_dict"]["tag"].item() == 2
    assert "estimated_quantiles" in upb and "difficulty_scores" not in upb
    assert "difficulty_scores" in difficulty and "estimated_quantiles" not in difficulty
    assert difficulty["class_labels"] == labels.tolist()
    assert torch.equal(
        difficulty["original_autoif_row_indices"], torch.arange(5)
    )
    assert summary_path.is_file()
    assert np.intersect1d(fit_calls[0], fit_calls[1]).size == 0
