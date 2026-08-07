"""Regression tests for causal feature, reveal, and event-time indexing."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from src.dataset_utils import data_utils
from src.dataset_utils.datasets import PartialSequenceDataset, SurvivalDataset
from src.dataset_utils.temporal import (
    build_causal_turn_features,
    event_metadata_from_labels,
    labels_from_event_times,
    normalize_event_times,
)
from src.dataset_utils.real_data import (
    _contiguous_embedding_indices,
    _validate_completed_sequences,
)
from src.train_model.models.transformer_survival_model import DiscreteSurvivalLoss
from src.train_model.active_learning import ActiveLearner
from src.predictive_bounds.budget_allocators.projected_optimization_utils import (
    expected_acquisition_cost as dapro_expected_acquisition_cost,
)
from src.predictive_bounds.budget_allocators.risk_controlled_budget import (
    cumulative_policy_costs,
)
from src.predictive_bounds.budget_allocators.vectorized_adaptive_allocator_patch import (
    _process_probabilities_and_required_mask,
    expected_acquisition_cost as locally_adaptive_expected_acquisition_cost,
)
from src.predictive_bounds.budget_allocators.budget_allocator import BudgetAllocationResult
from evaluation.estimate import (
    CumulativeJailbreakRateMetric,
    IPCWTrajectorySimulator,
    compute_oracle_metric,
)
from src.predictive_bounds.ablations.phase1_optimization import _dapro_scores


def test_causal_features_use_current_prompt_and_previous_response_only():
    prompts = torch.tensor([[[10.0], [20.0], [30.0]]])
    responses = torch.tensor([[[1.0], [2.0], [3.0]]])

    features = build_causal_turn_features(prompts, responses)

    assert torch.equal(
        features,
        torch.tensor([[[10.0, 0.0], [20.0, 1.0], [30.0, 2.0]]]),
    )


def test_event_metadata_is_one_based_and_uses_t_plus_one_for_no_event():
    labels = torch.tensor(
        [
            [1, 0, 0],
            [0, 0, 1],
            [0, 0, 0],
            [0, 1, 1],
        ]
    )

    event_times, observed = event_metadata_from_labels(labels)

    assert event_times.tolist() == [1, 3, 4, 2]
    assert observed.tolist() == [1, 1, 0, 1]
    assert torch.equal(labels_from_event_times(event_times, 3), labels.bool()) is False
    expected_first_events = torch.tensor(
        [[1, 0, 0], [0, 0, 1], [0, 0, 0], [0, 1, 0]], dtype=torch.bool
    )
    assert torch.equal(labels_from_event_times(event_times, 3), expected_first_events)


def test_legacy_censored_time_is_normalized_without_moving_observed_events():
    legacy_times = torch.tensor([1, 3, 3])
    observed = torch.tensor([1, 1, 0])

    normalized = normalize_event_times(legacy_times, observed, horizon=3)

    assert normalized.tolist() == [1, 3, 4]


def test_real_data_rejects_file_gaps_and_truncated_negative_sequences():
    assert _contiguous_embedding_indices(
        ["attack_log_1_embeddings.pt", "attack_log_0_embeddings.pt"]
    ) == [0, 1]
    with pytest.raises(ValueError, match="contiguous"):
        _contiguous_embedding_indices(
            ["attack_log_0_embeddings.pt", "attack_log_2_embeddings.pt"]
        )

    _validate_completed_sequences(
        [torch.tensor([1, 2, 10]), torch.tensor([1, 1, 1, 1])], max_time=4
    )
    with pytest.raises(ValueError, match="missing turns as negatives"):
        _validate_completed_sequences(
            [torch.tensor([1, 2]), torch.tensor([1, 1, 1])], max_time=3
        )


def test_partial_sequence_reveal_order_matches_online_protocol():
    x = torch.tensor([[[10.0], [20.0], [30.0]]])
    y = torch.tensor([[0, 1, 0]])
    dataset = PartialSequenceDataset(x, y, torch.tensor([2]), "tiny")

    x_initial, label_initial, _ = dataset[0]
    assert x_initial.squeeze(-1).tolist() == [10.0, 0.0, 0.0]
    assert label_initial == {
        "is_event": False,
        "event_time": 4,
        "censor_time": 0,
        "obs_len": 1,
    }

    assert dataset.observe_next_step([0]) == [False]
    x_after_negative, label_after_negative, _ = dataset[0]
    assert x_after_negative.squeeze(-1).tolist() == [10.0, 20.0, 0.0]
    assert label_after_negative["censor_time"] == 1
    assert label_after_negative["obs_len"] == 2

    assert dataset.observe_next_step([0]) == [True]
    x_after_event, label_after_event, _ = dataset[0]
    assert x_after_event.squeeze(-1).tolist() == [10.0, 20.0, 0.0]
    assert label_after_event["is_event"] is True
    assert label_after_event["event_time"] == 2
    assert label_after_event["censor_time"] == 2
    assert label_after_event["obs_len"] == 2


def test_final_turn_event_is_not_conflated_with_censoring():
    x = torch.zeros((2, 3, 1))
    y = torch.tensor([[0, 0, 1], [0, 0, 0]])
    data = SurvivalDataset(x, y, torch.tensor([3, 4]), "tiny")

    assert data.delta.tolist() == [1, 0]


def test_no_event_sequence_reveals_exactly_one_label_per_step():
    x = torch.ones((1, 3, 1))
    y = torch.zeros((1, 3), dtype=torch.long)
    dataset = PartialSequenceDataset(x, y, torch.tensor([4]), "tiny")

    assert dataset[0][1]["censor_time"] == 0
    for expected_known in (1, 2, 3):
        assert dataset.observe_next_step([0]) == [False]
        assert dataset[0][1]["censor_time"] == expected_known
    assert dataset.label_known[0].tolist() == [True, True, True]


def test_survival_loss_maps_one_based_event_time_to_zero_based_class():
    probs = torch.tensor(
        [
            [
                [0.70, 0.10, 0.10, 0.10],
                [0.00, 0.50, 0.25, 0.25],
                [0.00, 0.00, 0.40, 0.60],
            ],
            [
                [0.10, 0.20, 0.60, 0.10],
                [0.00, 0.20, 0.60, 0.20],
                [0.00, 0.00, 0.75, 0.25],
            ],
        ],
        dtype=torch.float64,
    )
    loss = DiscreteSurvivalLoss(censored_mode="full_survival")(
        probs,
        true_times=torch.tensor([1, 3]),
        event_indicators=torch.tensor([1, 1]),
    )

    # Event at time 1 uses class 0 and only x[0]. Event at time 3 uses class 2
    # from x[0], x[1], and x[2].
    expected = -(
        math.log(0.70) + math.log(0.60) + math.log(0.60) + math.log(0.75)
    ) / 4
    assert math.isclose(loss.item(), expected, rel_tol=1e-12)


def test_survival_loss_censor_threshold_counts_revealed_negative_labels():
    probs = torch.tensor(
        [[[0.40, 0.30, 0.20, 0.10], [0.00, 0.50, 0.25, 0.25], [0.00, 0.00, 0.60, 0.40]]],
        dtype=torch.float64,
    )
    loss = DiscreteSurvivalLoss(censored_mode="full_survival")(
        probs,
        true_times=torch.tensor([1]),
        event_indicators=torch.tensor([0]),
    )

    # One known negative rules out class 0; x[0] and newly visible x[1] are valid.
    expected = -(math.log(0.60) + math.log(1.00)) / 2
    assert math.isclose(loss.item(), expected, rel_tol=1e-12)


def test_allocators_share_the_same_one_based_active_length_contract():
    event_times = torch.tensor([1, 2, 4, 5])
    prior_horizons = torch.tensor([4, 2, 3, 4])
    active_lengths = torch.minimum(event_times, prior_horizons)
    continuation = torch.full((4, 4), 0.5, dtype=torch.float64)

    _, required_mask, _, _ = _process_probabilities_and_required_mask(
        torch.ones((4, 4), dtype=torch.float64),
        prior_horizons,
        event_times,
        lam=1.0,
        pi_func=lambda _: continuation,
    )
    assert required_mask.sum(dim=1).tolist() == active_lengths.tolist()

    local_total = locally_adaptive_expected_acquisition_cost(
        torch.ones((4, 4), dtype=torch.float64),
        prior_horizons,
        event_times,
        lam=1.0,
        pi_func=lambda _: continuation,
    ).item()
    dapro_mean = dapro_expected_acquisition_cost(continuation, active_lengths)
    cumulative = continuation.cumprod(dim=1)[0].numpy()
    fixed_policy_rows = cumulative_policy_costs(
        cumulative[None, :], active_lengths.numpy()
    )[:, 0]

    assert math.isclose(local_total / len(event_times), dapro_mean, rel_tol=1e-12)
    assert np.allclose(fixed_policy_rows, [0.5, 0.75, 0.875, 0.9375])
    assert math.isclose(dapro_mean, fixed_policy_rows.mean(), rel_tol=1e-12)


def test_quantile_dapro_score_uses_positive_one_based_counts():
    conditional_grid = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=torch.float64
    )

    scores = _dapro_scores(conditional_grid, "quantile")

    assert torch.isfinite(scores).all()
    assert scores.tolist() == [1.0, 0.5]


def test_get_data_jointly_shuffles_every_row_aligned_field(monkeypatch):
    split_ids = [np.array([0, 1]), np.array([2]), np.array([3, 4])]

    def fields(offset=0, *, shape_tail=(), dtype=np.int64):
        return tuple(
            np.asarray(ids + offset, dtype=dtype).reshape((len(ids),) + shape_tail)
            for ids in split_ids
        )

    p = fields()
    x = tuple(
        np.asarray(ids, dtype=np.float32).reshape(len(ids), 1, 1)
        for ids in split_ids
    )
    y = fields(shape_tail=(1,))
    event_times = fields(10)
    observed = fields(20)
    counts = fields(30)
    horizons = fields(40)

    monkeypatch.setattr(
        data_utils,
        "generate_real_data",
        lambda *_args, **_kwargs: (
            *p,
            *x,
            *y,
            *event_times,
            *observed,
            *counts,
            *horizons,
        ),
    )

    loaded = data_utils.get_data(
        True, "cpu", "unused", "unused", load_x=True, seed=7
    )
    (
        p_train,
        p_cal,
        p_test,
        x_train,
        x_cal,
        x_test,
        y_train,
        y_cal,
        y_test,
        t_train,
        t_cal,
        t_test,
        e_train,
        e_cal,
        e_test,
        b_train,
        b_cal,
        b_test,
        n_train,
        n_cal,
        n_test,
    ) = loaded

    for splits in zip(
        (p_train, p_cal, p_test),
        (x_train, x_cal, x_test),
        (y_train, y_cal, y_test),
        (t_train, t_cal, t_test),
        (e_train, e_cal, e_test),
        (b_train, b_cal, b_test),
        (n_train, n_cal, n_test),
    ):
        p_part, x_part, y_part, t_part, e_part, b_part, n_part = splits
        ids = p_part.long()
        assert torch.equal(x_part[:, 0, 0].long(), ids)
        assert torch.equal(y_part[:, 0].long(), ids)
        assert torch.equal(t_part.long(), ids + 10)
        assert torch.equal(e_part.long(), ids + 20)
        assert torch.equal(b_part.long(), ids + 30)
        assert torch.equal(n_part.long(), ids + 40)
        assert x_part.dtype == torch.float32


def test_metric_simulator_observes_event_on_last_acquired_turn():
    class FixedAllocator:
        def allocate_budget(self, *_args):
            return BudgetAllocationResult(
                f=torch.empty(0),
                C=torch.tensor([1, 2, 0]),
                C_probs=torch.tensor([0.5, 0.25, 0.0]),
            )

    event_times = torch.tensor([1, 2, 4])
    prediction = SimpleNamespace(
        probability_est=torch.empty((3, 3)),
        quantile_est=torch.empty((3, 1)),
    )
    data = IPCWTrajectorySimulator.simulate(
        FixedAllocator(), torch.empty((3, 3, 1)), prediction, event_times, max_time=3
    )

    assert data.Delta_i.tolist() == [True, True, False]
    assert data.Y_i.tolist() == [1, 2, 0]
    estimate = CumulativeJailbreakRateMetric(oracle_cjr=2 / 3).compute(data)
    assert math.isfinite(estimate["estimated_cjr"])
    assert math.isclose(estimate["estimated_cjr"], 200.0)


def test_oracle_metrics_include_events_on_the_final_turn():
    metrics = compute_oracle_metric(torch.tensor([1, 3, 4]), max_time=3)

    assert math.isclose(metrics["cjr"], 2 / 3, rel_tol=1e-7)
    assert math.isclose(metrics["rmttu"], 2.0)


class _FirstAcquisition:
    name = "first"

    def select(self, _model, _dataset, pool_indices, k, batch_size):
        return list(pool_indices)[:k]


def _tiny_active_learner(dataset):
    return ActiveLearner(
        model_class=lambda: nn.Linear(1, 1),
        loss_fn=lambda *_args: torch.tensor(0.0),
        dataset=dataset,
        seed_indices=[],
        val_indices=[],
        pool_indices=list(range(len(dataset))),
        acquisition=_FirstAcquisition(),
        device=torch.device("cpu"),
    )


def test_active_learner_keeps_final_label_available_after_final_x_is_revealed():
    dataset = PartialSequenceDataset(
        torch.ones((1, 2, 1)), torch.tensor([[0, 1]]), torch.tensor([2]), "tiny"
    )
    learner = _tiny_active_learner(dataset)
    acquire_once = learner._ActiveLearner__conduct_one_batch_one_time_acquire

    acquire_once([0])
    assert dataset.obs_lens.tolist() == [2]
    assert dataset.label_known.tolist() == [[True, False]]
    assert learner.pool_indices == [0]
    assert learner._count_events_observed() == 0

    acquire_once([0])
    assert dataset.label_known.tolist() == [[True, True]]
    assert learner.pool_indices == []
    assert learner._count_events_observed() == 1


def test_active_learning_checkpoint_restores_exact_label_reveal_state(tmp_path):
    original_data = PartialSequenceDataset(
        torch.ones((1, 3, 1)), torch.tensor([[0, 0, 1]]), torch.tensor([3]), "tiny"
    )
    original = _tiny_active_learner(original_data)
    original._ActiveLearner__conduct_one_batch_one_time_acquire([0])
    original.store_state(nn.Linear(1, 1), tmp_path, round=2)

    restored_data = PartialSequenceDataset(
        torch.ones((1, 3, 1)), torch.tensor([[0, 0, 1]]), torch.tensor([3]), "tiny"
    )
    restored = _tiny_active_learner(restored_data)
    loaded = restored.load_state(tmp_path, update_steps=True)

    assert loaded is not None
    assert restored_data.obs_lens.tolist() == [2]
    assert restored_data.label_known.tolist() == [[True, False, False]]
    assert restored_data[0][1]["censor_time"] == 1


def test_full_time_acquisition_stops_at_events_and_respects_exact_budget():
    dataset = PartialSequenceDataset(
        torch.ones((2, 3, 1)),
        torch.tensor([[0, 1, 0], [0, 0, 0]]),
        torch.tensor([2, 4]),
        "tiny",
    )
    learner = _tiny_active_learner(dataset)
    acquire_full = learner._ActiveLearner__conduct_one_batch_full_time_acquire

    acquired = acquire_full([0, 1], total_budget_left=4)

    assert acquired == 4
    assert learner.total_acquisitions == 4
    assert dataset.label_known.tolist() == [[True, True, False], [True, True, False]]
    assert learner.pool_indices == [1]
