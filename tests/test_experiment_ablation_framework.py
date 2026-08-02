"""Correctness tests for the reusable paper-ablation framework."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.predictive_bounds.calibration.oracle_survival_calibration import (
    OracleSurvivalCalibration,
    OracleSurvivalUPBCalibration,
)
from src.predictive_bounds.experiments.allocation_focus.construct import (
    rank_quartiles,
)
from src.predictive_bounds.experiments.common.results import (
    merge_sharded_bounds,
    result_roots,
    write_seed_manifest,
)
from src.predictive_bounds.experiments.distribution_shift.construct_calibrated_bound import (
    parse_args as parse_distribution_shift_args,
    validate_attacker_only_shift,
    validate_shift_design,
)
from src.predictive_bounds.experiments.distribution_shift import utils as shift_utils
from src.predictive_bounds.experiments.distribution_shift.run_matrix import (
    load_matrix as load_shift_matrix,
)
from src.predictive_bounds.experiments.full_bounds.make_tables import (
    render_latex_tables,
)
from src.predictive_bounds.experiments.full_bounds.config import (
    all_experiment_configs,
)
from src.predictive_bounds.experiments.judge_noise.noise import (
    corrupt_event_times,
)
from src.predictive_bounds.experiments.variance_components.design import (
    variance_jobs,
)
from src.predictive_bounds.experiments.matrix_runner import (
    commands_for_config as matrix_commands_for_config,
    parse_args as parse_matrix_args,
)
from src.train_model.models.utils import SurvivalModelPrediction
from src.train_model.train_model import (
    allocate_uniform_training_budget,
    predict_trajectory_probabilities_in_batches,
)


def test_infinite_budget_lpb_oracle_has_unit_weights_and_full_cost():
    taus = torch.tensor([0.1, 0.2, 0.3])
    quantiles = torch.tensor([
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0],
    ])
    times = torch.tensor([1, 4, 6])
    prediction = SurvivalModelPrediction(quantiles, torch.zeros(3, 6, 6))
    calibration = OracleSurvivalCalibration(taus, tau_prior=0.2)
    calibration.calibrate(None, times, prediction)
    metrics = calibration.compute_metrics(prediction, torch.tensor([0.2]))

    assert calibration.name == "oracle_survival_calibration"
    assert calibration.allocation_result.total_budget_used == 11
    assert metrics["mean_weight"] == 1.0
    assert metrics["max_weight"] == 1.0
    assert metrics["mean_a_weighted_inverse_probability_minus_one_0"] == 0.0
    assert metrics["infinite_budget_oracle"] == 1


def test_infinite_budget_upb_oracle_uses_upper_tail_and_full_cost():
    taus = torch.tensor([0.5, 0.7, 0.9])
    quantiles = torch.tensor([
        [2.0, 3.0, 4.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0],
    ])
    times = torch.tensor([1, 5, 4])
    prediction = SurvivalModelPrediction(quantiles, torch.zeros(3, 5, 5))
    calibration = OracleSurvivalUPBCalibration(taus, tau_prior=0.9)
    calibration.calibrate(None, times, prediction)
    metrics = calibration.compute_metrics(prediction, torch.tensor([0.7]))

    assert calibration.name == "oracle_survival_upb_calibration"
    assert calibration.allocation_result.total_budget_used == 10
    assert metrics["mean_weight"] == 1.0
    assert metrics["mean_a_weighted_inverse_probability_minus_one_0"] == 0.0
    assert metrics["infinite_budget_oracle"] == 1


def test_oracle_is_never_bolded_in_full_comparison_table():
    rows = []
    for method, calibration_name, value in [
        ("Constant", "calibration_random_adaptive_optimized_allocation", 0.90),
        ("Infinite-Budget Oracle", "oracle_survival_calibration", 0.90),
    ]:
        for seed in (0, 1):
            rows.append({
                "configuration": "toy__qwen",
                "dataset_display": "Toy",
                "bound_type": "LPB",
                "target_model": "Qwen2.5",
                "method": method,
                "calibration_name": calibration_name,
                "coverage_pct": 100 * value,
                "coverage_diff_pct": 0.0,
                "mean_weight": 1.0,
                "mean_a_weight": 0.1,
                "budget_used_per_sample": 20.0,
                "target_coverage_pct": 90.0,
                "target_budget": 20.0,
                "seed": seed,
            })
    latex = render_latex_tables(pd.DataFrame(rows))
    oracle_line = next(
        line for line in latex.splitlines() if line.startswith("Infinite-Budget Oracle")
    )
    assert "textbf" not in oracle_line


def test_oracle_is_not_bolded_even_when_it_is_the_only_table_row():
    rows = [{
        "configuration": "toy__qwen",
        "dataset_display": "Toy",
        "bound_type": "LPB",
        "target_model": "Qwen2.5",
        "method": "Infinite-Budget Oracle",
        "calibration_name": "oracle_survival_calibration",
        "coverage_pct": 90.0,
        "coverage_diff_pct": 0.0,
        "mean_weight": 1.0,
        "mean_a_weight": 0.1,
        "budget_used_per_sample": 100.0,
        "target_coverage_pct": 90.0,
        "target_budget": 20.0,
        "seed": seed,
    } for seed in (0, 1)]
    latex = render_latex_tables(pd.DataFrame(rows))
    oracle_line = next(
        line for line in latex.splitlines() if line.startswith("Infinite-Budget Oracle")
    )
    assert "textbf" not in oracle_line


def test_judge_false_negatives_delay_events_and_false_positives_advance_them():
    times = torch.tensor([1, 3, 5])
    false_negative = corrupt_event_times(
        times, 4, false_negative_rate=1.0, false_positive_rate=0.0, seed=1
    )
    assert false_negative.event_times.tolist() == [5, 5, 5]

    false_positive = corrupt_event_times(
        torch.tensor([2, 4, 5]), 4,
        false_negative_rate=0.0, false_positive_rate=1.0, seed=2,
    )
    assert torch.all(false_positive.event_times < torch.tensor([2, 4, 5]))
    assert false_positive.false_positive_rows.all()


def test_combined_judge_noise_applies_false_positives_after_false_negatives():
    result = corrupt_event_times(
        torch.tensor([1]), 4,
        false_negative_rate=1.0, false_positive_rate=1.0, seed=7,
    )
    assert result.false_negative_rows.item()
    assert result.false_positive_rows.item()
    assert 1 <= result.event_times.item() <= 4


def test_uniform_training_budget_is_exact_and_drops_unrevealed_times():
    selection = allocate_uniform_training_budget(
        [1, 2, 5], horizon=4, total_budget=5, seed=0
    )
    assert selection.budget_spent == selection.requested_budget == 5
    assert selection.retained_indices.tolist() == [0, 1]
    assert selection.observed_steps.tolist() == [1, 2, 2]


def test_batched_prediction_cache_preserves_tensor_and_row_order():
    class IdentityModel:
        def eval(self):
            return self

        def predict_proba(self, batch):
            return batch + 0.5

    first = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)
    second = torch.arange(12, 20, dtype=torch.float32).reshape(2, 2, 2)
    actual = predict_trajectory_probabilities_in_batches(
        IdentityModel(), (first, second), "cpu", batch_size=2
    )
    assert torch.equal(actual, torch.cat([first, second]) + 0.5)


def test_rank_quartiles_are_balanced_and_tie_safe():
    labels = rank_quartiles(np.ones(12))
    assert [int(np.sum(labels == f"Q{i}")) for i in range(1, 5)] == [3, 3, 3, 3]


def test_attacker_shift_rejects_target_or_judge_changes():
    source = "attack_default_attack_qwen_lm_target_target_a_judge_judge_a"
    valid = "attack_other_attack_qwen_lm_target_target_a_judge_judge_a"
    validate_attacker_only_shift(source, valid)
    invalid = "attack_other_attack_qwen_lm_target_target_b_judge_judge_a"
    try:
        validate_attacker_only_shift(source, invalid)
    except ValueError as error:
        assert "only the attacker" in str(error)
    else:
        raise AssertionError("A target-model change was accepted as attacker-only shift.")


def test_training_shift_supports_different_source_and_evaluation_datasets():
    args = parse_distribution_shift_args([
        "--shift-type", "train_calibration_test_shift",
        "--model-dataset-name", "dataset_toxicity",
        "--calibration-dataset-name", "dataset_red_team",
        "--test-dataset-name", "dataset_red_team",
        "--model-dataset-setup", "shared_setup",
        "--calibration-dataset-setup", "shared_setup",
        "--test-dataset-setup", "shared_setup",
    ])
    validate_shift_design(args)
    assert args.model_dataset_name == "dataset_toxicity"
    assert args.calibration_dataset_name == args.test_dataset_name == "dataset_red_team"


def test_attacker_shift_rejects_dataset_changes():
    args = parse_distribution_shift_args([
        "--shift-type", "attacker_shift",
        "--model-dataset-name", "dataset_red_team",
        "--calibration-dataset-name", "dataset_red_team",
        "--test-dataset-name", "dataset_toxicity",
        "--model-dataset-setup", "attack_a_lm_target_t_judge_j",
        "--calibration-dataset-setup", "attack_a_lm_target_t_judge_j",
        "--test-dataset-setup", "attack_b_lm_target_t_judge_j",
    ])
    try:
        validate_shift_design(args)
    except ValueError as error:
        assert "not the dataset" in str(error)
    else:
        raise AssertionError("A dataset change was accepted as attacker-only shift.")


def test_prediction_pool_uses_cross_domain_loader_when_only_dataset_changes(
        monkeypatch,
):
    calls = []

    def cross_loader(**kwargs):
        calls.append(kwargs)
        return "cross-domain"

    monkeypatch.setattr(
        shift_utils, "setup_source_model_evaluation_data", cross_loader
    )
    result = shift_utils.load_prediction_pool(
        model_dataset_name="source",
        evaluation_dataset_name="target",
        model_dataset_setup="same_setup",
        evaluation_dataset_setup="same_setup",
        device="cpu",
        taus_range=torch.tensor([0.1]),
        m_upper_bound=200,
    )
    assert result == "cross-domain"
    assert calls[0]["model_dataset_name"] == "source"
    assert calls[0]["evaluation_dataset_name"] == "target"


def test_shift_matrix_filters_explicit_designs(tmp_path):
    path = tmp_path / "shifts.json"
    path.write_text("""[
      {
        "shift_type": "train_calibration_test_shift",
        "model_dataset_setup": "source",
        "calibration_dataset_setup": "target",
        "test_dataset_setup": "target"
      },
      {
        "shift_type": "attacker_shift",
        "model_dataset_setup": "attack_a_lm_target_t_judge_j",
        "calibration_dataset_setup": "attack_a_lm_target_t_judge_j",
        "test_dataset_setup": "attack_b_lm_target_t_judge_j"
      }
    ]""", encoding="utf-8")
    selected = load_shift_matrix(path, "attacker_shift")
    assert len(selected) == 1
    assert selected[0]["test_dataset_setup"].startswith("attack_b")


def test_autoif_matrix_runs_only_autoif_configs_and_paper_methods():
    args = parse_matrix_args([
        "--experiment", "autoif_cross_class", "--dry-run"
    ])
    configs = {config.key: config for config in all_experiment_configs()}
    command = matrix_commands_for_config(configs["autoif__qwen"], args)[0]
    assert "--paper-methods-only" in command
    assert "--bound-type" in command and "lpb" in command
    assert matrix_commands_for_config(configs["toxicity__qwen"], args) == []


def test_variance_design_contains_all_one_factor_and_crossed_jobs():
    jobs = variance_jobs(replicates=5, crossed_groups=3, suffix_prefix="test")
    assert len(jobs) == 4 + 3 * 3
    assert {job.design for job in jobs} == {
        "all_random", "acquisition_only", "policy_only", "data_split_only",
        "policy_x_acquisition", "data_x_acquisition", "data_x_policy",
    }


def test_manifest_merge_refuses_partial_method_sets(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    name = "toy"
    methods = ("method_a", "method_b")
    for seed in (0, 1):
        write_seed_manifest(name, seed, "lpb", methods, {"experiment_type": "toy"})
        temporary, _ = result_roots("lpb", name)
        for method in methods:
            directory = temporary / method
            directory.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({
                "seed": [seed], "calibration_name": [method],
                "target_coverage": [0.9], "coverage": [0.9],
            }).to_csv(directory / f"seed={seed}.csv", index=False)
    output = merge_sharded_bounds(
        name, (0, 2), "lpb", expected_metadata={"experiment_type": "toy"}
    )
    assert len(pd.read_csv(output)) == 4

    (Path("results/tmp_calibration_results") / name / "method_b" / "seed=1.csv").unlink()
    try:
        merge_sharded_bounds(name, (0, 2), "lpb")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("A partial method set was silently merged.")
