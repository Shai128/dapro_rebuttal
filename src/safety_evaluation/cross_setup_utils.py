import hashlib
import json
from pathlib import Path

import torch


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]


def _format_number(value: float) -> str:
    return format(float(value), ".12g")


def get_cross_setup_experiment_name(
        dataset_name: str,
        model_dataset_setup: str,
        evaluation_dataset_setup: str,
        budget_per_sample: float,
        cal_size: int,
        tau_prior: float,
        gamma: float,
) -> str:
    """Return the shared, path-safe name used by construction, merging, and plotting."""
    model_key = _short_hash(model_dataset_setup)
    evaluation_key = _short_hash(evaluation_dataset_setup)
    return (
        f"cross_setup_{dataset_name}"
        f"_model-{model_key}"
        f"_eval-{evaluation_key}"
        f"_budget-{_format_number(budget_per_sample)}"
        f"_cal-{int(cal_size)}"
        f"_tau-{_format_number(tau_prior)}"
        f"_gamma-{_format_number(gamma)}"
    )


def get_cross_setup_metadata(
        dataset_name: str,
        model_dataset_setup: str,
        evaluation_dataset_setup: str,
) -> dict:
    return {
        "experiment_type": "cross_setup",
        "dataset_name": dataset_name,
        "model_dataset_setup": model_dataset_setup,
        "evaluation_dataset_setup": evaluation_dataset_setup,
    }


def setup_cross_setup_experiment_data(
        cal_size: int,
        is_real: bool,
        device,
        dataset_name: str,
        model_dataset_setup: str,
        evaluation_dataset_setup: str,
        taus_range: torch.Tensor,
        m_upper_bound: float,
):
    """Apply a source-setup survival model to calibration/test data from another setup."""
    from src.dataset_utils.data_utils import get_data
    from src.safety_evaluation.utils.utils import compute_probabilities_and_quantiles

    if model_dataset_setup == evaluation_dataset_setup:
        raise ValueError(
            "Cross-setup evaluation requires different model and evaluation dataset setups."
        )

    cache_key = _short_hash(
        f"{is_real}\0{dataset_name}\0{model_dataset_setup}\0{evaluation_dataset_setup}"
    )
    prediction_dir = Path("alg_playground_model") / "cross_setup" / cache_key
    prediction_path = prediction_dir / "probability_est_cal_test.pt"
    load_x = not prediction_path.exists()

    source_data = get_data(
        is_real,
        device,
        dataset_name,
        model_dataset_setup,
        load_x=load_x,
    )
    source_x_train = source_data[3]
    source_y_train = source_data[6]
    source_t_tilde_train = source_data[9]
    if load_x:
        n_seed = int(len(source_y_train) * 0.9)
        checkpoint_path = (
            Path("saved_models")
            / "al"
            / "transformer"
            / model_dataset_setup
            / "dummy"
            / f"seed_{n_seed}_budget_10"
            / "seed=0"
            / "al_state_latest.pt"
        )
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                "The source survival-model checkpoint is required before cross-setup "
                f"prediction: {checkpoint_path.resolve()}"
            )

    evaluation_data = get_data(
        is_real,
        device,
        dataset_name,
        evaluation_dataset_setup,
        load_x=load_x,
    )
    evaluation_p_cal = evaluation_data[1]
    evaluation_p_test = evaluation_data[2]
    evaluation_x_cal = evaluation_data[4]
    evaluation_x_test = evaluation_data[5]
    evaluation_y_train = evaluation_data[6]
    evaluation_t_tilde_cal = evaluation_data[10]
    evaluation_t_tilde_test = evaluation_data[11]
    del source_data, evaluation_data

    source_max_time = int(source_y_train.shape[1])
    evaluation_max_time = int(evaluation_y_train.shape[1])
    if source_max_time != evaluation_max_time:
        raise ValueError(
            "The source model and evaluation setup have incompatible time horizons: "
            f"{source_max_time} != {evaluation_max_time}."
        )

    if load_x:
        source_feature_dim = int(source_x_train.shape[-1])
        evaluation_feature_dim = int(evaluation_x_cal.shape[-1])
        if source_feature_dim != evaluation_feature_dim:
            raise ValueError(
                "The source model and evaluation setup have incompatible feature dimensions: "
                f"{source_feature_dim} != {evaluation_feature_dim}."
            )
        if int(evaluation_x_cal.shape[1]) != source_max_time or int(evaluation_x_test.shape[1]) != source_max_time:
            raise ValueError(
                "Evaluation embeddings must have the same time dimension as the source model."
            )

    prediction_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = prediction_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            get_cross_setup_metadata(
                dataset_name,
                model_dataset_setup,
                evaluation_dataset_setup,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    source_model_dir = (
        Path("alg_playground_model")
        / f"is_real_{is_real}_dataset_{dataset_name}_dataset_{model_dataset_setup}"
    )
    source_model_dir.mkdir(parents=True, exist_ok=True)

    quantile_est_cal_test, probability_est, conditional_grid = compute_probabilities_and_quantiles(
        evaluation_x_cal,
        source_x_train,
        evaluation_x_test,
        str(prediction_path),
        dataset_name,
        model_dataset_setup,
        evaluation_p_cal,
        evaluation_p_test,
        source_max_time,
        str(source_model_dir / "model.pt"),
        str(source_model_dir / "history.png"),
        is_real,
        source_t_tilde_train,
        taus_range,
        m_upper_bound,
        device,
    )
    del source_x_train, evaluation_x_cal, evaluation_x_test

    t_tilde_cal_test = torch.cat(
        [evaluation_t_tilde_cal, evaluation_t_tilde_test]
    ).clone()
    expected_samples = len(t_tilde_cal_test)
    if len(probability_est) != expected_samples or len(quantile_est_cal_test) != expected_samples:
        raise ValueError(
            "Cached cross-setup predictions do not match the evaluation dataset size. "
            f"Expected {expected_samples}, got probabilities={len(probability_est)} and "
            f"quantiles={len(quantile_est_cal_test)}. Remove {prediction_path} and rerun."
        )
    if not 0 < cal_size < expected_samples:
        raise ValueError(
            f"cal_size must be between 1 and {expected_samples - 1}; got {cal_size}."
        )

    test_size = expected_samples - cal_size
    return (
        source_max_time,
        t_tilde_cal_test,
        quantile_est_cal_test,
        probability_est,
        conditional_grid,
        test_size,
    )
