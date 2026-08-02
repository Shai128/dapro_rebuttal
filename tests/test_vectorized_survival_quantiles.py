import pytest
import torch

from src.predictive_bounds.survival_utils.compute_mean_time_given_pmf import (
    compute_quantile_survival_time,
    compute_quantiles_survival_time,
)


@pytest.mark.parametrize(
    "tail_distribution",
    ["constant", "geometric", "exponential", "power", "linear"],
)
def test_vectorized_quantiles_match_scalar_implementation(tail_distribution):
    generator = torch.Generator().manual_seed(13)
    probabilities = torch.rand(
        (7, 2, 11),
        generator=generator,
        dtype=torch.float64,
    )
    quantiles = torch.tensor(
        [0.001, 0.05, 0.20, 0.50, 0.90, 0.999],
        dtype=torch.float64,
    )

    actual = compute_quantiles_survival_time(
        probabilities,
        quantiles,
        tail_distribution=tail_distribution,
        quantile_chunk_size=2,
    )
    expected = torch.stack(
        [
            compute_quantile_survival_time(
                probabilities,
                quantile=float(quantile),
                tail_distribution=tail_distribution,
            )
            for quantile in quantiles
        ],
        dim=-1,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_vectorized_quantiles_validate_inputs():
    probabilities = torch.ones((2, 1, 4))

    with pytest.raises(ValueError, match="positive"):
        compute_quantiles_survival_time(
            probabilities,
            [0.5],
            quantile_chunk_size=0,
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        compute_quantiles_survival_time(probabilities, [-0.1, 0.5])
