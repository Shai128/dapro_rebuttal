"""Causal first-event label perturbations for the judge-noise experiment."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.dataset_utils.temporal import (
    event_metadata_from_labels,
    labels_from_event_times,
)


@dataclass(frozen=True)
class NoiseResult:
    event_times: torch.Tensor
    false_negative_rows: torch.Tensor
    false_positive_rows: torch.Tensor


def corrupt_event_times(
        event_times: torch.Tensor,
        horizon: int,
        *,
        false_negative_rate: float,
        false_positive_rate: float,
        seed: int,
) -> NoiseResult:
    """Flip first-event labels on calibration rows only.

    Rates are sample-level corruption probabilities.  A false negative removes
    the observed first event, moving its time to the ``T+1`` no-event sentinel.
    A false positive chooses one uniformly random observed negative turn before
    the current stopping point (or anywhere in a censored trajectory), so on
    its own the first event can only move earlier. With both mechanisms enabled,
    false negatives are applied first; false-positive eligibility is then
    computed from those intermediate labels, and final event times are
    recomputed from the perturbed label tensor.
    """
    for label, rate in (
        ("false_negative_rate", false_negative_rate),
        ("false_positive_rate", false_positive_rate),
    ):
        if not 0 <= rate <= 1:
            raise ValueError(f"{label} must lie in [0, 1].")
    times = torch.as_tensor(event_times).reshape(-1).to(torch.long)
    labels_cpu = labels_from_event_times(times, horizon).cpu()
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    observed_event = times <= horizon
    fn_rows = observed_event.cpu() & (
        torch.rand(len(times), generator=generator) < false_negative_rate
    )
    if bool(fn_rows.any()):
        rows = torch.arange(len(times))[fn_rows]
        columns = times.cpu()[fn_rows] - 1
        labels_cpu[rows, columns] = False

    times_after_fn, _ = event_metadata_from_labels(labels_cpu)
    fp_selected = torch.rand(len(times), generator=generator) < false_positive_rate
    fp_rows = torch.zeros(len(times), dtype=torch.bool)
    for row in torch.where(fp_selected)[0].tolist():
        # Only negative labels revealed before the current stopping point are
        # eligible. An FN row is censored at this intermediate stage, so FP can
        # create a later event on that row in the combined-noise condition.
        observed_negative_count = (
            horizon if times_after_fn[row] == horizon + 1
            else max(0, int(times_after_fn[row]) - 1)
        )
        if observed_negative_count == 0:
            continue
        turn = int(torch.randint(
            observed_negative_count, (1,), generator=generator
        ).item())
        labels_cpu[row, turn] = True
        fp_rows[row] = True
    perturbed, _ = event_metadata_from_labels(labels_cpu)
    return NoiseResult(
        event_times=perturbed.to(event_times.device),
        false_negative_rows=fn_rows.to(event_times.device),
        false_positive_rows=fp_rows.to(event_times.device),
    )
