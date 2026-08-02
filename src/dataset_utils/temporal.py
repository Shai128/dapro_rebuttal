"""Canonical timing conversions for sequential predictive-bound data.

The repository uses zero-based tensor indices and one-based event counts:

* ``x[i, t]`` is available at the start of turn ``t``.
* ``y[i, t]`` is revealed at the end of turn ``t``.
* after a negative ``y[i, t]``, ``x[i, t + 1]`` becomes available.
* ``t_tilde[i] == t + 1`` when the first event is ``y[i, t] == 1``.
* ``t_tilde[i] == T + 1`` when no event occurs within the ``T`` turns.

Keeping these conversions here prevents individual data loaders, models, and
allocation methods from independently adding or subtracting one.
"""

from __future__ import annotations

from typing import Tuple

import torch


def build_causal_turn_features(
    prompt_embeddings: torch.Tensor,
    response_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Return ``[current prompt, previous response]`` features for every turn.

    Both inputs must have shape ``[N, T, D]``.  The previous-response block is
    zero at ``t=0`` and contains ``response_embeddings[:, t - 1]`` thereafter.
    Consequently, no response from the current or a future turn leaks into
    ``x[:, t]``.
    """
    if prompt_embeddings.ndim != 3 or response_embeddings.ndim != 3:
        raise ValueError("prompt and response embeddings must both have shape [N, T, D]")
    if prompt_embeddings.shape != response_embeddings.shape:
        raise ValueError(
            "prompt and response embeddings must have identical shapes; got "
            f"{tuple(prompt_embeddings.shape)} and {tuple(response_embeddings.shape)}"
        )

    previous_responses = torch.zeros_like(response_embeddings)
    previous_responses[:, 1:] = response_embeddings[:, :-1]
    return torch.cat((prompt_embeddings, previous_responses), dim=-1)


def event_metadata_from_labels(labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Derive one-based first-event times and event indicators from ``[N, T]`` labels."""
    if labels.ndim != 2:
        raise ValueError(f"labels must have shape [N, T], got {tuple(labels.shape)}")

    event_mask = labels.bool()
    event_observed = event_mask.any(dim=1)
    horizon = labels.shape[1]
    first_event_time = event_mask.to(torch.int64).argmax(dim=1) + 1
    no_event_time = torch.full_like(first_event_time, horizon + 1)
    event_times = torch.where(event_observed, first_event_time, no_event_time)
    return event_times, event_observed.to(torch.int64)


def labels_from_event_times(event_times: torch.Tensor, horizon: int) -> torch.Tensor:
    """Reconstruct first-event labels from one-based times with ``T + 1`` censoring."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    times = torch.as_tensor(event_times)
    if times.ndim != 1:
        raise ValueError(f"event_times must have shape [N], got {tuple(times.shape)}")
    if torch.is_floating_point(times) and not torch.equal(times, times.round()):
        raise ValueError("event_times must contain integer-valued counts")

    times = times.to(dtype=torch.long)
    if bool(((times < 1) | (times > horizon + 1)).any()):
        raise ValueError(f"event_times must lie in [1, {horizon + 1}]")

    labels = torch.zeros((len(times), horizon), dtype=torch.bool, device=times.device)
    observed = times <= horizon
    if bool(observed.any()):
        rows = torch.arange(len(times), device=times.device)[observed]
        labels[rows, times[observed] - 1] = True
    return labels


def normalize_event_times(
    event_times: torch.Tensor,
    event_observed: torch.Tensor,
    horizon: int,
) -> torch.Tensor:
    """Normalize legacy censored times to the canonical ``T + 1`` sentinel."""
    times = torch.as_tensor(event_times).clone()
    observed = torch.as_tensor(event_observed, device=times.device).bool()
    if times.ndim != 1 or observed.shape != times.shape:
        raise ValueError("event_times and event_observed must be one-dimensional and aligned")
    if bool(((times[observed] < 1) | (times[observed] > horizon)).any()):
        raise ValueError(f"observed event times must lie in [1, {horizon}]")
    times[~observed] = horizon + 1
    return times
