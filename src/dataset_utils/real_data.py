"""Build cached real-data tensors with a causal, turn-aligned feature layout.

For zero-based turn ``t``, ``x[i, t]`` concatenates the prompt embedding for
turn ``t`` and the response embedding from turn ``t - 1`` (zero at ``t=0``).
The label ``y[i, t]`` describes the response produced during turn ``t`` and is
therefore revealed only after the allocation decision for that turn.  Event
times are one-based; ``T + 1`` denotes no event inside the observed horizon.
"""

import glob
import os
import re

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.dataset_utils.temporal import (
    build_causal_turn_features,
    event_metadata_from_labels,
    normalize_event_times,
)
from src.multi_turn_data_generation.utils.utils import get_records

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _normalize_cached_times(event_times, event_observed, labels):
    """Read old caches safely while exposing the canonical ``T + 1`` sentinel."""
    normalized = normalize_event_times(
        torch.as_tensor(event_times),
        torch.as_tensor(event_observed),
        horizon=labels.shape[1],
    )
    return normalized.cpu().numpy()


def _remove_misaligned_records(records, prompt_embeddings, response_embeddings):
    """Drop rare duplicate/out-of-order rows while keeping all three arrays aligned."""
    if records is None:
        raise ValueError("attack log is empty or unreadable")
    if len(prompt_embeddings) != len(response_embeddings):
        raise ValueError("prompt and response embedding sequences have different lengths")

    record_i = 0
    while record_i < len(records):
        if records[record_i]["iteration"] != record_i:
            records.pop(record_i)
            prompt_embeddings = torch.cat(
                (prompt_embeddings[:record_i], prompt_embeddings[record_i + 1:]), dim=0
            )
            response_embeddings = torch.cat(
                (response_embeddings[:record_i], response_embeddings[record_i + 1:]), dim=0
            )
            continue
        record_i += 1

    if len(records) != len(prompt_embeddings):
        raise ValueError(
            "record and embedding counts differ after duplicate removal: "
            f"{len(records)} versus {len(prompt_embeddings)}"
        )
    return records, prompt_embeddings, response_embeddings


def _contiguous_embedding_indices(file_paths):
    """Return sorted numeric IDs and fail before silently misaligning file rows."""
    if not file_paths:
        raise FileNotFoundError("no attack_log_*_embeddings.pt files were found")
    pattern = re.compile(r"attack_log_(\d+)_embeddings\.pt$")
    indices = []
    for path in file_paths:
        match = pattern.search(os.path.basename(path))
        if match is None:
            raise ValueError(f"unexpected embedding filename: {path}")
        indices.append(int(match.group(1)))
    indices.sort()
    expected = list(range(len(indices)))
    if indices != expected:
        missing = sorted(set(expected) - set(indices))
        raise ValueError(
            "embedding file indices must be contiguous from zero so rows stay "
            f"aligned with attack logs; found {indices[:5]}...; missing {missing[:10]}"
        )
    return indices


def _validate_completed_sequences(all_scores, max_time):
    """Reject truncated non-event conversations before zero-padding them."""
    for sample_index, sample_scores in enumerate(all_scores):
        if len(sample_scores) == 0:
            raise ValueError(f"sample {sample_index} has no turns")
        stopped_on_event = bool(sample_scores[-1] == 10)
        if len(sample_scores) < max_time and not stopped_on_event:
            raise ValueError(
                f"sample {sample_index} has only {len(sample_scores)} of {max_time} turns "
                "and did not end in an event; refusing to treat missing turns as negatives"
            )


def generate_real_data(dataset_name, dataset_setup, load_x=True):
    """Load or construct deterministic train/calibration/test real-data splits."""
    data_store_dir = os.path.join(BASE_DIR, 'datasets/real_data', dataset_name, dataset_setup)
    if not os.path.exists(data_store_dir):
        print(f"warning, data does not exist in: {os.path.abspath(data_store_dir)}")
    if os.path.exists(f"{data_store_dir}/n_samples_test.npy"):
        p_train = np.load(f"{data_store_dir}/p_train.npy", allow_pickle=True)
        p_cal = np.load(f"{data_store_dir}/p_cal.npy", allow_pickle=True)
        p_test = np.load(f"{data_store_dir}/p_test.npy", allow_pickle=True)

        if load_x:
            x_train = np.load(f"{data_store_dir}/x_train.npy", allow_pickle=True)
            x_cal = np.load(f"{data_store_dir}/x_cal.npy", allow_pickle=True)
            x_test = np.load(f"{data_store_dir}/x_test.npy", allow_pickle=True)
        else:
            x_train = x_cal = x_test = None

        y_train = np.load(f"{data_store_dir}/y_train.npy", allow_pickle=True)
        y_cal = np.load(f"{data_store_dir}/y_cal.npy", allow_pickle=True)
        y_test = np.load(f"{data_store_dir}/y_test.npy", allow_pickle=True)

        t_tilde_train = np.load(f"{data_store_dir}/t_tilde_train.npy", allow_pickle=True)
        t_tilde_cal = np.load(f"{data_store_dir}/t_tilde_cal.npy", allow_pickle=True)
        t_tilde_test = np.load(f"{data_store_dir}/t_tilde_test.npy", allow_pickle=True)

        e_train = np.load(f"{data_store_dir}/e_train.npy", allow_pickle=True)
        e_cal = np.load(f"{data_store_dir}/e_cal.npy", allow_pickle=True)
        e_test = np.load(f"{data_store_dir}/e_test.npy", allow_pickle=True)

        b_train = np.load(f"{data_store_dir}/b_train.npy", allow_pickle=True)
        b_cal = np.load(f"{data_store_dir}/b_cal.npy", allow_pickle=True)
        b_test = np.load(f"{data_store_dir}/b_test.npy", allow_pickle=True)

        n_samples_train = np.load(f"{data_store_dir}/n_samples_train.npy", allow_pickle=True)
        n_samples_cal = np.load(f"{data_store_dir}/n_samples_cal.npy", allow_pickle=True)
        n_samples_test = np.load(f"{data_store_dir}/n_samples_test.npy", allow_pickle=True)
        t_tilde_train = _normalize_cached_times(t_tilde_train, e_train, y_train)
        t_tilde_cal = _normalize_cached_times(t_tilde_cal, e_cal, y_cal)
        t_tilde_test = _normalize_cached_times(t_tilde_test, e_test, y_test)

        return (
            p_train, p_cal, p_test, x_train, x_cal, x_test, y_train, y_cal, y_test, t_tilde_train, t_tilde_cal,
            t_tilde_test,
            e_train, e_cal, e_test, b_train, b_cal, b_test, n_samples_train, n_samples_cal, n_samples_test)
    print(f"data does not exist in {os.path.abspath(f'{data_store_dir}/n_samples_test.npy')}, so creating it")
    embedding_data_dir = f'./results/embedding/{dataset_name}/{dataset_setup}'
    all_prompt_embeddings = []
    all_response_embeddings = []
    search_pattern = os.path.join(embedding_data_dir, "attack_log_*_embeddings.pt")
    valid_files = glob.glob(search_pattern)
    file_indices = _contiguous_embedding_indices(valid_files)
    n_files = len(file_indices)
    error_files = []
    for i in tqdm(file_indices, desc="loading embeddings"):
        file_name = f'attack_log_{i}_embeddings.pt'
        file_path = os.path.join(embedding_data_dir, file_name)
        if not os.path.exists(file_path):
            raise Exception(f"File {file_name} does not exist in {embedding_data_dir}")
        try:
            file = torch.load(file_path, map_location="cpu")
            all_prompt_embeddings.append(file['embeddings'])
            all_response_embeddings.append(file['responses_embeddings'])
        except Exception:
            error_files.append(file_path)
    if error_files:
        print(f"error loading files:\n{error_files}")
        raise RuntimeError("error loading embedding files")

    attack_data_dir = os.path.join("results", dataset_name, dataset_setup)
    save_dir = attack_data_dir

    scores = []
    for i in tqdm(range(n_files), desc="loading records and scores"):
        file_name = f'attack_log_{i}.jsonl'
        if not os.path.exists(os.path.join(embedding_data_dir, f'attack_log_{i}_embeddings.pt')) or \
                not os.path.exists(os.path.join(attack_data_dir, file_name)):
            raise Exception(f"File {file_name} does not exist in {attack_data_dir} or in {embedding_data_dir}")
        records = get_records(i, save_dir)
        records, all_prompt_embeddings[i], all_response_embeddings[i] = _remove_misaligned_records(
            records,
            all_prompt_embeddings[i],
            all_response_embeddings[i],
        )
        none_idxs = [j for j in range(len(records)) if records[j]['score'] is None]
        for idx in none_idxs:
            print(f"warning, in index {i}, {idx} is None")
            records[idx]['score'] = 0
        scores.append(torch.tensor([record['score'] for record in records]))

    max_time = max(e.shape[0] for e in all_prompt_embeddings)
    _validate_completed_sequences(scores, max_time)
    for i in range(len(all_prompt_embeddings)):
        all_prompt_embeddings[i] = F.pad(all_prompt_embeddings[i],
                                         (0, 0, 0, max_time - all_prompt_embeddings[i].shape[0]), "constant", 0)
        all_response_embeddings[i] = F.pad(all_response_embeddings[i],
                                           (0, 0, 0, max_time - all_response_embeddings[i].shape[0]), "constant", 0)

    all_prompt_embeddings = torch.stack(all_prompt_embeddings)
    all_response_embeddings = torch.stack(all_response_embeddings)

    all_x = build_causal_turn_features(all_prompt_embeddings, all_response_embeddings)

    for i, sample_scores in enumerate(scores):
        scores[i] = F.pad(
            sample_scores,
            (0, max_time - sample_scores.shape[0]),
            "constant",
            0,
        )

    all_scores = torch.stack(scores)
    all_y = all_scores == 10
    b = all_y.sum(dim=1).to(torch.float32)

    t_tilde, e = event_metadata_from_labels(all_y)

    x = all_x.numpy()
    p = np.zeros(len(all_x))
    y = all_y.numpy()
    t_tilde = t_tilde.numpy()
    e = e.numpy()
    b = b.numpy()
    n_samples = max_time * np.ones(len(x), dtype=int)
    assert x.shape[0] == y.shape[0] and x.shape[1] == y.shape[1]

    (
        p_train,
        p_test,
        x_train,
        x_test,
        y_train,
        y_test,
        t_tilde_train,
        t_tilde_test,
        e_train,
        e_test,
        b_train,
        b_test,
        n_samples_train,
        n_samples_test,
    ) = train_test_split(p, x, y, t_tilde, e, b, n_samples, test_size=0.2, random_state=42)

    (
        p_train,
        p_cal,
        x_train,
        x_cal,
        y_train,
        y_cal,
        t_tilde_train,
        t_tilde_cal,
        e_train,
        e_cal,
        b_train,
        b_cal,
        n_samples_train,
        n_samples_cal,
    ) = train_test_split(p_train, x_train, y_train, t_tilde_train, e_train, b_train, n_samples_train, test_size=0.5,
                         random_state=42)

    os.makedirs(data_store_dir, exist_ok=True)
    np.save(f"{data_store_dir}/p_train", p_train)
    np.save(f"{data_store_dir}/p_cal", p_cal)
    np.save(f"{data_store_dir}/p_test", p_test)
    np.save(f"{data_store_dir}/x_train", x_train)
    np.save(f"{data_store_dir}/x_cal", x_cal)
    np.save(f"{data_store_dir}/x_test", x_test)
    np.save(f"{data_store_dir}/y_train", y_train)
    np.save(f"{data_store_dir}/y_cal", y_cal)
    np.save(f"{data_store_dir}/y_test", y_test)
    np.save(f"{data_store_dir}/t_tilde_train", t_tilde_train)
    np.save(f"{data_store_dir}/t_tilde_cal", t_tilde_cal)
    np.save(f"{data_store_dir}/t_tilde_test", t_tilde_test)
    np.save(f"{data_store_dir}/e_train", e_train)
    np.save(f"{data_store_dir}/e_cal", e_cal)
    np.save(f"{data_store_dir}/e_test", e_test)
    np.save(f"{data_store_dir}/b_train", b_train)
    np.save(f"{data_store_dir}/b_cal", b_cal)
    np.save(f"{data_store_dir}/b_test", b_test)
    np.save(f"{data_store_dir}/n_samples_train", n_samples_train)
    np.save(f"{data_store_dir}/n_samples_cal", n_samples_cal)
    np.save(f"{data_store_dir}/n_samples_test", n_samples_test)

    return (
        p_train, p_cal, p_test, x_train, x_cal, x_test, y_train, y_cal, y_test, t_tilde_train, t_tilde_cal,
        t_tilde_test,
        e_train, e_cal, e_test, b_train, b_cal, b_test, n_samples_train, n_samples_cal, n_samples_test)

