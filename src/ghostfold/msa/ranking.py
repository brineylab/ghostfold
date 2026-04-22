"""MSA subsampling strategies for selecting representative sequence subsets.

All strategies return a list of sequences with the query (or first sequence)
guaranteed as the first element when include_query=True.
"""
import random
from typing import Literal

import numpy as np

from ghostfold.msa.neff import calculate_neff

_Strategy = Literal["farthest_first", "max_coverage", "column_entropy", "neff_contribution", "random"]


def _to_matrix(sequences: list[str]) -> np.ndarray:
    """Convert a list of equal-length sequences to a uint8 NumPy matrix (N, L)."""
    L = len(sequences[0])
    return np.frombuffer(
        b"".join(s.encode() for s in sequences), dtype=np.uint8
    ).reshape(len(sequences), L)


def _hamming_distances(query_row: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Return (N,) array of Hamming *distances* (1 − identity) from query_row."""
    return 1.0 - (query_row == matrix).mean(axis=1)


# ---------------------------------------------------------------------------
# Individual strategy implementations
# ---------------------------------------------------------------------------

def _farthest_first(sequences: list[str], n_select: int, query_idx: int) -> list[int]:
    """Greedy MaxMin: iteratively pick the sequence furthest from the current set."""
    arr = _to_matrix(sequences)
    selected = [query_idx]
    # min distance of each candidate to the current selected set
    min_dist = _hamming_distances(arr[query_idx], arr)
    min_dist[query_idx] = -1.0  # exclude already-selected

    while len(selected) < n_select:
        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)
        new_dist = _hamming_distances(arr[next_idx], arr)
        min_dist = np.minimum(min_dist, new_dist)
        min_dist[next_idx] = -1.0

    return selected


def _max_coverage(sequences: list[str], n_select: int, query_idx: int) -> list[int]:
    """Greedy: each step picks the sequence covering the most new non-gap positions."""
    arr = _to_matrix(sequences)
    gap_byte = ord("-")
    covered = (arr[query_idx] != gap_byte)  # positions covered so far
    selected = [query_idx]
    remaining = set(range(len(sequences))) - {query_idx}

    while len(selected) < n_select and remaining:
        best_idx, best_new = -1, -1
        for i in remaining:
            new = int(np.sum((arr[i] != gap_byte) & ~covered))
            if new > best_new:
                best_new, best_idx = new, i
        if best_idx == -1:
            break
        selected.append(best_idx)
        covered |= arr[best_idx] != gap_byte
        remaining.remove(best_idx)

    # Fill remainder with random sequences if needed
    if len(selected) < n_select and remaining:
        extra = random.sample(list(remaining), min(n_select - len(selected), len(remaining)))
        selected.extend(extra)

    return selected


def _column_entropy_scores(sequences: list[str]) -> np.ndarray:
    """Score each sequence by its per-column Shannon entropy contribution."""
    arr = _to_matrix(sequences)
    N, L = arr.shape
    scores = np.zeros(N)

    for col in range(L):
        col_data = arr[:, col]
        vals, counts = np.unique(col_data, return_counts=True)
        probs = counts / N
        col_entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
        # Sequences that carry a *rare* character in this column contribute more
        for v, c in zip(vals, counts):
            if c < N:  # not universal
                mask = col_data == v
                scores[mask] += col_entropy / max(c, 1)

    return scores


def _column_entropy_select(sequences: list[str], n_select: int, query_idx: int) -> list[int]:
    scores = _column_entropy_scores(sequences)
    scores[query_idx] = float("inf")  # always keep query first
    ranked = list(np.argsort(-scores))
    # Ensure query_idx is first
    ranked = [query_idx] + [i for i in ranked if i != query_idx]
    return ranked[:n_select]


def _neff_contribution_select(sequences: list[str], n_select: int, query_idx: int) -> list[int]:
    """Greedy: add the sequence that maximises ΔNEFF at each step."""
    selected_seqs = [sequences[query_idx]]
    selected_idx = [query_idx]
    remaining = list(range(len(sequences)))
    remaining.remove(query_idx)
    current_neff = calculate_neff(selected_seqs)

    while len(selected_idx) < n_select and remaining:
        best_idx, best_delta = -1, -float("inf")
        for i in remaining:
            candidate = selected_seqs + [sequences[i]]
            try:
                new_neff = calculate_neff(candidate)
            except ValueError:
                continue
            delta = new_neff - current_neff
            if delta > best_delta:
                best_delta, best_idx = delta, i
        if best_idx == -1:
            break
        selected_idx.append(best_idx)
        selected_seqs.append(sequences[best_idx])
        current_neff += best_delta
        remaining.remove(best_idx)

    if len(selected_idx) < n_select and remaining:
        extra = random.sample(remaining, min(n_select - len(selected_idx), len(remaining)))
        selected_idx.extend(extra)

    return selected_idx


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rank_and_subsample(
    sequences: list[str],
    n_select: int,
    strategy: _Strategy = "farthest_first",
    query: str | None = None,
) -> list[str]:
    """Select a representative subset of *sequences* of size *n_select*.

    Args:
        sequences: Input MSA sequences (all the same length for alignment-
                   based strategies; first sequence treated as query if
                   *query* is None).
        n_select:  Number of sequences to return (capped at len(sequences)).
        strategy:  Subsampling algorithm:
                   - ``farthest_first``    — greedy MaxMin Hamming diversity
                   - ``max_coverage``      — maximise non-gap position coverage
                   - ``column_entropy``    — maximise per-column entropy signal
                   - ``neff_contribution`` — greedily maximise ΔNEFF
                   - ``random``            — uniform random sample
        query:     If provided, this sequence is forced as element 0.
                   If None, sequences[0] is used.

    Returns:
        List of selected sequences with query as the first element.
    """
    if not sequences:
        return []

    n_select = min(n_select, len(sequences))

    if query is not None and query not in sequences:
        sequences = [query] + sequences
        query_idx = 0
    elif query is not None:
        query_idx = sequences.index(query)
    else:
        query_idx = 0

    if n_select <= 1:
        return [sequences[query_idx]]

    if strategy == "random":
        pool = [i for i in range(len(sequences)) if i != query_idx]
        chosen = random.sample(pool, min(n_select - 1, len(pool)))
        indices = [query_idx] + chosen
    elif strategy == "farthest_first":
        indices = _farthest_first(sequences, n_select, query_idx)
    elif strategy == "max_coverage":
        indices = _max_coverage(sequences, n_select, query_idx)
    elif strategy == "column_entropy":
        indices = _column_entropy_select(sequences, n_select, query_idx)
    elif strategy == "neff_contribution":
        indices = _neff_contribution_select(sequences, n_select, query_idx)
    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Choose from: "
            "farthest_first, max_coverage, column_entropy, neff_contribution, random."
        )

    return [sequences[i] for i in indices]
