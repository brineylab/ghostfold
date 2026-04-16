import re
import time
from difflib import SequenceMatcher
from typing import List, Optional

import numpy as np

from ghostfold.core.logging import get_logger

logger = get_logger("filters")

AA_LETTERS = "ACDEFGHIKLMNPQRSTVWYX"


def clean_repeats(
        sequences: List[str],
        mono_repeat_threshold: int = 5,
        dipeptide_repeat_threshold: int = 3,
        replacement_char: str = 'X',
        mono_replacement_char: Optional[str] = None,
        dipeptide_replacement_char: Optional[str] = None
) -> List[str]:
    """Replaces mono-amino acid and dipeptide repeat stretches in sequences."""
    mono_pattern = re.compile(rf"([A-Z])\1{{{mono_repeat_threshold - 1},}}")
    dipeptide_pattern = re.compile(
        rf"(([A-Z]{{2}}))(\1){{{dipeptide_repeat_threshold - 1},}}"
    )

    cleaned_sequences = []
    modified_count = 0

    mono_char = mono_replacement_char or replacement_char
    dipeptide_char = dipeptide_replacement_char or replacement_char

    for idx, seq in enumerate(sequences):
        original_seq = seq
        changes = []

        def replace_mono(m: re.Match) -> str:
            aa = m.group(1)
            length = len(m.group())
            changes.append(f"poly-{aa} ({length}x)")
            return mono_char * length

        seq = mono_pattern.sub(replace_mono, seq)

        def replace_dipeptide(m: re.Match) -> str:
            motif = m.group(1)
            full = m.group(0)
            count = len(full) // 2
            changes.append(f"{motif * dipeptide_repeat_threshold} ({count}x)")
            return dipeptide_char * len(full)

        seq = dipeptide_pattern.sub(replace_dipeptide, seq)

        if seq != original_seq:
            modified_count += 1

        cleaned_sequences.append(seq)

    return cleaned_sequences


def sequence_entropy(seq: str) -> float:
    """Calculates the Shannon entropy of a protein sequence based on amino acid frequencies."""
    if not seq:
        return 0.0
    # Fix 3: vectorized — bincount over byte values then index AA_LETTERS bytes
    # Matches original behaviour: only AA_LETTERS chars counted; denominator = full length
    arr = np.frombuffer(seq.encode(), dtype=np.uint8)
    aa_bytes = np.frombuffer(AA_LETTERS.encode(), dtype=np.uint8)
    counts = np.bincount(arr, minlength=256)
    aa_counts = counts[aa_bytes]
    probs = aa_counts / len(seq)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def is_similar(a: str, b: str, threshold: float = 0.95) -> bool:
    """Checks if two sequences are similar based on Hamming identity threshold.

    Requires equal-length strings (aligned sequences). Falls back to
    SequenceMatcher for unequal lengths.
    """
    if len(a) != len(b):
        return SequenceMatcher(None, a, b).ratio() > threshold
    arr_a = np.frombuffer(a.encode(), dtype=np.uint8)
    arr_b = np.frombuffer(b.encode(), dtype=np.uint8)
    return float((arr_a == arr_b).mean()) > threshold


def deduplicate(sequences: List[str], threshold: float = 0.95) -> List[str]:
    """Removes near-duplicate sequences using vectorized Hamming identity.

    All sequences must be the same length (standard for aligned MSAs).
    Falls back to sequential comparison for unequal-length inputs.
    """
    if not sequences:
        return []
    if len(set(len(s) for s in sequences)) > 1:
        # Unequal lengths: fall back to original O(n²) logic
        unique: List[str] = []
        for s in sequences:
            if not any(is_similar(s, u, threshold) for u in unique):
                unique.append(s)
        return unique

    L = len(sequences[0])
    arr = np.frombuffer(
        b"".join(s.encode() for s in sequences), dtype=np.uint8
    ).reshape(-1, L)  # (N, L)

    unique_indices: List[int] = [0]
    unique_arr = arr[[0]]  # (1, L)

    for i in range(1, len(arr)):
        row = arr[i:i+1]  # (1, L)
        identity = (row == unique_arr).mean(axis=1)  # (num_unique,)
        if not np.any(identity > threshold):
            unique_indices.append(i)
            unique_arr = arr[unique_indices]

    return [sequences[i] for i in unique_indices]


def filter_sequences(
    sequences: List[str],
    expected_length: int,
    entropy_threshold: float = 2.0,
    similarity_threshold: float = 0.95
) -> List[str]:
    """Filters a list of sequences, reporting the count and time for each step."""
    initial_count = len(sequences)
    if initial_count == 0:
        return []

    logger.info(f"--- Starting filtration of {initial_count} sequences ---")

    timing_data = {}
    processed_sequences = sequences

    start_total_time = time.time()
    last_time = start_total_time

    # Step 1: Clean Repeats
    processed_sequences = clean_repeats(processed_sequences)
    current_time = time.time()
    timing_data['Clean Repeats'] = {'count': len(processed_sequences), 'time': current_time - last_time}
    last_time = current_time

    # Step 2: Length Filter
    processed_sequences = [s for s in processed_sequences if len(s) == expected_length]
    current_time = time.time()
    timing_data['Length Match'] = {'count': len(processed_sequences), 'time': current_time - last_time}
    last_time = current_time

    # Step 3: Entropy Filter
    processed_sequences = [s for s in processed_sequences if sequence_entropy(s) > entropy_threshold]
    current_time = time.time()
    timing_data['Entropy Filter'] = {'count': len(processed_sequences), 'time': current_time - last_time}
    last_time = current_time

    # Step 4: Deduplicate (remove near-identical sequences)
    if len(processed_sequences) > 1:
        processed_sequences = deduplicate(processed_sequences, similarity_threshold)
    current_time = time.time()
    timing_data['Deduplicate'] = {'count': len(processed_sequences), 'time': current_time - last_time}
    last_time = current_time

    # --- Reporting ---
    logger.info(f"{'Step':<20} | {'Removed':>9} | {'Remaining':>11} | {'Time (s)':>10}")
    logger.info("-" * 59)

    last_count = initial_count
    for step_name, data in timing_data.items():
        current_count = data['count']
        time_taken = data['time']
        removed = last_count - current_count
        logger.info(f"{step_name:<20} | {removed:>9} | {current_count:>11} | {time_taken:>10.2f}")
        last_count = current_count

    total_time = time.time() - start_total_time
    final_count = len(processed_sequences)
    logger.info("-" * 59)
    logger.info(f"Total time: {total_time:.2f}s. Final count: {final_count} (removed {initial_count - final_count}).")
    logger.info("--- Filtration complete ---")

    return processed_sequences
