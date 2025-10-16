# utils/filters.py
import re
import time
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

AA_LETTERS = "ACDEFGHIKLMNPQRSTVWYX"


def clean_repeats(
        sequences: List[str],
        mono_repeat_threshold: int = 5,
        dipeptide_repeat_threshold: int = 3,
        replacement_char: str = 'X',
        mono_replacement_char: Optional[str] = None,
        dipeptide_replacement_char: Optional[str] = None
) -> List[str]:
    """
    Replaces mono-amino acid and dipeptide repeat stretches in sequences.
    Logs what was removed and how many sequences were modified.

    Args:
        sequences (List[str]): Protein sequences.
        mono_repeat_threshold (int): Threshold for single AA repeats (e.g., EEEEEE).
        dipeptide_repeat_threshold (int): Threshold for dipeptide repeats (e.g., GSGSGS).
        replacement_char (str): Default character to replace both repeat types.
        mono_replacement_char (Optional[str]): Override for mono-AA repeats.
        dipeptide_replacement_char (Optional[str]): Override for dipeptide repeats.

    Returns:
        List[str]: Cleaned protein sequences.
    """
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
            # print(f"Seq {idx+1}: removed " + "; ".join(changes)) # Optional: uncomment for verbose logging

        cleaned_sequences.append(seq)

    # print(f"{modified_count} out of {len(sequences)} sequences were modified to remove repeat motifs.")
    return cleaned_sequences


def sequence_entropy(seq: str) -> float:
    """
    Calculates the Shannon entropy of a protein sequence based on amino acid frequencies.

    Args:
        seq (str): The protein sequence.

    Returns:
        float: The calculated entropy.
    """
    if not seq:
        return 0.0
    aa_counts = np.array([seq.count(aa) for aa in AA_LETTERS])
    probs = aa_counts / len(seq)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def is_similar(a: str, b: str, threshold: float = 0.95) -> bool:
    """
    Checks if two sequences are similar based on a given threshold using SequenceMatcher.

    Args:
        a (str): First sequence.
        b (str): Second sequence.
        threshold (float): Similarity threshold (default: 0.95).

    Returns:
        bool: True if sequences are similar, False otherwise.
    """
    return SequenceMatcher(None, a, b).ratio() > threshold


def deduplicate(sequences: List[str], threshold: float = 0.95) -> List[str]:
    """
    Removes duplicate sequences from a list based on a similarity threshold.

    Args:
        sequences (List[str]): List of protein sequences.
        threshold (float): Similarity threshold for deduplication (default: 0.95).

    Returns:
        List[str]: List of unique protein sequences.
    """
    unique: List[str] = []
    for s in sequences:
        if not any(is_similar(s, u, threshold) for u in unique):
            unique.append(s)
    return unique


# def is_reasonable_aa_composition(seq: str, max_fraction: float = 0.05) -> bool:
#     """Check if any single amino acid exceeds max allowed frequency."""
#     aa_freq = {aa: seq.count(aa)/len(seq) for aa in AA_LETTERS}
#     return all(f <= max_fraction for f in aa_freq.values())

# def cluster_filter(sequences: List[str], threshold: float = 0.5) -> List[str]:
#     if len(sequences) <= 1:
#         return sequences
#     vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 3))
#     X = vectorizer.fit_transform(sequences)
#     sim = cosine_similarity(X)
#     to_remove = set()
#     for i in range(len(sim)):
#         for j in range(i + 1, len(sim)):
#             if sim[i, j] > threshold:
#                 to_remove.add(j)
#     return [s for i, s in enumerate(sequences) if i not in to_remove]


def filter_sequences(
    sequences: List[str],
    expected_length: int,
    entropy_threshold: float = 2.0,
    similarity_threshold: float = 0.95
) -> List[str]:
    """
    Filters a list of sequences, reporting the count and time for each step.

    Args:
        sequences (List[str]): The initial list of protein sequences.
        expected_length (int): The required length for sequences to be kept.
        entropy_threshold (float): The minimum entropy a sequence must have.
        similarity_threshold (float): The threshold for deduplication.

    Returns:
        List[str]: The list of filtered protein sequences.
    """
    initial_count = len(sequences)
    if initial_count == 0:
        return []

    print(f"\n--- Starting filtration of {initial_count} sequences ---")
    
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

    # # Step 4: Deduplication
    # processed_sequences = deduplicate(processed_sequences, similarity_threshold)
    # current_time = time.time()
    # timing_data['Deduplication'] = {'count': len(processed_sequences), 'time': current_time - last_time}
    
    # --- Reporting ---
    print(f"{'Step':<20} | {'Removed':>9} | {'Remaining':>11} | {'Time (s)':>10}")
    print("-" * 59)
    
    last_count = initial_count
    for step_name, data in timing_data.items():
        current_count = data['count']
        time_taken = data['time']
        removed = last_count - current_count
        print(f"{step_name:<20} | {removed:>9} | {current_count:>11} | {time_taken:>10.2f}")
        last_count = current_count
        
    total_time = time.time() - start_total_time
    final_count = len(processed_sequences)
    print("-" * 59)
    print(f"Total time: {total_time:.2f}s. Final count: {final_count} (removed {initial_count - final_count}).")
    print("--- Filtration complete ---\n")
    
    return processed_sequences
