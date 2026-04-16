import concurrent.futures
import csv
import math
import pathlib
from typing import List, Optional, Tuple

import numpy as np

from ghostfold.core.logging import get_logger

logger = get_logger("neff")


def parse_a3m(file_path: pathlib.Path) -> List[str]:
    """Parses a single a3m file to extract all sequences.

    Uses list accumulation instead of string concatenation to avoid O(n²)
    behavior for sequences spanning many lines.

    Args:
        file_path: The path to the .a3m file.

    Returns:
        A list of strings, where each string is a sequence from the MSA.

    Raises:
        FileNotFoundError: If the specified file_path does not exist.
    """
    if not file_path.is_file():
        raise FileNotFoundError(f"Error: The file '{file_path}' was not found.")

    sequences: List[str] = []
    current_parts: List[str] = []

    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    if current_parts:
                        sequences.append("".join(current_parts))
                    current_parts = []
                else:
                    current_parts.append(
                        "".join(c for c in line if c.isupper() or c == '-')
                    )
            if current_parts:
                sequences.append("".join(current_parts))
        return sequences
    except Exception as e:
        logger.error(f"An error occurred while reading {file_path}: {e}")
        return []


def calculate_neff(sequences: List[str], identity_threshold: float = 0.5) -> float:
    """Calculates the Neff score for a list of aligned sequences.

    Args:
        sequences: A list of strings representing the multiple sequence alignment.
        identity_threshold: The sequence identity threshold (default is 0.5).

    Returns:
        The calculated Neff value as a float.

    Raises:
        ValueError: If sequences have inconsistent lengths, zero length,
                    or identity_threshold is not between 0 and 1.
    """
    if not sequences:
        return 0.0
    N, L = len(sequences), len(sequences[0])
    if L == 0:
        raise ValueError("Error: Sequences cannot have a length of 0.")
    if not all(len(s) == L for s in sequences):
        raise ValueError("Error: All sequences must have the same length.")
    if not (0.0 <= identity_threshold <= 1.0):
        raise ValueError("Error: identity_threshold must be between 0.0 and 1.0.")

    _NEFF_CHUNK = 256  # rows per chunk; caps peak memory at chunk × N × L bytes

    try:
        msa_arr = np.frombuffer(
            b"".join(s.encode() for s in sequences), dtype=np.uint8
        ).reshape(N, L)

        weights = np.zeros(N, dtype=np.float64)
        for i in range(0, N, _NEFF_CHUNK):
            block = msa_arr[i:i + _NEFF_CHUNK]          # (chunk, L)
            # identity: (chunk, N) — fraction of positions that match
            id_mat = (block[:, None, :] == msa_arr[None, :, :]).mean(axis=2)
            above = id_mat >= identity_threshold          # (chunk, N) bool

            # zero out self-comparisons on the diagonal block
            chunk_size = block.shape[0]
            row_idx = np.arange(chunk_size)
            col_idx = np.arange(i, i + chunk_size)
            above[row_idx, col_idx] = False

            weights[i:i + chunk_size] = 1.0 / (1.0 + above.sum(axis=1).astype(float))

        return (1.0 / math.sqrt(L)) * float(weights.sum())
    except Exception as e:
        logger.error(f"An unexpected error occurred during Neff calculation: {e}")
        return 0.0


def process_single_file(msa_file: pathlib.Path) -> Optional[Tuple[str, float]]:
    """Worker function to parse one MSA file and calculate its Neff score."""
    try:
        protein_name = msa_file.parent.name
        sequences = parse_a3m(msa_file)
        if not sequences:
            return None
        neff_value = calculate_neff(sequences)
        return (protein_name, neff_value)
    except Exception as e:
        logger.error(f"Failed to process {msa_file}: {e}")
        return None


def run_neff_calculation_in_parallel(root_dir: str) -> None:
    """Finds all .a3m files and calculates their Neff scores in parallel.

    Args:
        root_dir: The path to the root directory.

    Raises:
        FileNotFoundError: If the root_dir does not exist or is not a directory.
    """
    root_path = pathlib.Path(root_dir)
    if not root_path.is_dir():
        raise FileNotFoundError(f"Error: The directory '{root_dir}' was not found.")

    a3m_files = sorted(list(root_path.glob('msa/*/*.a3m')))
    if not a3m_files:
        logger.info(f"No '.a3m' files found matching 'msa/*/*.a3m' under '{root_dir}'.")
        return

    logger.info(f"Found {len(a3m_files)} MSA files. Starting parallel processing...")
    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_results = executor.map(process_single_file, a3m_files)
        results = [res for res in future_results if res is not None]

    if not results:
        logger.info("Processing complete, but no valid results were generated.")
        return

    output_csv_path = root_path / "neff_results.csv"
    sorted_results = sorted(results)

    try:
        with open(output_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['pdb', 'NAD'])
            for protein_name, neff_value in sorted_results:
                writer.writerow([protein_name, f"{neff_value:.2f}"])

        logger.info(f"Success! Processed {len(sorted_results)} files.")
        logger.info(f"Results saved to: {output_csv_path}")
    except IOError as e:
        logger.error(f"Error writing to CSV file: {e}")
