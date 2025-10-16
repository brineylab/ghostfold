import concurrent.futures
import csv
import math
import pathlib
import sys
from typing import List, Optional, Tuple

def parse_a3m(file_path: pathlib.Path) -> List[str]:
    """Parses a single a3m file to extract all sequences.

    This function reads a file in a3m or FASTA-like format and returns a list
    of the sequences. It assumes each sequence in the alignment has the same
    length.

    Args:
        file_path: The path to the .a3m file.

    Returns:
        A list of strings, where each string is a sequence from the MSA.

    Raises:
        FileNotFoundError: If the specified file_path does not exist.
    """
    # --- Input Validation ---
    if not file_path.is_file():
        raise FileNotFoundError(f"Error: The file '{file_path}' was not found.")

    sequences: List[str] = []
    current_sequence: str = ""

    # --- Core Logic ---
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    if current_sequence:
                        sequences.append(current_sequence)
                    current_sequence = ""
                else:
                    current_sequence += "".join(filter(lambda x: x.isalpha() or x == '-', line))
            if current_sequence:
                sequences.append(current_sequence)
        return sequences
    except Exception as e:
        # --- Error Handling ---
        print(f"An error occurred while reading {file_path}: {e}", file=sys.stderr)
        return []

def calculate_neff(sequences: List[str], identity_threshold: float = 0.5) -> float:
    """Calculates the Neff score for a list of aligned sequences.

    Args:
        sequences: A list of strings representing the multiple sequence alignment.
        identity_threshold: The sequence identity threshold (default is 0.8).

    Returns:
        The calculated Neff value as a float.

    Raises:
        ValueError: If sequences have inconsistent lengths, have zero length,
                    or if the identity_threshold is not between 0 and 1.
    """
    # --- Input Validation ---
    if not sequences:
        return 0.0
    N, L = len(sequences), len(sequences[0])
    if L == 0:
        raise ValueError("Error: Sequences cannot have a length of 0.")
    if not all(len(s) == L for s in sequences):
        raise ValueError("Error: All sequences must have the same length.")
    if not (0.0 <= identity_threshold <= 1.0):
        raise ValueError("Error: identity_threshold must be between 0.0 and 1.0.")

    # --- Core Logic ---
    try:
        total_sum = 0.0
        for n in range(N):
            similar_sequences = 0
            for m in range(N):
                if n == m:
                    continue
                identical_positions = sum(1 for c1, c2 in zip(sequences[n], sequences[m]) if c1 == c2)
                if (identical_positions / L) >= identity_threshold:
                    similar_sequences += 1
            total_sum += 1.0 / (1.0 + similar_sequences)
        return (1.0 / math.sqrt(L)) * total_sum
    except Exception as e:
        # --- Error Handling ---
        print(f"An unexpected error occurred during Neff calculation: {e}", file=sys.stderr)
        return 0.0

def process_single_file(msa_file: pathlib.Path) -> Optional[Tuple[str, float]]:
    """
    Worker function to parse one MSA file and calculate its Neff score.
    
    This function is designed to be called by a parallel process pool.

    Args:
        msa_file: The path to the .a3m file to process.

    Returns:
        A tuple containing the protein name and its Neff score, or None if
        processing fails.
    """
    try:
        # UPDATED: The protein name is the name of the file's parent directory.
        protein_name = msa_file.parent.name
        
        sequences = parse_a3m(msa_file)
        if not sequences:
            return None

        neff_value = calculate_neff(sequences)
        return (protein_name, neff_value)
    except Exception as e:
        print(f"Failed to process {msa_file}: {e}", file=sys.stderr)
        return None

def run_neff_calculation_in_parallel(root_dir: str) -> None:
    """
    Finds all .a3m files and calculates their Neff scores in parallel, saving
    the output to a CSV file.

    Args:
        root_dir: The path to the root directory (the 'main_folder').

    Raises:
        FileNotFoundError: If the root_dir does not exist or is not a directory.
    """
    # --- Input Validation ---
    root_path = pathlib.Path(root_dir)
    if not root_path.is_dir():
        raise FileNotFoundError(f"Error: The directory '{root_dir}' was not found.")

    # --- Core Logic ---
    # UPDATED: The search pattern now reflects the new directory structure.
    a3m_files = sorted(list(root_path.glob('msa/*/*.a3m')))
    if not a3m_files:
        print(f"No '.a3m' files found matching 'msa/*/*.a3m' under '{root_dir}'.")
        return

    print(f"Found {len(a3m_files)} MSA files. Starting parallel processing...")
    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_results = executor.map(process_single_file, a3m_files)
        results = [res for res in future_results if res is not None]

    # --- Output to CSV File ---
    if not results:
        print("Processing complete, but no valid results were generated.")
        return

    output_csv_path = root_path / "neff_results.csv"
    sorted_results = sorted(results)

    try:
        with open(output_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['pdb', 'NAD']) # Write header
            for protein_name, neff_value in sorted_results:
                writer.writerow([protein_name, f"{neff_value:.2f}"])
        
        print(f"\n✅ Success! Processed {len(sorted_results)} files.")
        print(f"Results saved to: {output_csv_path}")
    except IOError as e:
        print(f"\n❌ Error writing to CSV file: {e}", file=sys.stderr)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Calculates Neff score in parallel for MSAs in a directory.", file=sys.stderr)
        print("\nUsage: python your_script_name.py <path_to_main_folder>", file=sys.stderr)
        sys.exit(1)

    main_folder_path = sys.argv[1]
    run_neff_calculation_in_parallel(main_folder_path)
