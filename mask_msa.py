# mask_msa.py
import argparse
import random
import string
import sys
from pathlib import Path

def mask_a3m_file(
    input_path: Path, output_path: Path, mask_fraction: float
) -> None:
    """Reads an A3M file and writes a masked version to a new file.

    This function correctly parses multi-line FASTA/A3M entries. It identifies
    the first entry (the query) and writes it to the new file without any
    changes. For all subsequent entries in the MSA, it replaces each
    uppercase amino acid in their sequence lines with 'X' based on the
    provided probability (mask_fraction).

    Args:
        input_path: The path to the source A3M file.
        output_path: The path where the masked A3M file will be saved.
        mask_fraction: The fraction (0.0 to 1.0) of residues to mask.

    Raises:
        FileNotFoundError: If the input_path does not exist.
        ValueError: If the mask_fraction is not between 0.0 and 1.0 or the
                    input file is not a valid FASTA/A3M format.
        IOError: If there is an issue reading from or writing to the files.
        TypeError: If arguments are of the incorrect type.
    """
    # --- 1. Input Validation ---
    if not isinstance(input_path, Path):
        raise TypeError("input_path must be a Path object.")
    if not input_path.is_file():
        raise FileNotFoundError(f"Error: Input file not found at '{input_path}'")

    if not isinstance(output_path, Path):
        raise TypeError("output_path must be a Path object.")

    if not isinstance(mask_fraction, float):
        raise TypeError("mask_fraction must be a float.")
    if not 0.0 <= mask_fraction <= 1.0:
        raise ValueError("Error: mask_fraction must be between 0.0 and 1.0.")

    # --- 2. Core Logic in a try/except block ---
    try:
        lines = input_path.read_text().splitlines()

        if mask_fraction == 0.0:
            print(f"INFO: Mask fraction is 0. Copying '{input_path}' to '{output_path}'.")
            output_path.write_text("\n".join(lines) + "\n")
            return

        # Find the start of each sequence entry
        header_indices = [i for i, line in enumerate(lines) if line.startswith('>')]
        if not header_indices:
            raise ValueError(f"Error: No FASTA headers ('>') found in '{input_path}'.")

        modified_lines = []
        amino_acids = set(string.ascii_uppercase)

        # Process each entry based on its header's index
        for i, header_idx in enumerate(header_indices):
            # Determine the lines for the current entry (from this header to the next)
            start_index = header_idx
            end_index = header_indices[i + 1] if i + 1 < len(header_indices) else len(lines)
            entry_lines = lines[start_index:end_index]

            header = entry_lines[0]
            sequence_lines = entry_lines[1:]

            modified_lines.append(header)

            # If it's the first entry (the query), add its sequence lines unmodified
            if i == 0:
                modified_lines.extend(sequence_lines)
                continue

            # For all subsequent entries, apply masking to their sequence lines
            for seq_line in sequence_lines:
                masked_line = [
                    'X' if char in amino_acids and random.random() < mask_fraction else char
                    for char in seq_line
                ]
                modified_lines.append("".join(masked_line))

        output_path.write_text("\n".join(modified_lines) + "\n")

    except IOError as e:
        raise IOError(f"Error processing files: {e}") from e


def main():
    """Main function to parse arguments and run the masking script."""
    parser = argparse.ArgumentParser(
        description="Create a masked copy of an A3M file, preserving the first sequence."
    )
    parser.add_argument(
        "--input_path",
        type=Path,
        required=True,
        help="Path to the source .a3m file."
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Path to write the new masked .a3m file."
    )
    parser.add_argument(
        "--mask_fraction",
        type=float,
        required=True,
        help="Fraction of residues to mask (e.g., 0.15 for 15%%)."
    )

    args = parser.parse_args()

    try:
        mask_a3m_file(args.input_path, args.output_path, args.mask_fraction)
        print(f"✅ Successfully created masked file '{args.output_path}'.")
    except (FileNotFoundError, ValueError, IOError, TypeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
