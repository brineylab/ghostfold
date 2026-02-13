import random
import string
from pathlib import Path


def mask_a3m_file(
    input_path: Path, output_path: Path, mask_fraction: float
) -> None:
    """Reads an A3M file and writes a masked version to a new file.

    The first entry (query) is preserved unchanged. For all subsequent entries,
    each uppercase amino acid is replaced with 'X' based on mask_fraction.

    Args:
        input_path: The path to the source A3M file.
        output_path: The path where the masked A3M file will be saved.
        mask_fraction: The fraction (0.0 to 1.0) of residues to mask.

    Raises:
        FileNotFoundError: If the input_path does not exist.
        ValueError: If mask_fraction is not between 0.0 and 1.0.
        TypeError: If arguments are of the incorrect type.
    """
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

    try:
        lines = input_path.read_text().splitlines()

        if mask_fraction == 0.0:
            print(f"INFO: Mask fraction is 0. Copying '{input_path}' to '{output_path}'.")
            output_path.write_text("\n".join(lines) + "\n")
            return

        header_indices = [i for i, line in enumerate(lines) if line.startswith('>')]
        if not header_indices:
            raise ValueError(f"Error: No FASTA headers ('>') found in '{input_path}'.")

        modified_lines = []
        amino_acids = set(string.ascii_uppercase)

        for i, header_idx in enumerate(header_indices):
            start_index = header_idx
            end_index = header_indices[i + 1] if i + 1 < len(header_indices) else len(lines)
            entry_lines = lines[start_index:end_index]

            header = entry_lines[0]
            sequence_lines = entry_lines[1:]

            modified_lines.append(header)

            # First entry (query) is preserved unchanged
            if i == 0:
                modified_lines.extend(sequence_lines)
                continue

            for seq_line in sequence_lines:
                masked_line = [
                    'X' if char in amino_acids and random.random() < mask_fraction else char
                    for char in seq_line
                ]
                modified_lines.append("".join(masked_line))

        output_path.write_text("\n".join(modified_lines) + "\n")

    except IOError as e:
        raise IOError(f"Error processing files: {e}") from e
