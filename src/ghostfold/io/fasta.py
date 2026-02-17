from __future__ import annotations

import os
from pathlib import Path
from typing import List, Union

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from ghostfold.core.logging import get_logger

logger = get_logger("fasta")

FASTA_EXTENSIONS = {".fasta", ".fa"}


def collect_fasta_paths(fasta_path: str | Path, recursive: bool = False) -> List[Path]:
    """Collect FASTA file paths from a file or directory.

    Args:
        fasta_path: Path to a FASTA file or directory containing FASTA files.
        recursive: If True, search directories recursively for FASTA files.

    Returns:
        Sorted, deduplicated list of FASTA file paths.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If a directory contains no matching FASTA files.
    """
    fasta_path = Path(fasta_path)
    if not fasta_path.exists():
        raise FileNotFoundError(f"Path not found: {fasta_path}")
    if fasta_path.is_file():
        if recursive:
            logger.warning(
                f"--recursive has no effect when the input path is a file: {fasta_path}"
            )
        return [fasta_path]
    # Directory
    paths: List[Path] = []
    for ext in sorted(FASTA_EXTENSIONS):
        pattern = f"**/*{ext}" if recursive else f"*{ext}"
        paths.extend(fasta_path.glob(pattern))
    paths = sorted(set(paths))
    if not paths:
        raise ValueError(
            f"No FASTA files ({', '.join(sorted(FASTA_EXTENSIONS))}) "
            f"found in directory: {fasta_path}"
        )
    return paths


def read_fasta_from_path(
    fasta_path: str | Path, recursive: bool = False
) -> List[SeqRecord]:
    """Read sequences from a FASTA file or directory of FASTA files.

    Args:
        fasta_path: Path to a FASTA file or directory containing FASTA files.
        recursive: If True, search directories recursively for FASTA files.

    Returns:
        List of SeqRecord objects in deterministic order (sorted by file path,
        then order within each file).
    """
    file_paths = collect_fasta_paths(fasta_path, recursive=recursive)
    records: List[SeqRecord] = []
    for fp in file_paths:
        records.extend(read_fasta(fp))
    return records


def read_fasta(file_path: str | Path) -> List[SeqRecord]:
    """Reads sequences from a FASTA file."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {file_path}")
    return list(SeqIO.parse(str(file_path), "fasta"))


def write_fasta(output_path: str | Path, records: List[SeqRecord]) -> None:
    """Writes a list of SeqRecord objects to a FASTA file."""
    with open(output_path, "w") as f:
        SeqIO.write(records, f, "fasta")


def append_fasta(
    sequences_to_append: List[Union[str, SeqRecord]], output_path: str | Path
) -> None:
    """Appends sequences to an existing FASTA file. Creates the file if it doesn't exist."""
    records_to_append = []
    for i, seq_item in enumerate(sequences_to_append):
        if isinstance(seq_item, str):
            records_to_append.append(
                SeqRecord(Seq(seq_item), id=f"appended_seq_{i}", description="")
            )
        elif isinstance(seq_item, SeqRecord):
            records_to_append.append(seq_item)
        else:
            logger.warning(
                f"Skipping unknown sequence type {type(seq_item)} during append."
            )

    mode = "a" if os.path.exists(output_path) else "w"
    with open(output_path, mode) as f:
        SeqIO.write(records_to_append, f, "fasta")


def create_project_dir(project_name: str, header: str) -> str:
    """Creates a directory for a sequence header inside a project's 'msa' folder.

    The created directory structure is: project_name/msa/<safe_header>
    """
    safe_header = "".join(c if c.isalnum() else "_" for c in header)
    project_dir = os.path.join(project_name, "msa", safe_header)
    os.makedirs(project_dir, exist_ok=True)
    return project_dir


def concatenate_fasta_files(
    file_paths: List[str],
    output_path: str,
) -> None:
    """Concatenates content of multiple FASTA files into a single output file.

    Args:
        file_paths: List of paths to input FASTA files.
        output_path: Path to the output concatenated FASTA file.
    """
    if not file_paths:
        logger.info("No files to concatenate. Skipping concatenation.")
        return

    records_to_write: List[SeqRecord] = []
    for fname in file_paths:
        if os.path.exists(fname) and os.path.getsize(fname) > 0:
            try:
                for record in SeqIO.parse(fname, "fasta"):
                    records_to_write.append(record)
            except Exception as e:
                logger.warning(f"Could not parse FASTA file '{fname}': {e}. Skipping.")
        else:
            logger.info(f"Skipping empty or non-existent file: {fname}")

    if records_to_write:
        write_fasta(output_path, records_to_write)
        logger.info(
            f"Successfully created {output_path} with {len(records_to_write)} records."
        )
    else:
        logger.info(f"No valid records found from input files to write to {output_path}.")
