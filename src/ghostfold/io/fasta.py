from __future__ import annotations

import os
from pathlib import Path
from typing import List, Union

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from ghostfold.core.logging import get_logger

logger = get_logger("fasta")


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
