# utils/io.py (Corrected version)
import os
from typing import List, Union

from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from Bio.Seq import Seq
from rich.console import Console

def read_fasta(file_path: str) -> List[SeqRecord]:
    """Reads sequences from a FASTA file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"FASTA file not found: {file_path}")
    return list(SeqIO.parse(file_path, "fasta"))


def write_fasta(output_path: str, records: List[SeqRecord]) -> None:
    """
    Writes a list of SeqRecord objects to a FASTA file.

    Args:
        output_path: The path to the output FASTA file.
        records: A list of Biopython SeqRecord objects.
    """
    with open(output_path, 'w') as f:
        SeqIO.write(records, f, "fasta")


def append_fasta(sequences_to_append: List[Union[str, SeqRecord]], output_path: str) -> None:
    """
    Appends sequences to an existing FASTA file.
    If the file does not exist, it creates it.

    Args:
        sequences_to_append: A list of sequence strings or SeqRecord objects to append.
        output_path: The path to the FASTA file.
    """
    # Convert strings to SeqRecord objects if necessary
    records_to_append = []
    for i, seq_item in enumerate(sequences_to_append):
        if isinstance(seq_item, str):
            # Generate a generic ID, or you might want to infer it or pass alongside
            records_to_append.append(SeqRecord(Seq(seq_item), id=f"appended_seq_{i}", description=""))
        elif isinstance(seq_item, SeqRecord):
            records_to_append.append(seq_item)
        else:
            print(f"Warning: Skipping unknown sequence type {type(seq_item)} during append.")

    mode = 'a' if os.path.exists(output_path) else 'w'
    with open(output_path, mode) as f:
        SeqIO.write(records_to_append, f, "fasta")


def create_project_dir(project_name: str, header: str) -> str:
    """
    Creates a directory for a sequence header inside a project's 'msa' folder.
    The created directory structure is: project_name/msa/<safe_header>
    """
    safe_header = "".join(c if c.isalnum() else "_" for c in header)
    project_dir = os.path.join(project_name, "msa", safe_header)
    os.makedirs(project_dir, exist_ok=True)
    return project_dir


def concatenate_fasta_files(
    file_paths: List[str], 
    output_path: str, 
    console: Console
) -> None:
    """
    Concatenates content of multiple FASTA files into a single output file.

    Args:
        file_paths: List of paths to input FASTA files.
        output_path: Path to the output concatenated FASTA file.
        console: Rich Console object for logging.
    """
    if not file_paths:
        console.print("[yellow]No files to concatenate. Skipping concatenation.[/yellow]")
        return

    records_to_write: List[SeqRecord] = []
    for fname in file_paths:
        if os.path.exists(fname) and os.path.getsize(fname) > 0:
            try:
                for record in SeqIO.parse(fname, "fasta"):
                    records_to_write.append(record)
            except Exception as e:
                console.print(f"[bold yellow]Warning:[/bold yellow] Could not parse FASTA file [red]'{fname}'[/red]: [red]{e}[/red]. Skipping this file.")
        else:
            console.print(f"[yellow]Skipping empty or non-existent file: [dim]{fname}[/dim][/yellow]")
    
    if records_to_write:
        # Assuming you have a 'write_fasta' function in this module
        write_fasta(output_path, records_to_write)
        console.print(f"[bold green]Successfully created [link={output_path}]{output_path}[/link] with {len(records_to_write)} records.[/bold green]")
    else:
        console.print(f"[bold yellow]No valid records found from input files to write to [red]{output_path}[/red].[/bold yellow]")
