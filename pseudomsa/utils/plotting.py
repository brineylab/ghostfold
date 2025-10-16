# utils/plotting.py

import os
from typing import List
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from rich.console import Console

# Assuming these are your existing utility functions
from .msa_coverage import plot_msa_coverage
from .coevolution import get_coevolution_numpy, plot_coevolution

def generate_optional_plots(
    sequences: List[str],
    full_len: int,
    img_dir: str,
    base_name: str,
    custom_colors: List[str],
    plot_msa: bool,
    plot_coevolution: bool,
    console: Console
) -> None:
    """
    Generates MSA coverage and/or coevolution plots if requested.

    Args:
        sequences: List of sequence strings.
        full_len: The expected full length of sequences for coverage plot.
        img_dir: Directory to save plots.
        base_name: Base name for output plot files (e.g., 'unfiltered').
        custom_colors: List of hexadecimal colors for plots.
        plot_msa: If True, MSA coverage maps will be plotted.
        plot_coevolution: If True, coevolution maps will be plotted.
        console: Rich Console object for logging.
    """
    if not sequences:
        console.print(f"[yellow]No sequences provided for [bold]{base_name}[/bold] plots.[/yellow]")
        return

    # MSA coverage plot
    if plot_msa:
        sequences_for_coverage_plot = [seq for seq in sequences if len(seq) == full_len]
        if sequences_for_coverage_plot:
            msa_coverage_path = os.path.join(img_dir, f'msa_coverage_{base_name}.png')
            console.print(f"Generating MSA coverage plot for [bold]{base_name}[/bold] sequences...")
            plot_msa_coverage(sequences_for_coverage_plot, save_path=msa_coverage_path, custom_colors=custom_colors)
            console.print(f"MSA coverage plot saved to [link={msa_coverage_path}]{msa_coverage_path}[/link]")
        else:
            console.print(f"[yellow]No [bold]{base_name}[/bold] sequences of query length [cyan]{full_len}[/cyan] found for MSA coverage plot.[/yellow]")
    else:
        console.print(f"[italic gray]MSA coverage plot generation skipped for [bold]{base_name}[/bold].[/italic gray]")

    # Coevolution plot
    if plot_coevolution:
        seq_records = [SeqRecord(Seq(s), id=f"seq_{i}", description="") for i, s in enumerate(sequences)]
        if len(seq_records) > 1:  # Coevolution needs at least 2 sequences
            console.print(f"Generating coevolution map from [bold]{base_name}[/bold] MSA.")
            coevol_matrix = get_coevolution_numpy(seq_records)
            plot_path = os.path.join(img_dir, f'coevolution_{base_name}_msa.png')
            plot_coevolution(coevol_matrix, plot_path)
            console.print(f"Coevolution map saved to [link={plot_path}]{plot_path}[/link]")
        else:
            console.print(f"[yellow]Skipping coevolution plot for [bold]{base_name}[/bold] as fewer than 2 sequences are available.[/yellow]")
    else:
        console.print(f"[italic gray]Coevolution map generation skipped for [bold]{base_name}[/bold].[/italic gray]")
