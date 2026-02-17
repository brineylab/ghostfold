from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import typer

app = typer.Typer(help="Generate pseudoMSAs from single sequences.")


@app.callback(invoke_without_command=True)
def msa(
    project_name: str = typer.Option(..., "--project-name", help="Name of the main project directory."),
    fasta_path: Path = typer.Option(..., "--fasta-path", exists=True, help="Path to a FASTA file or directory containing FASTA files."),
    recursive: bool = typer.Option(False, "--recursive", help="Recursively search directories for FASTA files."),
    config: Optional[Path] = typer.Option(None, "--config", exists=True, help="Path to YAML config (overrides defaults)."),
    coverage: Optional[List[float]] = typer.Option(None, "--coverage", help="Coverage values (can be specified multiple times). Default: 1.0"),
    num_runs: int = typer.Option(1, "--num-runs", help="Number of independent runs per query sequence."),
    plot_msa_coverage: bool = typer.Option(False, "--plot-msa-coverage", help="Generate MSA coverage plots."),
    no_coevolution_maps: bool = typer.Option(False, "--no-coevolution-maps", help="Do not generate coevolution maps."),
    evolve_msa: bool = typer.Option(False, "--evolve-msa", help="Enable MSA evolution using the mutator module."),
    mutation_rates: str = typer.Option(
        '{"MEGABLAST": 5, "PAM250": 20, "BLOSUM62": 10}',
        "--mutation-rates",
        help="JSON string for mutation rates.",
    ),
    sample_percentage: float = typer.Option(1.0, "--sample-percentage", help="Percentage of sequences to sample for evolution."),
) -> None:
    """Generate pseudoMSAs from single sequences using ProstT5."""
    from ghostfold.core.logging import setup_logging, get_console
    from ghostfold.core.config import load_config
    from ghostfold.core.pipeline import run_pipeline

    log_path = setup_logging(project_name)
    get_console().print(f"[dim]Log file: {log_path}[/dim]")

    cfg = load_config(config)
    coverage_list = list(coverage) if coverage else [1.0]

    run_pipeline(
        project=project_name,
        fasta_path=str(fasta_path),
        config=cfg,
        coverage_list=coverage_list,
        evolve_msa=evolve_msa,
        mutation_rates_str=mutation_rates,
        sample_percentage=sample_percentage,
        plot_msa=plot_msa_coverage,
        plot_coevolution=not no_coevolution_maps,
        num_runs=num_runs,
        recursive=recursive,
    )
