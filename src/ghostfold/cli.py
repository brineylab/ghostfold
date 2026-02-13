"""Typer CLI entrypoint for GhostFold."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer

from ghostfold import __version__
from ghostfold.config import (
    DEFAULT_MUTATION_RATES,
    MSAWorkflowConfig,
    MaskWorkflowConfig,
    NeffWorkflowConfig,
    PipelineWorkflowConfig,
)
from ghostfold.errors import GhostfoldError
from ghostfold.services import (
    run_mask_workflow,
    run_msa_workflow,
    run_neff_workflow,
    run_pipeline_workflow,
)

app = typer.Typer(
    add_completion=False,
    help="GhostFold command line interface.",
    no_args_is_help=True,
)


def _exit_on_error(exc: GhostfoldError) -> None:
    typer.secho(f"ERROR: {exc}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def _run_pipeline_command(config: PipelineWorkflowConfig) -> None:
    try:
        run_pipeline_workflow(config)
    except GhostfoldError as exc:
        _exit_on_error(exc)


def _run_msa_command(config: MSAWorkflowConfig) -> None:
    try:
        run_msa_workflow(config)
    except GhostfoldError as exc:
        _exit_on_error(exc)


def _run_mask_command(config: MaskWorkflowConfig) -> None:
    try:
        run_mask_workflow(config)
    except GhostfoldError as exc:
        _exit_on_error(exc)


def _run_neff_command(config: NeffWorkflowConfig) -> None:
    try:
        run_neff_workflow(config)
    except GhostfoldError as exc:
        _exit_on_error(exc)


@app.command("version")
def version_command() -> None:
    """Print the installed GhostFold version."""
    typer.echo(__version__)


@app.command("run")
def run_command(
    project_name: str = typer.Option(..., "--project-name", "--project_name", help="Main project directory name."),
    fasta_file: Optional[Path] = typer.Option(
        None,
        "--fasta-file",
        "--fasta_file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Input FASTA file (required except in --fold-only mode).",
    ),
    msa_only: bool = typer.Option(False, "--msa-only", help="Run only MSA generation."),
    fold_only: bool = typer.Option(False, "--fold-only", help="Run only fold stage from existing MSAs."),
    subsample: bool = typer.Option(False, "--subsample", help="Enable MSA subsampling mode for folding."),
    mask_msa: Optional[str] = typer.Option(
        None,
        "--mask-msa",
        "--mask_msa",
        help="Mask a fraction of MSA residues (examples: 0, 0.15, 1.0).",
    ),
) -> None:
    """Run full/msa/fold project workflow with shell-compatible mode flags."""
    _run_pipeline_command(
        PipelineWorkflowConfig(
            project_name=project_name,
            fasta_file=fasta_file,
            msa_only=msa_only,
            fold_only=fold_only,
            subsample=subsample,
            mask_msa=mask_msa,
        )
    )


@app.command("full")
def full_command(
    project_name: str = typer.Option(..., "--project-name", "--project_name", help="Main project directory name."),
    fasta_file: Path = typer.Option(
        ...,
        "--fasta-file",
        "--fasta_file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Input FASTA file.",
    ),
    subsample: bool = typer.Option(False, "--subsample", help="Enable MSA subsampling mode for folding."),
    mask_msa: Optional[str] = typer.Option(
        None,
        "--mask-msa",
        "--mask_msa",
        help="Mask a fraction of MSA residues (examples: 0, 0.15, 1.0).",
    ),
) -> None:
    """Run full pipeline (MSA generation + folding)."""
    _run_pipeline_command(
        PipelineWorkflowConfig(
            project_name=project_name,
            fasta_file=fasta_file,
            subsample=subsample,
            mask_msa=mask_msa,
        )
    )


@app.command("fold")
def fold_command(
    project_name: str = typer.Option(..., "--project-name", "--project_name", help="Main project directory name."),
    subsample: bool = typer.Option(False, "--subsample", help="Enable MSA subsampling mode for folding."),
    mask_msa: Optional[str] = typer.Option(
        None,
        "--mask-msa",
        "--mask_msa",
        help="Mask a fraction of MSA residues (examples: 0, 0.15, 1.0).",
    ),
) -> None:
    """Run fold-only mode on an existing project MSA directory."""
    _run_pipeline_command(
        PipelineWorkflowConfig(
            project_name=project_name,
            fold_only=True,
            subsample=subsample,
            mask_msa=mask_msa,
        )
    )


@app.command("msa")
def msa_command(
    project_name: str = typer.Option(..., "--project-name", help="Main project directory name."),
    fasta_file: Path = typer.Option(..., "--fasta-file", exists=True, file_okay=True, dir_okay=False, help="Input FASTA file."),
    config: Path = typer.Option(Path("config.yaml"), "--config", exists=True, file_okay=True, dir_okay=False, help="YAML config file."),
    coverage: Optional[List[float]] = typer.Option(None, "--coverage", help="Coverage values (repeat flag to pass multiple)."),
    num_runs: int = typer.Option(1, "--num-runs", min=1, help="Independent runs per input sequence."),
    plot_msa_coverage: bool = typer.Option(False, "--plot-msa-coverage", help="Generate MSA coverage plots."),
    no_coevolution_maps: bool = typer.Option(False, "--no-coevolution-maps", help="Disable coevolution maps."),
    evolve_msa: bool = typer.Option(False, "--evolve-msa", help="Enable MSA evolution."),
    mutation_rates: str = typer.Option(DEFAULT_MUTATION_RATES, "--mutation-rates", help="JSON mutation rate map."),
    sample_percentage: float = typer.Option(1.0, "--sample-percentage", min=0.0, max=1.0, help="Fraction of filtered sequences to mutate."),
) -> None:
    """Run the pseudoMSA generation workflow service directly."""
    _run_msa_command(
        MSAWorkflowConfig(
            project_name=project_name,
            fasta_file=fasta_file,
            config_path=config,
            coverage=coverage,
            num_runs=num_runs,
            plot_msa_coverage=plot_msa_coverage,
            no_coevolution_maps=no_coevolution_maps,
            evolve_msa=evolve_msa,
            mutation_rates=mutation_rates,
            sample_percentage=sample_percentage,
        )
    )


@app.command("mask")
def mask_command(
    input_path: Path = typer.Option(
        ...,
        "--input-path",
        "--input_path",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Input A3M file.",
    ),
    output_path: Path = typer.Option(..., "--output-path", "--output_path", help="Output masked A3M file."),
    mask_fraction: float = typer.Option(
        ...,
        "--mask-fraction",
        "--mask_fraction",
        min=0.0,
        max=1.0,
        help="Fraction of residues to mask.",
    ),
) -> None:
    """Mask an A3M file while preserving the first sequence."""
    _run_mask_command(
        MaskWorkflowConfig(
            input_path=input_path,
            output_path=output_path,
            mask_fraction=mask_fraction,
        )
    )


@app.command("neff")
def neff_command(
    project_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, help="Project directory containing msa/"),
) -> None:
    """Calculate Neff/NAD metrics for project MSAs."""
    _run_neff_command(NeffWorkflowConfig(project_dir=project_dir))


@app.command("mask_msa")
def mask_msa_command(
    input_path: Path = typer.Option(
        ...,
        "--input_path",
        "--input-path",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Path to the source .a3m file.",
    ),
    output_path: Path = typer.Option(
        ...,
        "--output_path",
        "--output-path",
        help="Path to write the masked .a3m file.",
    ),
    mask_fraction: float = typer.Option(
        ...,
        "--mask_fraction",
        "--mask-fraction",
        min=0.0,
        max=1.0,
        help="Fraction of residues to mask.",
    ),
) -> None:
    """Compatibility alias for the legacy `mask_msa.py` script."""
    _run_mask_command(
        MaskWorkflowConfig(
            input_path=input_path,
            output_path=output_path,
            mask_fraction=mask_fraction,
        )
    )


@app.command("calculate_neff")
def calculate_neff_command(
    project_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, help="Project directory containing msa/"),
) -> None:
    """Compatibility alias for the legacy `calculate_neff.py` script."""
    _run_neff_command(NeffWorkflowConfig(project_dir=project_dir))


if __name__ == "__main__":
    app()
