from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="Run the full GhostFold pipeline: MSA generation + ColabFold prediction.")


@app.callback(invoke_without_command=True)
def run(
    project_name: str = typer.Option(..., "--project-name", help="Name of the main project directory."),
    fasta_file: Path = typer.Option(..., "--fasta-file", exists=True, help="Path to query FASTA file."),
    config: Optional[Path] = typer.Option(None, "--config", exists=True, help="Path to YAML config (overrides defaults)."),
    subsample: bool = typer.Option(False, "--subsample", help="Enable MSA subsampling mode for ColabFold."),
    mask_fraction: Optional[float] = typer.Option(None, "--mask-fraction", min=0.0, max=1.0, help="Fraction of MSA residues to mask (0.0-1.0)."),
    num_gpus: Optional[int] = typer.Option(None, "--num-gpus", min=1, help="Override GPU count (auto-detected if not set)."),
) -> None:
    """Run full pipeline: MSA generation then ColabFold structure prediction."""
    from ghostfold.core.gpu import detect_gpus, run_parallel_msa
    from ghostfold.core.colabfold import run_colabfold

    gpus = num_gpus if num_gpus is not None else detect_gpus()

    run_parallel_msa(
        project_name=project_name,
        fasta_file=str(fasta_file),
        num_gpus=gpus,
        config_path=str(config) if config else None,
    )

    run_colabfold(
        project_name=project_name,
        num_gpus=gpus,
        subsample=subsample,
        mask_fraction=mask_fraction,
    )
