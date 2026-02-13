from __future__ import annotations

from typing import Optional

import typer

app = typer.Typer(help="Run ColabFold structure prediction on existing MSAs.")


@app.callback(invoke_without_command=True)
def fold(
    project_name: str = typer.Option(..., "--project-name", help="Name of the project directory containing MSAs."),
    subsample: bool = typer.Option(False, "--subsample", help="Enable MSA subsampling mode."),
    mask_fraction: Optional[float] = typer.Option(None, "--mask-fraction", min=0.0, max=1.0, help="Fraction of MSA residues to mask (0.0-1.0)."),
    num_gpus: Optional[int] = typer.Option(None, "--num-gpus", min=1, help="Override GPU count (auto-detected if not set)."),
) -> None:
    """Run ColabFold on existing MSAs for structure prediction."""
    from ghostfold.core.gpu import detect_gpus
    from ghostfold.core.colabfold import run_colabfold

    gpus = num_gpus if num_gpus is not None else detect_gpus()
    run_colabfold(
        project_name=project_name,
        num_gpus=gpus,
        subsample=subsample,
        mask_fraction=mask_fraction,
    )
