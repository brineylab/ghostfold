from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="Run ColabFold structure prediction on existing MSAs.")


@app.callback(invoke_without_command=True)
def fold(
    project_name: str = typer.Option(
        ...,
        "--project-name",
        help="Name of the project directory containing MSAs.",
    ),
    subsample: bool = typer.Option(False, "--subsample", help="Enable MSA subsampling mode."),
    mask_fraction: Optional[float] = typer.Option(
        None,
        "--mask-fraction",
        min=0.0,
        max=1.0,
        help="Fraction of MSA residues to mask (0.0-1.0).",
    ),
    num_gpus: Optional[int] = typer.Option(
        None,
        "--num-gpus",
        min=1,
        help="Override GPU count (auto-detected if not set).",
    ),
    colabfold_env: str = typer.Option(
        "colabfold",
        "--colabfold-env",
        help="Legacy mamba environment name for ColabFold fallback.",
    ),
    localcolabfold_dir: Optional[Path] = typer.Option(
        None,
        "--localcolabfold-dir",
        help="Path to localcolabfold pixi checkout (default: ./localcolabfold).",
    ),
) -> None:
    """Run ColabFold on existing MSAs for structure prediction."""
    from ghostfold.core.logging import setup_logging, get_console
    from ghostfold.core.gpu import detect_gpus
    from ghostfold.core.colabfold import run_colabfold
    from ghostfold.core.colabfold_env import ColabFoldSetupError, ensure_colabfold_ready

    log_path = setup_logging(project_name)
    get_console().print(f"[dim]Log file: {log_path}[/dim]")

    gpus = num_gpus if num_gpus is not None else detect_gpus()
    try:
        ensure_colabfold_ready(
            colabfold_env=colabfold_env,
            localcolabfold_dir=localcolabfold_dir,
        )
    except ColabFoldSetupError as exc:
        typer.secho(f"Warning: {exc}", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(code=1)

    run_colabfold(
        project_name=project_name,
        num_gpus=gpus,
        subsample=subsample,
        mask_fraction=mask_fraction,
        colabfold_env=colabfold_env,
        localcolabfold_dir=localcolabfold_dir,
    )
