from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="Install and configure a local ColabFold environment.")


@app.callback(invoke_without_command=True)
def install_colabfold(
    colabfold_env: str = typer.Option(
        "colabfold",
        "--colabfold-env",
        help="Mamba environment name to create/use for ColabFold.",
    ),
    data_dir: Optional[Path] = typer.Option(
        None,
        "--data-dir",
        help="Directory for ColabFold data and model cache (default: ./localcolabfold).",
    ),
) -> None:
    """Install ColabFold and required dependencies into a dedicated mamba env."""
    from ghostfold.core.colabfold_env import ColabFoldSetupError
    from ghostfold.core.colabfold_install import install_colabfold as install_colabfold_runtime

    try:
        target_dir = install_colabfold_runtime(
            colabfold_env=colabfold_env,
            data_dir=data_dir,
        )
    except ColabFoldSetupError as exc:
        typer.secho(f"Warning: {exc}", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(code=1)

    typer.secho("ColabFold installation completed successfully.", fg=typer.colors.GREEN)
    typer.echo(f"Environment: {colabfold_env}")
    typer.echo(f"Data directory: {target_dir}")
