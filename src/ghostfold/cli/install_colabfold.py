from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="Install and configure a local ColabFold environment.")


@app.callback(invoke_without_command=True)
def install_colabfold(
    localcolabfold_dir: Optional[Path] = typer.Option(
        None,
        "--localcolabfold-dir",
        help="Path to localcolabfold checkout (default: ./localcolabfold).",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Show full installer command output.",
    ),
) -> None:
    """Install and configure localcolabfold using pixi."""
    from ghostfold.core.colabfold_env import ColabFoldSetupError
    from ghostfold.core.colabfold_install import install_colabfold as install_colabfold_runtime

    try:
        target_dir = install_colabfold_runtime(
            localcolabfold_dir=localcolabfold_dir,
            verbose=verbose,
            progress_cb=typer.echo,
        )
    except ColabFoldSetupError as exc:
        typer.secho(f"Warning: {exc}", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(code=1)

    typer.secho("ColabFold installation completed successfully.", fg=typer.colors.GREEN)
    typer.echo(f"localcolabfold directory: {target_dir}")
