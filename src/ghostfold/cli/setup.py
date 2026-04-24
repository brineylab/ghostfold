from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from ghostfold.core.setup import GhostFoldSetupError, run_setup
from ghostfold.core.logging import get_console

app = typer.Typer(help="Install ColabFold, download AF2 weights, and cache ProstT5.")


@app.callback(invoke_without_command=True)
def setup(
    colabfold_dir: Path = typer.Option(
        Path("localcolabfold"),
        "--colabfold-dir",
        help="Directory to install ColabFold into (default: ./localcolabfold).",
    ),
    skip_weights: bool = typer.Option(
        False,
        "--skip-weights",
        help="Skip AlphaFold2 weight download.",
    ),
    hf_token: Optional[str] = typer.Option(
        None,
        "--hf-token",
        help="HuggingFace access token for ProstT5 (alternative to huggingface-cli login).",
        envvar="HF_TOKEN",
    ),
) -> None:
    """Bootstrap pixi, install ColabFold, download AF2 weights, and cache ProstT5."""
    console = get_console()
    console.print(f"[bold]GhostFold Setup[/bold] — installing to [cyan]{colabfold_dir.resolve()}[/cyan]")
    if skip_weights:
        console.print("[dim]  --skip-weights: AF2 weight download skipped[/dim]")

    try:
        run_setup(
            colabfold_dir=colabfold_dir,
            skip_weights=skip_weights,
            hf_token=hf_token,
        )
    except GhostFoldSetupError as exc:
        typer.secho(f"\nSetup failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    console.print("\n[green]Setup complete.[/green] Run: [bold]ghostfold run --help[/bold]")
    console.print("[dim]  Add ~/.pixi/bin to ~/.bashrc if pixi was freshly installed.[/dim]")
