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
    force: bool = typer.Option(
        False,
        "--force",
        help="Force reinstall of ColabFold env and libraries. AF2 and ProstT5 weights are reused if already present.",
    ),
) -> None:
    """Install ColabFold, download AF2 weights, and cache ProstT5."""
    console = get_console()
    console.print(f"[bold]GhostFold Setup[/bold] — installing to [cyan]{colabfold_dir.resolve()}[/cyan]")
    if skip_weights:
        console.print("[dim]  --skip-weights: AF2 weight download skipped[/dim]")
    if force:
        console.print("[dim]  --force: ColabFold env will be removed and reinstalled (weights reused)[/dim]")

    try:
        mamba_fresh = run_setup(
            colabfold_dir=colabfold_dir,
            skip_weights=skip_weights,
            hf_token=hf_token,
            force=force,
        )
    except GhostFoldSetupError as exc:
        typer.secho(f"\nSetup failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    console.print("\n[green]Setup complete.[/green] Run: [bold]ghostfold run --help[/bold]")
    console.print("[dim]  Tip: use [bold]--force[/bold] to reinstall libraries if something breaks (weights reused).[/dim]")
    console.print("[dim]  Tip: use [bold]--skip-weights[/bold] to skip AF2 weight download.[/dim]")
    if mamba_fresh:
        console.print("[dim]  micromamba installed to ~/micromamba/bin — add it to PATH or open a new terminal.[/dim]")
