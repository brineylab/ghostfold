import typer

from ghostfold._version import __version__
from ghostfold.cli import msa, fold, run, mask, neff

app = typer.Typer(
    name="ghostfold",
    no_args_is_help=True,
    help="GhostFold: database-free protein folding from single sequences using structure-aware synthetic MSAs.",
)

app.add_typer(msa.app, name="msa")
app.add_typer(fold.app, name="fold")
app.add_typer(run.app, name="run")
app.add_typer(mask.app, name="mask")
app.add_typer(neff.app, name="neff")


def _version_callback(value: bool) -> None:
    if value:
        print(f"ghostfold {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show the version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """GhostFold: database-free protein folding with synthetic MSAs."""
