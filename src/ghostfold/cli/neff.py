from pathlib import Path

import typer

app = typer.Typer(help="Calculate Neff scores for MSA files.")


@app.callback(invoke_without_command=True)
def neff(
    project_dir: Path = typer.Argument(..., exists=True, help="Path to the project directory containing MSA files."),
) -> None:
    """Calculate Neff scores for all A3M files in a project directory."""
    from ghostfold.msa.neff import run_neff_calculation_in_parallel

    run_neff_calculation_in_parallel(str(project_dir))
