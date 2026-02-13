from pathlib import Path

import typer

app = typer.Typer(help="Mask residues in A3M/FASTA files.")


@app.callback(invoke_without_command=True)
def mask(
    input_path: Path = typer.Option(..., "--input-path", exists=True, help="Path to the source .a3m file."),
    output_path: Path = typer.Option(..., "--output-path", help="Path to write the masked .a3m file."),
    mask_fraction: float = typer.Option(..., "--mask-fraction", min=0.0, max=1.0, help="Fraction of residues to mask (0.0-1.0)."),
) -> None:
    """Create a masked copy of an A3M file, preserving the query sequence."""
    from ghostfold.msa.mask import mask_a3m_file

    mask_a3m_file(input_path, output_path, mask_fraction)
    typer.echo(f"Successfully created masked file '{output_path}'.")
