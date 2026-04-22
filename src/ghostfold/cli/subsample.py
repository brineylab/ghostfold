from pathlib import Path
from typing import Annotated

import typer

from ghostfold.core.logging import get_logger
from ghostfold.msa.neff import calculate_neff, parse_a3m
from ghostfold.msa.ranking import rank_and_subsample

logger = get_logger("cli.subsample")

app = typer.Typer(
    name="subsample",
    no_args_is_help=True,
    help="Select a representative subset from a large MSA for AF2 evoformer input.",
)


@app.callback(invoke_without_command=True)
def subsample(
    a3m: Annotated[Path, typer.Option("--a3m", help="Input A3M file.", exists=True, readable=True)],
    n: Annotated[int, typer.Option("--n", help="Number of sequences to select (including query).")],
    output: Annotated[Path, typer.Option("--output", help="Output A3M path.")],
    strategy: Annotated[
        str,
        typer.Option(
            "--strategy",
            help=(
                "Subsampling strategy: farthest_first, max_coverage, "
                "column_entropy, neff_contribution, random."
            ),
        ),
    ] = "farthest_first",
    include_query: Annotated[
        bool,
        typer.Option(
            "--include-query/--no-include-query",
            help="Always keep the first sequence (query) in the output.",
        ),
    ] = True,
    report: Annotated[
        bool,
        typer.Option("--report", help="Print Neff before and after to stdout."),
    ] = False,
) -> None:
    """Subsample a large MSA to *n* representative sequences.

    The first sequence in the A3M is treated as the query and is always
    preserved as element 0 when --include-query is set (the default).

    Example::

        ghostfold subsample \\
            --a3m jackhmmer.a3m \\
            --n 512 \\
            --strategy farthest_first \\
            --output representative.a3m \\
            --report
    """
    sequences = parse_a3m(a3m)
    if not sequences:
        typer.echo(f"Error: no sequences found in {a3m}", err=True)
        raise typer.Exit(1)

    if report:
        neff_before = calculate_neff(sequences)
        typer.echo(f"Neff before: {neff_before:.3f}  ({len(sequences)} sequences)")

    query_seq = sequences[0] if include_query else None
    selected = rank_and_subsample(
        sequences=sequences,
        n_select=n,
        strategy=strategy,
        query=query_seq,
    )

    # Reconstruct A3M headers (parse_a3m strips them; use generic labels)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as fh:
        for i, seq in enumerate(selected):
            label = "query" if i == 0 and include_query else f"seq_{i}"
            fh.write(f">{label}\n{seq}\n")

    if report:
        neff_after = calculate_neff(selected)
        typer.echo(
            f"Neff after:  {neff_after:.3f}  ({len(selected)} sequences)  "
            f"[strategy={strategy}]"
        )

    logger.info(f"Wrote {len(selected)} sequences to {output}")
