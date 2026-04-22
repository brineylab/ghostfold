#!/usr/bin/env python
"""Hypothesis-driven pseudoMSA benchmarking study.

Runs every selected generation strategy against a set of proteins with
known reference structures, then writes a results.csv with:

    protein_id, strategy, variant_param, n_sequences, neff,
    gen_time_s, peak_vram_gb, ptm, mean_plddt, rmsd, tm_score

Usage::

    python benchmarks/bench_pseudomsa.py \\
        --bench-dir bench/ \\
        --out-dir bench_results/ \\
        --strategies all \\
        --device cuda \\
        --precision bf16

    # With structure prediction (requires local ColabFold):
    python benchmarks/bench_pseudomsa.py ... --fold --colabfold-gpus 1

Bench directory layout::

    bench/
      queries.fasta      # 15 sequences; FASTA headers = protein IDs
      1ABC.pdb           # reference ground-truth structures
      2XYZ.pdb
      ...
"""
from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)

_ALL_STRATEGIES = ["encoder_perturb", "diverse_beam", "round_trip", "3di_perturb"]

_DEFAULT_CONFIGS: dict[str, dict] = {
    "encoder_perturb": {
        "noise_scales": [0.05, 0.15, 0.35],
        "num_return_sequences": 5,
        "decode_conf": {"temperature": 0.7, "top_k": 20, "top_p": 0.95},
    },
    "diverse_beam": {
        "num_beams": 8,
        "diversity_penalty": 1.0,
        "decode_conf": {"temperature": 0.7, "top_k": 20, "top_p": 0.95},
    },
    "round_trip": {
        "n_seeds": 8,
        "n_rounds": 4,
        "decode_conf": {"temperature": 0.7, "top_k": 20, "top_p": 0.95},
    },
    "3di_perturb": {
        "mutation_rates": [0.05, 0.15, 0.25],
        "n_3di_seeds": 3,
        "num_return_sequences": 5,
        "decode_conf": {"temperature": 0.7, "top_k": 20, "top_p": 0.95},
    },
}


@app.command()
def main(
    bench_dir: Annotated[
        Path,
        typer.Option("--bench-dir", help="Directory with queries.fasta and reference PDBs.", exists=True),
    ],
    out_dir: Annotated[
        Path,
        typer.Option("--out-dir", help="Output directory for A3Ms, ColabFold runs, and results.csv."),
    ],
    strategies: Annotated[
        str,
        typer.Option(
            "--strategies",
            help=(
                "Comma-separated strategy names or 'all'. "
                f"Available: {', '.join(_ALL_STRATEGIES)}"
            ),
        ),
    ] = "all",
    device: Annotated[str, typer.Option("--device", help="Torch device (cuda, cpu).")] = "cuda",
    precision: Annotated[
        str, typer.Option("--precision", help="Model precision: bf16, fp16, int8, int4.")
    ] = "bf16",
    fold: Annotated[
        bool,
        typer.Option("--fold/--no-fold", help="Run ColabFold on generated MSAs."),
    ] = False,
    colabfold_gpus: Annotated[
        int, typer.Option("--colabfold-gpus", help="Number of GPUs for ColabFold.")
    ] = 1,
    proteins: Annotated[
        Optional[str],
        typer.Option(
            "--proteins",
            help="Comma-separated protein IDs to run (default: all in queries.fasta).",
        ),
    ] = None,
) -> None:
    """Run the pseudoMSA generation benchmarking study."""
    import torch

    from ghostfold.benchmark.runner import run_benchmark
    from ghostfold.core.pipeline import _load_model

    selected_strategies = (
        _ALL_STRATEGIES if strategies.strip().lower() == "all"
        else [s.strip() for s in strategies.split(",")]
    )

    invalid = [s for s in selected_strategies if s not in _ALL_STRATEGIES]
    if invalid:
        typer.echo(f"Unknown strategies: {invalid}. Available: {_ALL_STRATEGIES}", err=True)
        raise typer.Exit(1)

    protein_ids = (
        [p.strip() for p in proteins.split(",")]
        if proteins else None
    )

    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    typer.echo(f"Loading ProstT5 ({precision}) on {dev}...")
    tokenizer, model = _load_model(dev, precision=precision)
    model.eval()

    typer.echo(
        f"Running {len(selected_strategies)} strategies "
        f"× {'all proteins' if protein_ids is None else len(protein_ids)} proteins"
    )

    configs = {name: _DEFAULT_CONFIGS[name] for name in selected_strategies}

    results = run_benchmark(
        bench_dir=bench_dir,
        out_dir=out_dir,
        strategy_names=selected_strategies,
        strategy_configs=configs,
        model=model,
        tokenizer=tokenizer,
        device=dev,
        protein_ids=protein_ids,
        run_colabfold=fold,
        colabfold_gpus=colabfold_gpus,
    )

    csv_path = out_dir / "results.csv"
    typer.echo(f"\nDone. {len(results)} rows written to {csv_path}")
    _print_summary(results)


def _print_summary(results: list[dict]) -> None:
    """Print a compact per-strategy summary to stdout."""
    from collections import defaultdict

    by_strategy: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        by_strategy[row["strategy"]].append(row)

    header = f"{'Strategy':<20} {'N':>6} {'Neff':>8} {'Time(s)':>8} {'VRAM(GB)':>9}"
    typer.echo("\n" + header)
    typer.echo("-" * len(header))
    for strategy, rows in sorted(by_strategy.items()):
        n_seqs = sum(r["n_sequences"] for r in rows) / len(rows)
        neff = sum(r["neff"] for r in rows) / len(rows)
        t = sum(r["gen_time_s"] for r in rows) / len(rows)
        vram = sum(r["peak_vram_gb"] for r in rows) / len(rows)
        typer.echo(f"{strategy:<20} {n_seqs:>6.0f} {neff:>8.3f} {t:>8.1f} {vram:>9.2f}")


if __name__ == "__main__":
    app()
