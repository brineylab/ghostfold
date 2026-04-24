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

_ALL_STRATEGIES = [
    "baseline",
    "encoder_only_3di_sub",
    "temperature_sweep",
    "embedding_walk_full",
    "embedding_walk_encoder",
    "encoder_perturb",
    "round_trip",
    "3di_perturb",
    "single_sequence",
    "cnn_3di_predict",
    "cnn_aa_predict",
]

_DEFAULT_CONFIGS: dict[str, dict] = {
    "baseline": {
        "num_return_sequences": 100,
        "inference_batch_size": 8,
        "decode_conf": {"temperature": 0.7, "top_k": 20, "top_p": 0.95},
        "coverage_values": [1.0],
    },
    "encoder_only_3di_sub": {
        "mutation_rates": [0.05, 0.10, 0.20, 0.30, 0.40],
        "variants_per_rate": 10,
        "num_return_sequences": 3,
        "decode_conf": {"temperature": 0.7, "top_k": 20, "top_p": 0.95},
    },
    "temperature_sweep": {
        "temperatures": [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0],
        "num_return_sequences": 20,
        "base_decode_conf": {"top_p": 0.85, "top_k": 3, "repetition_penalty": 1.2},
    },
    "embedding_walk_full": {
        "noise_scales": [0.03, 0.07, 0.12, 0.20, 0.35],
        "depth_decay": 0.8,
        "num_return_sequences": 25,
        "decode_conf": {"temperature": 0.7, "top_k": 20, "top_p": 0.95},
    },
    "embedding_walk_encoder": {
        "n_pca_components": 10,
        "step_sizes": [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0],
        "n_noise_samples": 50,
        "noise_sigma": 0.1,
        "num_return_sequences": 15,
        "decode_conf": {"temperature": 0.7, "top_k": 20, "top_p": 0.95},
    },
    # 8 noise scales × 17 seqs = 136 raw; subsampled to target_n after length filter
    "encoder_perturb": {
        "noise_scales": [0.03, 0.07, 0.12, 0.18, 0.25, 0.35, 0.50, 0.70],
        "num_return_sequences": 17,
        "decode_conf": {"temperature": 0.7, "top_k": 20, "top_p": 0.95},
    },
    # 16 seeds × 9 rounds = 144 raw; subsampled to target_n
    "round_trip": {
        "n_seeds": 16,
        "n_rounds": 9,
        "decode_conf": {"temperature": 0.7, "top_k": 20, "top_p": 0.95},
    },
    # 4 seeds × 6 rates × 6 seqs = 144 raw; subsampled to target_n
    "3di_perturb": {
        "mutation_rates": [0.05, 0.10, 0.15, 0.20, 0.30, 0.40],
        "n_3di_seeds": 4,
        "num_return_sequences": 6,
        "decode_conf": {"temperature": 0.7, "top_k": 20, "top_p": 0.95},
    },
    # single_sequence: no generation at all — just the query in the A3M
    "single_sequence": {},
    # 5 rates × 4 perturbations × 7 seqs = 140 raw; no separate encoder model needed
    "cnn_3di_predict": {
        "mutation_rates": [0.05, 0.10, 0.20, 0.30, 0.40],
        "n_perturbations": 4,
        "num_return_sequences": 7,
        "decode_conf": {"temperature": 0.7, "top_k": 20, "top_p": 0.95},
    },
    # 128 3Di samples → 128 CNN-predicted AA sequences
    "cnn_aa_predict": {
        "n_3di_seeds": 128,
        "decode_conf": {"temperature": 1.0, "top_k": 20, "top_p": 0.95},
    },
}


@app.command()
def main(
    bench_dir: Annotated[
        Path,
        typer.Option(
            "--bench-dir",
            help="Benchmark directory with fasta/<id>.fasta and pdb/<id>.pdb subdirs.",
            exists=True,
        ),
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
    target_n_sequences: Annotated[
        int,
        typer.Option(
            "--target-n",
            help="Subsample each strategy's output to this many sequences for fair comparison.",
        ),
    ] = 128,
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

    # Load optional models; each is kept on CPU until the runner moves it to GPU
    # so that strategies which don't need a model don't pay the VRAM cost.
    encoder_model = None
    cnn_3di = None
    cnn_aa = None

    needs_encoder = any(
        s in selected_strategies for s in ("encoder_only_3di_sub", "embedding_walk_encoder")
    )
    if needs_encoder:
        import torch as _torch
        from transformers import T5EncoderModel
        typer.echo("Loading T5EncoderModel for encoder-only strategies (bf16)...")
        encoder_model = T5EncoderModel.from_pretrained(
            "Rostlab/ProstT5_fp16",
            torch_dtype=_torch.bfloat16,
            cache_dir=None,
        ).cpu().eval()  # start on CPU; runner moves to GPU when needed

    needs_cnn_3di = any(
        s in selected_strategies for s in ("encoder_only_3di_sub", "cnn_3di_predict")
    )
    if needs_cnn_3di:
        from ghostfold.msa.strategies.encoder_only_3di_sub import load_cnn_3di
        typer.echo("Loading CNN 3Di head (downloading if needed)...")
        cnn_3di = load_cnn_3di(dev)

    needs_cnn_aa = "cnn_aa_predict" in selected_strategies
    if needs_cnn_aa:
        from ghostfold.msa.strategies.encoder_only_3di_sub import load_cnn_aa
        typer.echo("Loading CNN AA head (downloading if needed)...")
        cnn_aa = load_cnn_aa(dev)

    typer.echo(
        f"Running {len(selected_strategies)} strategies "
        f"x {'all' if protein_ids is None else len(protein_ids)} proteins"
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
        target_n_sequences=target_n_sequences,
        encoder_model=encoder_model,
        cnn_3di=cnn_3di,
        cnn_aa=cnn_aa,
    )

    csv_path = out_dir / "results.csv"
    typer.echo(f"\nDone. {len(results)} rows written to {csv_path}")




if __name__ == "__main__":
    app()
