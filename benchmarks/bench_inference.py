#!/usr/bin/env python3
"""Benchmark ProstT5 inference across precision modes.

Usage:
    python benchmarks/bench_inference.py --fasta proteins.fasta
    python benchmarks/bench_inference.py --fasta proteins.fasta --precisions bf16,fp16 --runs 3
    python benchmarks/bench_inference.py --fasta proteins.fasta --precisions bf16,fp16,int8,int4 --output results.csv
"""
from __future__ import annotations

import argparse
import csv
import gc
import statistics
import sys
import time
from pathlib import Path
from typing import List

# Ensure src/ is on the path when run from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _peak_vram_gb() -> float:
    """Return peak allocated VRAM in GB since last reset. Returns 0.0 on CPU."""
    try:
        import torch
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    except Exception:
        return 0.0


def _reset_vram_stats() -> None:
    try:
        import torch
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _free_gpu() -> None:
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def _run_one(fasta_path: str, precision: str, config: dict, device) -> dict:
    """Run full generate -> filter -> neff for one precision. Returns metrics dict."""
    from ghostfold.core.pipeline import _load_model
    from ghostfold.msa.generation import generate_sequences_for_coverages_batched
    from ghostfold.msa.filters import filter_sequences
    from ghostfold.msa.neff import calculate_neff
    from ghostfold.io.fasta import read_fasta_from_path
    from ghostfold.core.pipeline import generate_decoding_configs

    records = read_fasta_from_path(fasta_path)
    if not records:
        raise ValueError(f"No sequences found in {fasta_path}")

    record = records[0]
    query_seq = str(record.seq)
    full_len = len(query_seq)

    # Load model
    _reset_vram_stats()
    t0 = time.perf_counter()
    tokenizer, model = _load_model(device, precision=precision)
    load_time = time.perf_counter() - t0
    load_vram = _peak_vram_gb()

    # Generate
    decoding_configs = generate_decoding_configs(config.get("decoding_params", {}))
    num_return_sequences = config.get("num_return_sequences", 5)
    multiplier = config.get("multiplier", 1)
    inference_batch_size = config.get("inference_batch_size", 4)
    coverage_list = [1.0]

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        _reset_vram_stats()
        t1 = time.perf_counter()
        raw_seqs = generate_sequences_for_coverages_batched(
            query_seq=query_seq,
            full_len=full_len,
            decoding_configs=decoding_configs,
            num_return_sequences=num_return_sequences,
            multiplier=multiplier,
            coverage_list=coverage_list,
            model=model,
            tokenizer=tokenizer,
            device=device,
            project_dir=tmpdir,
            inference_batch_size=inference_batch_size,
        )
        gen_time = time.perf_counter() - t1
        gen_vram = _peak_vram_gb()

    # Filter
    all_seqs = [query_seq] + raw_seqs
    filtered = filter_sequences(all_seqs, full_len)
    raw_count = len(all_seqs)
    filtered_count = len(filtered)
    filter_rate = filtered_count / raw_count if raw_count > 0 else 0.0

    # Neff (primary quality metric)
    neff = calculate_neff(filtered) if filtered else 0.0

    return {
        "load_time_s": round(load_time, 3),
        "gen_time_s": round(gen_time, 3),
        "peak_vram_gb": round(max(load_vram, gen_vram), 3),
        "raw_seqs": raw_count,
        "filtered_seqs": filtered_count,
        "filter_rate": round(filter_rate, 3),
        "neff": round(neff, 4),
    }


def _print_table(rows: list[dict], precisions: list[str]) -> None:
    """Print Rich summary table to terminal."""
    try:
        from rich.table import Table
        from rich.console import Console
        from rich import box

        console = Console()
        table = Table(title="Inference Benchmark Results", box=box.SIMPLE_HEAD)
        cols = ["precision", "load_time_s", "gen_time_s", "peak_vram_gb",
                "raw_seqs", "filtered_seqs", "filter_rate", "neff"]
        for col in cols:
            table.add_column(col, justify="right" if col != "precision" else "left")

        # Find best (min) for perf cols, best (max) for quality cols
        best = {
            "load_time_s": min(r["load_time_s"] for r in rows),
            "gen_time_s": min(r["gen_time_s"] for r in rows),
            "peak_vram_gb": min(r["peak_vram_gb"] for r in rows),
            "neff": max(r["neff"] for r in rows),
        }

        for row in rows:
            cells = []
            for col in cols:
                val = row.get(col, "N/A")
                cell = str(val)
                if col in best and val == best[col]:
                    cell = f"[bold green]{cell}[/bold green]"
                elif col == "neff":
                    cell = f"[cyan]{cell}[/cyan]"
                cells.append(cell)
            table.add_row(*cells)

        console.print(table)
    except ImportError:
        header = "\t".join(["precision", "load_s", "gen_s", "vram_gb",
                             "raw", "filtered", "filter_rate", "neff"])
        print(header)
        for row in rows:
            print("\t".join(str(row.get(c, "")) for c in
                            ["precision", "load_time_s", "gen_time_s", "peak_vram_gb",
                             "raw_seqs", "filtered_seqs", "filter_rate", "neff"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark GhostFold T5 inference across precisions.")
    parser.add_argument("--fasta", required=True, help="Path to FASTA file.")
    parser.add_argument(
        "--precisions", default="bf16,fp16",
        help="Comma-separated precision modes to benchmark. Default: bf16,fp16. "
             "Add int8,int4 after: pip install -e '.[quant]'"
    )
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of timed runs per precision (first discarded as warmup). Default: 3")
    parser.add_argument("--output", default=None,
                        help="Path to write CSV results. Default: bench_results.csv")
    args = parser.parse_args()

    precisions = [p.strip() for p in args.precisions.split(",")]
    output_path = args.output or "bench_results.csv"

    import torch
    from ghostfold.core.config import load_config
    from ghostfold.core.pipeline import _MODEL_CACHE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(None)  # load defaults

    summary_rows: List[dict] = []

    for precision in precisions:
        print(f"\n[{precision}] Starting benchmark ({args.runs} runs, first discarded)...")

        run_results = []
        for run_idx in range(args.runs):
            _MODEL_CACHE.clear()
            _free_gpu()
            gc.collect()

            try:
                metrics = _run_one(args.fasta, precision, config, device)
                if run_idx == 0:
                    print("  [warmup run discarded]")
                else:
                    run_results.append(metrics)
                    print(f"  run {run_idx}: gen={metrics['gen_time_s']}s  "
                          f"vram={metrics['peak_vram_gb']}GB  neff={metrics['neff']}")
            except ImportError as e:
                print(f"  SKIP {precision}: {e}")
                break
            except Exception as e:
                print(f"  ERROR on run {run_idx} for {precision}: {e}")
                break

        if not run_results:
            continue

        avg: dict = {"precision": precision}
        for key in ["load_time_s", "gen_time_s", "peak_vram_gb",
                    "raw_seqs", "filtered_seqs", "filter_rate", "neff"]:
            vals = [r[key] for r in run_results if key in r]
            avg[key] = round(statistics.mean(vals), 4) if vals else "N/A"

        summary_rows.append(avg)

    if not summary_rows:
        print("No results collected. Check errors above.")
        sys.exit(1)

    fieldnames = ["precision", "load_time_s", "gen_time_s", "peak_vram_gb",
                  "raw_seqs", "filtered_seqs", "filter_rate", "neff"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nResults written to: {output_path}")

    _print_table(summary_rows, precisions)


if __name__ == "__main__":
    main()
