#!/usr/bin/env python
"""Benchmark low-Neff pseudoMSA sampling with ProstT5."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated, Optional

import torch
import typer
from transformers import GenerationConfig, LogitsProcessorList
from transformers.modeling_outputs import BaseModelOutput

from ghostfold.benchmark.low_neff import (
    RESULT_FIELDS,
    SUMMARY_FIELDS,
    SamplingVariant,
    build_generation_config_kwargs,
    filter_exact_length,
    generate_embedding_variants,
    generate_temperature_variants,
    make_result_row,
    normalize_cache_implementations,
    select_first_valid,
    select_lowest_neff,
    summarize_best_rows,
    summary_rows,
    write_a3m,
    write_csv,
)
from ghostfold.benchmark.runner import _batch_fold_and_score, _discover_proteins, _find_ref_pdb
from ghostfold.core.pipeline import _load_model
from ghostfold.msa.model import FiniteLogitsProcessor, generate_3di, preprocess_sequence
from ghostfold.msa.neff import calculate_neff

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _peak_vram_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    return 0.0


def _reset_vram() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _safe_variant_dir(row: dict) -> str:
    text = f"{row['strategy']}__{row['variant_param']}__cache={row['cache_implementation']}"
    return "".join(ch if ch.isalnum() or ch in "._=-" else "_" for ch in text)


def _decode_aa_microbatch(
    fold_seqs: list[str],
    tokenizer,
    model,
    device: torch.device,
    num_return_sequences: int,
    decode_conf: dict,
    cache_implementation: str | None,
) -> list[str]:
    inputs = [preprocess_sequence(list(seq), "<fold2AA>") for seq in fold_seqs]
    ids = tokenizer(inputs, add_special_tokens=True, padding="longest", return_tensors="pt").to(device)
    max_len = ids.input_ids.shape[1] + 1
    kwargs = build_generation_config_kwargs(decode_conf, cache_implementation)
    gen_cfg = GenerationConfig(
        max_length=max_len,
        num_return_sequences=num_return_sequences,
        num_beams=1,
        do_sample=True,
        **kwargs,
    )
    with torch.no_grad():
        outputs = model.generate(
            input_ids=ids.input_ids,
            attention_mask=ids.attention_mask,
            generation_config=gen_cfg,
            logits_processor=LogitsProcessorList([FiniteLogitsProcessor()]),
        )
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return [seq.replace(" ", "") for seq in decoded]


def _do_selection(
    query_seq: str,
    candidates: list[str],
    target_n: int,
    selection_mode: str,
    allow_duplicates: bool,
) -> list[str]:
    if selection_mode == "first_valid":
        result = select_first_valid(candidates, target_n)
    else:
        result = select_lowest_neff(
            query_seq,
            candidates,
            target_n,
            neff_fn=calculate_neff,
            dedupe=not allow_duplicates,
        )
    return result.sequences


def _run_temperature_variant(
    protein_id: str,
    query_seq: str,
    variant: SamplingVariant,
    tokenizer,
    model,
    device: torch.device,
    target_n: int,
    candidate_n: int,
    microbatch: int,
    selection_mode: str,
    allow_duplicates: bool,
    out_dir: Path,
) -> dict:
    row = make_result_row(
        protein_id=protein_id,
        strategy=variant.strategy,
        variant_param=variant.variant_param,
        cache_implementation=str(variant.cache_implementation),
        target_n=target_n,
        candidate_n_requested=candidate_n,
    )
    _reset_vram()
    t0 = time.perf_counter()
    candidates: list[str] = []
    try:
        seed_3di = generate_3di(
            [list(query_seq)],
            tokenizer,
            model,
            device,
            1,
            {"temperature": 1.0, "top_k": 3, "top_p": 0.85},
        )
        while len(candidates) < candidate_n:
            remaining = candidate_n - len(candidates)
            batch_returns = min(microbatch, remaining)
            candidates.extend(
                _decode_aa_microbatch(
                    seed_3di,
                    tokenizer,
                    model,
                    device,
                    batch_returns,
                    variant.decode_conf,
                    variant.cache_implementation,
                )
            )
    except torch.cuda.OutOfMemoryError:
        if microbatch <= 1:
            row["status"] = "oom"
            row["error"] = "CUDA OOM at microbatch=1"
            row["peak_vram_gb"] = round(_peak_vram_gb(), 3)
            return row
        torch.cuda.empty_cache()
        return _run_temperature_variant(
            protein_id, query_seq, variant, tokenizer, model, device,
            target_n, candidate_n, max(1, microbatch // 2),
            selection_mode, allow_duplicates, out_dir,
        )
    except Exception as exc:
        row["status"] = "skipped"
        row["error"] = str(exc)
        row["peak_vram_gb"] = round(_peak_vram_gb(), 3)
        return row

    gen_time = time.perf_counter() - t0
    filtered = filter_exact_length(candidates, len(query_seq), dedupe=not allow_duplicates)
    t1 = time.perf_counter()
    selected = _do_selection(query_seq, filtered.sequences, target_n, selection_mode, allow_duplicates)
    selection_time = time.perf_counter() - t1
    neff = calculate_neff([query_seq] + selected) if selected else 0.0
    a3m_path = out_dir / "a3m" / protein_id / _safe_variant_dir(row) / "pstMSA.a3m"
    a3m_path.parent.mkdir(parents=True, exist_ok=True)
    write_a3m(a3m_path, protein_id, query_seq, selected)
    row.update(
        {
            "raw_candidates": filtered.raw_count,
            "valid_candidates": filtered.valid_count,
            "selected_sequences": len(selected),
            "neff": round(neff, 4),
            "gen_time_s": round(gen_time, 3),
            "peak_vram_gb": round(_peak_vram_gb(), 3),
            "selection_time_s": round(selection_time, 3),
            "status": "ok",
        }
    )
    return row


def _run_embedding_variant(
    protein_id: str,
    query_seq: str,
    variant: SamplingVariant,
    tokenizer,
    model,
    device: torch.device,
    target_n: int,
    candidate_n: int,
    microbatch: int,
    selection_mode: str,
    allow_duplicates: bool,
    out_dir: Path,
) -> dict:
    row = make_result_row(
        protein_id=protein_id,
        strategy=variant.strategy,
        variant_param=variant.variant_param,
        cache_implementation=str(variant.cache_implementation),
        target_n=target_n,
        candidate_n_requested=candidate_n,
    )
    _reset_vram()
    t0 = time.perf_counter()
    candidates: list[str] = []
    sigma = float(variant.extra["sigma"])
    depth_decay = float(variant.extra.get("depth_decay", 0.8))
    try:
        seq_input = preprocess_sequence(list(query_seq), "<AA2fold>")
        ids = tokenizer([seq_input], padding="longest", return_tensors="pt").to(device)
        max_len = ids.input_ids.shape[1] + 1
        blocks = list(model.encoder.block)
        n_layers = len(blocks)

        while len(candidates) < candidate_n:
            hooks = []

            def _make_hook(layer_idx: int):
                decay = depth_decay ** (n_layers - 1 - layer_idx)

                def hook(_module, _input, output):
                    h = output[0] if isinstance(output, tuple) else output
                    if not isinstance(h, torch.Tensor):
                        return output
                    noisy = h + torch.randn_like(h) * sigma * decay
                    if isinstance(output, tuple):
                        return (noisy,) + output[1:]
                    return noisy

                return hook

            for idx, block in enumerate(blocks):
                hooks.append(block.register_forward_hook(_make_hook(idx)))
            try:
                with torch.no_grad():
                    enc_out = model.encoder(input_ids=ids.input_ids, attention_mask=ids.attention_mask)
            finally:
                for hook in hooks:
                    hook.remove()

            remaining = candidate_n - len(candidates)
            batch_returns = min(microbatch, remaining)
            kwargs = build_generation_config_kwargs(variant.decode_conf, variant.cache_implementation)
            gen_cfg = GenerationConfig(
                max_length=max_len,
                min_length=max_len - 2,
                num_return_sequences=batch_returns,
                num_beams=1,
                do_sample=True,
                **kwargs,
            )
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=ids.input_ids,
                    attention_mask=ids.attention_mask,
                    encoder_outputs=BaseModelOutput(last_hidden_state=enc_out.last_hidden_state),
                    generation_config=gen_cfg,
                    logits_processor=LogitsProcessorList([FiniteLogitsProcessor()]),
                )
            threedi = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            threedi = [seq.replace(" ", "") for seq in threedi]
            candidates.extend(
                _decode_aa_microbatch(
                    threedi, tokenizer, model, device, 1,
                    variant.decode_conf, variant.cache_implementation,
                )
            )
    except torch.cuda.OutOfMemoryError:
        if microbatch <= 1:
            row["status"] = "oom"
            row["error"] = "CUDA OOM at microbatch=1"
            row["peak_vram_gb"] = round(_peak_vram_gb(), 3)
            return row
        torch.cuda.empty_cache()
        return _run_embedding_variant(
            protein_id, query_seq, variant, tokenizer, model, device,
            target_n, candidate_n, max(1, microbatch // 2),
            selection_mode, allow_duplicates, out_dir,
        )
    except Exception as exc:
        row["status"] = "skipped"
        row["error"] = str(exc)
        row["peak_vram_gb"] = round(_peak_vram_gb(), 3)
        return row

    gen_time = time.perf_counter() - t0
    filtered = filter_exact_length(candidates, len(query_seq), dedupe=not allow_duplicates)
    t1 = time.perf_counter()
    selected = _do_selection(query_seq, filtered.sequences, target_n, selection_mode, allow_duplicates)
    selection_time = time.perf_counter() - t1
    neff = calculate_neff([query_seq] + selected) if selected else 0.0
    a3m_path = out_dir / "a3m" / protein_id / _safe_variant_dir(row) / "pstMSA.a3m"
    a3m_path.parent.mkdir(parents=True, exist_ok=True)
    write_a3m(a3m_path, protein_id, query_seq, selected)
    row.update(
        {
            "raw_candidates": filtered.raw_count,
            "valid_candidates": filtered.valid_count,
            "selected_sequences": len(selected),
            "neff": round(neff, 4),
            "gen_time_s": round(gen_time, 3),
            "peak_vram_gb": round(_peak_vram_gb(), 3),
            "selection_time_s": round(selection_time, 3),
            "status": "ok",
        }
    )
    return row


def _run_optional_folding(
    rows: list[dict],
    bench_dir: Path,
    out_dir: Path,
    colabfold_gpus: int,
) -> None:
    from rich.progress import Progress

    pending = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        a3m_path = out_dir / "a3m" / row["protein_id"] / _safe_variant_dir(row) / "pstMSA.a3m"
        ref_pdb = _find_ref_pdb(bench_dir, row["protein_id"])
        pending.append((row, a3m_path, a3m_path.parent, ref_pdb))
    if not pending:
        return
    with Progress(transient=False) as progress:
        task = progress.add_task("[green]ColabFold[/]", total=len(pending))
        _batch_fold_and_score(pending, out_dir, colabfold_gpus, progress, task)


@app.command()
def main(
    bench_dir: Annotated[
        Path,
        typer.Option("--bench-dir", exists=True, help="Benchmark dir with fasta/ and optional pdb/."),
    ],
    out_dir: Annotated[Path, typer.Option("--out-dir", help="Output directory.")],
    proteins: Annotated[
        Optional[str],
        typer.Option("--proteins", help="Comma-separated protein IDs. Default: all."),
    ] = None,
    target_n: Annotated[int, typer.Option("--target-n", help="Selected MSA sequence count.")] = 128,
    candidate_n: Annotated[int, typer.Option("--candidate-n", help="Requested valid candidate count per variant.")] = 1024,
    microbatch: Annotated[int, typer.Option("--microbatch", help="Generation microbatch size.")] = 16,
    device: Annotated[str, typer.Option("--device", help="Torch device: cuda or cpu.")] = "cuda",
    precision: Annotated[str, typer.Option("--precision", help="Model precision: bf16, fp16, int8, int4.")] = "bf16",
    max_vram_gb: Annotated[
        Optional[float],
        typer.Option("--max-vram-gb", help="VRAM cap used for summary ranking."),
    ] = None,
    cache_implementations: Annotated[
        str,
        typer.Option(
            "--cache-implementations",
            help="Comma-separated cache modes: default,offloaded,offloaded_static,quantized.",
        ),
    ] = "default",
    strategies: Annotated[
        str,
        typer.Option(
            "--strategies",
            help="Comma-separated: temperature_low_neff,embedding_walk_low_neff,all.",
        ),
    ] = "all",
    selection_mode: Annotated[
        str,
        typer.Option("--selection-mode", help="lowest_neff or first_valid."),
    ] = "lowest_neff",
    allow_duplicates: Annotated[
        bool,
        typer.Option("--allow-duplicates/--dedupe", help="Allow duplicate generated sequences."),
    ] = False,
    fold: Annotated[
        bool,
        typer.Option("--fold/--no-fold", help="Run ColabFold on selected MSAs."),
    ] = False,
    colabfold_gpus: Annotated[int, typer.Option("--colabfold-gpus")] = 1,
    min_plddt: Annotated[
        Optional[float],
        typer.Option("--min-plddt", help="Minimum pLDDT for fold-aware summary."),
    ] = None,
    min_tm_score: Annotated[
        Optional[float],
        typer.Option("--min-tm-score", help="Minimum TM-score for fold-aware summary."),
    ] = None,
) -> None:
    """Run low-Neff pseudoMSA sampling benchmark."""
    if target_n <= 0:
        raise typer.BadParameter("--target-n must be positive")
    if candidate_n <= 0:
        raise typer.BadParameter("--candidate-n must be positive")
    if microbatch <= 0:
        raise typer.BadParameter("--microbatch must be positive")
    if selection_mode not in {"lowest_neff", "first_valid"}:
        raise typer.BadParameter("--selection-mode must be lowest_neff or first_valid")

    selected_strategies = (
        ["temperature_low_neff", "embedding_walk_low_neff"]
        if strategies.strip().lower() == "all"
        else [name.strip() for name in strategies.split(",") if name.strip()]
    )
    invalid_strategies = [
        name for name in selected_strategies
        if name not in {"temperature_low_neff", "embedding_walk_low_neff"}
    ]
    if invalid_strategies:
        raise typer.BadParameter(f"Unknown strategies: {invalid_strategies}")

    cache_modes = normalize_cache_implementations(
        [part.strip() for part in cache_implementations.split(",")]
    )
    protein_ids = [p.strip() for p in proteins.split(",")] if proteins else None

    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    typer.echo(f"Loading ProstT5 ({precision}) on {dev}...")
    tokenizer, model = _load_model(dev, precision=precision)
    model.eval()

    discovered = [
        (pid, seq)
        for pid, seq in _discover_proteins(bench_dir)
        if protein_ids is None or pid in protein_ids
    ]

    all_variants: list[SamplingVariant] = []
    if "temperature_low_neff" in selected_strategies:
        all_variants.extend(generate_temperature_variants(cache_modes))
    if "embedding_walk_low_neff" in selected_strategies:
        all_variants.extend(generate_embedding_variants(cache_modes))

    rows: list[dict] = []
    for protein_id, query_seq in discovered:
        for variant in all_variants:
            typer.echo(
                f"{protein_id} {variant.strategy} {variant.variant_param} "
                f"cache={variant.cache_implementation}"
            )
            if variant.strategy == "temperature_low_neff":
                row = _run_temperature_variant(
                    protein_id=protein_id,
                    query_seq=query_seq,
                    variant=variant,
                    tokenizer=tokenizer,
                    model=model,
                    device=dev,
                    target_n=target_n,
                    candidate_n=candidate_n,
                    microbatch=microbatch,
                    selection_mode=selection_mode,
                    allow_duplicates=allow_duplicates,
                    out_dir=out_dir,
                )
            else:
                row = _run_embedding_variant(
                    protein_id=protein_id,
                    query_seq=query_seq,
                    variant=variant,
                    tokenizer=tokenizer,
                    model=model,
                    device=dev,
                    target_n=target_n,
                    candidate_n=candidate_n,
                    microbatch=microbatch,
                    selection_mode=selection_mode,
                    allow_duplicates=allow_duplicates,
                    out_dir=out_dir,
                )
            rows.append(row)
            write_csv(out_dir / "results.csv", rows, RESULT_FIELDS)

    if fold:
        _run_optional_folding(rows, bench_dir, out_dir, colabfold_gpus)
        write_csv(out_dir / "results.csv", rows, RESULT_FIELDS)

    summary = summarize_best_rows(
        rows,
        max_vram_gb=max_vram_gb,
        min_mean_plddt=min_plddt,
        min_tm_score=min_tm_score,
    )
    write_csv(out_dir / "summary.csv", summary_rows(summary), SUMMARY_FIELDS)
    typer.echo(f"Wrote {len(rows)} rows to {out_dir / 'results.csv'}")


if __name__ == "__main__":
    app()
