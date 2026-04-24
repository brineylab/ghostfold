"""Per-protein × per-strategy benchmark orchestration."""
import csv
import os
import random
import time
from pathlib import Path
from typing import Any

import torch
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from ghostfold.core.logging import get_console, get_logger
from ghostfold.msa.neff import calculate_neff
from ghostfold.msa.strategies import STRATEGIES, BaseStrategy

logger = get_logger("benchmark.runner")

_CSV_FIELDS = [
    "protein_id",
    "strategy",
    "variant_param",
    "n_sequences",
    "neff",
    "gen_time_s",
    "peak_vram_gb",
    "ptm",
    "mean_plddt",
    "rmsd",
    "tm_score",
]


def _discover_proteins(bench_dir: Path) -> list[tuple[str, str]]:
    from ghostfold.io.fasta import read_fasta_from_path
    fasta_dir = bench_dir / "fasta"
    proteins: list[tuple[str, str]] = []
    for fasta_path in sorted(fasta_dir.glob("*.fasta")):
        records = list(read_fasta_from_path(fasta_path))
        if records:
            proteins.append((fasta_path.stem, str(records[0].seq)))
    return proteins


def _find_ref_pdb(bench_dir: Path, protein_id: str) -> Path | None:
    pdb = bench_dir / "pdb" / f"{protein_id}.pdb"
    return pdb if pdb.exists() else None


def _write_a3m(path: Path, query_id: str, query_seq: str, sequences: list[str]) -> None:
    with open(path, "w") as fh:
        fh.write(f">{query_id}\n{query_seq}\n")
        for i, seq in enumerate(sequences):
            fh.write(f">generated_{i}\n{seq}\n")


def _measure_vram_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    return 0.0


def _reset_vram() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _variant_param(strategy_name: str, config: dict) -> str:
    if strategy_name == "encoder_perturb":
        return f"sigmas={config.get('noise_scales', [])}"
    if strategy_name == "diverse_beam":
        return f"beams={config.get('num_beams', 8)},dp={config.get('diversity_penalty', 1.0)}"
    if strategy_name == "round_trip":
        return f"rounds={config.get('n_rounds', 4)},seeds={config.get('n_seeds', 8)}"
    if strategy_name == "3di_perturb":
        return f"rates={config.get('mutation_rates', [])}"
    return ""


def _subsample(sequences: list[str], target: int) -> list[str]:
    """Random subsample down to *target* without replacement."""
    if len(sequences) <= target:
        return sequences
    return random.sample(sequences, target)


def _generate_msa(
    protein_id: str,
    query_seq: str,
    strategy: "BaseStrategy",
    strategy_config: dict,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    out_dir: Path,
    target_n_sequences: int,
    progress: Progress,
    task_id: TaskID,
) -> tuple[dict[str, Any], Path]:
    """Generate MSA for one protein × strategy. Returns (row_dict, a3m_path)."""
    run_dir = out_dir / protein_id / strategy.name
    run_dir.mkdir(parents=True, exist_ok=True)

    progress.update(task_id, description=f"[cyan]{protein_id}[/] [yellow]{strategy.name}[/] generating…")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _reset_vram()
    t0 = time.perf_counter()

    try:
        sequences = strategy.generate_msa(query_seq, model, tokenizer, device, strategy_config)
    except Exception as exc:
        logger.error(f"{protein_id}/{strategy.name}: generation failed — {exc}")
        sequences = []

    gen_time = time.perf_counter() - t0
    peak_vram = _measure_vram_gb()

    sequences = [s for s in sequences if len(s) == len(query_seq)]

    raw_n = len(sequences)
    if raw_n > target_n_sequences:
        sequences = _subsample(sequences, target_n_sequences)
        logger.info(
            f"{protein_id}/{strategy.name}: subsampled {raw_n} → {len(sequences)} sequences"
        )
    elif raw_n < target_n_sequences:
        logger.warning(
            f"{protein_id}/{strategy.name}: only {raw_n} sequences generated "
            f"(target {target_n_sequences})"
        )

    progress.update(task_id, description=f"[cyan]{protein_id}[/] [yellow]{strategy.name}[/] Neff…")

    neff = 0.0
    if sequences:
        try:
            neff = calculate_neff([query_seq] + sequences)
        except Exception:
            pass

    msa_subdir = run_dir / "msa" / protein_id
    msa_subdir.mkdir(parents=True, exist_ok=True)
    a3m_path = msa_subdir / "pstMSA.a3m"
    _write_a3m(a3m_path, protein_id, query_seq, sequences)

    row: dict[str, Any] = {
        "protein_id": protein_id,
        "strategy": strategy.name,
        "variant_param": _variant_param(strategy.name, strategy_config),
        "n_sequences": len(sequences),
        "neff": round(neff, 4),
        "gen_time_s": round(gen_time, 3),
        "peak_vram_gb": round(peak_vram, 3),
        "ptm": None,
        "mean_plddt": None,
        "rmsd": None,
        "tm_score": None,
    }

    return row, a3m_path


def _batch_fold_and_score(
    pending: list[tuple[dict, Path, Path, Path | None]],
    out_dir: Path,
    colabfold_gpus: int,
    progress: Progress,
    task_id: TaskID,
) -> None:
    """Fold all MSAs in a single ColabFold run, then parse scores back into rows."""
    from ghostfold.benchmark.colabfold_parser import parse_best_scores
    from ghostfold.benchmark.structure_metrics import compute_structure_metrics
    from ghostfold.core.colabfold import run_colabfold

    batch_root = out_dir / "_cf_batch"
    msa_stage = batch_root / "msa"
    msa_stage.mkdir(parents=True, exist_ok=True)

    staging_map: dict[str, tuple[dict, Path | None]] = {}

    for row, a3m_path, _run_dir, ref_pdb in pending:
        if not a3m_path.exists():
            continue
        dirname = f"{row['protein_id']}__{row['strategy']}"
        link_dir = msa_stage / dirname
        link_dir.mkdir(exist_ok=True)
        link_path = link_dir / "pstMSA.a3m"
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        os.symlink(a3m_path.resolve(), link_path)
        staging_map[dirname] = (row, ref_pdb)

    if not staging_map:
        logger.warning("No valid MSAs to fold.")
        return

    progress.update(task_id, description=f"[green]ColabFold[/] folding {len(staging_map)} MSAs…", total=len(staging_map), completed=0)

    try:
        run_colabfold(
            project_name=str(batch_root),
            num_gpus=colabfold_gpus,
            subsample=False,
        )
    except Exception as exc:
        logger.error(f"Batch ColabFold run failed: {exc}")
        return

    preds_root = batch_root / "subsample_1" / "preds"

    for i, (dirname, (row, ref_pdb)) in enumerate(staging_map.items()):
        progress.update(task_id, description=f"[green]Scoring[/] {dirname}…", completed=i + 1)
        cf_out = preds_root / dirname
        if not cf_out.exists():
            logger.warning(f"No preds dir found for {dirname}")
            continue

        try:
            scores = parse_best_scores(cf_out)
            row["ptm"] = scores.get("ptm")
            row["mean_plddt"] = scores.get("mean_plddt")
        except FileNotFoundError:
            logger.warning(f"No rank_001 JSON in {cf_out}")
            continue

        if ref_pdb and scores.get("best_pdb"):
            try:
                metrics = compute_structure_metrics(scores["best_pdb"], ref_pdb)
                row["rmsd"] = round(metrics["rmsd"], 4)
                row["tm_score"] = round(metrics["tm_score"], 4)
            except Exception as exc:
                logger.warning(f"Structure comparison failed for {dirname}: {exc}")


def _print_summary(results: list[dict], console: Any) -> None:
    """Render a per-strategy summary table to the console."""
    from collections import defaultdict

    by_strategy: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        by_strategy[row["strategy"]].append(row)

    table = Table(title="Benchmark Summary", show_lines=False)
    table.add_column("Strategy", style="cyan", no_wrap=True)
    table.add_column("N", justify="right")
    table.add_column("Neff", justify="right")
    table.add_column("Time (s)", justify="right")
    table.add_column("VRAM (GB)", justify="right")
    table.add_column("pTM", justify="right")
    table.add_column("pLDDT", justify="right")
    table.add_column("RMSD", justify="right")
    table.add_column("TM-score", justify="right")

    def _avg(rows: list[dict], key: str) -> str:
        vals = [r[key] for r in rows if r[key] is not None]
        return f"{sum(vals)/len(vals):.3f}" if vals else "—"

    for strategy, rows in sorted(by_strategy.items()):
        table.add_row(
            strategy,
            _avg(rows, "n_sequences"),
            _avg(rows, "neff"),
            _avg(rows, "gen_time_s"),
            _avg(rows, "peak_vram_gb"),
            _avg(rows, "ptm"),
            _avg(rows, "mean_plddt"),
            _avg(rows, "rmsd"),
            _avg(rows, "tm_score"),
        )

    console.print(table)


def run_benchmark(
    bench_dir: Path,
    out_dir: Path,
    strategy_names: list[str],
    strategy_configs: dict[str, dict],
    model: Any,
    tokenizer: Any,
    device: torch.device,
    encoder_model: Any | None = None,
    cnn_3di: Any | None = None,
    protein_ids: list[str] | None = None,
    run_colabfold: bool = False,
    colabfold_gpus: int = 1,
    target_n_sequences: int = 128,
) -> list[dict]:
    """Run the full benchmark matrix and write results.csv to *out_dir*.

    Phase 1: Generate MSAs for all proteins × strategies, subsampled to
             *target_n_sequences* for a fair cross-strategy comparison.
    Phase 2: If run_colabfold, fold all MSAs in a single ColabFold invocation.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    proteins = [
        (pid, seq)
        for pid, seq in _discover_proteins(bench_dir)
        if protein_ids is None or pid in protein_ids
    ]
    total_jobs = len(proteins) * len(strategy_names)

    console = get_console()
    results: list[dict] = []
    pending: list[tuple[dict, Path, Path, Path | None]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        overall = progress.add_task(
            f"[bold]MSA generation[/] (0/{total_jobs})", total=total_jobs
        )
        detail = progress.add_task("", total=1)

        for pid, query_seq in proteins:
            ref_pdb = _find_ref_pdb(bench_dir, pid)

            for name in strategy_names:
                cls = STRATEGIES.get(name)
                if cls is None:
                    logger.warning(f"Unknown strategy '{name}' — skipping.")
                    progress.update(overall, advance=1)
                    continue

                strategy = cls()
                cfg = dict(strategy_configs.get(name, {}))
                cfg["encoder_model"] = encoder_model
                cfg["cnn_3di"] = cnn_3di

                row, a3m_path = _generate_msa(
                    protein_id=pid,
                    query_seq=query_seq,
                    strategy=strategy,
                    strategy_config=cfg,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    out_dir=out_dir,
                    target_n_sequences=target_n_sequences,
                    progress=progress,
                    task_id=detail,
                )
                results.append(row)
                run_dir = out_dir / pid / name
                pending.append((row, a3m_path, run_dir, ref_pdb))
                _save_csv(out_dir / "results.csv", results)

                progress.update(
                    overall,
                    advance=1,
                    description=(
                        f"[bold]MSA generation[/] — "
                        f"last: [cyan]{pid}[/]/[yellow]{name}[/] "
                        f"n={row['n_sequences']} neff={row['neff']:.3f}"
                    ),
                )

        progress.update(detail, description="[dim]MSA generation complete[/]", completed=1)

        if run_colabfold:
            fold_task = progress.add_task(
                f"[green]ColabFold[/] — {len(pending)} MSAs queued",
                total=len(pending),
            )
            _batch_fold_and_score(pending, out_dir, colabfold_gpus, progress, fold_task)
            _save_csv(out_dir / "results.csv", results)

    _print_summary(results, console)
    return results


def _save_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
