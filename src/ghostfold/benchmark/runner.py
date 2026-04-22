"""Per-protein × per-strategy benchmark orchestration."""
import csv
import time
from pathlib import Path
from typing import Any

import torch

from ghostfold.core.logging import get_logger
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


def _write_a3m(path: Path, query_id: str, query_seq: str, sequences: list[str]) -> None:
    """Write a minimal A3M file with query first and generated sequences after."""
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
    """Encode the key hyperparameter for this strategy run as a short string."""
    if strategy_name == "encoder_perturb":
        return f"sigmas={config.get('noise_scales', [])}"
    if strategy_name == "diverse_beam":
        return f"beams={config.get('num_beams', 8)},dp={config.get('diversity_penalty', 1.0)}"
    if strategy_name == "round_trip":
        return f"rounds={config.get('n_rounds', 4)},seeds={config.get('n_seeds', 8)}"
    if strategy_name == "3di_perturb":
        return f"rates={config.get('mutation_rates', [])}"
    return ""


def run_single(
    protein_id: str,
    query_seq: str,
    strategy: BaseStrategy,
    strategy_config: dict,
    model,
    tokenizer,
    device: torch.device,
    out_dir: Path,
    reference_pdb: Path | None = None,
    run_colabfold: bool = False,
    colabfold_gpus: int = 1,
) -> dict[str, Any]:
    """Run one protein × strategy combination and return a result row dict."""
    run_dir = out_dir / protein_id / strategy.name
    run_dir.mkdir(parents=True, exist_ok=True)

    _reset_vram()
    t0 = time.perf_counter()

    try:
        sequences = strategy.generate_msa(query_seq, model, tokenizer, device, strategy_config)
    except Exception as exc:
        logger.error(f"{protein_id}/{strategy.name}: generation failed — {exc}")
        sequences = []

    gen_time = time.perf_counter() - t0
    peak_vram = _measure_vram_gb()

    # Filter to query length (strategies may return varying-length sequences)
    sequences = [s for s in sequences if len(s) == len(query_seq)]

    neff = 0.0
    if sequences:
        try:
            neff = calculate_neff([query_seq] + sequences)
        except Exception:
            pass

    a3m_path = run_dir / "msa.a3m"
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

    if run_colabfold and sequences:
        _run_fold_and_score(row, a3m_path, run_dir, reference_pdb, colabfold_gpus)

    return row


def _run_fold_and_score(
    row: dict,
    a3m_path: Path,
    run_dir: Path,
    reference_pdb: Path | None,
    colabfold_gpus: int,
) -> None:
    """Run ColabFold, parse scores, compute structure metrics; mutates *row*."""
    from ghostfold.benchmark.colabfold_parser import parse_best_scores
    from ghostfold.benchmark.structure_metrics import compute_structure_metrics
    from ghostfold.core.colabfold import run_colabfold

    cf_out = run_dir / "colabfold_out"
    cf_out.mkdir(exist_ok=True)

    try:
        run_colabfold(
            project_name=str(run_dir),
            num_gpus=colabfold_gpus,
            subsample=False,
        )
    except Exception as exc:
        logger.error(f"ColabFold failed for {row['protein_id']}/{row['strategy']}: {exc}")
        return

    try:
        scores = parse_best_scores(cf_out)
        row["ptm"] = scores.get("ptm")
        row["mean_plddt"] = scores.get("mean_plddt")
    except FileNotFoundError:
        logger.warning(f"No rank_001 JSON found in {cf_out}")
        return

    if reference_pdb and scores.get("best_pdb"):
        try:
            metrics = compute_structure_metrics(scores["best_pdb"], reference_pdb)
            row["rmsd"] = round(metrics["rmsd"], 4)
            row["tm_score"] = round(metrics["tm_score"], 4)
        except Exception as exc:
            logger.warning(f"Structure comparison failed: {exc}")


def run_benchmark(
    bench_dir: Path,
    out_dir: Path,
    strategy_names: list[str],
    strategy_configs: dict[str, dict],
    model,
    tokenizer,
    device: torch.device,
    protein_ids: list[str] | None = None,
    run_colabfold: bool = False,
    colabfold_gpus: int = 1,
) -> list[dict]:
    """Run the full benchmark matrix and write results.csv to *out_dir*.

    Args:
        bench_dir:        Directory with queries.fasta and {id}.pdb files.
        out_dir:          Output root; per-run subdirs created automatically.
        strategy_names:   Names from STRATEGIES to benchmark.
        strategy_configs: Dict mapping strategy name → config dict passed to
                          generate_msa().
        model/tokenizer/device: Loaded ProstT5 model.
        protein_ids:      Subset of proteins to run; None = all in queries.fasta.
        run_colabfold:    Whether to fold the generated MSAs.
        colabfold_gpus:   GPU count passed to run_colabfold.
    """
    from ghostfold.io.fasta import read_fasta_from_path

    out_dir.mkdir(parents=True, exist_ok=True)
    queries_fasta = bench_dir / "queries.fasta"
    records = read_fasta_from_path(queries_fasta)

    results: list[dict] = []

    for record in records:
        pid = record.id
        if protein_ids and pid not in protein_ids:
            continue
        query_seq = str(record.seq)
        ref_pdb = bench_dir / f"{pid}.pdb"
        ref_pdb = ref_pdb if ref_pdb.exists() else None

        for name in strategy_names:
            cls = STRATEGIES.get(name)
            if cls is None:
                logger.warning(f"Unknown strategy '{name}' — skipping.")
                continue
            strategy = cls()
            cfg = strategy_configs.get(name, {})
            logger.info(f"Running {pid} × {name}")
            row = run_single(
                protein_id=pid,
                query_seq=query_seq,
                strategy=strategy,
                strategy_config=cfg,
                model=model,
                tokenizer=tokenizer,
                device=device,
                out_dir=out_dir,
                reference_pdb=ref_pdb,
                run_colabfold=run_colabfold,
                colabfold_gpus=colabfold_gpus,
            )
            results.append(row)
            _save_csv(out_dir / "results.csv", results)

    return results


def _save_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
