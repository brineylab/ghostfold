# Low-Neff PseudoMSA Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `benchmarks/bench_pseudomsa_low_neff.py`, an experiment runner that samples many ProstT5 pseudoMSA candidates from temperature and embedding-walk variants, selects lower-Neff subsets, tracks VRAM, and writes result summaries.

**Architecture:** Put pure, testable logic in `src/ghostfold/benchmark/low_neff.py`: config-grid construction, filtering, selection, row ranking, and A3M/CSV helpers. Keep model inference and CLI orchestration in `benchmarks/bench_pseudomsa_low_neff.py`. Reuse existing GhostFold loaders, Neff calculation, benchmark protein discovery, and ColabFold batch helpers instead of changing production CLI code.

**Tech Stack:** Python 3.10+, Typer, PyTorch, Hugging Face Transformers `GenerationConfig`, existing `ghostfold.msa.neff.calculate_neff`, existing `ghostfold.core.pipeline._load_model`, pytest, Ruff.

---

## File Structure

- Create `src/ghostfold/benchmark/low_neff.py`: pure utilities and small dataclasses for the low-Neff experiment.
- Create `benchmarks/bench_pseudomsa_low_neff.py`: Typer CLI, model loading, sampling loops, VRAM measurement, optional folding.
- Create `tests/test_low_neff_benchmark.py`: unit tests for pure utilities only.
- Do not modify `benchmarks/bench_pseudomsa.py`, `src/ghostfold/msa/strategies/*`, or GhostFold CLI modules.

---

### Task 1: Add Pure Low-Neff Utilities

**Files:**
- Create: `src/ghostfold/benchmark/low_neff.py`
- Test: `tests/test_low_neff_benchmark.py`

- [ ] **Step 1: Write failing tests for config grids, filtering, selection, and summary ranking**

Create `tests/test_low_neff_benchmark.py` with:

```python
from ghostfold.benchmark.low_neff import (
    Candidate,
    filter_exact_length,
    generate_temperature_variants,
    select_first_valid,
    select_lowest_neff,
    summarize_best_rows,
)


def fake_neff(sequences):
    unique = set(sequences)
    return float(len(unique))


def test_generate_temperature_variants_includes_modern_sampling_knobs():
    variants = list(generate_temperature_variants(cache_implementations=["default"]))

    assert variants
    assert all(v.strategy == "temperature_low_neff" for v in variants)
    assert any("min_p" in v.decode_conf for v in variants)
    assert any("typical_p" in v.decode_conf for v in variants)
    assert variants[0].cache_implementation == "default"


def test_filter_exact_length_removes_wrong_lengths_and_tracks_counts():
    candidates = ["ACDE", "ACD", "ACDE", "ACDEF"]

    result = filter_exact_length(candidates, expected_length=4, dedupe=True)

    assert result.raw_count == 4
    assert result.valid_count == 2
    assert result.sequences == ["ACDE"]


def test_select_first_valid_returns_prefix():
    candidates = ["AAAA", "AAAT", "AATT"]

    selected = select_first_valid(candidates, target_n=2)

    assert selected == ["AAAA", "AAAT"]


def test_select_lowest_neff_prefers_redundant_candidates():
    query = "AAAA"
    candidates = ["AAAT", "AAAT", "AATT", "TTTT"]

    selected = select_lowest_neff(
        query_seq=query,
        candidates=candidates,
        target_n=3,
        neff_fn=fake_neff,
        allow_duplicates=True,
        candidate_window=4,
    )

    assert selected == ["AAAT", "AAAT", "AATT"]
    assert fake_neff([query] + selected) == 3.0


def test_summarize_best_rows_picks_fixed_count_and_vram_best():
    rows = [
        {
            "protein_id": "p1",
            "strategy": "a",
            "target_n": 3,
            "selected_sequences": 3,
            "neff": 5.0,
            "peak_vram_gb": 7.9,
            "mean_plddt": None,
            "tm_score": None,
        },
        {
            "protein_id": "p1",
            "strategy": "b",
            "target_n": 3,
            "selected_sequences": 3,
            "neff": 2.0,
            "peak_vram_gb": 8.5,
            "mean_plddt": None,
            "tm_score": None,
        },
        {
            "protein_id": "p1",
            "strategy": "c",
            "target_n": 3,
            "selected_sequences": 2,
            "neff": 1.0,
            "peak_vram_gb": 6.0,
            "mean_plddt": None,
            "tm_score": None,
        },
    ]

    summary = summarize_best_rows(rows, max_vram_gb=8.0)

    assert summary["fixed_count_best"]["strategy"] == "b"
    assert summary["vram_aware_best"]["strategy"] == "a"
    assert summary["fold_aware_best"] is None
```

- [ ] **Step 2: Run tests and verify they fail because module does not exist**

Run:

```bash
pytest tests/test_low_neff_benchmark.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'ghostfold.benchmark.low_neff'`.

- [ ] **Step 3: Implement pure utility module**

Create `src/ghostfold/benchmark/low_neff.py`:

```python
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence


@dataclass(frozen=True)
class SamplingVariant:
    strategy: str
    variant_param: str
    decode_conf: dict
    cache_implementation: str
    extra: dict


@dataclass(frozen=True)
class FilterResult:
    sequences: list[str]
    raw_count: int
    valid_count: int


@dataclass(frozen=True)
class Candidate:
    sequence: str
    source: str = ""


RESULT_FIELDS = [
    "protein_id",
    "strategy",
    "variant_param",
    "cache_implementation",
    "target_n",
    "candidate_n_requested",
    "raw_candidates",
    "valid_candidates",
    "selected_sequences",
    "neff",
    "gen_time_s",
    "peak_vram_gb",
    "selection_time_s",
    "ptm",
    "mean_plddt",
    "rmsd",
    "tm_score",
    "status",
    "error",
]

SUMMARY_FIELDS = ["objective", *RESULT_FIELDS]


def generate_temperature_variants(
    cache_implementations: Sequence[str],
) -> Iterable[SamplingVariant]:
    temperatures = [0.3, 0.5, 0.7, 0.9]
    top_ks = [1, 3, 5, 10]
    top_ps = [0.70, 0.80, 0.90]
    modern_knobs = [
        {"min_p": 0.05},
        {"typical_p": 0.80},
        {"eta_cutoff": 6e-4},
        {"epsilon_cutoff": 6e-4},
    ]

    for cache_impl in cache_implementations:
        for temp in temperatures:
            for top_k in top_ks:
                for top_p in top_ps:
                    base = {
                        "temperature": temp,
                        "top_k": top_k,
                        "top_p": top_p,
                        "repetition_penalty": 1.15,
                    }
                    yield SamplingVariant(
                        strategy="temperature_low_neff",
                        variant_param=f"temp={temp},top_k={top_k},top_p={top_p}",
                        decode_conf=dict(base),
                        cache_implementation=cache_impl,
                        extra={},
                    )
                    for knob in modern_knobs:
                        conf = dict(base)
                        conf.update(knob)
                        knob_name, knob_value = next(iter(knob.items()))
                        yield SamplingVariant(
                            strategy="temperature_low_neff",
                            variant_param=(
                                f"temp={temp},top_k={top_k},top_p={top_p},"
                                f"{knob_name}={knob_value}"
                            ),
                            decode_conf=conf,
                            cache_implementation=cache_impl,
                            extra={},
                        )


def generate_embedding_variants(
    cache_implementations: Sequence[str],
) -> Iterable[SamplingVariant]:
    sigmas = [0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
    decode_conf = {
        "temperature": 0.5,
        "top_k": 5,
        "top_p": 0.85,
        "repetition_penalty": 1.15,
    }
    for cache_impl in cache_implementations:
        for sigma in sigmas:
            yield SamplingVariant(
                strategy="embedding_walk_low_neff",
                variant_param=f"sigma={sigma}",
                decode_conf=dict(decode_conf),
                cache_implementation=cache_impl,
                extra={"sigma": sigma, "depth_decay": 0.8},
            )


def normalize_cache_implementations(values: Sequence[str]) -> list[str]:
    allowed = {"default", "offloaded", "offloaded_static", "quantized"}
    normalized = []
    for value in values:
        name = value.strip()
        if not name:
            continue
        if name not in allowed:
            raise ValueError(f"Unsupported cache implementation: {name}")
        normalized.append(name)
    return normalized or ["default"]


def filter_exact_length(
    sequences: Sequence[str],
    expected_length: int,
    dedupe: bool = True,
) -> FilterResult:
    raw = list(sequences)
    valid = [seq for seq in raw if len(seq) == expected_length]
    valid_count = len(valid)
    if dedupe:
        seen: set[str] = set()
        deduped = []
        for seq in valid:
            if seq in seen:
                continue
            seen.add(seq)
            deduped.append(seq)
        valid = deduped
    return FilterResult(sequences=valid, raw_count=len(raw), valid_count=valid_count)


def hamming_distance(left: str, right: str) -> int:
    return sum(a != b for a, b in zip(left, right)) + abs(len(left) - len(right))


def select_first_valid(candidates: Sequence[str], target_n: int) -> list[str]:
    return list(candidates[:target_n])


def select_lowest_neff(
    query_seq: str,
    candidates: Sequence[str],
    target_n: int,
    neff_fn: Callable[[Sequence[str]], float],
    allow_duplicates: bool,
    candidate_window: int = 128,
) -> list[str]:
    pool = list(candidates)
    if not allow_duplicates:
        seen: set[str] = set()
        unique = []
        for seq in pool:
            if seq in seen:
                continue
            seen.add(seq)
            unique.append(seq)
        pool = unique

    pool.sort(key=lambda seq: (hamming_distance(query_seq, seq), seq))
    selected: list[str] = []

    while pool and len(selected) < target_n:
        window = pool[:candidate_window]
        best_idx = 0
        best_neff = float("inf")
        for idx, seq in enumerate(window):
            score = neff_fn([query_seq] + selected + [seq])
            if score < best_neff:
                best_neff = score
                best_idx = idx
        selected.append(pool.pop(best_idx))

    return selected


def best_lowest_neff(rows: Sequence[dict]) -> dict | None:
    eligible = [row for row in rows if row.get("neff") is not None]
    if not eligible:
        return None
    return min(eligible, key=lambda row: (float(row["neff"]), row.get("gen_time_s") or 0.0))


def summarize_best_rows(
    rows: Sequence[dict],
    max_vram_gb: float | None = None,
    min_plddt: float | None = None,
    min_tm_score: float | None = None,
) -> dict[str, dict | None]:
    fixed_rows = [
        row for row in rows
        if row.get("selected_sequences") == row.get("target_n")
    ]
    vram_rows = [
        row for row in fixed_rows
        if max_vram_gb is None or float(row.get("peak_vram_gb") or 0.0) <= max_vram_gb
    ]
    fold_rows = []
    for row in fixed_rows:
        plddt = row.get("mean_plddt")
        tm_score = row.get("tm_score")
        if plddt is None and tm_score is None:
            continue
        if min_plddt is not None and (plddt is None or float(plddt) < min_plddt):
            continue
        if min_tm_score is not None and (tm_score is None or float(tm_score) < min_tm_score):
            continue
        fold_rows.append(row)

    return {
        "fixed_count_best": best_lowest_neff(fixed_rows),
        "fold_aware_best": best_lowest_neff(fold_rows),
        "vram_aware_best": best_lowest_neff(vram_rows),
    }


def write_a3m(path: Path, query_id: str, query_seq: str, sequences: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.write(f">{query_id}\n{query_seq}\n")
        for idx, seq in enumerate(sequences):
            fh.write(f">generated_{idx}\n{seq}\n")


def write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def summary_rows(summary: dict[str, dict | None]) -> list[dict]:
    rows = []
    for objective, row in summary.items():
        if row is None:
            continue
        out = {"objective": objective}
        out.update(row)
        rows.append(out)
    return rows
```

- [ ] **Step 4: Run utility tests**

Run:

```bash
pytest tests/test_low_neff_benchmark.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit utilities**

Run:

```bash
git add src/ghostfold/benchmark/low_neff.py tests/test_low_neff_benchmark.py
git commit -m "Add low-Neff benchmark utilities"
```

Expected: commit created.

---

### Task 2: Add Low-Neff Benchmark CLI Skeleton

**Files:**
- Create: `benchmarks/bench_pseudomsa_low_neff.py`
- Modify: `tests/test_low_neff_benchmark.py`

- [ ] **Step 1: Add test that script help works without model loading**

Append to `tests/test_low_neff_benchmark.py`:

```python
from typer.testing import CliRunner

from benchmarks.bench_pseudomsa_low_neff import app


def test_low_neff_script_help_exposes_core_options():
    runner = CliRunner()

    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "--candidate-n" in result.output
    assert "--microbatch" in result.output
    assert "--max-vram-gb" in result.output
    assert "--cache-implementations" in result.output
```

- [ ] **Step 2: Run focused test and verify import failure**

Run:

```bash
pytest tests/test_low_neff_benchmark.py::test_low_neff_script_help_exposes_core_options -q
```

Expected: FAIL with `ModuleNotFoundError` or import error for `benchmarks.bench_pseudomsa_low_neff`.

- [ ] **Step 3: Add CLI skeleton**

Create `benchmarks/bench_pseudomsa_low_neff.py`:

```python
#!/usr/bin/env python
"""Benchmark low-Neff pseudoMSA sampling with ProstT5."""
from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from ghostfold.benchmark.low_neff import normalize_cache_implementations

app = typer.Typer(add_completion=False, no_args_is_help=True)


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

    cache_modes = normalize_cache_implementations(
        [part.strip() for part in cache_implementations.split(",")]
    )
    protein_ids = [p.strip() for p in proteins.split(",")] if proteins else None

    typer.echo(
        "low-Neff benchmark configured: "
        f"proteins={'all' if protein_ids is None else len(protein_ids)} "
        f"target_n={target_n} candidate_n={candidate_n} "
        f"microbatch={microbatch} cache={cache_modes} fold={fold}"
    )
    typer.echo("Model sampling implementation added in later tasks.")


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run help test**

Run:

```bash
pytest tests/test_low_neff_benchmark.py::test_low_neff_script_help_exposes_core_options -q
```

Expected: PASS.

- [ ] **Step 5: Commit CLI skeleton**

Run:

```bash
git add benchmarks/bench_pseudomsa_low_neff.py tests/test_low_neff_benchmark.py
git commit -m "Add low-Neff benchmark CLI skeleton"
```

Expected: commit created.

---

### Task 3: Implement Temperature Low-Neff Sampling

**Files:**
- Modify: `benchmarks/bench_pseudomsa_low_neff.py`
- Modify: `src/ghostfold/benchmark/low_neff.py`
- Test: `tests/test_low_neff_benchmark.py`

- [ ] **Step 1: Add tests for cache config attachment and row construction helper**

Append to `tests/test_low_neff_benchmark.py`:

```python
from ghostfold.benchmark.low_neff import build_generation_config_kwargs, make_result_row


def test_build_generation_config_kwargs_omits_default_cache():
    kwargs = build_generation_config_kwargs({"temperature": 0.5}, "default")

    assert kwargs == {"temperature": 0.5}


def test_build_generation_config_kwargs_adds_non_default_cache():
    kwargs = build_generation_config_kwargs({"temperature": 0.5}, "offloaded")

    assert kwargs["temperature"] == 0.5
    assert kwargs["cache_implementation"] == "offloaded"


def test_make_result_row_uses_schema_defaults():
    row = make_result_row(
        protein_id="p1",
        strategy="temperature_low_neff",
        variant_param="temp=0.5",
        cache_implementation="default",
        target_n=128,
        candidate_n_requested=1024,
    )

    assert row["protein_id"] == "p1"
    assert row["selected_sequences"] == 0
    assert row["status"] == "pending"
    assert set(row) == set(RESULT_FIELDS)
```

Also update the import list at the top:

```python
from ghostfold.benchmark.low_neff import (
    Candidate,
    RESULT_FIELDS,
    build_generation_config_kwargs,
    filter_exact_length,
    generate_temperature_variants,
    make_result_row,
    select_first_valid,
    select_lowest_neff,
    summarize_best_rows,
)
```

- [ ] **Step 2: Run tests and verify missing functions fail**

Run:

```bash
pytest tests/test_low_neff_benchmark.py -q
```

Expected: FAIL with import errors for `build_generation_config_kwargs` and `make_result_row`.

- [ ] **Step 3: Add generation config and result-row helpers**

Add to `src/ghostfold/benchmark/low_neff.py`:

```python
def build_generation_config_kwargs(
    decode_conf: dict,
    cache_implementation: str,
) -> dict:
    kwargs = dict(decode_conf)
    if cache_implementation != "default":
        kwargs["cache_implementation"] = cache_implementation
    return kwargs


def make_result_row(
    protein_id: str,
    strategy: str,
    variant_param: str,
    cache_implementation: str,
    target_n: int,
    candidate_n_requested: int,
) -> dict:
    row = {field: None for field in RESULT_FIELDS}
    row.update(
        {
            "protein_id": protein_id,
            "strategy": strategy,
            "variant_param": variant_param,
            "cache_implementation": cache_implementation,
            "target_n": target_n,
            "candidate_n_requested": candidate_n_requested,
            "raw_candidates": 0,
            "valid_candidates": 0,
            "selected_sequences": 0,
            "neff": 0.0,
            "gen_time_s": 0.0,
            "peak_vram_gb": 0.0,
            "selection_time_s": 0.0,
            "status": "pending",
            "error": "",
        }
    )
    return row
```

- [ ] **Step 4: Implement temperature sampler in benchmark script**

Replace the CLI-only body in `benchmarks/bench_pseudomsa_low_neff.py` with orchestration helpers. Keep the existing Typer signature and add imports:

```python
import time
from collections.abc import Sequence

import torch
from transformers import GenerationConfig, LogitsProcessorList

from ghostfold.benchmark.low_neff import (
    RESULT_FIELDS,
    SUMMARY_FIELDS,
    build_generation_config_kwargs,
    filter_exact_length,
    generate_temperature_variants,
    make_result_row,
    select_first_valid,
    select_lowest_neff,
    summarize_best_rows,
    summary_rows,
    write_a3m,
    write_csv,
)
from ghostfold.benchmark.runner import _discover_proteins
from ghostfold.core.pipeline import _load_model
from ghostfold.msa.model import FiniteLogitsProcessor, generate_3di, preprocess_sequence
from ghostfold.msa.neff import calculate_neff
```

Add helpers above `main()`:

```python
def _peak_vram_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    return 0.0


def _reset_vram() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _decode_aa_microbatch(
    fold_seqs: Sequence[str],
    tokenizer,
    model,
    device: torch.device,
    num_return_sequences: int,
    decode_conf: dict,
    cache_implementation: str,
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


def _select_candidates(
    query_seq: str,
    candidates: Sequence[str],
    target_n: int,
    selection_mode: str,
    allow_duplicates: bool,
) -> list[str]:
    if selection_mode == "first_valid":
        return select_first_valid(candidates, target_n)
    return select_lowest_neff(
        query_seq=query_seq,
        candidates=candidates,
        target_n=target_n,
        neff_fn=calculate_neff,
        allow_duplicates=allow_duplicates,
    )


def _run_temperature_variant(
    protein_id: str,
    query_seq: str,
    variant,
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
        cache_implementation=variant.cache_implementation,
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
            protein_id,
            query_seq,
            variant,
            tokenizer,
            model,
            device,
            target_n,
            candidate_n,
            max(1, microbatch // 2),
            selection_mode,
            allow_duplicates,
            out_dir,
        )
    except Exception as exc:
        row["status"] = "skipped"
        row["error"] = str(exc)
        row["peak_vram_gb"] = round(_peak_vram_gb(), 3)
        return row

    gen_time = time.perf_counter() - t0
    filtered = filter_exact_length(candidates, len(query_seq), dedupe=not allow_duplicates)
    t1 = time.perf_counter()
    selected = _select_candidates(
        query_seq,
        filtered.sequences,
        target_n,
        selection_mode,
        allow_duplicates,
    )
    selection_time = time.perf_counter() - t1
    neff = calculate_neff([query_seq] + selected) if selected else 0.0
    a3m_path = out_dir / "a3m" / protein_id / _safe_variant_dir(row) / "pstMSA.a3m"
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


def _safe_variant_dir(row: dict) -> str:
    text = f"{row['strategy']}__{row['variant_param']}__cache={row['cache_implementation']}"
    return "".join(ch if ch.isalnum() or ch in "._=-" else "_" for ch in text)
```

Then replace the final echo-only body in `main()` after validation with:

```python
    import torch

    cache_modes = normalize_cache_implementations(
        [part.strip() for part in cache_implementations.split(",")]
    )
    protein_ids = [p.strip() for p in proteins.split(",")] if proteins else None

    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    typer.echo(f"Loading ProstT5 ({precision}) on {dev}...")
    tokenizer, model = _load_model(dev, precision=precision)
    model.eval()

    discovered = [
        (pid, seq)
        for pid, seq in _discover_proteins(bench_dir)
        if protein_ids is None or pid in protein_ids
    ]
    variants = list(generate_temperature_variants(cache_modes))
    rows: list[dict] = []

    for protein_id, query_seq in discovered:
        for variant in variants:
            typer.echo(f"{protein_id} {variant.strategy} {variant.variant_param} cache={variant.cache_implementation}")
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
            rows.append(row)
            write_csv(out_dir / "results.csv", rows, RESULT_FIELDS)

    summary = summarize_best_rows(rows, max_vram_gb=max_vram_gb, min_plddt=min_plddt, min_tm_score=min_tm_score)
    write_csv(out_dir / "summary.csv", summary_rows(summary), SUMMARY_FIELDS)
    typer.echo(f"Wrote {len(rows)} rows to {out_dir / 'results.csv'}")
```

- [ ] **Step 5: Run tests**

Run:

```bash
pytest tests/test_low_neff_benchmark.py -q
```

Expected: PASS.

- [ ] **Step 6: Run script help**

Run:

```bash
python benchmarks/bench_pseudomsa_low_neff.py --help
```

Expected: exit 0 and output includes `--candidate-n`.

- [ ] **Step 7: Commit temperature implementation**

Run:

```bash
git add benchmarks/bench_pseudomsa_low_neff.py src/ghostfold/benchmark/low_neff.py tests/test_low_neff_benchmark.py
git commit -m "Add temperature low-Neff benchmark sampling"
```

Expected: commit created.

---

### Task 4: Add Embedding-Walk Low-Neff Sampling

**Files:**
- Modify: `benchmarks/bench_pseudomsa_low_neff.py`
- Modify: `src/ghostfold/benchmark/low_neff.py`
- Test: `tests/test_low_neff_benchmark.py`

- [ ] **Step 1: Add test for embedding variants**

Append to `tests/test_low_neff_benchmark.py`:

```python
from ghostfold.benchmark.low_neff import generate_embedding_variants


def test_generate_embedding_variants_use_conservative_sigmas():
    variants = list(generate_embedding_variants(["default"]))

    assert variants[0].strategy == "embedding_walk_low_neff"
    assert variants[0].extra["sigma"] == 0.005
    assert variants[-1].extra["sigma"] == 0.10
    assert all(v.decode_conf["temperature"] == 0.5 for v in variants)
```

- [ ] **Step 2: Run test**

Run:

```bash
pytest tests/test_low_neff_benchmark.py::test_generate_embedding_variants_use_conservative_sigmas -q
```

Expected: PASS if Task 1 added `generate_embedding_variants`; otherwise implement it from Task 1 snippet now.

- [ ] **Step 3: Add strategy filter CLI option**

Modify `main()` signature in `benchmarks/bench_pseudomsa_low_neff.py`:

```python
    strategies: Annotated[
        str,
        typer.Option(
            "--strategies",
            help="Comma-separated: temperature_low_neff,embedding_walk_low_neff,all.",
        ),
    ] = "all",
```

Add validation in `main()`:

```python
    selected_strategies = (
        ["temperature_low_neff", "embedding_walk_low_neff"]
        if strategies.strip().lower() == "all"
        else [name.strip() for name in strategies.split(",") if name.strip()]
    )
    invalid = [
        name for name in selected_strategies
        if name not in {"temperature_low_neff", "embedding_walk_low_neff"}
    ]
    if invalid:
        raise typer.BadParameter(f"Unknown strategies: {invalid}")
```

- [ ] **Step 4: Add embedding walk generation helper**

Add to `benchmarks/bench_pseudomsa_low_neff.py`:

```python
from transformers.modeling_outputs import BaseModelOutput
```

Add helper:

```python
def _run_embedding_variant(
    protein_id: str,
    query_seq: str,
    variant,
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
        cache_implementation=variant.cache_implementation,
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
                    threedi,
                    tokenizer,
                    model,
                    device,
                    1,
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
        return _run_embedding_variant(
            protein_id,
            query_seq,
            variant,
            tokenizer,
            model,
            device,
            target_n,
            candidate_n,
            max(1, microbatch // 2),
            selection_mode,
            allow_duplicates,
            out_dir,
        )
    except Exception as exc:
        row["status"] = "skipped"
        row["error"] = str(exc)
        row["peak_vram_gb"] = round(_peak_vram_gb(), 3)
        return row

    gen_time = time.perf_counter() - t0
    filtered = filter_exact_length(candidates, len(query_seq), dedupe=not allow_duplicates)
    t1 = time.perf_counter()
    selected = _select_candidates(query_seq, filtered.sequences, target_n, selection_mode, allow_duplicates)
    selection_time = time.perf_counter() - t1
    neff = calculate_neff([query_seq] + selected) if selected else 0.0
    a3m_path = out_dir / "a3m" / protein_id / _safe_variant_dir(row) / "pstMSA.a3m"
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
```

- [ ] **Step 5: Wire selected variants in `main()`**

Replace:

```python
    variants = list(generate_temperature_variants(cache_modes))
```

with:

```python
    variants = []
    if "temperature_low_neff" in selected_strategies:
        variants.extend(generate_temperature_variants(cache_modes))
    if "embedding_walk_low_neff" in selected_strategies:
        variants.extend(generate_embedding_variants(cache_modes))
```

Replace call to `_run_temperature_variant(...)` inside loop with:

```python
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
```

- [ ] **Step 6: Run tests and help**

Run:

```bash
pytest tests/test_low_neff_benchmark.py -q
python benchmarks/bench_pseudomsa_low_neff.py --help
```

Expected: tests PASS; help includes `--strategies`.

- [ ] **Step 7: Commit embedding implementation**

Run:

```bash
git add benchmarks/bench_pseudomsa_low_neff.py src/ghostfold/benchmark/low_neff.py tests/test_low_neff_benchmark.py
git commit -m "Add embedding-walk low-Neff sampling"
```

Expected: commit created.

---

### Task 5: Add Optional ColabFold Scoring

**Files:**
- Modify: `benchmarks/bench_pseudomsa_low_neff.py`
- Modify: `src/ghostfold/benchmark/low_neff.py`
- Test: `tests/test_low_neff_benchmark.py`

- [ ] **Step 1: Add test for fold-aware summary thresholds**

Append to `tests/test_low_neff_benchmark.py`:

```python
def test_summarize_best_rows_applies_fold_thresholds():
    rows = [
        {
            "strategy": "low_quality",
            "target_n": 2,
            "selected_sequences": 2,
            "neff": 1.0,
            "peak_vram_gb": 7.0,
            "mean_plddt": 60.0,
            "tm_score": 0.7,
        },
        {
            "strategy": "high_quality",
            "target_n": 2,
            "selected_sequences": 2,
            "neff": 2.0,
            "peak_vram_gb": 7.0,
            "mean_plddt": 80.0,
            "tm_score": 0.8,
        },
    ]

    summary = summarize_best_rows(rows, min_plddt=70.0, min_tm_score=0.75)

    assert summary["fold_aware_best"]["strategy"] == "high_quality"
```

- [ ] **Step 2: Run test**

Run:

```bash
pytest tests/test_low_neff_benchmark.py::test_summarize_best_rows_applies_fold_thresholds -q
```

Expected: PASS if Task 1 summary logic present.

- [ ] **Step 3: Add fold helper to script**

Add imports:

```python
from ghostfold.benchmark.runner import _batch_fold_and_score, _find_ref_pdb
from rich.progress import Progress
```

Add helper:

```python
def _run_optional_folding(
    rows: list[dict],
    bench_dir: Path,
    out_dir: Path,
    colabfold_gpus: int,
) -> None:
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
```

- [ ] **Step 4: Call fold helper before summary**

In `main()`, before summary construction, add:

```python
    if fold:
        _run_optional_folding(rows, bench_dir, out_dir, colabfold_gpus)
        write_csv(out_dir / "results.csv", rows, RESULT_FIELDS)
```

- [ ] **Step 5: Run tests**

Run:

```bash
pytest tests/test_low_neff_benchmark.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit fold wiring**

Run:

```bash
git add benchmarks/bench_pseudomsa_low_neff.py tests/test_low_neff_benchmark.py
git commit -m "Add optional folding to low-Neff benchmark"
```

Expected: commit created.

---

### Task 6: Final Validation and Cleanup

**Files:**
- Modify only if validation finds issues: `benchmarks/bench_pseudomsa_low_neff.py`, `src/ghostfold/benchmark/low_neff.py`, `tests/test_low_neff_benchmark.py`

- [ ] **Step 1: Run focused tests**

Run:

```bash
pytest tests/test_low_neff_benchmark.py -q
```

Expected: all tests PASS.

- [ ] **Step 2: Run script help**

Run:

```bash
python benchmarks/bench_pseudomsa_low_neff.py --help
```

Expected: exit 0. Output includes `--candidate-n`, `--microbatch`, `--strategies`, `--cache-implementations`, and `--max-vram-gb`.

- [ ] **Step 3: Run Ruff on touched files**

Run:

```bash
ruff check benchmarks/bench_pseudomsa_low_neff.py src/ghostfold/benchmark/low_neff.py tests/test_low_neff_benchmark.py
```

Expected: PASS. If Ruff reports import ordering or line-length issues, edit only the reported files and rerun the command.

- [ ] **Step 4: Run full tests if runtime is reasonable**

Run:

```bash
pytest -q
```

Expected: PASS. If environment lacks optional runtime dependencies, record the exact failure and keep focused test/Ruff evidence.

- [ ] **Step 5: Commit validation fixes**

If any files changed during cleanup:

```bash
git add benchmarks/bench_pseudomsa_low_neff.py src/ghostfold/benchmark/low_neff.py tests/test_low_neff_benchmark.py
git commit -m "Polish low-Neff benchmark experiment"
```

Expected: commit created only if cleanup changed files.

---

## Self-Review Notes

- Spec coverage: plan covers new script, temperature low-Neff sampler, embedding-walk low-Neff sampler, VRAM measurement, cache implementations, candidate filtering, low-Neff selection, `results.csv`, `summary.csv`, A3M output, optional folding, and pure unit tests.
- Scope: one script plus one helper module. No production CLI/default behavior changes.
- Type consistency: `SamplingVariant`, row field names, and summary objective names match across tasks.
- Known implementation risk: Hugging Face cache options may not support this installed Transformers/T5 path. Plan handles this by row-level skip on exception and records `status`/`error`.
