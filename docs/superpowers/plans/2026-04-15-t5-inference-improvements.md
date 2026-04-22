# T5 Inference Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve ProstT5 inference speed, VRAM usage, and throughput on a single RTX 4090 16GB via Flash Attention 2 / SDPA kernel selection, `max-autotune` compile, and bitsandbytes INT8/INT4 quantization — with a benchmark harness that measures Neff (primary), wall time, and peak VRAM across all precision modes.

**Architecture:** `_load_model()` in `pipeline.py` is the single choke point — add `precision` param there, resolve attention backend and compile strategy at load time, update cache key to include precision. Thread `precision` through `run_pipeline()` and the CLI. New standalone `benchmarks/bench_inference.py` imports these functions directly and runs each precision in isolation.

**Tech Stack:** PyTorch 2.0+ (SDPA built-in), `flash-attn` (optional, manual install), `bitsandbytes>=0.41.0` (optional, `pip install -e ".[quant]"`), Rich (already a dep), Typer (already a dep)

---

## File Map

| Status | File | Change |
|--------|------|--------|
| Modify | `src/ghostfold/core/pipeline.py` | `_load_model()` gains `precision` param; FA2/SDPA selection; `max-autotune` compile; cache key update; `run_pipeline()` gains `precision` param |
| Modify | `src/ghostfold/cli/msa.py` | Add `--precision` option, pass to `run_pipeline()` |
| Modify | `src/ghostfold/cli/run.py` | Add `--precision` option, pass through to MSA stage |
| Modify | `pyproject.toml` | Add `[quant]` optional dep group |
| Create | `benchmarks/bench_inference.py` | Standalone benchmark script |
| Create | `tests/test_model_loading.py` | Unit tests for `_load_model()` changes |
| Create | `tests/test_inference_precision.py` | Slow integration tests for full pipeline at bf16/fp16 |

---

## Task 1: Add `[quant]` optional dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add quant optional dependency group**

In `pyproject.toml`, after the existing `[project.optional-dependencies]` dev block, add:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov",
    "ruff",
]
quant = ["bitsandbytes>=0.41.0"]
```

- [ ] **Step 2: Verify editable install still works**

```bash
pip install -e ".[dev]"
```

Expected: no errors, existing `ghostfold --help` still works.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "build: add [quant] optional dep group for bitsandbytes"
```

---

## Task 2: Refactor `_load_model()` — precision param + attention backend + compile

**Files:**
- Modify: `src/ghostfold/core/pipeline.py:43-74`

- [ ] **Step 1: Write failing unit test for precision param and cache key**

Create `tests/test_model_loading.py`:

```python
"""Unit tests for _load_model precision parameter and cache key logic."""
import importlib.util
from unittest.mock import MagicMock, patch
import pytest


def _make_mock_model(supports_bf16=True):
    """Return a mock (tokenizer, model) pair."""
    tokenizer = MagicMock()
    model = MagicMock()
    model.to = MagicMock(return_value=model)
    model.half = MagicMock(return_value=model)
    model.eval = MagicMock(return_value=model)
    return tokenizer, model


@pytest.fixture(autouse=True)
def clear_model_cache():
    """Clear the module-level model cache before each test."""
    from ghostfold.core import pipeline
    pipeline._MODEL_CACHE.clear()
    yield
    pipeline._MODEL_CACHE.clear()


@patch("ghostfold.core.pipeline.AutoModelForSeq2SeqLM")
@patch("ghostfold.core.pipeline.T5Tokenizer")
def test_cache_key_includes_precision(mock_tokenizer_cls, mock_model_cls):
    """Cache key must be '{model}:{device}:{precision}' to allow concurrent precisions."""
    import torch
    from ghostfold.core import pipeline

    mock_tokenizer_cls.from_pretrained.return_value = MagicMock()
    mock_model = MagicMock()
    mock_model.to = MagicMock(return_value=mock_model)
    mock_model.half = MagicMock(return_value=mock_model)
    mock_model.eval = MagicMock(return_value=mock_model)
    mock_model_cls.from_pretrained.return_value = mock_model

    device = torch.device("cpu")
    pipeline._load_model(device, precision="bf16")
    pipeline._load_model(device, precision="fp16")

    assert len(pipeline._MODEL_CACHE) == 2
    keys = list(pipeline._MODEL_CACHE.keys())
    assert any("bf16" in k for k in keys)
    assert any("fp16" in k for k in keys)


@patch("ghostfold.core.pipeline.AutoModelForSeq2SeqLM")
@patch("ghostfold.core.pipeline.T5Tokenizer")
def test_load_model_bf16_default_succeeds(mock_tokenizer_cls, mock_model_cls):
    """Default precision=bf16 loads without error (regression guard)."""
    import torch
    from ghostfold.core import pipeline

    mock_tokenizer_cls.from_pretrained.return_value = MagicMock()
    mock_model = MagicMock()
    mock_model.to = MagicMock(return_value=mock_model)
    mock_model.half = MagicMock(return_value=mock_model)
    mock_model.eval = MagicMock(return_value=mock_model)
    mock_model_cls.from_pretrained.return_value = mock_model

    device = torch.device("cpu")
    tokenizer, model = pipeline._load_model(device, precision="bf16")
    assert tokenizer is not None
    assert model is not None


def test_invalid_precision_raises():
    """Unsupported precision value must raise ValueError immediately."""
    import torch
    from ghostfold.core import pipeline

    device = torch.device("cpu")
    with pytest.raises(ValueError, match="precision"):
        pipeline._load_model(device, precision="fp8")


def test_int8_without_bitsandbytes_raises():
    """precision='int8' without bitsandbytes installed must raise ImportError with pip hint."""
    import sys
    import torch
    from ghostfold.core import pipeline

    # Simulate bitsandbytes not installed
    with patch.dict(sys.modules, {"bitsandbytes": None}):
        with pytest.raises(ImportError, match="bitsandbytes"):
            pipeline._load_model(torch.device("cpu"), precision="int8")


def test_int4_without_bitsandbytes_raises():
    """precision='int4' without bitsandbytes installed must raise ImportError with pip hint."""
    import sys
    import torch
    from ghostfold.core import pipeline

    with patch.dict(sys.modules, {"bitsandbytes": None}):
        with pytest.raises(ImportError, match="bitsandbytes"):
            pipeline._load_model(torch.device("cpu"), precision="int4")
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_model_loading.py -v
```

Expected: FAIL — `_load_model` doesn't accept `precision` param yet.

- [ ] **Step 3: Implement `_load_model()` changes in `pipeline.py`**

Replace the existing `_load_model` function (lines 43–74) and update the `_MODEL_CACHE` comment on line 40:

```python
import importlib.util

# Module-level model cache: key is "{model_name}:{device_type}:{precision}"
_MODEL_CACHE: Dict[str, Tuple[Any, Any]] = {}


def _load_model(
    device: Any,
    model_name: str = MODEL_NAME,
    precision: str = "bf16",
) -> Tuple[Any, Any]:
    """Return (tokenizer, model) for the given device and precision, loading only on first call.

    Precision options:
      bf16  — bfloat16 (default, Ampere+)
      fp16  — float16
      int8  — bitsandbytes 8-bit quantization (requires pip install -e '.[quant]')
      int4  — bitsandbytes NF4 4-bit quantization (requires pip install -e '.[quant]')

    Attention backend: Flash Attention 2 if available, else SDPA (PyTorch built-in).
    torch.compile: max-autotune for bf16/fp16 only (skipped for quantized models).
    """
    import torch

    _VALID_PRECISIONS = ("bf16", "fp16", "int8", "int4")
    if precision not in _VALID_PRECISIONS:
        raise ValueError(
            f"Invalid precision '{precision}'. Must be one of: {_VALID_PRECISIONS}"
        )

    key = f"{model_name}:{device.type}:{precision}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    logger.info(f"Loading T5 model '{model_name}' on {device.type.upper()} with precision={precision}...")

    # Resolve attention backend
    if importlib.util.find_spec("flash_attn") is not None:
        attn_impl = "flash_attention_2"
        logger.info("Flash Attention 2 detected — using flash_attention_2 backend.")
    else:
        attn_impl = "sdpa"
        logger.info("flash-attn not found — falling back to SDPA backend.")

    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False, legacy=True)

    if precision in ("int8", "int4"):
        # Validate bitsandbytes is available before attempting to load
        if importlib.util.find_spec("bitsandbytes") is None:
            raise ImportError(
                f"precision='{precision}' requires bitsandbytes. "
                "Install with: pip install -e '.[quant]'"
            )
        from transformers import BitsAndBytesConfig
        if precision == "int8":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        else:  # int4
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            attn_implementation=attn_impl,
            device_map="auto",
        )
        logger.info(f"Model loaded with {precision} quantization (bitsandbytes).")
        logger.debug("torch.compile skipped for quantized models (bitsandbytes incompatibility).")
    else:
        torch_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            attn_implementation=attn_impl,
        ).to(device)
        logger.info(f"Model loaded with {precision} precision.")
        try:
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            model = torch.compile(model, mode="max-autotune")
            logger.info("Model compiled with torch.compile (max-autotune).")
        except Exception as e:
            logger.warning(f"torch.compile unavailable, using eager mode: {e}")

    model.eval()
    logger.info("Model and tokenizer loaded successfully.")
    _MODEL_CACHE[key] = (tokenizer, model)
    return _MODEL_CACHE[key]
```

Also update `run_pipeline()` signature (line 276) to accept and pass `precision`:

```python
def run_pipeline(
    project: str,
    fasta_path: str,
    config: dict,
    coverage_list: List[float],
    evolve_msa: bool,
    mutation_rates_str: str,
    sample_percentage: float,
    plot_msa: bool,
    plot_coevolution: bool,
    hex_colors: List[str] = MSA_COLORS,
    num_runs: int = 1,
    recursive: bool = False,
    show_progress: bool = True,
    precision: str = "bf16",
) -> None:
```

And update the `_load_model` call inside `run_pipeline()` (currently line 301):

```python
    try:
        tokenizer, model = _load_model(device, precision=precision)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_model_loading.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ghostfold/core/pipeline.py tests/test_model_loading.py
git commit -m "feat(pipeline): add precision param to _load_model with FA2/SDPA and max-autotune"
```

---

## Task 3: Add `--precision` flag to `msa` and `run` CLI subcommands

**Files:**
- Modify: `src/ghostfold/cli/msa.py`
- Modify: `src/ghostfold/cli/run.py`

- [ ] **Step 1: Write failing CLI tests**

Add to `tests/test_cli.py` (or create if it doesn't have these):

```python
from typer.testing import CliRunner
from ghostfold.cli.app import app
from unittest.mock import patch

runner = CliRunner()


def test_msa_accepts_precision_flag():
    """--precision flag must be accepted by the msa subcommand."""
    with patch("ghostfold.core.pipeline.run_pipeline") as mock_run:
        result = runner.invoke(app, [
            "msa",
            "--project-name", "test_proj",
            "--fasta-path", "tests/fixtures/test.fasta",
            "--precision", "fp16",
        ])
    # If flag is unrecognised typer exits with code 2
    assert result.exit_code != 2, f"Unrecognised flag. Output:\n{result.output}"


def test_msa_precision_default_is_bf16():
    """msa subcommand must pass precision='bf16' to run_pipeline by default."""
    with patch("ghostfold.cli.msa.run_pipeline") as mock_run, \
         patch("ghostfold.cli.msa.setup_logging"), \
         patch("ghostfold.cli.msa.load_config", return_value={}), \
         patch("ghostfold.cli.msa.get_console"):
        runner.invoke(app, [
            "msa",
            "--project-name", "test_proj",
            "--fasta-path", "tests/fixtures/test.fasta",
        ])
        call_kwargs = mock_run.call_args.kwargs if mock_run.called else {}
        assert call_kwargs.get("precision", "bf16") == "bf16"


def test_msa_precision_invalid_rejected():
    """--precision with invalid value must exit with non-zero code."""
    result = runner.invoke(app, [
        "msa",
        "--project-name", "test_proj",
        "--fasta-path", "tests/fixtures/test.fasta",
        "--precision", "fp32",
    ])
    assert result.exit_code != 0
```

You will need a minimal fixture FASTA. Create `tests/fixtures/test.fasta` if it doesn't exist:

```
>test_seq
ACDEFGHIKLMNPQRSTVWY
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_cli.py::test_msa_accepts_precision_flag tests/test_cli.py::test_msa_precision_default_is_bf16 -v
```

Expected: FAIL — `--precision` not yet a valid flag.

- [ ] **Step 3: Add `--precision` to `msa.py`**

Replace `src/ghostfold/cli/msa.py` with:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import typer

app = typer.Typer(help="Generate pseudoMSAs from single sequences.")

_VALID_PRECISIONS = ["bf16", "fp16", "int8", "int4"]


@app.callback(invoke_without_command=True)
def msa(
    project_name: str = typer.Option(..., "--project-name", help="Name of the main project directory."),
    fasta_path: Path = typer.Option(..., "--fasta-path", exists=True, help="Path to a FASTA file or directory containing FASTA files."),
    recursive: bool = typer.Option(False, "--recursive", help="Recursively search directories for FASTA files."),
    config: Optional[Path] = typer.Option(None, "--config", exists=True, help="Path to YAML config (overrides defaults)."),
    coverage: Optional[List[float]] = typer.Option(None, "--coverage", help="Coverage values (can be specified multiple times). Default: 1.0"),
    num_runs: int = typer.Option(1, "--num-runs", help="Number of independent runs per query sequence."),
    plot_msa_coverage: bool = typer.Option(False, "--plot-msa-coverage", help="Generate MSA coverage plots."),
    no_coevolution_maps: bool = typer.Option(False, "--no-coevolution-maps", help="Do not generate coevolution maps."),
    evolve_msa: bool = typer.Option(False, "--evolve-msa", help="Enable MSA evolution using the mutator module."),
    mutation_rates: str = typer.Option(
        '{"MEGABLAST": 5, "PAM250": 20, "BLOSUM62": 10}',
        "--mutation-rates",
        help="JSON string for mutation rates.",
    ),
    sample_percentage: float = typer.Option(1.0, "--sample-percentage", help="Percentage of sequences to sample for evolution."),
    precision: str = typer.Option(
        "bf16",
        "--precision",
        help="Model precision: bf16, fp16, int8, int4. int8/int4 require pip install -e '.[quant]'.",
    ),
) -> None:
    """Generate pseudoMSAs from single sequences using ProstT5."""
    from ghostfold.core.logging import setup_logging, get_console
    from ghostfold.core.config import load_config
    from ghostfold.core.pipeline import run_pipeline

    if precision not in _VALID_PRECISIONS:
        typer.echo(f"Error: --precision must be one of {_VALID_PRECISIONS}. Got: '{precision}'", err=True)
        raise typer.Exit(code=1)

    log_path = setup_logging(project_name)
    get_console().print(f"[dim]Log file: {log_path}[/dim]")

    cfg = load_config(config)
    coverage_list = list(coverage) if coverage else [1.0]

    run_pipeline(
        project=project_name,
        fasta_path=str(fasta_path),
        config=cfg,
        coverage_list=coverage_list,
        evolve_msa=evolve_msa,
        mutation_rates_str=mutation_rates,
        sample_percentage=sample_percentage,
        plot_msa=plot_msa_coverage,
        plot_coevolution=not no_coevolution_maps,
        num_runs=num_runs,
        recursive=recursive,
        precision=precision,
    )
```

- [ ] **Step 4: Add `--precision` to `run.py`**

In `src/ghostfold/cli/run.py`, add the precision option to the `run()` function signature (after `localcolabfold_dir`):

```python
    precision: str = typer.Option(
        "bf16",
        "--precision",
        help="Model precision for MSA generation: bf16, fp16, int8, int4. int8/int4 require pip install -e '.[quant]'.",
    ),
```

And in the function body, add the precision validation before `log_path = setup_logging(...)`:

```python
    _VALID_PRECISIONS = ["bf16", "fp16", "int8", "int4"]
    if precision not in _VALID_PRECISIONS:
        typer.echo(f"Error: --precision must be one of {_VALID_PRECISIONS}. Got: '{precision}'", err=True)
        raise typer.Exit(code=1)
```

And pass precision to `run_parallel_msa()`. First check its signature:

In `run.py` the call to `run_parallel_msa` currently does not pass precision. Update it:

```python
    run_parallel_msa(
        project_name=project_name,
        fasta_path=str(fasta_path),
        num_gpus=gpus,
        config_path=str(config) if config else None,
        log_file_path=str(log_path),
        recursive=recursive,
        precision=precision,
    )
```

Then check `src/ghostfold/core/gpu.py` to confirm `run_parallel_msa` passes kwargs through to `run_pipeline`. Read that file and add `precision` to its signature if needed (see Task 4).

- [ ] **Step 5: Run CLI tests**

```bash
pytest tests/test_cli.py -v
```

Expected: all tests PASS including new precision flag tests.

- [ ] **Step 6: Smoke-test CLI help**

```bash
ghostfold msa --help
```

Expected: `--precision` listed in options with description.

- [ ] **Step 7: Commit**

```bash
git add src/ghostfold/cli/msa.py src/ghostfold/cli/run.py tests/test_cli.py tests/fixtures/test.fasta
git commit -m "feat(cli): add --precision flag to msa and run subcommands"
```

---

## Task 4: Thread `precision` through `run_parallel_msa` in `gpu.py`

**Files:**
- Modify: `src/ghostfold/core/gpu.py`

- [ ] **Step 1: Read `gpu.py` to understand current signature**

```bash
cat src/ghostfold/core/gpu.py
```

Find `run_parallel_msa` — it likely calls `run_pipeline` in worker processes. Note the exact call site.

- [ ] **Step 2: Add `precision` param to `run_parallel_msa` and its internal `run_pipeline` call**

In `src/ghostfold/core/gpu.py`, find the `run_parallel_msa` function. Add `precision: str = "bf16"` to its signature. Find where it calls `run_pipeline` (directly or via a worker function) and add `precision=precision` to that call.

The exact edit depends on the current code. Pattern to find and update:

```python
# Before (find this pattern):
run_pipeline(
    project=...,
    fasta_path=...,
    ...
)

# After (add precision kwarg):
run_pipeline(
    project=...,
    fasta_path=...,
    ...
    precision=precision,
)
```

- [ ] **Step 3: Run full test suite to catch any breakage**

```bash
pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add src/ghostfold/core/gpu.py
git commit -m "feat(gpu): thread precision param through run_parallel_msa to run_pipeline"
```

---

## Task 5: Create benchmark harness

**Files:**
- Create: `benchmarks/bench_inference.py`

- [ ] **Step 1: Create the benchmark script**

Create `benchmarks/bench_inference.py`:

```python
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
    """Run full generate → filter → neff for one precision. Returns metrics dict."""
    import torch
    from ghostfold.core.pipeline import _load_model
    from ghostfold.msa.generation import generate_sequences_for_coverages_batched
    from ghostfold.msa.filters import filter_sequences
    from ghostfold.msa.neff import calculate_neff
    from ghostfold.io.fasta import read_fasta_from_path
    from ghostfold.core.pipeline import generate_decoding_configs

    records = read_fasta_from_path(fasta_path)
    if not records:
        raise ValueError(f"No sequences found in {fasta_path}")

    # Use first sequence for per-run metrics; caller averages across runs
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

    import tempfile, os
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
        # Fallback: plain text
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
            # Clear cache so each run loads fresh (important for int8/int4 VRAM measurement)
            _MODEL_CACHE.clear()
            _free_gpu()
            gc.collect()

            try:
                metrics = _run_one(args.fasta, precision, config, device)
                if run_idx == 0:
                    print(f"  [warmup run discarded]")
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

        # Average across timed runs
        avg: dict = {"precision": precision}
        for key in ["load_time_s", "gen_time_s", "peak_vram_gb",
                    "raw_seqs", "filtered_seqs", "filter_rate", "neff"]:
            vals = [r[key] for r in run_results if key in r]
            avg[key] = round(statistics.mean(vals), 4) if vals else "N/A"

        summary_rows.append(avg)

    if not summary_rows:
        print("No results collected. Check errors above.")
        sys.exit(1)

    # Write CSV
    fieldnames = ["precision", "load_time_s", "gen_time_s", "peak_vram_gb",
                  "raw_seqs", "filtered_seqs", "filter_rate", "neff"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nResults written to: {output_path}")

    # Print Rich table
    _print_table(summary_rows, precisions)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script imports without error**

```bash
python -c "import benchmarks.bench_inference" 2>&1 || python benchmarks/bench_inference.py --help
```

Expected: help text printed, no ImportError.

- [ ] **Step 3: Commit**

```bash
git add benchmarks/bench_inference.py
git commit -m "feat(benchmarks): add bench_inference.py for precision vs Neff benchmarking"
```

---

## Task 6: Add slow integration tests for bf16/fp16 precision

**Files:**
- Create: `tests/test_inference_precision.py`

- [ ] **Step 1: Create `tests/fixtures/test.fasta` if not already present**

```
>test_protein
ACDEFGHIKLMNPQRSTVWYACDEFGHIKLM
```

(31 AA — short enough for fast CPU inference in tests)

- [ ] **Step 2: Create the integration test file**

Create `tests/test_inference_precision.py`:

```python
"""Integration tests for T5 inference at different precisions.

Marked @pytest.mark.slow — skipped in normal CI.
Run with: pytest tests/test_inference_precision.py -v -m slow
"""
import pytest
import tempfile
import os


@pytest.fixture(scope="module")
def short_fasta(tmp_path_factory):
    """Write a short protein FASTA for inference tests."""
    d = tmp_path_factory.mktemp("fixtures")
    fasta = d / "short.fasta"
    fasta.write_text(">test_protein\nACDEFGHIKLMNPQRSTVWYACDEFGHIKLM\n")
    return str(fasta)


@pytest.fixture(autouse=True)
def clear_model_cache():
    from ghostfold.core import pipeline
    pipeline._MODEL_CACHE.clear()
    yield
    pipeline._MODEL_CACHE.clear()


@pytest.mark.slow
@pytest.mark.parametrize("precision", ["bf16", "fp16"])
def test_full_pipeline_neff_positive(short_fasta, precision):
    """Full generate → filter → neff pipeline must produce neff > 0 at bf16 and fp16."""
    import torch
    from ghostfold.core.pipeline import _load_model, generate_decoding_configs
    from ghostfold.msa.generation import generate_sequences_for_coverages_batched
    from ghostfold.msa.filters import filter_sequences
    from ghostfold.msa.neff import calculate_neff
    from ghostfold.io.fasta import read_fasta_from_path
    from ghostfold.core.config import load_config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(None)

    tokenizer, model = _load_model(device, precision=precision)
    assert model is not None
    assert tokenizer is not None

    records = read_fasta_from_path(short_fasta)
    query_seq = str(records[0].seq)
    full_len = len(query_seq)

    decoding_configs = generate_decoding_configs(config.get("decoding_params", {}))

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_seqs = generate_sequences_for_coverages_batched(
            query_seq=query_seq,
            full_len=full_len,
            decoding_configs=decoding_configs,
            num_return_sequences=config.get("num_return_sequences", 5),
            multiplier=config.get("multiplier", 1),
            coverage_list=[1.0],
            model=model,
            tokenizer=tokenizer,
            device=device,
            project_dir=tmpdir,
            inference_batch_size=config.get("inference_batch_size", 4),
        )

    all_seqs = [query_seq] + raw_seqs
    filtered = filter_sequences(all_seqs, full_len)
    assert len(filtered) > 0, f"No sequences passed filter at precision={precision}"

    neff = calculate_neff(filtered)
    assert neff > 0.0, f"Neff=0 at precision={precision}; filtered={len(filtered)} seqs"


@pytest.mark.slow
def test_int8_requires_bitsandbytes_or_skips():
    """int8 precision: passes if bitsandbytes installed, skips cleanly if not."""
    bitsandbytes = pytest.importorskip(
        "bitsandbytes",
        reason="bitsandbytes not installed; skipping int8 test (pip install -e '.[quant]')"
    )
    import torch
    from ghostfold.core.pipeline import _load_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = _load_model(device, precision="int8")
    assert model is not None


@pytest.mark.slow
def test_int4_requires_bitsandbytes_or_skips():
    """int4 precision: passes if bitsandbytes installed, skips cleanly if not."""
    bitsandbytes = pytest.importorskip(
        "bitsandbytes",
        reason="bitsandbytes not installed; skipping int4 test (pip install -e '.[quant]')"
    )
    import torch
    from ghostfold.core.pipeline import _load_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = _load_model(device, precision="int4")
    assert model is not None
```

- [ ] **Step 3: Verify unit tests still pass (slow tests skipped)**

```bash
pytest tests/ -v -m "not slow"
```

Expected: all unit tests PASS, slow integration tests skipped.

- [ ] **Step 4: Commit**

```bash
git add tests/test_inference_precision.py tests/fixtures/test.fasta
git commit -m "test: add slow integration tests for bf16/fp16/int8/int4 precision pipeline"
```

---

## Task 7: Final verification

- [ ] **Step 1: Full fast test suite**

```bash
pytest tests/ -v -m "not slow"
```

Expected: all tests PASS, 0 failures.

- [ ] **Step 2: Lint**

```bash
ruff check src tests benchmarks
```

Expected: no errors.

- [ ] **Step 3: Verify CLI help shows `--precision`**

```bash
ghostfold msa --help
ghostfold run --help
```

Expected: `--precision` option listed in both.

- [ ] **Step 4: Smoke-test benchmark script help**

```bash
python benchmarks/bench_inference.py --help
```

Expected: help printed, `--fasta`, `--precisions`, `--runs`, `--output` all listed.

- [ ] **Step 5: Commit**

```bash
git add .
git commit -m "chore: final lint and verification pass for T5 inference improvements"
```
