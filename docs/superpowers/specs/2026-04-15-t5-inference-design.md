# T5 Inference Improvements — Design Spec

**Date:** 2026-04-15
**Status:** Approved
**Target:** GhostFold v1 (ProstT5, single RTX 4090 16GB)

## Problem

ProstT5 inference is the primary bottleneck in GhostFold. Three overlapping constraints:

- **Latency:** wall time per protein is high for interactive use
- **Throughput:** sequences/hour on batch jobs limited by sequential generate() calls
- **VRAM:** 16GB is tight for ProstT5 (~3B params) + batch overhead; limits effective batch size

## Goals

- Reduce latency and peak VRAM usage
- Enable larger effective batch sizes
- Maintain or improve MSA quality (primary metric: Neff)
- Provide a benchmarking harness to compare precision modes head-to-head

## Non-Goals

- Replacing ProstT5 with another model
- vLLM/TGI inference engine (deferred to v2)
- Multi-GPU inference

## Approach

Two parallel tracks:

**Track A — Kernel upgrades (no new deps)**
- Flash Attention 2 with SDPA fallback
- `torch.compile(mode="max-autotune")` for bf16/fp16

**Track B — Quantization backends (optional deps)**
- bitsandbytes INT8 and INT4 (NF4) precision modes
- `--precision` CLI flag
- Benchmark harness comparing all precisions on Neff, wall time, peak VRAM

## Architecture

```
Track A: Kernel upgrades
  src/ghostfold/msa/model.py     → FA2/SDPA selection + max-autotune compile

Track B: Quantization
  src/ghostfold/msa/model.py     → precision-aware load path
  src/ghostfold/cli/app.py       → --precision flag (msa/fold/run subcommands)
  pyproject.toml                 → [quant] optional dep group

Benchmark harness (new):
  benchmarks/bench_inference.py  → standalone script
```

## Component Details

### `src/ghostfold/msa/model.py`

`load_model()` gains a `precision` parameter (`Literal["bf16", "fp16", "int8", "int4"]`, default `"bf16"`).

**Precision → load config mapping:**

| precision | torch_dtype | BitsAndBytesConfig |
|-----------|-------------|-------------------|
| `bf16` | bfloat16 | none |
| `fp16` | float16 | none |
| `int8` | bfloat16 (compute) | `load_in_8bit=True` |
| `int4` | bfloat16 (compute) | `load_in_4bit=True, bnb_4bit_quant_type="nf4"` |

**Attention backend selection (all precisions):**
- FA2 available (`importlib.util.find_spec("flash_attn")`) → `attn_implementation="flash_attention_2"`
- FA2 not available → `attn_implementation="sdpa"` (always available in PyTorch 2.0+)
- Log one INFO line on fallback; no error

**torch.compile:**
- `mode="max-autotune"` for bf16/fp16 only
- Skipped for int8/int4 (bitsandbytes + compile conflict)
- Log DEBUG line when skipped

**Model cache key:** `"{model}:{device}:{precision}"` — allows different precisions to coexist in cache during benchmarking.

### `src/ghostfold/cli/app.py`

`--precision` option added to `msa`, `fold`, and `run` subcommands. Default `"bf16"`. Passed through `run_pipeline()` → `load_model()`.

### `pyproject.toml`

```toml
[project.optional-dependencies]
quant = ["bitsandbytes>=0.41.0"]
```

`flash-attn` intentionally excluded — requires matching CUDA toolkit, user installs manually.

### `benchmarks/bench_inference.py`

Standalone script. Usage:
```
python benchmarks/bench_inference.py --fasta <file> --precisions bf16,fp16,int8,int4 --runs 3
```

**Per-precision loop:**
1. `load_model(precision=p)` — record load time, peak VRAM after load
2. `generate_sequences_for_coverages_batched()` — record wall time, peak VRAM during gen
3. `filter_sequences()` — record raw → filtered count
4. `calculate_neff()` — primary quality metric
5. `torch.cuda.reset_peak_memory_stats()` — reset between runs
6. `del model; gc.collect(); torch.cuda.empty_cache()` — clean slate

**VRAM tracking:** `torch.cuda.max_memory_allocated()` before/after each stage.

**CSV output columns:**
```
precision, load_time_s, gen_time_s, peak_vram_gb, raw_seqs, filtered_seqs, filter_rate, neff
```

**Rich summary table** printed to terminal. Neff column highlighted; best value per column bolded.

**Multi-sequence FASTA:** per-sequence rows + aggregate mean row.

**`--runs N`** (default 3): averages timing across N runs. First run discarded (compile warmup for bf16/fp16).

## Error Handling

| Scenario | Behavior |
|----------|----------|
| `flash-attn` not installed | Silent fallback to sdpa, one INFO log line |
| `bitsandbytes` not installed, precision=int8/int4 | Clear error: `"precision='int8' requires bitsandbytes: pip install -e '.[quant]'"` |
| Precision fails to load in benchmark | Skip that row, print warning, continue remaining precisions |
| int8/int4 + torch.compile | Compile silently skipped, DEBUG log explains why |

## Testing

**`tests/test_model_loading.py`** (unit, runs in CI):
- `load_model(precision="bf16")` succeeds — regression guard
- FA2 unavailable → falls back to sdpa without error
- `precision="int8"` without bitsandbytes → raises `ImportError` with pip hint
- Cache key includes precision

**`tests/test_inference_precision.py`** (integration, `@pytest.mark.slow`, skipped in CI):
- Runs short synthetic sequence (~15 AA) through full generate → filter → neff at bf16 and fp16
- Asserts: `neff > 0`, `filtered_seqs > 0`, no exceptions
- int8/int4: `pytest.importorskip("bitsandbytes")`

**CI:** unchanged. `pytest -m "not slow"` keeps fast path.

## Quality Metric

**Primary:** Neff — effective number of sequences in the filtered MSA. Must not degrade meaningfully vs. bf16 baseline for int8/int4 to be acceptable.

**Secondary:** filter_rate (raw → filtered ratio) — catches precision modes generating low-entropy junk. Wall time and peak VRAM are performance metrics only.

## Dependencies

| Package | Required | Install |
|---------|----------|---------|
| `bitsandbytes>=0.41.0` | For int8/int4 | `pip install -e ".[quant]"` |
| `flash-attn` | Optional | Manual (CUDA toolkit match required) |
