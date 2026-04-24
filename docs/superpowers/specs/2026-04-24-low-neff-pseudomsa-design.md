# Low-Neff PseudoMSA Sampling Benchmark Design

## Goal

Create an experimental benchmark script that explores ProstT5 sampling settings for generating many pseudoMSA candidates while keeping VRAM low and optimizing for lower GhostFold Neff. In this project, lower measured Neff is treated as better because benchmark results show lower Neff MSAs can produce better AF2 models.

The script will focus only on descendants of the current `temperature_sweep` and `embedding_walk_full` approaches in `benchmarks/bench_pseudomsa.py`.

## Non-Goals

This experiment will not change the production GhostFold CLI, default config, or existing benchmark script. It will not include encoder-only, CNN, round-trip, 3Di perturbation, or baseline methods. It will not train or fine-tune models.

## Script

Add `benchmarks/bench_pseudomsa_low_neff.py`.

Example:

```bash
python benchmarks/bench_pseudomsa_low_neff.py \
  --bench-dir bench \
  --out-dir bench_low_neff \
  --proteins 6LF2_B \
  --target-n 128 \
  --candidate-n 1024 \
  --microbatch 16 \
  --precision bf16 \
  --device cuda \
  --max-vram-gb 8
```

Core options:

- `--bench-dir`: benchmark input directory with `fasta/` and optional `pdb/`.
- `--out-dir`: output directory.
- `--proteins`: optional comma-separated protein IDs.
- `--target-n`: selected output sequence count for fixed-count comparison.
- `--candidate-n`: desired raw valid candidate pool size per variant.
- `--microbatch`: generation chunk size used to bound peak VRAM.
- `--device`: `cuda` or `cpu`.
- `--precision`: passed to existing `_load_model`.
- `--max-vram-gb`: optional VRAM budget for the VRAM-aware objective.
- `--fold/--no-fold`: optional ColabFold scoring.
- `--colabfold-gpus`: number of GPUs for ColabFold if folding.

## Sampling Families

### Low-Neff Temperature Sampling

Start from the current `temperature_sweep` flow:

1. Generate one or more AA-to-3Di seed sequences from the query.
2. Decode fold-to-AA candidates from each 3Di seed.
3. Sweep conservative decoding configs intended to produce close variants.
4. Filter by exact query length.
5. Select the lowest-Neff subset of size `target_n`.

Candidate decoding grid:

- Low and moderate temperatures, for example `0.3`, `0.5`, `0.7`, `0.9`.
- Small `top_k`, for example `1`, `3`, `5`, `10`.
- Conservative `top_p`, for example `0.70`, `0.80`, `0.90`.
- Optional modern truncation controls where supported by installed Transformers: `min_p`, `typical_p`, `epsilon_cutoff`, `eta_cutoff`, `top_h`.
- Optional `repetition_penalty`.

The script will validate each candidate `GenerationConfig` by constructing it before model inference. Unsupported generation parameters will skip that config with a warning that names the parameter and variant.

### Low-Neff Embedding Walk

Use `embedding_walk_full` mechanics, but run in memory-bounded chunks:

1. Tokenize query once.
2. For each small sigma and seed, perturb encoder hidden states through layer hooks.
3. Decode a small microbatch from the perturbed encoder output.
4. Backtranslate 3Di to AA in small chunks.
5. Online filter by length.
6. Stop when `candidate_n` valid candidates are collected or configs are exhausted.

Default sigmas will bias conservative local moves: `0.005`, `0.01`, `0.02`, `0.03`, `0.05`, `0.07`, `0.10`.

## VRAM Controls

Each sampler variant will run with bounded microbatches. Peak VRAM will be measured with `torch.cuda.reset_peak_memory_stats()` and `torch.cuda.max_memory_allocated()`.

The script will expose Hugging Face generation cache choices. Each option will be validated before inference:

- default cache behavior
- `offloaded`
- `offloaded_static`
- `quantized`

Cache implementations may trade speed for memory. Unsupported cache options will be skipped with a warning that names the cache implementation and exception.

## Candidate Selection

The script will support two selection modes:

- `lowest_neff`: greedily build a size-`target_n` subset that minimizes calculated Neff.
- `first_valid`: baseline selector for debugging and speed comparison.

The first implementation can use a pragmatic greedy heuristic:

1. Keep query sequence fixed as the first sequence for Neff calculation.
2. Deduplicate candidates unless `--allow-duplicates` is set.
3. Start from candidates most similar to the query.
4. Iteratively add the candidate that gives the lowest resulting Neff among a sampled candidate window.

This avoids exhaustive subset search, which is not practical for 1024+ candidates.

## Metrics

Each run row in `results.csv` will include:

- `protein_id`
- `strategy`
- `variant_param`
- `cache_implementation`
- `target_n`
- `candidate_n_requested`
- `raw_candidates`
- `valid_candidates`
- `selected_sequences`
- `neff`
- `gen_time_s`
- `peak_vram_gb`
- `selection_time_s`
- `ptm`
- `mean_plddt`
- `rmsd`
- `tm_score`

Write selected MSAs under:

```text
<out_dir>/a3m/<protein_id>/<strategy_variant>/pstMSA.a3m
```

Write `summary.csv` with three objective views:

- `fixed_count_best`: lowest Neff with exactly `target_n` selected sequences.
- `fold_aware_best`: lowest Neff among rows meeting fold-quality thresholds when `--fold` is enabled.
- `vram_aware_best`: lowest Neff among rows at or below `--max-vram-gb`.

## Fold-Aware Objective

If `--fold` is enabled, use the existing benchmark runner's ColabFold staging/parsing path unless that path cannot be imported. The fold-aware summary will only rank rows that have valid fold metrics.

Default thresholds:

- If `--fold-baseline single_sequence` is provided, generate and fold a single-sequence A3M first, then require `mean_plddt >= baseline_mean_plddt`.
- If `--min-plddt` is provided, require `mean_plddt >= --min-plddt`.
- `tm_score` must be at least the configured minimum if `--min-tm-score` is provided.
- If no threshold is provided and no baseline exists, report fold metrics without filtering.

## Error Handling

Generation config failures will mark the row as skipped and continue. CUDA OOM will reduce microbatch once by half, retry, then skip if it still fails. Length-filter failures will produce a row with zero selected sequences and `neff=0`.

## Tests

Add focused tests for pure logic only:

- Config grid construction skips unsupported knobs cleanly.
- Candidate filtering keeps exact-length sequences.
- Lowest-Neff selector returns exactly `target_n` when enough candidates exist.
- Summary ranking chooses lowest Neff for fixed-count and VRAM-aware objectives.

Model inference and ColabFold runs will remain benchmark-only, not unit tests.

## Acceptance Criteria

- `python benchmarks/bench_pseudomsa_low_neff.py --help` works.
- Script can run one protein with `--no-fold` and write `results.csv`, `summary.csv`, and selected A3M files.
- Existing tests still pass.
- Ruff passes on new Python files.
