# Multimer MSA Cartesian Product Pairing

**Date:** 2026-04-30  
**Status:** Approved

## Problem

Current `write_multimer_pst_msa` writes per-chain sequences as gap-padded unpaired blocks. Gap-padded rows give AF2-Multimer weak coevolutionary signal and cause unstable structure prediction inference.

## Solution

Replace gap-padded unpaired blocks with a Cartesian-product paired block. Per-chain MSAs are combined via Cartesian product to produce fully-concatenated paired sequences with no gap padding. The original concat_seqs (full-complex ProstT5 translations) are retained as they carry the strongest signal.

## Architecture

### New module: `src/ghostfold/msa/pairing.py`

Single public function:

```python
def build_paired_msa(
    per_chain_seqs: List[List[str]],
    n_subsets: int = 20,
    subset_size: int = 175,
    top_k: int = 5,
    neff_threshold: float = 0.8,
) -> List[str]:
```

**Algorithm:**

1. `itertools.product(*per_chain_seqs)` → iterator (never materialized)
2. Reservoir-sample `n_subsets` subsets of `subset_size` seqs from the iterator
3. `calculate_neff()` on each subset (reuses `msa/neff.py`)
4. Rank by Neff descending, take top `top_k`
5. Merge top-k subsets + `deduplicate()` (reuses `msa/filters.py`)
6. Return `List[str]` of fully-concatenated paired sequences

### Modified: `pipeline.py::write_multimer_pst_msa`

- Call `build_paired_msa(per_chain_seqs)` to get paired sequences
- Paired block written = `concat_seqs + paired_seqs`
- Remove all gap-padded unpaired block writes
- Heterooligomer and homooligomer paths both use this paired block

### Unchanged

- `generate_multimer_sequences` — still returns `(concat_seqs, per_chain_seqs)`
- `process_multimer_run` — no change
- `msa/neff.py`, `msa/filters.py` — reused as-is

## Data Flow

```
generate_multimer_sequences()
  → concat_seqs: List[str]          # full-complex ProstT5 translations
  → per_chain_seqs: List[List[str]] # per-chain ProstT5 translations

build_paired_msa(per_chain_seqs)
  → paired_seqs: List[str]          # Cartesian-product paired, Neff-selected, deduped

write_multimer_pst_msa()
  paired block = concat_seqs + paired_seqs
  NO unpaired/gap-padded blocks
```

## Edge Cases

| Condition | Behavior |
|-----------|----------|
| Single chain | Cartesian product = that list; return as-is |
| Any chain empty | Product is empty; return `[]`; caller writes concat_seqs only |
| Product smaller than subset_size | Sample without replacement up to available size |
| All subsets identical Neff | Top-k via stable sort; arbitrary but deterministic |
| Dedup reduces to 0 | Return `[]`, log warning |

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `n_subsets` | 20 | Number of random subsets sampled from Cartesian product |
| `subset_size` | 175 | Sequences per subset (midpoint of 150–200 range) |
| `top_k` | 5 | Top subsets by Neff to merge |
| `neff_threshold` | 0.8 | Identity threshold passed to `calculate_neff` |

## Public API

Export `build_paired_msa` from `src/ghostfold/__init__.py`.

## Tests

New `tests/test_pairing.py`:

- `test_cartesian_product_two_chains` — 3-seq chains → output seqs are valid concatenations
- `test_single_chain_passthrough` — 1 chain → returns chain seqs unchanged
- `test_empty_chain_returns_empty` — one empty chain → returns `[]`
- `test_top_k_selection` — mock `calculate_neff` with known values → top-5 picked correctly
- `test_dedup_applied` — inject duplicates across subsets → dedup runs
- `test_subset_size_respected` — product smaller than subset_size → no crash

Integration: extend `test_cli.py` multimer path to assert no gap-padded rows in output `.a3m`.
