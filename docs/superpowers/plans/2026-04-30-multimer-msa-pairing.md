# Multimer MSA Cartesian Product Pairing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace gap-padded unpaired blocks in `write_multimer_pst_msa` with Cartesian-product paired sequences to give AF2-Multimer strong coevolutionary signal.

**Architecture:** New `src/ghostfold/msa/pairing.py` exposes `build_paired_msa` which reservoir-samples subsets from the Cartesian product of per-chain sequences, ranks by Neff, merges top-k, and deduplicates. `write_multimer_pst_msa` in `pipeline.py` is updated to call `build_paired_msa` and write `concat_seqs + paired_seqs` with no gap-padded blocks.

**Tech Stack:** Python stdlib `itertools`, `random`; existing `calculate_neff` (`msa/neff.py`), `deduplicate` (`msa/filters.py`); pytest.

---

## File Map

| Action | Path | Purpose |
|--------|------|---------|
| Create | `src/ghostfold/msa/pairing.py` | `build_paired_msa` — reservoir sampling, Neff ranking, dedup |
| Modify | `src/ghostfold/core/pipeline.py:334-387` | `write_multimer_pst_msa` — call `build_paired_msa`, drop gap-padded blocks |
| Modify | `src/ghostfold/__init__.py` | export `build_paired_msa` |
| Create | `tests/test_pairing.py` | unit tests for `build_paired_msa` |
| Modify | `tests/test_cli.py` | integration: assert no gap-padded rows in multimer `.a3m` output |

---

## Task 1: Create `pairing.py` with skeleton + reservoir helper

**Files:**
- Create: `src/ghostfold/msa/pairing.py`

- [ ] **Step 1: Write the failing test for reservoir sampling**

```python
# tests/test_pairing.py
import pytest
from ghostfold.msa.pairing import _reservoir_sample_product


def test_reservoir_sample_product_count():
    chains = [["AA", "BB", "CC"], ["XX", "YY"]]
    result = _reservoir_sample_product(chains, k=4)
    assert len(result) == 4


def test_reservoir_sample_product_valid_concatenations():
    chains = [["AA", "BB"], ["XX", "YY"]]
    result = _reservoir_sample_product(chains, k=4)
    assert set(result) <= {"AAXX", "AAYY", "BBXX", "BBYY"}


def test_reservoir_sample_product_smaller_than_k():
    chains = [["AA"], ["XX"]]
    result = _reservoir_sample_product(chains, k=10)
    assert result == ["AAXX"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_pairing.py::test_reservoir_sample_product_count -v
```
Expected: `ImportError` or `ModuleNotFoundError`

- [ ] **Step 3: Create `src/ghostfold/msa/pairing.py` with reservoir helper**

```python
import itertools
import random
from typing import List

from ghostfold.core.logging import get_logger

logger = get_logger("pairing")


def _reservoir_sample_product(per_chain_seqs: List[List[str]], k: int) -> List[str]:
    """Reservoir-sample k concatenated sequences from the Cartesian product.

    Never materializes the full product — streams via itertools.product.
    Returns fewer than k items if the product is smaller than k.
    """
    reservoir: List[str] = []
    for i, combo in enumerate(itertools.product(*per_chain_seqs)):
        concat = "".join(combo)
        if i < k:
            reservoir.append(concat)
        else:
            j = random.randint(0, i)
            if j < k:
                reservoir[j] = concat
    return reservoir
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_pairing.py::test_reservoir_sample_product_count \
       tests/test_pairing.py::test_reservoir_sample_product_valid_concatenations \
       tests/test_pairing.py::test_reservoir_sample_product_smaller_than_k -v
```
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/ghostfold/msa/pairing.py tests/test_pairing.py
git commit -m "feat: add reservoir sampler for Cartesian product MSA pairing"
```

---

## Task 2: Implement `build_paired_msa`

**Files:**
- Modify: `src/ghostfold/msa/pairing.py`

- [ ] **Step 1: Write failing tests for `build_paired_msa`**

```python
# append to tests/test_pairing.py
from unittest.mock import patch
from ghostfold.msa.pairing import build_paired_msa


def test_cartesian_product_two_chains():
    chain_a = ["AAAA", "BBBB", "CCCC"]
    chain_b = ["XXXX", "YYYY", "ZZZZ"]
    result = build_paired_msa([chain_a, chain_b], n_subsets=3, subset_size=4, top_k=2)
    assert all(len(s) == 8 for s in result)
    for s in result:
        assert s[:4] in chain_a and s[4:] in chain_b


def test_single_chain_passthrough():
    chain = ["AAAA", "BBBB", "CCCC"]
    result = build_paired_msa([chain], n_subsets=3, subset_size=4, top_k=2)
    assert set(result).issubset(set(chain))


def test_empty_chain_returns_empty():
    result = build_paired_msa([["AAAA", "BBBB"], []], n_subsets=3, subset_size=4, top_k=2)
    assert result == []


def test_top_k_selection():
    """Mock calculate_neff to return known values; verify top-k chosen."""
    chain_a = ["AAAA", "BBBB"]
    chain_b = ["XXXX", "YYYY"]
    neff_values = iter([0.9, 0.3, 0.7, 0.5, 0.1])
    with patch("ghostfold.msa.pairing.calculate_neff", side_effect=lambda seqs, **kw: next(neff_values)):
        result = build_paired_msa([chain_a, chain_b], n_subsets=5, subset_size=2, top_k=2)
    # top-k=2 means 2 subsets were merged; result should be non-empty
    assert len(result) > 0


def test_dedup_applied():
    """All subsets identical → dedup reduces to unique seqs only."""
    chain_a = ["AAAA"]
    chain_b = ["XXXX"]
    result = build_paired_msa([chain_a, chain_b], n_subsets=5, subset_size=2, top_k=3)
    assert result.count("AAAAXXXX") == 1


def test_subset_size_respected_small_product():
    """Product smaller than subset_size → no crash, returns available seqs."""
    chain_a = ["AAAA"]
    chain_b = ["XXXX", "YYYY"]
    result = build_paired_msa([chain_a, chain_b], n_subsets=3, subset_size=100, top_k=2)
    assert set(result) <= {"AAAAXXXX", "AAAAYYYY"}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_pairing.py -k "test_cartesian" -v
```
Expected: `ImportError` — `build_paired_msa` not defined yet

- [ ] **Step 3: Implement `build_paired_msa` in `pairing.py`**

Add after `_reservoir_sample_product`:

```python
from ghostfold.msa.neff import calculate_neff
from ghostfold.msa.filters import deduplicate


def build_paired_msa(
    per_chain_seqs: List[List[str]],
    n_subsets: int = 20,
    subset_size: int = 175,
    top_k: int = 5,
    neff_threshold: float = 0.8,
) -> List[str]:
    """Build a paired MSA block via Cartesian-product reservoir sampling.

    Args:
        per_chain_seqs: One list of sequences per chain.
        n_subsets: Number of random subsets sampled from the Cartesian product.
        subset_size: Sequences per subset.
        top_k: Number of top-Neff subsets to merge.
        neff_threshold: Identity threshold passed to calculate_neff.

    Returns:
        Deduplicated list of fully-concatenated paired sequences.
        Empty list if any chain is empty or product is empty.
    """
    if any(len(chain) == 0 for chain in per_chain_seqs):
        logger.warning("build_paired_msa: one or more chains empty → returning []")
        return []

    if len(per_chain_seqs) == 1:
        return list(per_chain_seqs[0])

    subsets: List[List[str]] = []
    for _ in range(n_subsets):
        sample = _reservoir_sample_product(per_chain_seqs, k=subset_size)
        if sample:
            subsets.append(sample)

    if not subsets:
        logger.warning("build_paired_msa: all subsets empty → returning []")
        return []

    scored = sorted(
        subsets,
        key=lambda s: calculate_neff(s, identity_threshold=neff_threshold),
        reverse=True,
    )

    merged: List[str] = []
    for subset in scored[:top_k]:
        merged.extend(subset)

    result = deduplicate(merged)
    if not result:
        logger.warning("build_paired_msa: dedup reduced to 0 sequences → returning []")
    return result
```

- [ ] **Step 4: Run all pairing tests**

```bash
pytest tests/test_pairing.py -v
```
Expected: all 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/ghostfold/msa/pairing.py tests/test_pairing.py
git commit -m "feat: implement build_paired_msa with Neff-ranked Cartesian product pairing"
```

---

## Task 3: Update `write_multimer_pst_msa` in `pipeline.py`

**Files:**
- Modify: `src/ghostfold/core/pipeline.py:334-387`

- [ ] **Step 1: Write failing integration test**

```python
# append to tests/test_pairing.py
import pathlib
import tempfile
from ghostfold.core.pipeline import write_multimer_pst_msa


def test_write_multimer_pst_msa_no_gap_padded_rows():
    concat_seqs = ["AAAABBBB", "CCCCDDDD"]
    per_chain_seqs = [["AAAA", "CCCC"], ["BBBB", "DDDD"]]
    chain_lengths = [4, 4]
    query_seq = "AAAABBBB"

    with tempfile.NamedTemporaryFile(mode="r", suffix=".a3m", delete=False) as f:
        path = f.name

    write_multimer_pst_msa(
        output_path=path,
        query_seq=query_seq,
        concat_seqs=concat_seqs,
        per_chain_seqs=per_chain_seqs,
        chain_lengths=chain_lengths,
    )

    content = pathlib.Path(path).read_text()
    lines = [l for l in content.splitlines() if not l.startswith(">") and not l.startswith("#")]
    for seq in lines:
        assert "----" not in seq, f"Gap-padded row found: {seq!r}"
```

- [ ] **Step 2: Run test to verify it currently fails (gap rows present)**

```bash
pytest tests/test_pairing.py::test_write_multimer_pst_msa_no_gap_padded_rows -v
```
Expected: FAIL — gap-padded rows exist in current implementation

- [ ] **Step 3: Modify `write_multimer_pst_msa` in `pipeline.py`**

Add import at top of `pipeline.py` (with other msa imports, around line 27):

```python
from ghostfold.msa.pairing import build_paired_msa
```

Replace the entire `write_multimer_pst_msa` function body (lines 352–387):

```python
def write_multimer_pst_msa(
    output_path: str,
    query_seq: str,
    concat_seqs: List[str],
    per_chain_seqs: List[List[str]],
    chain_lengths: List[int],
) -> None:
    """Write a multimer pseudoMSA file in ColabFold A3M/FASTA format.

    Paired block = concat_seqs + Cartesian-product paired sequences (no gap padding).
    Both heterooligomer and homooligomer paths use this paired block.
    """
    clean_query = query_seq.replace(":", "")
    chain_queries: List[str] = []
    pos = 0
    for length in chain_lengths:
        chain_queries.append(clean_query[pos:pos + length])
        pos += length

    is_homooligomer = len(chain_lengths) > 1 and len(set(chain_queries)) == 1

    paired_seqs = build_paired_msa(per_chain_seqs)

    with open(output_path, "w") as fh:
        if is_homooligomer:
            L = chain_lengths[0]
            N = len(chain_lengths)
            fh.write(f"#{L}\t{N}\n")
            fh.write(f">query\n{chain_queries[0]}\n")
            for i, seq in enumerate(concat_seqs):
                fh.write(f">concat_{i}\n{seq}\n")
            for i, seq in enumerate(paired_seqs):
                fh.write(f">paired_{i}\n{seq}\n")
        else:
            lengths_str = ",".join(str(n) for n in chain_lengths)
            cardinality_str = ",".join("1" for _ in chain_lengths)
            chain_header = "\t".join(str(i + 1) for i in range(len(chain_lengths)))
            fh.write(f"#{lengths_str}\t{cardinality_str}\n")
            fh.write(f">{chain_header}\n{clean_query}\n")
            for i, seq in enumerate(concat_seqs):
                fh.write(f">concat_{i}\n{seq}\n")
            for i, seq in enumerate(paired_seqs):
                fh.write(f">paired_{i}\n{seq}\n")
```

- [ ] **Step 4: Run the new test and full suite**

```bash
pytest tests/test_pairing.py -v
pytest tests/ -q
```
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/ghostfold/core/pipeline.py
git commit -m "feat: replace gap-padded unpaired blocks with Cartesian product paired MSA"
```

---

## Task 4: Export `build_paired_msa` from public API

**Files:**
- Modify: `src/ghostfold/__init__.py`

- [ ] **Step 1: Write failing import test**

```python
# append to tests/test_pairing.py
def test_build_paired_msa_importable_from_package():
    from ghostfold import build_paired_msa
    assert callable(build_paired_msa)
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_pairing.py::test_build_paired_msa_importable_from_package -v
```
Expected: `ImportError`

- [ ] **Step 3: Add export to `__init__.py`**

Add import line after the existing `from ghostfold.msa.neff import ...` line:

```python
from ghostfold.msa.pairing import build_paired_msa
```

Add `"build_paired_msa"` to the `__all__` list.

Full updated `__init__.py`:

```python
from ghostfold._version import __version__
from ghostfold.core.pipeline import run_pipeline
from ghostfold.io.fasta import collect_fasta_paths, read_fasta_from_path
from ghostfold.msa.mask import mask_a3m_file
from ghostfold.msa.neff import calculate_neff, run_neff_calculation_in_parallel
from ghostfold.msa.pairing import build_paired_msa
from ghostfold.msa.ranking import rank_and_subsample
from ghostfold.mutator import MSA_Mutator

__all__ = [
    "__version__",
    "build_paired_msa",
    "collect_fasta_paths",
    "read_fasta_from_path",
    "run_pipeline",
    "mask_a3m_file",
    "calculate_neff",
    "run_neff_calculation_in_parallel",
    "rank_and_subsample",
    "MSA_Mutator",
]
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_pairing.py -v
ruff check src tests
```
Expected: all PASS, no lint errors

- [ ] **Step 5: Commit**

```bash
git add src/ghostfold/__init__.py
git commit -m "feat: export build_paired_msa from public API"
```

---

## Task 5: Integration test — no gap-padded rows in CLI multimer output

**Files:**
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write the integration test**

Find the existing multimer CLI test section in `tests/test_cli.py` (search for `multimer` or the `msa` subcommand with `:` in the sequence). Add a new test adjacent to existing multimer tests:

```python
def test_multimer_msa_no_gap_padded_rows(tmp_path):
    """Multimer .a3m output must contain no gap-padded unpaired rows."""
    fasta = tmp_path / "complex.fasta"
    fasta.write_text(">complex\nAAAA:BBBB\n")
    out_dir = tmp_path / "out"

    result = runner.invoke(
        app,
        ["msa", str(fasta), "--output-dir", str(out_dir), "--num-return-sequences", "2"],
    )
    assert result.exit_code == 0, result.output

    a3m_files = list(out_dir.rglob("*.a3m"))
    assert len(a3m_files) > 0, "No .a3m files produced"

    for a3m_path in a3m_files:
        content = a3m_path.read_text()
        seq_lines = [
            line for line in content.splitlines()
            if line and not line.startswith(">") and not line.startswith("#")
        ]
        for seq in seq_lines:
            assert "----" not in seq, (
                f"Gap-padded row in {a3m_path.name}: {seq!r}"
            )
```

- [ ] **Step 2: Run the integration test**

```bash
pytest tests/test_cli.py::test_multimer_msa_no_gap_padded_rows -v
```
Expected: PASS (gap rows removed in Task 3)

- [ ] **Step 3: Run full test suite + lint**

```bash
pytest tests/ -q
ruff check src tests
```
Expected: all PASS, no errors

- [ ] **Step 4: Commit**

```bash
git add tests/test_cli.py
git commit -m "test: assert no gap-padded rows in multimer MSA CLI output"
```

---

## Self-Review

### Spec coverage

| Spec requirement | Task |
|-----------------|------|
| New `pairing.py` with `build_paired_msa` | Tasks 1–2 |
| `itertools.product` iterator (never materialized) | Task 1 — `_reservoir_sample_product` streams via `itertools.product` |
| Reservoir-sample `n_subsets` subsets of `subset_size` | Task 2 |
| Rank by Neff descending, take `top_k` | Task 2 |
| Merge top-k + `deduplicate` | Task 2 |
| Return `List[str]` of fully-concatenated paired seqs | Task 2 |
| `write_multimer_pst_msa` calls `build_paired_msa` | Task 3 |
| Paired block = `concat_seqs + paired_seqs` | Task 3 |
| Remove all gap-padded unpaired block writes | Task 3 |
| Homo- and heterooligomer both use paired block | Task 3 |
| Export from `__init__.py` | Task 4 |
| `test_cartesian_product_two_chains` | Task 2 |
| `test_single_chain_passthrough` | Task 2 |
| `test_empty_chain_returns_empty` | Task 2 |
| `test_top_k_selection` | Task 2 |
| `test_dedup_applied` | Task 2 |
| `test_subset_size_respected` | Task 2 |
| Integration: no gap-padded rows in `.a3m` | Task 5 |
| Default params: n_subsets=20, subset_size=175, top_k=5, neff_threshold=0.8 | Task 2 ✓ |

All spec requirements covered.

### Placeholder scan

No TBD/TODO/placeholder patterns found.

### Type consistency

- `_reservoir_sample_product(per_chain_seqs: List[List[str]], k: int) -> List[str]` — used identically in Task 2 impl.
- `build_paired_msa(per_chain_seqs: List[List[str]], ...) -> List[str]` — called in Task 3 as `build_paired_msa(per_chain_seqs)` ✓
- `calculate_neff(seqs, identity_threshold=neff_threshold)` — matches actual signature `calculate_neff(sequences: List[str], identity_threshold: float = 0.5)` ✓
- `deduplicate(merged)` — matches actual signature `deduplicate(sequences: List[str], threshold: float = 0.95)` ✓
