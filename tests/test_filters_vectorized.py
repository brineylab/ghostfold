import time
import numpy as np
from ghostfold.msa.filters import filter_sequences, deduplicate

def test_filter_sequences_removes_near_duplicates():
    """filter_sequences must deduplicate near-identical sequences."""
    base = "ACDEFGHIKLMNPQRSTVWY"  # len=20, entropy > 2.0
    # 10 copies identical + 1 slightly different; after dedup only 2 should remain
    seqs = [base] * 10 + ["ACDEFGHIKLMNPQRSTVWX"]
    result = filter_sequences(seqs, expected_length=20, similarity_threshold=0.95)
    # both unique variants survive, duplicates gone
    assert len(result) == 2


def test_deduplicate_removes_exact_duplicates():
    seqs = ["ACDEFGHIKL"] * 5 + ["MNPQRSTVWY"]
    result = deduplicate(seqs, threshold=0.95)
    assert len(result) == 2
    assert "ACDEFGHIKL" in result
    assert "MNPQRSTVWY" in result

def test_deduplicate_keeps_dissimilar():
    seqs = ["AAAAAAAAAA", "CCCCCCCCCC"]
    result = deduplicate(seqs, threshold=0.95)
    assert len(result) == 2

def test_deduplicate_removes_near_identical():
    # 9/10 chars same = 0.9 identity — above 0.85 threshold, second removed
    seqs = ["ACDEFGHIKL", "ACDEFGHIKM"]
    result = deduplicate(seqs, threshold=0.85)
    assert len(result) == 1

def test_deduplicate_empty():
    assert deduplicate([], 0.95) == []

def test_deduplicate_single():
    assert deduplicate(["ACDE"], 0.95) == ["ACDE"]

def test_deduplicate_performance():
    """Numpy dedup of 1000 length-100 sequences should complete under 2 seconds."""
    rng = np.random.default_rng(42)
    aa = list("ACDEFGHIKLMNPQRSTVWY")
    seqs = ["".join(rng.choice(aa, 100)) for _ in range(1000)]
    t0 = time.time()
    result = deduplicate(seqs, threshold=0.95)
    elapsed = time.time() - t0
    assert elapsed < 2.0
    assert len(result) <= len(seqs)
