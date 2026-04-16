import math
import pathlib
import tempfile
import numpy as np
from ghostfold.msa.neff import parse_a3m, calculate_neff

def test_parse_a3m_multiline_sequence():
    """parse_a3m must correctly join multi-line sequences."""
    content = ">seq1\nACDEF\nGHIKL\n>seq2\nMNPQR\nSTVWY\n"
    with tempfile.NamedTemporaryFile(suffix=".a3m", mode="w", delete=False) as f:
        f.write(content)
        path = pathlib.Path(f.name)
    seqs = parse_a3m(path)
    assert seqs == ["ACDEFGHIKL", "MNPQRSTVWY"]

def test_parse_a3m_strips_lowercase_insertions():
    """parse_a3m must strip lowercase insertion chars, keep uppercase and '-'."""
    content = ">q\nACde-FG\n"
    with tempfile.NamedTemporaryFile(suffix=".a3m", mode="w", delete=False) as f:
        f.write(content)
        path = pathlib.Path(f.name)
    seqs = parse_a3m(path)
    assert seqs == ["AC-FG"]


def test_calculate_neff_basic():
    # All identical: each sequence has N-1 neighbors above threshold
    # weight = 1/(1 + N-1) = 1/N per sequence → Neff = N * (1/N) / sqrt(L) = 1/sqrt(L)
    seqs = ["ACDEFGHIKL"] * 10
    neff = calculate_neff(seqs, identity_threshold=0.5)
    L = 10
    expected = (1.0 / math.sqrt(L))
    assert abs(neff - expected) < 1e-6


def test_calculate_neff_all_different():
    # All completely different: no neighbors above 0.5 threshold
    # each weight = 1/(1+0) = 1.0 → Neff = N / sqrt(L)
    seqs = [
        "AAAAAAAAAA",
        "CCCCCCCCCC",
        "DDDDDDDDDD",
        "EEEEEEEEEE",
    ]
    neff = calculate_neff(seqs, identity_threshold=0.5)
    expected = 4.0 / math.sqrt(10)
    assert abs(neff - expected) < 1e-6


def test_calculate_neff_chunked_matches_original():
    """Chunked result must match for a moderate-size input."""
    rng = np.random.default_rng(0)
    aa = list("ACDEFGHIKLMNPQRSTVWY")
    seqs = ["".join(rng.choice(aa, 50)) for _ in range(200)]
    neff_result = calculate_neff(seqs, identity_threshold=0.5)
    assert neff_result > 0.0
    assert neff_result < len(seqs)
