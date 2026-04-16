"""Tests for Fix 3: vectorized sequence_entropy.

The vectorized implementation must:
- Exactly match original behavior for standard AA sequences
- Exactly match original behavior for gapped sequences (gaps excluded from counts,
  but full length used as denominator — same as original AA_LETTERS-based approach)
- Be faster for large batches
"""
import math
import time
import pytest
import numpy as np


def _reference_entropy(seq: str) -> float:
    """The original O(21n) implementation, used as ground truth."""
    AA_LETTERS = "ACDEFGHIKLMNPQRSTVWYX"
    if not seq:
        return 0.0
    aa_counts = np.array([seq.count(aa) for aa in AA_LETTERS])
    probs = aa_counts / len(seq)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


class TestEntropyVectorizationCorrectness:
    def test_empty_sequence(self):
        from ghostfold.msa.filters import sequence_entropy
        assert sequence_entropy("") == 0.0

    def test_single_aa_zero_entropy(self):
        from ghostfold.msa.filters import sequence_entropy
        result = sequence_entropy("AAAAAA")
        expected = _reference_entropy("AAAAAA")
        assert abs(result - expected) < 1e-10
        assert result == 0.0

    def test_diverse_sequence_matches_reference(self):
        from ghostfold.msa.filters import sequence_entropy
        seq = "ACDEFGHIKLMNPQRSTVWY"
        assert abs(sequence_entropy(seq) - _reference_entropy(seq)) < 1e-10

    def test_gapped_sequence_matches_reference(self):
        """Gaps ('-') must be excluded from AA counts but denominator is full length."""
        from ghostfold.msa.filters import sequence_entropy
        seq = "ACDEF-----"
        result = sequence_entropy(seq)
        expected = _reference_entropy(seq)
        assert abs(result - expected) < 1e-10

    def test_fully_gapped_sequence(self):
        """All gaps → no AA counts → entropy 0.0."""
        from ghostfold.msa.filters import sequence_entropy
        seq = "----------"
        result = sequence_entropy(seq)
        expected = _reference_entropy(seq)
        assert abs(result - expected) < 1e-10

    def test_mixed_coverage_sequence_matches_reference(self):
        """Realistic padded sequence (gaps on both ends)."""
        from ghostfold.msa.filters import sequence_entropy
        seq = "-----ACDEFGHIKL-----"
        result = sequence_entropy(seq)
        expected = _reference_entropy(seq)
        assert abs(result - expected) < 1e-10

    def test_uniform_distribution_entropy(self):
        """20 distinct AAs, each once: entropy = log2(20) adjusted for length."""
        from ghostfold.msa.filters import sequence_entropy
        seq = "ACDEFGHIKLMNPQRSTVWY"  # 20 chars, each unique
        result = sequence_entropy(seq)
        expected = _reference_entropy(seq)
        assert abs(result - expected) < 1e-10

    def test_x_character_treated_as_aa(self):
        """'X' is in AA_LETTERS — must be counted."""
        from ghostfold.msa.filters import sequence_entropy
        seq = "XXXXXX"
        result = sequence_entropy(seq)
        expected = _reference_entropy(seq)
        assert abs(result - expected) < 1e-10
