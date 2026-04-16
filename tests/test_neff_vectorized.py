"""Tests for Fix 1: vectorized Neff calculation.

The vectorized implementation must produce numerically identical results
to the reference O(n²) Python loop, and must not call the slow path.
"""
import math
import time
import pytest


class TestVectorizedNeffCorrectness:
    def test_single_sequence_matches_reference(self):
        from ghostfold.msa.neff import calculate_neff
        result = calculate_neff(["ACDEF"])
        expected = 1.0 / math.sqrt(5)
        assert abs(result - expected) < 1e-9

    def test_identical_sequences_match_reference(self):
        from ghostfold.msa.neff import calculate_neff
        seqs = ["ACDEF", "ACDEF", "ACDEF"]
        result = calculate_neff(seqs)
        expected = (1.0 / math.sqrt(5)) * (3 * (1.0 / 3.0))
        assert abs(result - expected) < 1e-9

    def test_diverse_sequences_match_reference(self):
        """Vectorized result must exactly equal the O(n²) reference."""
        from ghostfold.msa.neff import calculate_neff

        seqs = [
            "ACDEFGHIKL",
            "MNPQRSTVWY",
            "ACDEACDEAC",
            "GHIKLMNPQR",
            "STVWYACDEA",
        ]
        result = calculate_neff(seqs, identity_threshold=0.5)

        # Compute reference manually (O(n²) Python loop)
        N, L = len(seqs), len(seqs[0])
        total = 0.0
        for n in range(N):
            sim = 0
            for m in range(N):
                if n == m:
                    continue
                ident = sum(1 for c1, c2 in zip(seqs[n], seqs[m]) if c1 == c2) / L
                if ident >= 0.5:
                    sim += 1
            total += 1.0 / (1.0 + sim)
        expected = (1.0 / math.sqrt(L)) * total

        assert abs(result - expected) < 1e-9

    def test_sequences_with_gaps_match_reference(self):
        """Gaps ('-') must be treated as regular characters (equal to gap = match)."""
        from ghostfold.msa.neff import calculate_neff

        seqs = [
            "ACDE-GHIKL",
            "ACDE-GHIKL",
            "MNPQ-RSTVW",
        ]
        result = calculate_neff(seqs, identity_threshold=0.5)

        N, L = len(seqs), len(seqs[0])
        total = 0.0
        for n in range(N):
            sim = 0
            for m in range(N):
                if n == m:
                    continue
                ident = sum(1 for c1, c2 in zip(seqs[n], seqs[m]) if c1 == c2) / L
                if ident >= 0.5:
                    sim += 1
            total += 1.0 / (1.0 + sim)
        expected = (1.0 / math.sqrt(L)) * total

        assert abs(result - expected) < 1e-9

    def test_threshold_boundary_match_reference(self):
        """Sequences exactly at identity_threshold must be counted (>=, not >)."""
        from ghostfold.msa.neff import calculate_neff

        # 5 of 10 identical = 0.5 identity; threshold=0.5 → should be counted
        seqs = [
            "AAAAAXYZQW",
            "AAAAABBBBB",
        ]
        result = calculate_neff(seqs, identity_threshold=0.5)

        N, L = len(seqs), len(seqs[0])
        total = 0.0
        for n in range(N):
            sim = 0
            for m in range(N):
                if n == m:
                    continue
                ident = sum(1 for c1, c2 in zip(seqs[n], seqs[m]) if c1 == c2) / L
                if ident >= 0.5:
                    sim += 1
            total += 1.0 / (1.0 + sim)
        expected = (1.0 / math.sqrt(L)) * total

        assert abs(result - expected) < 1e-9


class TestVectorizedNeffPerformance:
    def test_large_msa_faster_than_quadratic_threshold(self):
        """1000 sequences of length 200 should complete in < 1 second (vectorized).
        The O(n²) Python loop takes ~5s on this input.
        """
        import random
        from ghostfold.msa.neff import calculate_neff

        rng = random.Random(42)
        aa = "ACDEFGHIKLMNPQRSTVWY-"
        seqs = ["".join(rng.choices(aa, k=200)) for _ in range(1000)]

        start = time.perf_counter()
        result = calculate_neff(seqs)
        elapsed = time.perf_counter() - start

        assert result > 0
        assert elapsed < 1.0, f"Too slow: {elapsed:.2f}s (expected < 1s, O(n²) takes ~5s)"
