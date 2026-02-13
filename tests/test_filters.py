import pytest

from ghostfold.msa.filters import (
    clean_repeats,
    sequence_entropy,
    is_similar,
    deduplicate,
    filter_sequences,
)


class TestCleanRepeats:
    def test_mono_repeat_replaced(self):
        seqs = ["AAAAAACDEFG"]  # 6 A's should be replaced (threshold=5)
        result = clean_repeats(seqs)
        assert result[0] == "XXXXXXCDEFG"

    def test_below_threshold_preserved(self):
        seqs = ["AAAACDEFG"]  # 4 A's, below threshold of 5
        result = clean_repeats(seqs)
        assert result[0] == "AAAACDEFG"

    def test_dipeptide_repeat(self):
        seqs = ["GSGSGSXYZ"]  # GS repeated 3 times, threshold=3
        result = clean_repeats(seqs)
        assert "GS" not in result[0][:6]

    def test_no_repeats_unchanged(self):
        seqs = ["ACDEFGHIKL"]
        result = clean_repeats(seqs)
        assert result[0] == "ACDEFGHIKL"


class TestSequenceEntropy:
    def test_empty_sequence(self):
        assert sequence_entropy("") == 0.0

    def test_single_aa_low_entropy(self):
        # All same AA — only one amino acid type, entropy is 0
        ent = sequence_entropy("AAAAAA")
        assert ent == 0.0

    def test_diverse_sequence_higher_entropy(self):
        diverse = "ACDEFGHIKLMNPQRSTVWY"
        uniform = "AAAAAAAAAA"
        assert sequence_entropy(diverse) > sequence_entropy(uniform)


class TestIsSimilar:
    def test_identical_sequences(self):
        assert is_similar("ACDEF", "ACDEF") is True

    def test_different_sequences(self):
        assert is_similar("ACDEF", "XXXXX") is False

    def test_custom_threshold(self):
        # These differ by 1 out of 5 = 80% identity
        assert is_similar("ACDEF", "ACDEX", threshold=0.7) is True
        assert is_similar("ACDEF", "ACDEX", threshold=0.9) is False


class TestDeduplicate:
    def test_removes_duplicates(self):
        seqs = ["ACDEF", "ACDEF", "XXXXX"]
        result = deduplicate(seqs)
        assert len(result) == 2

    def test_no_duplicates(self):
        seqs = ["ACDEF", "XXXXX"]
        result = deduplicate(seqs)
        assert len(result) == 2


class TestFilterSequences:
    def test_empty_input(self):
        assert filter_sequences([], 10) == []

    def test_length_filter(self):
        seqs = ["ACDEF", "ACDEFGHIKL", "AC"]
        result = filter_sequences(seqs, 5, entropy_threshold=0.0)
        assert all(len(s) == 5 for s in result)

    def test_full_pipeline(self):
        # Create sequences of length 10 with reasonable diversity
        seqs = ["ACDEFGHIKL", "MNPQRSTVWY", "ACDEACDEAC"]
        result = filter_sequences(seqs, 10, entropy_threshold=0.0)
        assert len(result) <= len(seqs)
        assert all(len(s) == 10 for s in result)
