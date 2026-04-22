"""Tests for MSA subsampling / ranking strategies."""
import pytest

from ghostfold.msa.ranking import rank_and_subsample

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SEQS_10 = [
    "MQCDGLDGADGTSNGQAGASGLAGG",  # query (index 0)
    "AQCDGLDGADGTSNGQAGASGLAGG",
    "MQCEGLDGADGTSNGQAGASGLAGG",
    "MQCDGLDNADGTSNGQAGASGLAGG",
    "MQCDGLDGADGTSNGQAGASGLAAA",
    "AAAAAAAAAAAAAAAAAAAAAAAAA",
    "LLLLLLLLLLLLLLLLLLLLLLLLL",
    "MQCDGLDGADKTSNGQAGASGLAGG",
    "IQCDGLDGADGTSNGQAGASGLAGG",
    "MQCDGLDGADGTSNGQAGASGLAGV",
]

L = len(SEQS_10[0])
assert all(len(s) == L for s in SEQS_10), "All test sequences must be the same length"


# ---------------------------------------------------------------------------
# Common behaviours across all strategies
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strategy", [
    "farthest_first", "max_coverage", "column_entropy", "neff_contribution", "random",
])
class TestCommonBehaviours:
    def test_returns_correct_count(self, strategy):
        result = rank_and_subsample(SEQS_10, 4, strategy=strategy)
        assert len(result) == 4

    def test_query_is_first(self, strategy):
        result = rank_and_subsample(SEQS_10, 4, strategy=strategy, query=SEQS_10[0])
        assert result[0] == SEQS_10[0]

    def test_no_duplicates(self, strategy):
        result = rank_and_subsample(SEQS_10, 5, strategy=strategy)
        assert len(set(result)) == len(result)

    def test_n_larger_than_pool_capped(self, strategy):
        result = rank_and_subsample(SEQS_10, 999, strategy=strategy)
        assert len(result) == len(SEQS_10)

    def test_empty_input(self, strategy):
        assert rank_and_subsample([], 5, strategy=strategy) == []

    def test_n_one_returns_query(self, strategy):
        result = rank_and_subsample(SEQS_10, 1, strategy=strategy, query=SEQS_10[0])
        assert result == [SEQS_10[0]]

    def test_all_sequences_valid(self, strategy):
        result = rank_and_subsample(SEQS_10, 5, strategy=strategy)
        for seq in result:
            assert seq in SEQS_10


# ---------------------------------------------------------------------------
# Strategy-specific properties
# ---------------------------------------------------------------------------

class TestFarthestFirst:
    def test_diverse_pair_selected_before_similar(self):
        # SEQS_10[5] (all-A) and SEQS_10[6] (all-L) are maximally different
        # from the query; they should be selected when n=3.
        result = rank_and_subsample(SEQS_10, 3, strategy="farthest_first", query=SEQS_10[0])
        assert SEQS_10[5] in result or SEQS_10[6] in result


class TestMaxCoverage:
    def test_returns_list(self):
        result = rank_and_subsample(SEQS_10, 4, strategy="max_coverage")
        assert len(result) == 4


class TestColumnEntropy:
    def test_returns_list(self):
        result = rank_and_subsample(SEQS_10, 4, strategy="column_entropy")
        assert len(result) == 4


class TestNeffContribution:
    def test_neff_increases_monotonically(self):
        from ghostfold.msa.neff import calculate_neff
        result = rank_and_subsample(SEQS_10, len(SEQS_10), strategy="neff_contribution")
        neffs = [calculate_neff(result[:i+1]) for i in range(1, len(result))]
        # Should be non-decreasing
        for a, b in zip(neffs, neffs[1:]):
            assert b >= a - 1e-9


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            rank_and_subsample(SEQS_10, 3, strategy="nonexistent_strategy")

    def test_external_query_prepended(self):
        external = "X" * L
        result = rank_and_subsample(SEQS_10, 3, strategy="farthest_first", query=external)
        assert result[0] == external
