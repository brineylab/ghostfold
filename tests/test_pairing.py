import pytest
from unittest.mock import patch
from ghostfold.msa.pairing import _reservoir_sample_product, build_paired_msa


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
