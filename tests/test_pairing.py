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
