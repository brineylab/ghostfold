import numpy as np
import time
from unittest.mock import patch
from ghostfold.viz.coevolution import one_hot_encode_msa, AA_IDX, NUM_AAS, get_coevolution_numpy, _COEVO_CACHE

def test_one_hot_encode_shape():
    seqs = ["ACD", "EFG"]
    result = one_hot_encode_msa(seqs)
    assert result.shape == (2, 3, NUM_AAS)
    assert result.dtype == np.float32

def test_one_hot_encode_correct_values():
    seqs = ["A"]
    result = one_hot_encode_msa(seqs)
    assert result[0, 0, AA_IDX['A']] == 1.0
    assert result[0, 0].sum() == 1.0

def test_one_hot_encode_unknown_aa():
    # Unknown AA should map to NUM_AAS-1 (the fallback index)
    seqs = ["Z"]  # Z not in AA_WITH_GAPS
    result = one_hot_encode_msa(seqs)
    assert result[0, 0, NUM_AAS - 1] == 1.0

def test_one_hot_encode_performance():
    """Vectorized one-hot for N=500, L=300 should complete under 0.5s."""
    rng = np.random.default_rng(7)
    aa = list("ACDEFGHIKLMNPQRSTVWY-X")
    seqs = ["".join(rng.choice(aa, 300)) for _ in range(500)]
    t0 = time.time()
    result = one_hot_encode_msa(seqs)
    elapsed = time.time() - t0
    assert elapsed < 0.5
    assert result.shape == (500, 300, NUM_AAS)

def test_coevolution_cache_hit():
    """Second call with identical sequences returns cached result without recomputing."""
    _COEVO_CACHE.clear()
    seqs = ["ACDEFGHIKL"] * 20 + ["MNPQRSTVWY"] * 20

    with patch("ghostfold.viz.coevolution._compute_coevo_matrix") as mock_compute:
        mock_compute.return_value = np.zeros((10, 10))
        get_coevolution_numpy(seqs)
        get_coevolution_numpy(seqs)  # second call — same seqs
        assert mock_compute.call_count == 1  # only computed once

def test_coevolution_cache_miss_on_different_seqs():
    """Different sequence sets each get their own computation."""
    _COEVO_CACHE.clear()
    seqs_a = ["ACDEFGHIKL"] * 20
    seqs_b = ["MNPQRSTVWY"] * 20

    with patch("ghostfold.viz.coevolution._compute_coevo_matrix") as mock_compute:
        mock_compute.return_value = np.zeros((10, 10))
        get_coevolution_numpy(seqs_a)
        get_coevolution_numpy(seqs_b)
        assert mock_compute.call_count == 2
