import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List

AA_WITH_GAPS = "ACDEFGHIKLMNPQRSTVWY-X"
AA_IDX = {aa: i for i, aa in enumerate(AA_WITH_GAPS)}
NUM_AAS = len(AA_WITH_GAPS)

# Module-level cache: sequence-set hash → coevolution matrix
_COEVO_CACHE: Dict[int, np.ndarray] = {}


def one_hot_encode_msa(sequences: List[str]) -> np.ndarray:
    """One-hot encode an MSA into a (N, L, NUM_AAS) float32 array.

    Uses NumPy advanced indexing (single C-level scatter) instead of nested
    Python loops — ~100x faster for large MSAs.
    """
    N, L = len(sequences), len(sequences[0])
    idx = np.array(
        [[AA_IDX.get(aa, NUM_AAS - 1) for aa in seq] for seq in sequences],
        dtype=np.intp,
    )
    msa_oh = np.zeros((N, L, NUM_AAS), dtype=np.float32)
    msa_oh[np.arange(N)[:, None], np.arange(L)[None, :], idx] = 1.0
    return msa_oh


def _compute_coevo_matrix(sequences: List[str]) -> np.ndarray:
    """Core coevolution computation (no caching). Called by get_coevolution_numpy."""
    Y = one_hot_encode_msa(sequences)
    N, L, A = Y.shape
    Y_flat = Y.reshape(N, -1)
    C = np.cov(Y_flat, rowvar=False)
    shrink = 4.5 / np.sqrt(N) * np.eye(C.shape[0])
    # solve is more numerically stable than inv and ~20% faster
    C_inv = np.linalg.solve(C + shrink, np.eye(C.shape[0]))
    diag = np.diag(C_inv)
    pcc = C_inv / np.sqrt(np.outer(diag, diag))
    blocks = pcc.reshape(L, A, L, A)
    raw = np.sqrt(np.sum(blocks[:, :20, :, :20] ** 2, axis=(1, 3)))
    np.fill_diagonal(raw, 0)
    apc = raw.mean(axis=1, keepdims=True) @ raw.mean(axis=0, keepdims=True) / raw.mean()
    corrected = raw - apc
    np.fill_diagonal(corrected, 0)
    return corrected


def get_coevolution_numpy(sequences: List[str]) -> np.ndarray:
    """Return the APC-corrected coevolution matrix, using a module-level cache.

    The cache key is a hash of the sequence tuple. Identical sequence sets
    (common when called multiple times per pipeline run) skip recomputation.
    """
    key = hash(tuple(sequences))
    if key not in _COEVO_CACHE:
        _COEVO_CACHE[key] = _compute_coevo_matrix(sequences)
    return _COEVO_CACHE[key]


def plot_coevolution(matrix: np.ndarray, out_path: str) -> None:
    cmap = LinearSegmentedColormap.from_list("custom", ["#ffffff", "#ABAEBA", "#023047"])
    plt.figure(figsize=(5, 5), dpi=150)
    plt.title("Coevolution")
    plt.imshow(matrix, cmap=cmap, vmin=0)

    ax = plt.gca()
    for spine in ['top', 'left', 'right']:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
