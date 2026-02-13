# utils/coevolution.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

AA_WITH_GAPS = "ACDEFGHIKLMNPQRSTVWY-X"
AA_IDX = {aa: i for i, aa in enumerate(AA_WITH_GAPS)}
NUM_AAS = len(AA_WITH_GAPS)

def one_hot_encode_msa(sequences):
    N, L = len(sequences), len(sequences[0])
    msa_oh = np.zeros((N, L, NUM_AAS), dtype=np.float32)
    for i, seq in enumerate(sequences):
        for j, aa in enumerate(seq):
            msa_oh[i, j, AA_IDX.get(aa, NUM_AAS - 1)] = 1.0
    return msa_oh

def get_coevolution_numpy(sequences):
    Y = one_hot_encode_msa(sequences)
    N, L, A = Y.shape
    Y_flat = Y.reshape(N, -1)
    C = np.cov(Y_flat, rowvar=False)
    shrink = 4.5 / np.sqrt(N) * np.eye(C.shape[0])
    C_inv = np.linalg.inv(C + shrink)
    diag = np.diag(C_inv)
    pcc = C_inv / np.sqrt(np.outer(diag, diag))
    blocks = pcc.reshape(L, A, L, A)
    raw = np.sqrt(np.sum(blocks[:, :20, :, :20] ** 2, axis=(1, 3)))
    np.fill_diagonal(raw, 0)
    apc = raw.mean(axis=1, keepdims=True) @ raw.mean(axis=0, keepdims=True) / raw.mean()
    corrected = raw - apc
    np.fill_diagonal(corrected, 0)
    return corrected

def plot_coevolution(matrix, out_path):
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
