import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def plot_msa_coverage(
    sequences,
    save_path=None,
    cmap_name="GnBu",
    custom_colors=None,
    dpi=150
):
    """Plot and optionally save the MSA coverage heatmap.

    Args:
        sequences (list of str): List of aligned sequences (first one is the query).
        save_path (str): Path to save the figure (PNG). If None, only shows the plot.
        cmap_name (str): Matplotlib colormap name (used if custom_colors is None).
        custom_colors (list of str): List of hex codes to define a custom colormap.
        dpi (int): Figure resolution.
    """
    if len(sequences) < 2:
        raise ValueError("At least two sequences (query + others) are required.")

    query_seq = np.array(list(sequences[0]))
    msa_array = np.array([list(seq) for seq in sequences])
    num_seqs, seq_len = msa_array.shape

    seq_identity = np.array([
        np.count_nonzero(query_seq == msa_array[i]) / seq_len
        for i in range(num_seqs)
    ])

    non_gap_mask = (msa_array != '-').astype(float)
    non_gap_mask[non_gap_mask == 0] = np.nan
    coverage_matrix = non_gap_mask * seq_identity[:, None]

    query_row = coverage_matrix[0:1]
    rest_of_sequences = coverage_matrix[1:]

    if len(rest_of_sequences) > 0:
        sequence_lengths = np.nansum(~np.isnan(rest_of_sequences), axis=1)
        sorted_idx = np.argsort(-sequence_lengths)
        sorted_rest = rest_of_sequences[sorted_idx]
        sorted_coverage = np.vstack([query_row, sorted_rest])
    else:
        sorted_coverage = query_row

    if custom_colors is not None:
        cmap = LinearSegmentedColormap.from_list("custom_cmap", custom_colors)
    else:
        from matplotlib import cm
        cmap = cm.get_cmap(cmap_name)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
    cax = ax.imshow(
        sorted_coverage,
        interpolation="nearest",
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=1,
        origin="lower"
    )

    position_coverage = (~np.isnan(sorted_coverage)).sum(0)
    ax.plot(position_coverage, color="#023047", linewidth=0.5)

    for spine in ['top', 'left', 'right']:
        ax.spines[spine].set_visible(False)

    ax.set_xlim(0, seq_len)
    ax.set_ylim(-0.5, num_seqs - 0.5)
    ax.set_xlabel("Positions")
    ax.set_ylabel("Sequences")

    cbar = fig.colorbar(cax, ax=ax, orientation="horizontal", pad=0.15, shrink=0.25)
    cbar.set_label("Sequence identity to query")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
