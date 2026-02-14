import os
from typing import List

from ghostfold.core.logging import get_logger

from .msa_coverage import plot_msa_coverage
from .coevolution import get_coevolution_numpy, plot_coevolution

logger = get_logger("plotting")


def generate_optional_plots(
    sequences: List[str],
    full_len: int,
    img_dir: str,
    base_name: str,
    custom_colors: List[str],
    plot_msa: bool,
    plot_coevolution_flag: bool,
) -> None:
    """Generates MSA coverage and/or coevolution plots if requested."""
    if not sequences:
        logger.info(f"No sequences provided for {base_name} plots.")
        return

    # MSA coverage plot
    if plot_msa:
        sequences_for_coverage_plot = [seq for seq in sequences if len(seq) == full_len]
        if sequences_for_coverage_plot:
            msa_coverage_path = os.path.join(img_dir, f'msa_coverage_{base_name}.png')
            logger.info(f"Generating MSA coverage plot for {base_name} sequences...")
            plot_msa_coverage(sequences_for_coverage_plot, save_path=msa_coverage_path, custom_colors=custom_colors)
            logger.info(f"MSA coverage plot saved to {msa_coverage_path}")
        else:
            logger.info(f"No {base_name} sequences of query length {full_len} found for MSA coverage plot.")
    else:
        logger.debug(f"MSA coverage plot generation skipped for {base_name}.")

    # Coevolution plot
    if plot_coevolution_flag:
        from Bio.SeqRecord import SeqRecord
        from Bio.Seq import Seq
        seq_records = [SeqRecord(Seq(s), id=f"seq_{i}", description="") for i, s in enumerate(sequences)]
        if len(seq_records) > 1:
            logger.info(f"Generating coevolution map from {base_name} MSA.")
            coevol_matrix = get_coevolution_numpy(seq_records)
            plot_path = os.path.join(img_dir, f'coevolution_{base_name}_msa.png')
            plot_coevolution(coevol_matrix, plot_path)
            logger.info(f"Coevolution map saved to {plot_path}")
        else:
            logger.info(f"Skipping coevolution plot for {base_name} as fewer than 2 sequences are available.")
    else:
        logger.debug(f"Coevolution map generation skipped for {base_name}.")
