import itertools
import random
from typing import List

from ghostfold.core.logging import get_logger
from ghostfold.msa.neff import calculate_neff
from ghostfold.msa.filters import deduplicate

logger = get_logger("pairing")


def _reservoir_sample_product(per_chain_seqs: List[List[str]], k: int) -> List[str]:
    """Reservoir-sample k concatenated sequences from the Cartesian product.

    Never materializes the full product — streams via itertools.product.
    Returns fewer than k items if the product is smaller than k.
    """
    reservoir: List[str] = []
    for i, combo in enumerate(itertools.product(*per_chain_seqs)):
        concat = "".join(combo)
        if i < k:
            reservoir.append(concat)
        else:
            j = random.randint(0, i)
            if j < k:
                reservoir[j] = concat
    return reservoir


def build_paired_msa(
    per_chain_seqs: List[List[str]],
    n_subsets: int = 20,
    subset_size: int = 175,
    top_k: int = 5,
    neff_threshold: float = 0.8,
) -> List[str]:
    """Build a paired MSA block via Cartesian-product reservoir sampling.

    Args:
        per_chain_seqs: One list of sequences per chain.
        n_subsets: Number of random subsets sampled from the Cartesian product.
        subset_size: Sequences per subset.
        top_k: Number of top-Neff subsets to merge.
        neff_threshold: Identity threshold passed to calculate_neff.

    Returns:
        Deduplicated list of fully-concatenated paired sequences.
        Empty list if any chain is empty or product is empty.
    """
    if any(len(chain) == 0 for chain in per_chain_seqs):
        logger.warning("build_paired_msa: one or more chains empty → returning []")
        return []

    if len(per_chain_seqs) == 1:
        return list(per_chain_seqs[0])

    subsets: List[List[str]] = []
    for _ in range(n_subsets):
        sample = _reservoir_sample_product(per_chain_seqs, k=subset_size)
        if sample:
            subsets.append(sample)

    if not subsets:
        logger.warning("build_paired_msa: all subsets empty → returning []")
        return []

    scored = sorted(
        subsets,
        key=lambda s: calculate_neff(s, identity_threshold=neff_threshold),
        reverse=True,
    )

    merged: List[str] = []
    for subset in scored[:top_k]:
        merged.extend(subset)

    result = deduplicate(merged)
    if not result:
        logger.warning("build_paired_msa: dedup reduced to 0 sequences → returning []")
    return result
