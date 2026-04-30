import itertools
import random
from typing import List

from ghostfold.core.logging import get_logger

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
