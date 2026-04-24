from .base import BaseStrategy


class SingleSequenceStrategy(BaseStrategy):
    """Fold the query sequence alone — no MSA, no generated sequences.

    Produces a single-sequence A3M containing only the query.  This serves
    as a complement to the baseline strategy: baseline tries (and sometimes
    OOMs) to generate a full MSA, while this approach skips generation
    entirely and lets ColabFold run in true single-sequence mode.
    """

    name = "single_sequence"
    models_needed: frozenset[str] = frozenset()  # no heavyweight models required

    def generate_msa(self, query_seq, model, tokenizer, device, config) -> list[str]:
        return []
