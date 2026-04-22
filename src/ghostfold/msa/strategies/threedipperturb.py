import random

import numpy as np

from ghostfold.msa.model import generate_3di, generate_aa

from .base import BaseStrategy

# 3Di alphabet as produced by ProstT5 (lowercase structural tokens)
_3DI_TOKENS = list("acdefghiklmnpqrstvwy")

# Approximate structural clusters derived from Foldseek 3Di semantics.
# Replace with the official Foldseek matrix (data/3di.mat in the Foldseek
# repository) for higher fidelity substitution probabilities.
_TOKEN_CLUSTER: dict[str, int] = {
    "a": 0, "g": 0, "s": 0, "t": 0,   # helix / compact backbone
    "f": 1, "i": 1, "l": 1, "v": 1,   # extended / hydrophobic
    "d": 2, "e": 2, "n": 2, "q": 2,   # polar charged loops
    "h": 3, "k": 3, "p": 3, "r": 3,   # turn / charged
    "c": 4, "m": 4, "w": 4, "y": 4,   # aromatic / special
}


def _build_3di_matrix() -> dict[str, dict[str, int]]:
    mat: dict[str, dict[str, int]] = {}
    for t1 in _3DI_TOKENS:
        mat[t1] = {}
        for t2 in _3DI_TOKENS:
            if t1 == t2:
                mat[t1][t2] = 6
            elif _TOKEN_CLUSTER[t1] == _TOKEN_CLUSTER[t2]:
                mat[t1][t2] = 2
            else:
                mat[t1][t2] = -2
    return mat


_3DI_SUBST_MATRIX: dict[str, dict[str, int]] = _build_3di_matrix()


def _substitution_probs(token: str) -> dict[str, float]:
    """Softmax-normalised substitution probabilities for a 3Di token."""
    row = _3DI_SUBST_MATRIX.get(token)
    if not row:
        return {}
    exp_scores = {t: np.exp(float(v)) for t, v in row.items()}
    total = sum(exp_scores.values())
    return {t: v / total for t, v in exp_scores.items()} if total > 0 else {}


def _mutate_3di(seq: str, mutation_rate: float) -> str:
    """Apply structural-context-aware mutations to a 3Di sequence."""
    tokens = list(seq)
    mutable = [i for i, t in enumerate(tokens) if t in _3DI_SUBST_MATRIX]
    n_mutate = max(1, int(len(mutable) * mutation_rate))
    if n_mutate > len(mutable):
        n_mutate = len(mutable)
    positions = random.sample(mutable, n_mutate)
    for pos in positions:
        probs = _substitution_probs(tokens[pos])
        if probs:
            tokens[pos] = random.choices(list(probs), weights=list(probs.values()), k=1)[0]
    return "".join(tokens)


class ThreeDiPerturbStrategy(BaseStrategy):
    """Perturb 3Di structural tokens before back-translating to amino acids.

    Mutations happen in structural (3Di) space using a cluster-based
    substitution matrix, so every resulting AA sequence is guaranteed to
    come from a structurally plausible 3Di template.  Multiple mutation
    rates let the pool span different evolutionary distances.

    For each 3Di seed and each mutation_rate: one mutated 3Di template →
    num_return_sequences AA sequences.
    Total ≈ n_3di_seeds × len(mutation_rates) × num_return_sequences.
    """

    name = "3di_perturb"

    def generate_msa(
        self,
        query_seq: str,
        model,
        tokenizer,
        device,
        config: dict,
    ) -> list[str]:
        mutation_rates: list[float] = config.get("mutation_rates", [0.05, 0.15, 0.25])
        n_3di_seeds: int = config.get("n_3di_seeds", 3)
        num_return_sequences: int = config.get("num_return_sequences", 5)
        decode_conf: dict = config.get(
            "decode_conf", {"temperature": 0.7, "top_k": 20, "top_p": 0.95}
        )

        # Generate a small pool of 3Di seeds from the query
        seed_3di = generate_3di(
            [query_seq], tokenizer, model, device, n_3di_seeds, decode_conf
        )

        all_aa: list[str] = []
        for threedi_seq in seed_3di:
            for rate in mutation_rates:
                mutated = _mutate_3di(threedi_seq, rate)
                aa_seqs = generate_aa(
                    [mutated], tokenizer, model, device, num_return_sequences, decode_conf
                )
                all_aa.extend(aa_seqs)

        return all_aa
