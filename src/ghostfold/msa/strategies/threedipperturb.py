import random

import numpy as np

from ghostfold.msa.model import generate_3di, generate_aa

from .base import BaseStrategy

# 3Di alphabet as produced by ProstT5 (lowercase structural tokens).
# X (gap/unknown) is excluded — never mutated to or from.
_3DI_TOKENS = list("acdefghiklmnpqrstvwy")

# Official Foldseek 3Di substitution matrix (from Foldseek data/3di.mat).
# Rows/columns ordered A C D E F G H I K L M N P Q R S T V W Y.
# Values are log-odds scores; higher = more structurally similar.
# ProstT5 emits lowercase tokens; keys here are lowercase accordingly.
_3DI_SUBST_MATRIX: dict[str, dict[str, int]] = {
    "a": {"a":  6, "c": -3, "d":  1, "e":  2, "f":  3, "g": -2, "h": -2, "i": -7, "k": -3, "l": -3, "m":-10, "n": -5, "p": -1, "q":  1, "r": -4, "s": -7, "t": -5, "v": -6, "w":  0, "y": -2},
    "c": {"a": -3, "c":  6, "d": -2, "e": -8, "f": -5, "g": -4, "h": -4, "i":-12, "k":-13, "l":  1, "m":-14, "n":  0, "p":  0, "q":  1, "r": -1, "s":  0, "t": -8, "v":  1, "w": -7, "y": -9},
    "d": {"a":  1, "c": -2, "d":  4, "e": -3, "f":  0, "g":  1, "h":  1, "i": -3, "k": -5, "l": -4, "m": -5, "n": -2, "p":  1, "q": -1, "r": -1, "s": -4, "t": -2, "v": -3, "w": -2, "y": -2},
    "e": {"a":  2, "c": -8, "d": -3, "e":  9, "f": -2, "g": -7, "h": -4, "i":-12, "k":-10, "l": -7, "m":-17, "n": -8, "p": -6, "q": -3, "r": -8, "s":-10, "t":-10, "v":-13, "w": -6, "y": -3},
    "f": {"a":  3, "c": -5, "d":  0, "e": -2, "f":  7, "g": -3, "h": -3, "i": -5, "k":  1, "l": -3, "m": -9, "n": -5, "p": -2, "q":  2, "r": -5, "s": -8, "t": -3, "v": -7, "w":  4, "y": -4},
    "g": {"a": -2, "c": -4, "d":  1, "e": -7, "f": -3, "g":  6, "h":  3, "i":  0, "k": -7, "l": -7, "m": -1, "n": -2, "p": -2, "q": -4, "r":  3, "s": -3, "t":  4, "v": -6, "w": -4, "y": -2},
    "h": {"a": -2, "c": -4, "d":  1, "e": -4, "f": -3, "g":  3, "h":  6, "i": -4, "k": -7, "l": -6, "m": -6, "n":  0, "p": -1, "q": -3, "r":  1, "s": -3, "t": -1, "v": -5, "w": -5, "y":  3},
    "i": {"a": -7, "c":-12, "d": -3, "e":-12, "f": -5, "g":  0, "h": -4, "i":  8, "k": -5, "l":-11, "m":  7, "n": -7, "p": -6, "q": -6, "r": -3, "s": -9, "t":  6, "v":-12, "w": -5, "y": -8},
    "k": {"a": -3, "c":-13, "d": -5, "e":-10, "f":  1, "g": -7, "h": -7, "i": -5, "k":  9, "l":-11, "m": -8, "n":-12, "p": -6, "q": -5, "r": -9, "s":-14, "t": -5, "v":-15, "w":  5, "y": -8},
    "l": {"a": -3, "c":  1, "d": -4, "e": -7, "f": -3, "g": -7, "h": -6, "i":-11, "k":-11, "l":  6, "m":-16, "n": -3, "p": -2, "q":  2, "r": -4, "s": -4, "t": -9, "v":  0, "w": -8, "y": -9},
    "m": {"a":-10, "c":-14, "d": -5, "e":-17, "f": -9, "g": -1, "h": -6, "i":  7, "k": -8, "l":-16, "m": 10, "n": -9, "p": -9, "q":-10, "r": -5, "s":-10, "t":  3, "v":-16, "w": -6, "y": -9},
    "n": {"a": -5, "c":  0, "d": -2, "e": -8, "f": -5, "g": -2, "h":  0, "i": -7, "k":-12, "l": -3, "m": -9, "n":  7, "p":  0, "q": -2, "r":  2, "s":  3, "t": -4, "v":  0, "w": -8, "y": -5},
    "p": {"a": -1, "c":  0, "d":  1, "e": -6, "f": -2, "g": -2, "h": -1, "i": -6, "k": -6, "l": -2, "m": -9, "n":  0, "p":  4, "q":  0, "r":  0, "s": -2, "t": -4, "v":  0, "w": -4, "y": -5},
    "q": {"a":  1, "c":  1, "d": -1, "e": -3, "f":  2, "g": -4, "h": -3, "i": -6, "k": -5, "l":  2, "m":-10, "n": -2, "p":  0, "q":  5, "r": -2, "s": -4, "t": -5, "v": -1, "w": -2, "y": -5},
    "r": {"a": -4, "c": -1, "d": -1, "e": -8, "f": -5, "g":  3, "h":  1, "i": -3, "k": -9, "l": -4, "m": -5, "n":  2, "p":  0, "q": -2, "r":  6, "s":  2, "t":  0, "v": -1, "w": -6, "y": -3},
    "s": {"a": -7, "c":  0, "d": -4, "e":-10, "f": -8, "g": -3, "h": -3, "i": -9, "k":-14, "l": -4, "m":-10, "n":  3, "p": -2, "q": -4, "r":  2, "s":  6, "t": -6, "v":  0, "w":-11, "y": -9},
    "t": {"a": -5, "c": -8, "d": -2, "e":-10, "f": -3, "g":  4, "h": -1, "i":  6, "k": -5, "l": -9, "m":  3, "n": -4, "p": -4, "q": -5, "r":  0, "s": -6, "t":  8, "v": -9, "w": -5, "y": -5},
    "v": {"a": -6, "c":  1, "d": -3, "e":-13, "f": -7, "g": -6, "h": -5, "i":-12, "k":-15, "l":  0, "m":-16, "n":  0, "p":  0, "q": -1, "r": -1, "s":  0, "t": -9, "v":  3, "w":-10, "y":-11},
    "w": {"a":  0, "c": -7, "d": -2, "e": -6, "f":  4, "g": -4, "h": -5, "i": -5, "k":  5, "l": -8, "m": -6, "n": -8, "p": -4, "q": -2, "r": -6, "s":-11, "t": -5, "v":-10, "w":  8, "y": -6},
    "y": {"a": -2, "c": -9, "d": -2, "e": -3, "f": -4, "g": -2, "h":  3, "i": -8, "k": -8, "l": -9, "m": -9, "n": -5, "p": -5, "q": -5, "r": -3, "s": -9, "t": -5, "v":-11, "w": -6, "y":  9},
}


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
