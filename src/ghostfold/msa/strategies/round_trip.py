import torch

from ghostfold.msa.model import generate_3di, generate_aa

from .base import BaseStrategy


class RoundTripStrategy(BaseStrategy):
    """Iterative AA→3Di→AA chains that simulate evolutionary drift.

    Each round introduces a small stochastic shift: sequences from early
    rounds are conservative variants; later rounds are more diverged.
    Population size stays constant at n_seeds across all rounds, so total
    output = n_seeds × n_rounds sequences.
    """

    name = "round_trip"

    def generate_msa(
        self,
        query_seq: str,
        model,
        tokenizer,
        device: torch.device,
        config: dict,
    ) -> list[str]:
        n_seeds: int = config.get("n_seeds", 8)
        n_rounds: int = config.get("n_rounds", 4)
        decode_conf: dict = config.get(
            "decode_conf", {"temperature": 0.7, "top_k": 20, "top_p": 0.95}
        )

        # Bootstrap: generate n_seeds starting sequences from the query
        fold_seqs = generate_3di([query_seq], tokenizer, model, device, n_seeds, decode_conf)
        current_seqs = generate_aa(fold_seqs, tokenizer, model, device, 1, decode_conf)
        all_seqs: list[str] = list(current_seqs)

        for _ in range(1, n_rounds):
            fold_seqs = generate_3di(current_seqs, tokenizer, model, device, 1, decode_conf)
            current_seqs = generate_aa(fold_seqs, tokenizer, model, device, 1, decode_conf)
            all_seqs.extend(current_seqs)

        return all_seqs
