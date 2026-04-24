from __future__ import annotations

from typing import List

import torch

from ghostfold.msa.model import generate_3di, generate_aa

from .base import BaseStrategy

_DEFAULT_REPETITION_PENALTY = 1.2
_DEFAULT_TEMPERATURES = [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0]


class TemperatureSweepStrategy(BaseStrategy):
    """Sweep temperature across a range during AA→3Di→AA translation."""

    name = "temperature_sweep"

    def generate_msa(
        self,
        query_seq: str,
        model,
        tokenizer,
        device: torch.device,
        config: dict,
    ) -> List[str]:
        temperatures: List[float] = config.get("temperatures", _DEFAULT_TEMPERATURES)
        num_return_sequences: int = config.get("num_return_sequences", 20)
        base_conf: dict = dict(config.get("base_decode_conf", {"top_p": 0.85, "top_k": 3}))
        base_conf.setdefault("repetition_penalty", _DEFAULT_REPETITION_PENALTY)

        seed_conf = dict(base_conf)
        seed_conf["temperature"] = 1.0
        threedi_seqs = generate_3di(
            [list(query_seq)], tokenizer, model, device, 1, seed_conf
        )

        all_aa: List[str] = []
        for temp in temperatures:
            conf = dict(base_conf)
            conf["temperature"] = temp
            aa_seqs = generate_aa(
                [list(s) for s in threedi_seqs],
                tokenizer,
                model,
                device,
                num_return_sequences,
                conf,
            )
            all_aa.extend(aa_seqs)
        return all_aa
