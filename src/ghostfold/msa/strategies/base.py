from abc import ABC, abstractmethod

import torch


class BaseStrategy(ABC):
    name: str = "base"

    @abstractmethod
    def generate_msa(
        self,
        query_seq: str,
        model,
        tokenizer,
        device: torch.device,
        config: dict,
    ) -> list[str]:
        """Return list of generated AA sequences (not including the query)."""
