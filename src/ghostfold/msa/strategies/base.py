from abc import ABC, abstractmethod

import torch


class BaseStrategy(ABC):
    name: str = "base"
    # Declare which heavyweight models this strategy requires on GPU.
    # The runner uses this to move unused models to CPU before generation,
    # freeing VRAM for strategies that only need a subset.
    # Valid tokens: "main" (full ProstT5), "encoder" (T5EncoderModel),
    #               "cnn_3di" (AA→3Di CNN), "cnn_aa" (3Di→AA CNN).
    models_needed: frozenset[str] = frozenset(["main"])

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
