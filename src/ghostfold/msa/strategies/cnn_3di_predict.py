"""CnnThreediPredictStrategy: CNN predicts 3Di from full-model encoder, then T5 decodes AA.

Unlike encoder_only_3di_sub, this strategy uses the full ProstT5 model's own
encoder (model.encoder) instead of a separately loaded T5EncoderModel.  This
saves ~5.6 GB of VRAM and is the preferred approach on GPUs with <24 GB.
"""

from __future__ import annotations

import torch

from ghostfold.msa.model import generate_aa
from ghostfold.msa.strategies.encoder_only_3di_sub import _3DI_ALPHABET
from ghostfold.msa.strategies.threedipperturb import _mutate_3di

from .base import BaseStrategy


def _predict_3di_from_main_encoder(
    query_seq: str,
    model,
    tokenizer,
    device: torch.device,
    cnn,
) -> str:
    """Use the full model's encoder + CNN to deterministically predict 3Di."""
    prefix = "<AA2fold>"
    seq_input = prefix + " " + " ".join(list(query_seq))
    ids = tokenizer([seq_input], padding="longest", return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.encoder(
            input_ids=ids.input_ids,
            attention_mask=ids.attention_mask,
        ).last_hidden_state  # (1, L+2, 1024)
    emb = emb[:, 1 : len(query_seq) + 1, :]  # strip prefix + EOS → (1, L, 1024)
    with torch.no_grad():
        logits = cnn(emb)  # (1, 20, L)
    indices = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
    return "".join(_3DI_ALPHABET[i] for i in indices)


class CnnThreediPredictStrategy(BaseStrategy):
    """CNN predicts 3Di (via full-model encoder), perturb, then T5 decodes AA.

    Pipeline:
        AA query → full ProstT5 encoder → CNN → base 3Di (deterministic)
        base 3Di → perturb at N mutation rates × M times → T5 decode AA

    Total sequences ≈ len(mutation_rates) × n_perturbations × num_return_sequences.
    Does NOT require a separately loaded T5EncoderModel.
    """

    name = "cnn_3di_predict"
    models_needed = frozenset(["main", "cnn_3di"])

    def generate_msa(
        self,
        query_seq: str,
        model,
        tokenizer,
        device: torch.device,
        config: dict,
    ) -> list[str]:
        cnn_3di = config.get("cnn_3di")
        if cnn_3di is None:
            raise ValueError("cnn_3di_predict requires 'cnn_3di' in config")

        mutation_rates: list[float] = config.get(
            "mutation_rates", [0.05, 0.10, 0.20, 0.30, 0.40]
        )
        n_perturbations: int = config.get("n_perturbations", 4)
        num_return_sequences: int = config.get("num_return_sequences", 7)
        decode_conf: dict = config.get(
            "decode_conf", {"temperature": 0.7, "top_k": 20, "top_p": 0.95}
        )

        base_3di = _predict_3di_from_main_encoder(
            query_seq, model, tokenizer, device, cnn_3di
        )

        all_aa: list[str] = []
        for rate in mutation_rates:
            for _ in range(n_perturbations):
                mutated = _mutate_3di(base_3di, rate)
                aa_seqs = generate_aa(
                    [mutated], tokenizer, model, device, num_return_sequences, decode_conf
                )
                all_aa.extend(aa_seqs)
        return all_aa
