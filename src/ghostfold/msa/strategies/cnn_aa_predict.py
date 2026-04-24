"""CnnAaPredictStrategy: T5 samples diverse 3Di sequences → CNN predicts AA directly.

This is the reverse of cnn_3di_predict: instead of using the CNN to predict 3Di
and then the T5 decoder for AA, we use the T5 decoder for 3Di (for diversity) and
then use the CNN to skip the T5 AA decoder entirely.  The CNN 3Di→AA head is
deterministic, so diversity comes exclusively from the upstream 3Di sampling.
"""

from __future__ import annotations

import torch

from ghostfold.msa.model import generate_3di
from ghostfold.msa.strategies.encoder_only_3di_sub import _AA_ALPHABET

from .base import BaseStrategy


def _cnn_predict_aa(
    threedi_seq: str,
    model,
    tokenizer,
    device: torch.device,
    cnn_aa,
) -> str:
    """Encode a 3Di sequence with the full model's encoder, then CNN → AA."""
    prefix = "<fold2AA>"
    seq_input = prefix + " " + " ".join(list(threedi_seq))
    ids = tokenizer([seq_input], padding="longest", return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.encoder(
            input_ids=ids.input_ids,
            attention_mask=ids.attention_mask,
        ).last_hidden_state  # (1, L+2, 1024)
    emb = emb[:, 1 : len(threedi_seq) + 1, :]  # strip prefix + EOS → (1, L, 1024)
    with torch.no_grad():
        logits = cnn_aa(emb)  # (1, 20, L)
    indices = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
    return "".join(_AA_ALPHABET[i] for i in indices)


class CnnAaPredictStrategy(BaseStrategy):
    """T5 decoder samples diverse 3Di → CNN 3Di→AA predicts AA directly.

    Pipeline:
        AA query → T5 AA→3Di decoder (sampled, n_3di_seeds times) → diverse 3Di pool
        each 3Di → full ProstT5 encoder → CNN → AA (deterministic per 3Di)

    Total sequences = n_3di_seeds (one CNN-predicted AA per 3Di sample).
    Diversity comes from the 3Di sampling step; CNN decoding is fast and
    sidesteps the T5 AA decoder.
    """

    name = "cnn_aa_predict"
    models_needed = frozenset(["main", "cnn_aa"])

    def generate_msa(
        self,
        query_seq: str,
        model,
        tokenizer,
        device: torch.device,
        config: dict,
    ) -> list[str]:
        cnn_aa = config.get("cnn_aa")
        if cnn_aa is None:
            raise ValueError("cnn_aa_predict requires 'cnn_aa' in config")

        n_3di_seeds: int = config.get("n_3di_seeds", 128)
        decode_conf: dict = config.get(
            "decode_conf", {"temperature": 1.0, "top_k": 20, "top_p": 0.95}
        )

        threedi_seqs = generate_3di(
            [query_seq], tokenizer, model, device, n_3di_seeds, decode_conf
        )

        all_aa: list[str] = []
        for threedi_seq in threedi_seqs:
            aa_seq = _cnn_predict_aa(threedi_seq, model, tokenizer, device, cnn_aa)
            all_aa.append(aa_seq)
        return all_aa
