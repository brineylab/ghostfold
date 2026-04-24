from pathlib import Path

import torch
import torch.nn as nn

from ghostfold.msa.model import generate_aa
from ghostfold.msa.strategies.threedipperturb import _mutate_3di

from .base import BaseStrategy

_3DI_ALPHABET = "acdefghiklmnpqrstvwy"
_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

_CNN_URL = "https://github.com/mheinzinger/ProstT5/raw/main/cnn_chkpnt/model.pt"
_CNN_CACHE = Path.home() / ".cache" / "ghostfold" / "cnn_3di.pt"

_CNN_AA_URL = "https://github.com/mheinzinger/ProstT5/raw/main/cnn_chkpnt_AA_CNN/model.pt"
_CNN_AA_CACHE = Path.home() / ".cache" / "ghostfold" / "cnn_aa.pt"


class _CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(1024, 32, kernel_size=(7, 1), padding=(3, 0)),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Conv2d(32, 20, kernel_size=(7, 1), padding=(3, 0)),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1).unsqueeze(dim=-1)
        return self.classifier(x).squeeze(dim=-1)


def load_cnn_3di(device: torch.device) -> _CNN:
    _CNN_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if not _CNN_CACHE.exists():
        from urllib import request
        request.urlretrieve(_CNN_URL, _CNN_CACHE)
    state = torch.load(_CNN_CACHE, map_location=device)
    model = _CNN()
    model.load_state_dict(state["state_dict"])
    return model.eval().to(device)


def load_cnn_aa(device: torch.device) -> _CNN:
    """Load the CNN that predicts AA directly from T5 encoder embeddings of 3Di input."""
    _CNN_AA_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if not _CNN_AA_CACHE.exists():
        from urllib import request
        request.urlretrieve(_CNN_AA_URL, _CNN_AA_CACHE)
    state = torch.load(_CNN_AA_CACHE, map_location=device)
    model = _CNN()
    model.load_state_dict(state["state_dict"])
    return model.eval().to(device)


def _predict_3di_encoder(
    query_seq: str,
    encoder_model,
    tokenizer,
    device: torch.device,
    cnn: nn.Module,
) -> str:
    prefix = "<AA2fold>"
    seq_input = prefix + " " + " ".join(list(query_seq))
    encoding = tokenizer(
        [seq_input], add_special_tokens=True, padding="longest", return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        emb = encoder_model(
            input_ids=encoding.input_ids,
            attention_mask=encoding.attention_mask,
        ).last_hidden_state  # (1, L+2, 1024)
    emb = emb[:, 1 : len(query_seq) + 1, :]  # (1, L, 1024)
    with torch.no_grad():
        logits = cnn(emb)  # (1, 20, L)
    indices = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()  # (L,)
    return "".join(_3DI_ALPHABET[i] for i in indices)


class EncoderOnly3DiSubStrategy(BaseStrategy):
    """Deterministic encoder-only 3Di → sub matrix variants → full decoder AA."""

    name = "encoder_only_3di_sub"
    models_needed = frozenset(["main", "encoder", "cnn_3di"])

    def generate_msa(
        self,
        query_seq: str,
        model,
        tokenizer,
        device: torch.device,
        config: dict,
    ) -> list[str]:
        encoder_model = config.get("encoder_model")
        cnn_3di = config.get("cnn_3di")
        mutation_rates: list[float] = config.get("mutation_rates", [0.05, 0.10, 0.20, 0.30, 0.40])
        variants_per_rate: int = config.get("variants_per_rate", 10)
        num_return_sequences: int = config.get("num_return_sequences", 3)
        decode_conf: dict = config.get("decode_conf", {"temperature": 0.7, "top_k": 20, "top_p": 0.95})

        if encoder_model is None or cnn_3di is None:
            raise ValueError("encoder_only_3di_sub requires 'encoder_model' and 'cnn_3di' in config")

        base_3di = _predict_3di_encoder(query_seq, encoder_model, tokenizer, device, cnn_3di)

        all_aa: list[str] = []
        for rate in mutation_rates:
            for _ in range(variants_per_rate):
                mutated = _mutate_3di(base_3di, rate)
                aa_seqs = generate_aa([mutated], tokenizer, model, device, num_return_sequences, decode_conf)
                all_aa.extend(aa_seqs)
        return all_aa
