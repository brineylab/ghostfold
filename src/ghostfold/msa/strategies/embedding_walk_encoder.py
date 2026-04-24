"""EmbeddingWalkEncoderStrategy: walk T5 encoder embedding space along PCA directions, then decode."""

from __future__ import annotations

import numpy as np
import torch
from transformers import GenerationConfig, LogitsProcessorList
from transformers.modeling_outputs import BaseModelOutput

from ghostfold.msa.model import FiniteLogitsProcessor, generate_aa, preprocess_sequence

from .base import BaseStrategy


def _get_embeddings(
    query_seq: str,
    encoder_model,
    tokenizer,
    device: torch.device,
) -> torch.Tensor:
    """Return (1, L, d_model) encoder embeddings for query_seq."""
    prefix = "<AA2fold>"
    seq_input = prefix + " " + " ".join(list(query_seq))
    encoding = tokenizer([seq_input], padding="longest", return_tensors="pt").to(device)
    with torch.no_grad():
        out = encoder_model(
            input_ids=encoding.input_ids,
            attention_mask=encoding.attention_mask,
        )
    return out.last_hidden_state  # (1, L, d_model)


def _pca_directions(
    base_emb: torch.Tensor,
    n_components: int,
    n_samples: int,
    sigma: float,
) -> np.ndarray:
    """Fit PCA on noise-augmented copies of base_emb. Return (n_components, L*d) directions."""
    L, d = base_emb.shape[1], base_emb.shape[2]
    flat = base_emb.squeeze(0).cpu().float().numpy().reshape(1, L * d)
    noise = np.random.randn(n_samples, L * d) * sigma
    samples = flat + noise
    samples -= samples.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(samples, full_matrices=False)
    return Vt[:n_components]


class EmbeddingWalkEncoderStrategy(BaseStrategy):
    """Walk encoder embedding space along PCA directions, then decode."""

    name = "embedding_walk_encoder"

    def generate_msa(
        self,
        query_seq: str,
        model,
        tokenizer,
        device: torch.device,
        config: dict,
    ) -> list[str]:
        encoder_model = config.get("encoder_model")
        if encoder_model is None:
            raise ValueError("embedding_walk_encoder requires 'encoder_model' in config")

        n_pca_components: int = config.get("n_pca_components", 10)
        step_sizes: list[float] = config.get("step_sizes", [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])
        n_noise_samples: int = config.get("n_noise_samples", 50)
        noise_sigma: float = config.get("noise_sigma", 0.1)
        num_return_sequences: int = config.get("num_return_sequences", 15)
        decode_conf: dict = config.get("decode_conf", {"temperature": 0.7, "top_k": 20, "top_p": 0.95})

        base_emb = _get_embeddings(query_seq, encoder_model, tokenizer, device)
        L, d = base_emb.shape[1], base_emb.shape[2]

        directions = _pca_directions(base_emb, n_pca_components, n_noise_samples, noise_sigma)

        seq_input = preprocess_sequence(query_seq, "<AA2fold>")
        ids = tokenizer([seq_input], padding="longest", return_tensors="pt").to(device)
        max_len = ids.input_ids.shape[1] + 1
        processors = LogitsProcessorList([FiniteLogitsProcessor()])

        flat_base = base_emb.squeeze(0).cpu().float().numpy().reshape(L * d)

        all_aa: list[str] = []
        for pc_idx in range(min(n_pca_components, len(directions))):
            direction = directions[pc_idx]
            for step in step_sizes:
                walked_flat = flat_base + step * direction
                walked = torch.tensor(walked_flat, dtype=base_emb.dtype, device=device).reshape(1, L, d)

                gen_cfg = GenerationConfig(
                    max_length=max_len,
                    min_length=max_len - 2,
                    num_return_sequences=num_return_sequences,
                    num_beams=1,
                    do_sample=True,
                    **decode_conf,
                )
                with torch.no_grad():
                    threedi_outputs = model.generate(
                        input_ids=ids.input_ids,
                        attention_mask=ids.attention_mask,
                        encoder_outputs=BaseModelOutput(last_hidden_state=walked),
                        generation_config=gen_cfg,
                        logits_processor=processors,
                    )

                threedi_seqs = tokenizer.batch_decode(
                    threedi_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                threedi_seqs = [s.replace(" ", "") for s in threedi_seqs]
                aa_seqs = generate_aa(threedi_seqs, tokenizer, model, device, 1, decode_conf)
                all_aa.extend(aa_seqs)

        return all_aa
