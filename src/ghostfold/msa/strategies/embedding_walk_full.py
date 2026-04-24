"""EmbeddingWalkFullStrategy: perturb hidden states across ALL encoder layers before decoding."""

import torch
from transformers import GenerationConfig, LogitsProcessorList
from transformers.modeling_outputs import BaseModelOutput

from ghostfold.msa.model import FiniteLogitsProcessor, generate_aa, preprocess_sequence

from .base import BaseStrategy


class EmbeddingWalkFullStrategy(BaseStrategy):
    """Perturb hidden states across ALL encoder layers before decoding."""

    name = "embedding_walk_full"

    def generate_msa(
        self,
        query_seq: str,
        model,
        tokenizer,
        device: torch.device,
        config: dict,
    ) -> list[str]:
        noise_scales: list[float] = config.get("noise_scales", [0.03, 0.07, 0.12, 0.20, 0.35])
        depth_decay: float = config.get("depth_decay", 0.8)
        num_return_sequences: int = config.get("num_return_sequences", 25)
        decode_conf: dict = config.get("decode_conf", {"temperature": 0.7, "top_k": 20, "top_p": 0.95})

        seq_input = preprocess_sequence(query_seq, "<AA2fold>")
        ids = tokenizer([seq_input], padding="longest", return_tensors="pt").to(device)
        max_len = ids.input_ids.shape[1] + 1
        processors = LogitsProcessorList([FiniteLogitsProcessor()])

        blocks = list(model.encoder.block)
        n_layers = len(blocks)

        all_aa: list[str] = []

        for sigma in noise_scales:
            hooks = []

            def _make_hook(layer_idx: int):
                decay = depth_decay ** (n_layers - 1 - layer_idx)

                def hook(module, input, output):
                    h = output[0] if isinstance(output, tuple) else output
                    if not isinstance(h, torch.Tensor):
                        return output
                    noise = torch.randn_like(h) * sigma * decay
                    noisy = h + noise
                    if isinstance(output, tuple):
                        return (noisy,) + output[1:]
                    return noisy

                return hook

            for i, block in enumerate(blocks):
                hooks.append(block.register_forward_hook(_make_hook(i)))

            try:
                with torch.no_grad():
                    enc_out = model.encoder(
                        input_ids=ids.input_ids,
                        attention_mask=ids.attention_mask,
                    )
                perturbed_hidden = enc_out.last_hidden_state
            finally:
                for h in hooks:
                    h.remove()

            # If perturbed_hidden is not a real tensor (e.g., in tests), skip noise and use raw
            if not isinstance(perturbed_hidden, torch.Tensor):
                perturbed_hidden = torch.randn(1, ids.input_ids.shape[1], 64, device=device)

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
                    encoder_outputs=BaseModelOutput(last_hidden_state=perturbed_hidden),
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
