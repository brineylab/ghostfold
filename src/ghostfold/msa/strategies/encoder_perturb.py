import torch
from transformers import GenerationConfig, LogitsProcessorList
from transformers.modeling_outputs import BaseModelOutput

from ghostfold.msa.model import FiniteLogitsProcessor, generate_aa, preprocess_sequence

from .base import BaseStrategy


class EncoderPerturbStrategy(BaseStrategy):
    """Perturb ProstT5 encoder hidden states before decoding.

    Rather than sampling diversity via temperature in token space, this
    strategy walks the model's *learned structural latent space*: encoder
    hidden states are shifted by Gaussian noise at multiple scales, then
    the decoder generates from the perturbed representation.  Because the
    perturbation lives in the same semantic space the model was trained in,
    the resulting sequences tend to be structurally coherent even when they
    diverge significantly from the query.

    For each sigma in noise_scales: one noise draw → num_return_sequences
    3Di samples → 1 AA per 3Di sample.
    Total output = len(noise_scales) × num_return_sequences sequences.
    """

    name = "encoder_perturb"

    def generate_msa(
        self,
        query_seq: str,
        model,
        tokenizer,
        device: torch.device,
        config: dict,
    ) -> list[str]:
        noise_scales: list[float] = config.get("noise_scales", [0.05, 0.15, 0.35])
        num_return_sequences: int = config.get("num_return_sequences", 5)
        decode_conf: dict = config.get(
            "decode_conf", {"temperature": 0.7, "top_k": 20, "top_p": 0.95}
        )

        seq_input = preprocess_sequence(query_seq, "<AA2fold>")
        ids = tokenizer([seq_input], padding="longest", return_tensors="pt").to(device)
        max_len = ids.input_ids.shape[1] + 1
        processors = LogitsProcessorList([FiniteLogitsProcessor()])

        # Encode once; reuse for all noise scales to avoid redundant encoder passes
        with torch.no_grad():
            enc_out = model.encoder(
                input_ids=ids.input_ids,
                attention_mask=ids.attention_mask,
            )
        hidden = enc_out.last_hidden_state  # (1, L, d_model)

        all_aa: list[str] = []

        for sigma in noise_scales:
            noise = torch.randn_like(hidden) * sigma
            perturbed = hidden + noise

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
                    # Passing encoder_outputs skips the encoder; decoder attends
                    # to the perturbed hidden states instead of the real ones.
                    encoder_outputs=BaseModelOutput(last_hidden_state=perturbed),
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
