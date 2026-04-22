import torch
from transformers import LogitsProcessorList

from ghostfold.msa.model import FiniteLogitsProcessor, generate_aa, preprocess_sequence

from .base import BaseStrategy


class DiverseBeamStrategy(BaseStrategy):
    """AA→3Di via grouped beam search, then 3Di→AA with sampling.

    diversity_penalty forces each beam group to generate distinct token sequences,
    giving structurally spread outputs without relying on temperature noise.
    num_beams == num_beam_groups means every returned sequence comes from a
    separate group — maximum forced divergence.
    """

    name = "diverse_beam"

    def generate_msa(
        self,
        query_seq: str,
        model,
        tokenizer,
        device: torch.device,
        config: dict,
    ) -> list[str]:
        num_beams: int = config.get("num_beams", 8)
        diversity_penalty: float = config.get("diversity_penalty", 1.0)
        decode_conf: dict = config.get(
            "decode_conf", {"temperature": 0.7, "top_k": 20, "top_p": 0.95}
        )

        seq_input = preprocess_sequence(query_seq, "<AA2fold>")
        ids = tokenizer([seq_input], padding="longest", return_tensors="pt").to(device)
        max_len = ids.input_ids.shape[1] + 1

        processors = LogitsProcessorList([FiniteLogitsProcessor()])

        with torch.no_grad():
            threedi_outputs = model.generate(
                input_ids=ids.input_ids,
                attention_mask=ids.attention_mask,
                max_length=max_len,
                num_beams=num_beams,
                num_beam_groups=num_beams,
                diversity_penalty=diversity_penalty,
                num_return_sequences=num_beams,
                do_sample=False,
                logits_processor=processors,
            )

        threedi_seqs = tokenizer.batch_decode(
            threedi_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        threedi_seqs = [s.replace(" ", "") for s in threedi_seqs]

        return generate_aa(threedi_seqs, tokenizer, model, device, 1, decode_conf)
