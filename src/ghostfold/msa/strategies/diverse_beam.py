import logging

import torch
from transformers import GenerationConfig, LogitsProcessorList

from ghostfold.msa.model import (
    FiniteLogitsProcessor,
    generate_3di,
    generate_aa,
    preprocess_sequence,
)

from .base import BaseStrategy

logger = logging.getLogger("ghostfold.benchmark.diverse_beam")


class DiverseBeamStrategy(BaseStrategy):
    """AA→3Di via grouped beam search, then 3Di→AA with sampling.

    Runs beam search in chunks of *chunk_beams* beam groups to cap peak VRAM.
    E.g. num_beams=48 with chunk_beams=2 runs 24 passes × 2-beam search.
    If chunk_beams=1 still OOMs, falls back to high-temperature sampling so
    the strategy always returns sequences rather than silently producing 0.
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
        num_beams: int = config.get("num_beams", 48)
        diversity_penalty: float = config.get("diversity_penalty", 1.0)
        chunk_beams: int = config.get("chunk_beams", 2)
        decode_conf: dict = config.get(
            "decode_conf", {"temperature": 0.7, "top_k": 20, "top_p": 0.95}
        )

        seq_input = preprocess_sequence(query_seq, "<AA2fold>")
        ids = tokenizer([seq_input], padding="longest", return_tensors="pt").to(device)
        max_len = ids.input_ids.shape[1] + 1
        processors = LogitsProcessorList([FiniteLogitsProcessor()])

        threedi_seqs: list[str] = []
        remaining = num_beams
        current_chunk = min(chunk_beams, remaining)
        beam_oom = False

        while remaining > 0:
            torch.cuda.empty_cache()
            try:
                gen_cfg = GenerationConfig(
                    max_length=max_len,
                    num_beams=current_chunk,
                    num_beam_groups=current_chunk,
                    diversity_penalty=diversity_penalty,
                    num_return_sequences=current_chunk,
                    do_sample=False,
                )
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=ids.input_ids,
                        attention_mask=ids.attention_mask,
                        generation_config=gen_cfg,
                        logits_processor=processors,
                        trust_remote_code=True,
                        custom_generate="transformers-community/group-beam-search",
                    )
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                torch.cuda.empty_cache()
                if current_chunk == 1:
                    logger.warning(
                        "diverse_beam: group beam search OOM even at chunk_beams=1; "
                        "falling back to high-temperature sampling for remaining %d sequences.",
                        remaining,
                    )
                    beam_oom = True
                    break
                current_chunk = max(1, current_chunk // 2)
                continue

            batch = tokenizer.batch_decode(
                outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            threedi_seqs.extend(s.replace(" ", "") for s in batch)
            remaining -= current_chunk
            current_chunk = min(current_chunk, remaining)

        # Sampling fallback: generate remaining sequences via temperature sampling
        # using generate_3di so they still pass through the structural space.
        if beam_oom and remaining > 0:
            torch.cuda.empty_cache()
            fallback_conf = {**decode_conf, "temperature": max(decode_conf.get("temperature", 0.7), 1.0)}
            try:
                sampled = generate_3di(
                    [query_seq], tokenizer, model, device, remaining, fallback_conf
                )
                threedi_seqs.extend(sampled)
            except Exception as exc:
                logger.error("diverse_beam sampling fallback failed: %s", exc)

        if not threedi_seqs:
            return []

        return generate_aa(threedi_seqs, tokenizer, model, device, 1, decode_conf)
