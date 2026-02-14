import os
import random
import torch
from typing import List, Dict, Any
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM

from ghostfold.core.logging import get_logger
from ghostfold.io.fasta import append_fasta
from .model import generate_3di, generate_aa

logger = get_logger("generation")


def pad_sequence(seq: str, start_idx: int, full_len: int) -> str:
    """Pads a sequence with hyphens to a specified full length."""
    return '-' * start_idx + seq + '-' * (full_len - start_idx - len(seq))


def _generate_and_save_sequences(
    aa_input: List[str],
    seq_dir: str,
    file_prefix: str,
    tokenizer: T5Tokenizer,
    model: AutoModelForSeq2SeqLM,
    device: torch.device,
    num_return_sequences: int,
    decode_conf: Dict[str, Any],
    inference_batch_size: int,
) -> List[str]:
    """Generates sequences (3Di or AA) in batches and saves them to a FASTA file."""
    all_generated_sequences = []

    def batch_generator(data: List[str], size: int):
        for i in range(0, len(data), size):
            yield data[i:i + size]

    for batch in batch_generator(aa_input, inference_batch_size):
        if file_prefix == 'AA23Di':
            generated_batch = generate_3di(batch, tokenizer, model, device, num_return_sequences, decode_conf)
        elif file_prefix == '3Di2AA':
            generated_batch = generate_aa(batch, tokenizer, model, device, num_return_sequences, decode_conf)
        else:
            raise ValueError(f"Invalid file_prefix: {file_prefix}. Must be 'AA23Di' or '3Di2AA'.")

        all_generated_sequences.extend(generated_batch)

    output_path = os.path.join(seq_dir, f'{file_prefix}.fasta')
    append_fasta(all_generated_sequences, output_path)
    return all_generated_sequences


def generate_sequences_for_coverage(
    query_seq: str,
    full_len: int,
    decoding_configs: List[Dict[str, Any]],
    num_return_sequences: int,
    multiplier: int,
    coverage: float,
    model: AutoModelForSeq2SeqLM,
    tokenizer: T5Tokenizer,
    device: torch.device,
    project_dir: str,
    inference_batch_size: int,
) -> List[str]:
    """Generates and pads sequences for a given coverage."""
    chunk_len = int(coverage * full_len)
    all_backtranslated: List[str] = []

    seq_3di_dir = os.path.join(project_dir, 'seqs', 'AA23Di')
    seq_if_dir = os.path.join(project_dir, 'seqs', '3Di2AA')
    os.makedirs(seq_3di_dir, exist_ok=True)
    os.makedirs(seq_if_dir, exist_ok=True)

    for decode_conf in decoding_configs:
        if full_len - chunk_len < 0:
            logger.warning(f"Chunk length ({chunk_len}) is greater than full sequence length ({full_len}). Skipping generation for this coverage.")
            continue
        start = random.randint(0, full_len - chunk_len)
        chunk = query_seq[start: start + chunk_len]

        try:
            # Stage 1: AA -> 3Di
            fold_translations = _generate_and_save_sequences(
                aa_input=[chunk], seq_dir=seq_3di_dir, file_prefix='AA23Di',
                tokenizer=tokenizer, model=model, device=device,
                num_return_sequences=num_return_sequences, decode_conf=decode_conf,
                inference_batch_size=inference_batch_size,
            )
            # Stage 2: 3Di -> AA
            backtranslated = _generate_and_save_sequences(
                aa_input=fold_translations, seq_dir=seq_if_dir, file_prefix='3Di2AA',
                tokenizer=tokenizer, model=model, device=device,
                num_return_sequences=multiplier, decode_conf=decode_conf,
                inference_batch_size=inference_batch_size,
            )
            padded = [pad_sequence(seq, start, full_len) for seq in backtranslated]
            all_backtranslated.extend(padded)

        except RuntimeError as e:
            logger.error(f"OOM Error on coverage {coverage}, chunk {start}:{start + chunk_len} — skipping. Error: {e}")
            torch.cuda.empty_cache()
            continue
        finally:
            torch.cuda.empty_cache()

    return all_backtranslated
