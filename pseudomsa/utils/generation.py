### utils/generation.py

import os
import random
import torch
from typing import List, Dict, Any
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM
from rich.console import Console
from rich.progress import Progress

from .io import append_fasta
from .model import generate_3di, generate_aa


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
    # multiplier: int,
    decode_conf: Dict[str, Any],
    inference_batch_size: int, # Add batch size for memory control
    console: Console, # Pass console for progress bar
) -> List[str]:
    """Generates sequences (3Di or AA) in batches and saves them to a FASTA file."""
    all_generated_sequences = []
    
    # Use a generator to create batches from the input list
    def batch_generator(data: List[str], size: int):
        for i in range(0, len(data), size):
            yield data[i:i + size]
            
    # Process the input in controlled mini-batches
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
    console: Console
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
            console.print(f"[bold yellow]Warning:[/bold yellow] Chunk length ([yellow]{chunk_len}[/yellow]) is greater than full sequence length ([yellow]{full_len}[/yellow]). Skipping generation for this coverage.")
            continue
        start = random.randint(0, full_len - chunk_len)
        chunk = query_seq[start: start + chunk_len]

        try:
            # Stage 1: AA -> 3Di. Generates `num_return_sequences` of 3Di sequences.
            fold_translations = _generate_and_save_sequences(
                aa_input=[chunk], seq_dir=seq_3di_dir, file_prefix='AA23Di',
                tokenizer=tokenizer, model=model, device=device,
                num_return_sequences=num_return_sequences, decode_conf=decode_conf,
                inference_batch_size=inference_batch_size, console=console
            )
            # Stage 2: 3Di -> AA. For each 3Di sequence, generate `multiplier` AA sequences.
            backtranslated = _generate_and_save_sequences(
                aa_input=fold_translations, seq_dir=seq_if_dir, file_prefix='3Di2AA',
                tokenizer=tokenizer, model=model, device=device,
                num_return_sequences= multiplier, decode_conf=decode_conf,
                inference_batch_size=inference_batch_size, console=console
            )
            padded = [pad_sequence(seq, start, full_len) for seq in backtranslated]
            all_backtranslated.extend(padded)

        except RuntimeError as e:
            console.print(f"[bold red]OOM Error:[/bold red] on coverage [cyan]{coverage}[/cyan], chunk [cyan]{start}:{start + chunk_len}[/cyan] — skipping. Error: [red]{e}[/red]")
            torch.cuda.empty_cache()
            continue
        finally:
            torch.cuda.empty_cache()

    return all_backtranslated
