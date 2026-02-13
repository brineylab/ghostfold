### utils/model.py
import torch
from transformers import LogitsProcessor, LogitsProcessorList
from transformers import PreTrainedTokenizer, PreTrainedModel
from typing import List, Dict, Any


class FiniteLogitsProcessor(LogitsProcessor):
    """Ensure logits have no NaN or Inf values before sampling."""
    def __call__(self, input_ids, scores):
        scores = torch.nan_to_num(scores, neginf=-1e9, posinf=1e9)
        return torch.clamp(scores, -1e9, 1e9)


def preprocess_sequence(sequence: List[str], prefix: str) -> str:
    """
    Preprocesses a sequence by joining its elements and adding a prefix.

    Args:
        sequence: A list of strings representing the sequence.
        prefix: The prefix to add to the sequence.

    Returns:
        The preprocessed sequence as a single string.
    """
    return prefix + ' ' + ' '.join(sequence)

def generate_3di(
    sequences: List[List[str]],
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    device: torch.device,
    num_return_sequences: int,
    decode_conf: Dict[str, Any]
) -> List[str]:
    """
    Generates 3D-interpretable sequences using a pre-trained model.

    Args:
        sequences: A list of sequences, where each sequence is a list of strings.
        tokenizer: The tokenizer for encoding the sequences.
        model: The pre-trained model for generation.
        device: The device (e.g., 'cuda' or 'cpu') to run the model on.
        num_return_sequences: The number of sequences to generate for each input.
        decode_conf: A dictionary of additional decoding configuration parameters.

    Returns:
        A list of generated 3D-interpretable sequences.
    """
    # Use a generator expression for `preprocess_sequence` to avoid creating
    # a full intermediate list, especially for very large `sequences`.
    # This is a minor optimization but good practice.
    inputs = (preprocess_sequence(seq, "<AA2fold>") for seq in sequences)

    ids = tokenizer(
        list(inputs),
        padding='longest',
        return_tensors='pt'
    ).to(device)

    # Calculate max_len directly based on the tokenized input's length.
    max_len = ids.input_ids.shape[1] + 1

    processors = LogitsProcessorList([FiniteLogitsProcessor()])
    
    with torch.no_grad():
        # Model generation benefits from `no_grad` for inference.
        # The `generate` method handles the actual inference on the GPU.
        outputs = model.generate(
            input_ids=ids.input_ids,
            attention_mask=ids.attention_mask,
            max_length=max_len,
            num_return_sequences=num_return_sequences,
            num_beams=num_return_sequences*3,
            early_stopping=True,
            do_sample=True,
            logits_processor=processors,
            **decode_conf
        )

    # `batch_decode` is efficient for decoding multiple sequences.
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    # Use a list comprehension for efficient post-processing.
    # `replace(" ", "")` is typically faster than `split(" ")` and `join`
    # if you just want to remove all spaces.
    decoded = [x.replace(" ", "") for x in decoded]
    return decoded


def generate_aa(
    fold_seqs: List[List[str]],
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    device: torch.device,
    num_return_sequences: int,
    decode_conf: Dict[str, Any]
) -> List[str]:
    """
    Generates amino acid sequences from fold sequences using a pre-trained model.

    Args:
        fold_seqs: A list of fold sequences, where each sequence is a list of strings.
        tokenizer: The tokenizer for encoding the sequences.
        model: The pre-trained model for generation.
        device: The device (e.g., 'cuda' or 'cpu') to run the model on.
        num_return_sequences: The number of sequences to generate for each input (typically 1 for this function).
        decode_conf: A dictionary of additional decoding configuration parameters.

    Returns:
        A list of generated amino acid sequences.
    """
    inputs = (preprocess_sequence(seq, "<fold2AA>") for seq in fold_seqs)

    ids = tokenizer(
        list(inputs),
        add_special_tokens=True,
        padding='longest',
        return_tensors='pt'
    ).to(device)

    max_len = ids.input_ids.shape[1] + 1

    processors = LogitsProcessorList([FiniteLogitsProcessor()])
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=ids.input_ids,
            attention_mask=ids.attention_mask,
            max_length=max_len,
            num_return_sequences=num_return_sequences,
            num_beams=1,
            # early_stopping=True,
            do_sample=True,
            logits_processor=processors,
            **decode_conf
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    decoded = [x.replace(" ", "") for x in decoded]
    return decoded
