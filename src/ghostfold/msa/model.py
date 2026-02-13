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
    """Preprocesses a sequence by joining its elements and adding a prefix."""
    return prefix + ' ' + ' '.join(sequence)


def generate_3di(
    sequences: List[List[str]],
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    device: torch.device,
    num_return_sequences: int,
    decode_conf: Dict[str, Any]
) -> List[str]:
    """Generates 3D-interpretable sequences using a pre-trained model."""
    inputs = (preprocess_sequence(seq, "<AA2fold>") for seq in sequences)

    ids = tokenizer(
        list(inputs),
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
            num_beams=num_return_sequences*3,
            early_stopping=True,
            do_sample=True,
            logits_processor=processors,
            **decode_conf
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
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
    """Generates amino acid sequences from fold sequences using a pre-trained model."""
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
            do_sample=True,
            logits_processor=processors,
            **decode_conf
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    decoded = [x.replace(" ", "") for x in decoded]
    return decoded
