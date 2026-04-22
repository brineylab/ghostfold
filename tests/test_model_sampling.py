"""Tests for Fix 5: generate_3di must use sampling (num_beams=1), not beam search.

generate_aa already uses num_beams=1, so only generate_3di is fixed.
"""
from unittest.mock import MagicMock
import torch


def _make_mock_model_and_tokenizer(seq_len=5, num_return=2):
    """Return minimal mocks for tokenizer + model that satisfy generate_3di."""
    tokenizer = MagicMock()
    # tokenizer(...) returns an object with .input_ids and .attention_mask
    encoded = MagicMock()
    encoded.input_ids = torch.zeros(1, seq_len, dtype=torch.long)
    encoded.attention_mask = torch.ones(1, seq_len, dtype=torch.long)
    tokenizer.return_value = encoded
    tokenizer.batch_decode.return_value = ["A C D E F"] * num_return

    model = MagicMock()
    model.generate.return_value = torch.zeros(num_return, seq_len, dtype=torch.long)

    device = MagicMock()
    encoded.to.return_value = encoded

    return tokenizer, model, device


class TestGenerate3DiUsesSampling:
    def test_num_beams_is_one(self):
        """Fix 5: generate_3di must call model.generate with num_beams=1."""
        from ghostfold.msa.model import generate_3di

        tokenizer, model, device = _make_mock_model_and_tokenizer(num_return=3)
        decode_conf = {"temperature": 1.0}

        generate_3di(
            sequences=[list("ACDEF")],
            tokenizer=tokenizer,
            model=model,
            device=device,
            num_return_sequences=3,
            decode_conf=decode_conf,
        )

        _, kwargs = model.generate.call_args
        assert kwargs.get("num_beams", 1) == 1, (
            f"num_beams should be 1 (sampling), got {kwargs.get('num_beams')}. "
            "Beam search is 3-5x slower than sampling."
        )

    def test_early_stopping_removed(self):
        """Fix 5: early_stopping must not be passed (only meaningful with beam search)."""
        from ghostfold.msa.model import generate_3di

        tokenizer, model, device = _make_mock_model_and_tokenizer(num_return=2)
        decode_conf = {"temperature": 1.0}

        generate_3di(
            sequences=[list("ACDEF")],
            tokenizer=tokenizer,
            model=model,
            device=device,
            num_return_sequences=2,
            decode_conf=decode_conf,
        )

        _, kwargs = model.generate.call_args
        assert "early_stopping" not in kwargs, (
            "early_stopping should be removed — it is only meaningful with beam search."
        )

    def test_do_sample_still_true(self):
        """Fix 5: do_sample=True must remain (needed for multinomial sampling)."""
        from ghostfold.msa.model import generate_3di

        tokenizer, model, device = _make_mock_model_and_tokenizer(num_return=2)

        generate_3di(
            sequences=[list("ACDEF")],
            tokenizer=tokenizer,
            model=model,
            device=device,
            num_return_sequences=2,
            decode_conf={"temperature": 1.0},
        )

        _, kwargs = model.generate.call_args
        assert kwargs.get("do_sample") is True

    def test_num_return_sequences_passed_correctly(self):
        """Fix 5: num_return_sequences must still be passed to model.generate."""
        from ghostfold.msa.model import generate_3di

        tokenizer, model, device = _make_mock_model_and_tokenizer(num_return=5)

        generate_3di(
            sequences=[list("ACDEF")],
            tokenizer=tokenizer,
            model=model,
            device=device,
            num_return_sequences=5,
            decode_conf={"temperature": 1.0},
        )

        _, kwargs = model.generate.call_args
        assert kwargs.get("num_return_sequences") == 5

    def test_generate_aa_still_uses_num_beams_one(self):
        """generate_aa already used num_beams=1; must not regress."""
        from ghostfold.msa.model import generate_aa

        tokenizer, model, device = _make_mock_model_and_tokenizer(num_return=2)

        generate_aa(
            fold_seqs=[list("ACDEF")],
            tokenizer=tokenizer,
            model=model,
            device=device,
            num_return_sequences=2,
            decode_conf={"temperature": 1.0},
        )

        _, kwargs = model.generate.call_args
        assert kwargs.get("num_beams", 1) == 1
