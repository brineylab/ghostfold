import torch
from unittest.mock import patch, MagicMock
from ghostfold.core.pipeline import _load_model, _MODEL_CACHE

def test_load_model_uses_bf16_when_supported():
    """_load_model should use bfloat16 when CUDA + bf16 supported."""
    _MODEL_CACHE.clear()
    mock_model = MagicMock()
    mock_model.eval.return_value = mock_model
    mock_model.to.return_value = mock_model
    mock_model.half.return_value = mock_model

    with patch("ghostfold.core.pipeline.AutoModelForSeq2SeqLM") as mock_cls, \
         patch("ghostfold.core.pipeline.T5Tokenizer"), \
         patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.is_bf16_supported", return_value=True):
        mock_cls.from_pretrained.return_value = mock_model
        device = torch.device("cuda")
        _load_model(device)

    mock_model.to.assert_any_call(torch.bfloat16)

def test_load_model_uses_fp16_fallback():
    """_load_model should fall back to fp16 when bf16 not supported."""
    _MODEL_CACHE.clear()
    mock_model = MagicMock()
    mock_model.eval.return_value = mock_model
    mock_model.to.return_value = mock_model
    mock_model.half.return_value = mock_model

    with patch("ghostfold.core.pipeline.AutoModelForSeq2SeqLM") as mock_cls, \
         patch("ghostfold.core.pipeline.T5Tokenizer"), \
         patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.is_bf16_supported", return_value=False):
        mock_cls.from_pretrained.return_value = mock_model
        device = torch.device("cuda")
        _load_model(device)

    mock_model.half.assert_called_once()
