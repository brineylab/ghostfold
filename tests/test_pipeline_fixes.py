import torch
from unittest.mock import patch, MagicMock
from ghostfold.core.pipeline import _load_model, _MODEL_CACHE


def test_load_model_uses_bf16_when_requested():
    """_load_model should pass torch_dtype=bfloat16 when precision='bf16'."""
    _MODEL_CACHE.clear()
    mock_model = MagicMock()
    mock_model.eval.return_value = mock_model
    mock_model.to.return_value = mock_model
    mock_model.half.return_value = mock_model

    with patch("ghostfold.core.pipeline.AutoModelForSeq2SeqLM") as mock_cls, \
         patch("ghostfold.core.pipeline.T5Tokenizer"):
        mock_cls.from_pretrained.return_value = mock_model
        device = torch.device("cpu")
        _load_model(device, precision="bf16")

    _, kwargs = mock_cls.from_pretrained.call_args
    assert kwargs.get("torch_dtype") == torch.bfloat16


def test_load_model_uses_fp16_when_requested():
    """_load_model should pass torch_dtype=float16 when precision='fp16'."""
    _MODEL_CACHE.clear()
    mock_model = MagicMock()
    mock_model.eval.return_value = mock_model
    mock_model.to.return_value = mock_model
    mock_model.half.return_value = mock_model

    with patch("ghostfold.core.pipeline.AutoModelForSeq2SeqLM") as mock_cls, \
         patch("ghostfold.core.pipeline.T5Tokenizer"):
        mock_cls.from_pretrained.return_value = mock_model
        device = torch.device("cpu")
        _load_model(device, precision="fp16")

    _, kwargs = mock_cls.from_pretrained.call_args
    assert kwargs.get("torch_dtype") == torch.float16
