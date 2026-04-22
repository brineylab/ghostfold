"""Unit tests for _load_model precision parameter and cache key logic."""
from unittest.mock import MagicMock, patch
import pytest


def _make_mock_model(supports_bf16=True):
    """Return a mock (tokenizer, model) pair."""
    tokenizer = MagicMock()
    model = MagicMock()
    model.to = MagicMock(return_value=model)
    model.half = MagicMock(return_value=model)
    model.eval = MagicMock(return_value=model)
    return tokenizer, model


@pytest.fixture(autouse=True)
def clear_model_cache():
    """Clear the module-level model cache before each test."""
    from ghostfold.core import pipeline
    pipeline._MODEL_CACHE.clear()
    yield
    pipeline._MODEL_CACHE.clear()


@patch("ghostfold.core.pipeline.AutoModelForSeq2SeqLM")
@patch("ghostfold.core.pipeline.T5Tokenizer")
def test_cache_key_includes_precision(mock_tokenizer_cls, mock_model_cls):
    """Cache key must be '{model}:{device}:{precision}' to allow concurrent precisions."""
    import torch
    from ghostfold.core import pipeline

    mock_tokenizer_cls.from_pretrained.return_value = MagicMock()
    mock_model = MagicMock()
    mock_model.to = MagicMock(return_value=mock_model)
    mock_model.half = MagicMock(return_value=mock_model)
    mock_model.eval = MagicMock(return_value=mock_model)
    mock_model_cls.from_pretrained.return_value = mock_model

    device = torch.device("cpu")
    pipeline._load_model(device, precision="bf16")
    pipeline._load_model(device, precision="fp16")

    assert len(pipeline._MODEL_CACHE) == 2
    keys = list(pipeline._MODEL_CACHE.keys())
    assert any("bf16" in k for k in keys)
    assert any("fp16" in k for k in keys)


@patch("ghostfold.core.pipeline.AutoModelForSeq2SeqLM")
@patch("ghostfold.core.pipeline.T5Tokenizer")
def test_load_model_bf16_default_succeeds(mock_tokenizer_cls, mock_model_cls):
    """Default precision=bf16 loads without error (regression guard)."""
    import torch
    from ghostfold.core import pipeline

    mock_tokenizer_cls.from_pretrained.return_value = MagicMock()
    mock_model = MagicMock()
    mock_model.to = MagicMock(return_value=mock_model)
    mock_model.half = MagicMock(return_value=mock_model)
    mock_model.eval = MagicMock(return_value=mock_model)
    mock_model_cls.from_pretrained.return_value = mock_model

    device = torch.device("cpu")
    tokenizer, model = pipeline._load_model(device, precision="bf16")
    assert tokenizer is not None
    assert model is not None


def test_invalid_precision_raises():
    """Unsupported precision value must raise ValueError immediately."""
    import torch
    from ghostfold.core import pipeline

    device = torch.device("cpu")
    with pytest.raises(ValueError, match="precision"):
        pipeline._load_model(device, precision="fp8")


def test_int8_without_bitsandbytes_raises():
    """precision='int8' without bitsandbytes installed must raise ImportError with pip hint."""
    import sys
    import torch
    from ghostfold.core import pipeline

    # Simulate bitsandbytes not installed
    with patch.dict(sys.modules, {"bitsandbytes": None}):
        with pytest.raises(ImportError, match="bitsandbytes"):
            pipeline._load_model(torch.device("cpu"), precision="int8")


def test_int4_without_bitsandbytes_raises():
    """precision='int4' without bitsandbytes installed must raise ImportError with pip hint."""
    import sys
    import torch
    from ghostfold.core import pipeline

    with patch.dict(sys.modules, {"bitsandbytes": None}):
        with pytest.raises(ImportError, match="bitsandbytes"):
            pipeline._load_model(torch.device("cpu"), precision="int4")
