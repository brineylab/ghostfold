from unittest.mock import patch, MagicMock
from ghostfold.msa.generation import generate_sequences_for_coverages_batched
import tempfile, os


def test_generate_sequences_for_coverages_batched_returns_list():
    """Batched generation returns a flat list of sequences."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    import torch
    device = torch.device("cpu")

    with patch("ghostfold.msa.generation._generate_and_save_sequences") as mock_gen, \
         tempfile.TemporaryDirectory() as tmpdir:
        mock_gen.return_value = ["AAAAAAAAAA"]
        result = generate_sequences_for_coverages_batched(
            query_seq="AAAAAAAAAA",
            full_len=10,
            decoding_configs=[{"temperature": 0.7}],
            num_return_sequences=1,
            multiplier=1,
            coverage_list=[1.0, 0.8],
            model=mock_model,
            tokenizer=mock_tokenizer,
            device=device,
            project_dir=tmpdir,
            inference_batch_size=4,
        )
    assert isinstance(result, list)
