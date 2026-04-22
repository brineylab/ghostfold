"""Integration tests for T5 inference at different precisions.

Marked @pytest.mark.slow -- skipped in normal CI.
Run with: pytest tests/test_inference_precision.py -v -m slow
"""
import pytest
import tempfile


@pytest.fixture(scope="module")
def short_fasta(tmp_path_factory):
    """Write a short protein FASTA for inference tests."""
    d = tmp_path_factory.mktemp("fixtures")
    fasta = d / "short.fasta"
    fasta.write_text(">test_protein\nACDEFGHIKLMNPQRSTVWYACDEFGHIKLM\n")
    return str(fasta)


@pytest.fixture(autouse=True)
def clear_model_cache():
    from ghostfold.core import pipeline
    pipeline._MODEL_CACHE.clear()
    yield
    pipeline._MODEL_CACHE.clear()


@pytest.mark.slow
@pytest.mark.parametrize("precision", ["bf16", "fp16"])
def test_full_pipeline_neff_positive(short_fasta, precision):
    """Full generate -> filter -> neff pipeline must produce neff > 0 at bf16 and fp16."""
    import torch
    from ghostfold.core.pipeline import _load_model, generate_decoding_configs
    from ghostfold.msa.generation import generate_sequences_for_coverages_batched
    from ghostfold.msa.filters import filter_sequences
    from ghostfold.msa.neff import calculate_neff
    from ghostfold.io.fasta import read_fasta_from_path
    from ghostfold.core.config import load_config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(None)

    tokenizer, model = _load_model(device, precision=precision)
    assert model is not None
    assert tokenizer is not None

    records = read_fasta_from_path(short_fasta)
    query_seq = str(records[0].seq)
    full_len = len(query_seq)

    decoding_configs = generate_decoding_configs(config.get("decoding_params", {}))

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_seqs = generate_sequences_for_coverages_batched(
            query_seq=query_seq,
            full_len=full_len,
            decoding_configs=decoding_configs,
            num_return_sequences=config.get("num_return_sequences", 5),
            multiplier=config.get("multiplier", 1),
            coverage_list=[1.0],
            model=model,
            tokenizer=tokenizer,
            device=device,
            project_dir=tmpdir,
            inference_batch_size=config.get("inference_batch_size", 4),
        )

    all_seqs = [query_seq] + raw_seqs
    filtered = filter_sequences(all_seqs, full_len)
    assert len(filtered) > 0, f"No sequences passed filter at precision={precision}"

    neff = calculate_neff(filtered)
    assert neff > 0.0, f"Neff=0 at precision={precision}; filtered={len(filtered)} seqs"


@pytest.mark.slow
def test_int8_requires_bitsandbytes_or_skips():
    """int8 precision: passes if bitsandbytes installed, skips cleanly if not."""
    pytest.importorskip(
        "bitsandbytes",
        reason="bitsandbytes not installed; skipping int8 test (pip install -e '.[quant]')"
    )
    import torch
    from ghostfold.core.pipeline import _load_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = _load_model(device, precision="int8")
    assert model is not None


@pytest.mark.slow
def test_int4_requires_bitsandbytes_or_skips():
    """int4 precision: passes if bitsandbytes installed, skips cleanly if not."""
    pytest.importorskip(
        "bitsandbytes",
        reason="bitsandbytes not installed; skipping int4 test (pip install -e '.[quant]')"
    )
    import torch
    from ghostfold.core.pipeline import _load_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = _load_model(device, precision="int4")
    assert model is not None
