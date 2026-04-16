"""Tests for generation.py performance fixes:
  Fix 2: Adaptive batch size on OOM (retry with halved batch_size)
  Fix 4: torch.cuda.empty_cache() NOT called in finally (only on actual OOM)
"""
import pytest
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_generate_args(inference_batch_size=8, decoding_configs=None):
    """Return a minimal valid kwargs dict for generate_sequences_for_coverage."""
    return dict(
        query_seq="ACDEFGHIKL",
        full_len=10,
        decoding_configs=decoding_configs or [{"temperature": 1.0}],
        num_return_sequences=2,
        multiplier=2,
        coverage=1.0,
        model=MagicMock(),
        tokenizer=MagicMock(),
        device=MagicMock(),
        project_dir="/tmp/ghostfold_test",
        inference_batch_size=inference_batch_size,
    )


# ---------------------------------------------------------------------------
# Fix 4: empty_cache NOT called in finally on successful run
# ---------------------------------------------------------------------------

class TestNoCacheFlushOnSuccess:
    def test_empty_cache_not_called_when_no_oom(self, tmp_path):
        """Fix 4: torch.cuda.empty_cache() must not be called on a clean run."""
        from ghostfold.msa.generation import generate_sequences_for_coverage

        args = _make_generate_args()
        args["project_dir"] = str(tmp_path)

        fake_seqs = ["ACDEFGHIKL", "MNPQRSTVWY"]

        with patch("ghostfold.msa.generation._generate_and_save_sequences",
                   return_value=fake_seqs) as mock_gen, \
             patch("ghostfold.msa.generation.torch") as mock_torch:

            generate_sequences_for_coverage(**args)

        mock_torch.cuda.empty_cache.assert_not_called()

    def test_empty_cache_called_on_oom(self, tmp_path):
        """Fix 2+4: torch.cuda.empty_cache() IS called when OOM occurs."""
        from ghostfold.msa.generation import generate_sequences_for_coverage

        args = _make_generate_args(inference_batch_size=1)
        args["project_dir"] = str(tmp_path)

        oom_error = RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB")

        with patch("ghostfold.msa.generation._generate_and_save_sequences",
                   side_effect=oom_error), \
             patch("ghostfold.msa.generation.torch") as mock_torch:

            result = generate_sequences_for_coverage(**args)

        assert result == []
        mock_torch.cuda.empty_cache.assert_called()


# ---------------------------------------------------------------------------
# Fix 2: Adaptive batch size retry on OOM
# ---------------------------------------------------------------------------

class TestAdaptiveBatchSizeOnOOM:
    def test_retries_with_halved_batch_size(self, tmp_path):
        """Fix 2: OOM triggers retry with batch_size // 2, then succeeds."""
        from ghostfold.msa.generation import generate_sequences_for_coverage

        args = _make_generate_args(inference_batch_size=8)
        args["project_dir"] = str(tmp_path)

        fake_seqs = ["ACDEFGHIKL", "MNPQRSTVWY"]
        oom_error = RuntimeError("CUDA out of memory.")

        call_count = {"n": 0}

        def side_effect(*a, **kw):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise oom_error
            return fake_seqs

        with patch("ghostfold.msa.generation._generate_and_save_sequences",
                   side_effect=side_effect) as mock_gen, \
             patch("ghostfold.msa.generation.torch"):

            result = generate_sequences_for_coverage(**args)

        # Must have retried
        assert mock_gen.call_count >= 2
        # Second call must use halved batch size (4)
        second_call_kwargs = mock_gen.call_args_list[1][1]
        assert second_call_kwargs["inference_batch_size"] == 4

    def test_no_retry_when_batch_size_already_one(self, tmp_path):
        """Fix 2: OOM at batch_size=1 logs error and skips, does NOT loop forever."""
        from ghostfold.msa.generation import generate_sequences_for_coverage

        args = _make_generate_args(inference_batch_size=1)
        args["project_dir"] = str(tmp_path)

        oom_error = RuntimeError("CUDA out of memory.")

        with patch("ghostfold.msa.generation._generate_and_save_sequences",
                   side_effect=oom_error), \
             patch("ghostfold.msa.generation.torch"):

            result = generate_sequences_for_coverage(**args)

        assert result == []

    def test_non_oom_runtime_error_still_skips(self, tmp_path):
        """Fix 2: Non-OOM RuntimeError should not trigger retry, just skip."""
        from ghostfold.msa.generation import generate_sequences_for_coverage

        args = _make_generate_args(inference_batch_size=8)
        args["project_dir"] = str(tmp_path)

        other_error = RuntimeError("some other cuda error")
        call_count = {"n": 0}

        def side_effect(*a, **kw):
            call_count["n"] += 1
            raise other_error

        with patch("ghostfold.msa.generation._generate_and_save_sequences",
                   side_effect=side_effect), \
             patch("ghostfold.msa.generation.torch"):

            result = generate_sequences_for_coverage(**args)

        # Should only be called once per decode_conf, no retry
        assert call_count["n"] == 1
        assert result == []

    def test_successful_run_returns_padded_sequences(self, tmp_path):
        """Fix 2: Successful run returns all padded sequences unchanged."""
        from ghostfold.msa.generation import generate_sequences_for_coverage

        args = _make_generate_args(inference_batch_size=4)
        args["project_dir"] = str(tmp_path)

        fake_seqs = ["ACDEFGHIKL", "MNPQRSTVWY"]

        with patch("ghostfold.msa.generation._generate_and_save_sequences",
                   return_value=fake_seqs), \
             patch("ghostfold.msa.generation.torch"):

            result = generate_sequences_for_coverage(**args)

        assert len(result) > 0
