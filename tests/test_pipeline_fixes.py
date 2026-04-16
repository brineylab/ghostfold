"""Tests for:
  Fix 6: Coverage levels parallelized via ThreadPoolExecutor
  Fix 7: Model + tokenizer cached at module level (no reload on second call)
"""
import pytest
from unittest.mock import MagicMock, patch, call
import threading


# ---------------------------------------------------------------------------
# Fix 6: Parallel coverage levels
# ---------------------------------------------------------------------------

class TestParallelCoverageLevels:
    def test_all_coverages_produce_results(self, tmp_path):
        """Fix 6: Results from ALL coverage levels must be collected."""
        from ghostfold.core.pipeline import process_sequence_run

        coverage_list = [0.25, 0.5, 0.75, 1.0]
        coverage_calls = []

        def fake_generate(query_seq, full_len, decoding_configs,
                          num_return_sequences, multiplier, coverage,
                          model, tokenizer, device, project_dir,
                          inference_batch_size):
            coverage_calls.append(coverage)
            # Return sequences of the right length for the query
            return [query_seq]  # full-length, passes filter

        with patch("ghostfold.core.pipeline.generate_sequences_for_coverage",
                   side_effect=fake_generate), \
             patch("ghostfold.core.pipeline.filter_sequences",
                   return_value=["ACDEFGHIKL"]), \
             patch("ghostfold.core.pipeline.write_fasta"), \
             patch("ghostfold.core.pipeline.generate_optional_plots"), \
             patch("ghostfold.core.pipeline.MSA_Mutator"):

            process_sequence_run(
                query_seq="ACDEFGHIKL",
                header="test",
                full_len=10,
                run_idx=1,
                base_project_dir=str(tmp_path),
                decoding_configs=[{"temperature": 1.0}],
                num_return_sequences=2,
                multiplier=1,
                coverage_list=coverage_list,
                model=MagicMock(),
                tokenizer=MagicMock(),
                device=MagicMock(),
                evolve_msa=False,
                mutation_rates_str='{"BLOSUM62": 5}',
                sample_percentage=1.0,
                hex_colors=["#fff"],
                plot_msa=False,
                plot_coevolution=False,
                inference_batch_size=4,
            )

        # All 4 coverage levels must have been processed
        assert sorted(coverage_calls) == sorted(coverage_list)

    def test_coverage_results_combined_in_output(self, tmp_path):
        """Fix 6: Sequences from each coverage level must all appear in total."""
        from ghostfold.core.pipeline import process_sequence_run

        coverage_list = [0.5, 1.0]
        # Return distinct sequences per coverage
        coverage_seqs = {
            0.5: ["AAAAAAAAAA"],
            1.0: ["CCCCCCCCCC"],
        }

        def fake_generate(query_seq, full_len, **kwargs):
            return coverage_seqs.get(kwargs["coverage"], [])

        captured_filter_input = []

        def fake_filter(seqs, expected_length, **kwargs):
            captured_filter_input.extend(seqs)
            return []  # all filtered out to stop processing early

        with patch("ghostfold.core.pipeline.generate_sequences_for_coverage",
                   side_effect=fake_generate), \
             patch("ghostfold.core.pipeline.filter_sequences",
                   side_effect=fake_filter), \
             patch("ghostfold.core.pipeline.write_fasta"), \
             patch("ghostfold.core.pipeline.generate_optional_plots"):

            process_sequence_run(
                query_seq="ACDEFGHIKL",
                header="test",
                full_len=10,
                run_idx=1,
                base_project_dir=str(tmp_path),
                decoding_configs=[{"temperature": 1.0}],
                num_return_sequences=2,
                multiplier=1,
                coverage_list=coverage_list,
                model=MagicMock(),
                tokenizer=MagicMock(),
                device=MagicMock(),
                evolve_msa=False,
                mutation_rates_str='{"BLOSUM62": 5}',
                sample_percentage=1.0,
                hex_colors=["#fff"],
                plot_msa=False,
                plot_coevolution=False,
                inference_batch_size=4,
            )

        # Sequences from both coverage levels must be present
        assert "AAAAAAAAAA" in captured_filter_input
        assert "CCCCCCCCCC" in captured_filter_input

    def test_generate_called_for_each_coverage(self, tmp_path):
        """Fix 6: generate_sequences_for_coverage must be called once per coverage."""
        from ghostfold.core.pipeline import process_sequence_run

        coverage_list = [0.5, 0.75, 1.0]
        call_count = {"n": 0}

        def fake_generate(**kwargs):
            call_count["n"] += 1
            return []

        with patch("ghostfold.core.pipeline.generate_sequences_for_coverage",
                   side_effect=lambda *a, **kw: fake_generate(**kw) or []), \
             patch("ghostfold.core.pipeline.filter_sequences", return_value=[]), \
             patch("ghostfold.core.pipeline.write_fasta"), \
             patch("ghostfold.core.pipeline.generate_optional_plots"):

            process_sequence_run(
                query_seq="ACDEFGHIKL",
                header="test",
                full_len=10,
                run_idx=1,
                base_project_dir=str(tmp_path),
                decoding_configs=[{"temperature": 1.0}],
                num_return_sequences=2,
                multiplier=1,
                coverage_list=coverage_list,
                model=MagicMock(),
                tokenizer=MagicMock(),
                device=MagicMock(),
                evolve_msa=False,
                mutation_rates_str='{"BLOSUM62": 5}',
                sample_percentage=1.0,
                hex_colors=["#fff"],
                plot_msa=False,
                plot_coevolution=False,
                inference_batch_size=4,
            )

        assert call_count["n"] == len(coverage_list)


# ---------------------------------------------------------------------------
# Fix 7: Model + tokenizer cache
# ---------------------------------------------------------------------------

class TestModelCache:
    def setup_method(self):
        """Clear the module-level cache before each test."""
        import ghostfold.core.pipeline as pipeline_mod
        if hasattr(pipeline_mod, "_MODEL_CACHE"):
            pipeline_mod._MODEL_CACHE.clear()

    def test_model_loaded_only_once_across_two_calls(self):
        """Fix 7: from_pretrained must be called only once for same device."""
        import ghostfold.core.pipeline as pipeline_mod
        import torch

        fake_tokenizer = MagicMock()
        fake_model = MagicMock()
        fake_model.to.return_value = fake_model

        device = torch.device("cpu")

        # Patch at the source module since _load_model uses local imports
        with patch("transformers.T5Tokenizer") as mock_tok_cls, \
             patch("transformers.AutoModelForSeq2SeqLM") as mock_model_cls:
            mock_tok_cls.from_pretrained.return_value = fake_tokenizer
            mock_model_cls.from_pretrained.return_value = fake_model

            tok1, mod1 = pipeline_mod._load_model(device)
            tok2, mod2 = pipeline_mod._load_model(device)

        assert mock_tok_cls.from_pretrained.call_count == 1
        assert mock_model_cls.from_pretrained.call_count == 1
        assert tok1 is tok2
        assert mod1 is mod2

    def test_different_devices_load_separately(self):
        """Fix 7: CPU and CUDA devices must be cached independently."""
        import ghostfold.core.pipeline as pipeline_mod
        import torch

        fake_model = MagicMock()
        fake_model.to.return_value = fake_model

        cpu_device = torch.device("cpu")
        cuda_device = MagicMock()
        cuda_device.type = "cuda"

        with patch("transformers.T5Tokenizer") as mock_tok_cls, \
             patch("transformers.AutoModelForSeq2SeqLM") as mock_model_cls:
            mock_tok_cls.from_pretrained.return_value = MagicMock()
            mock_model_cls.from_pretrained.return_value = fake_model

            pipeline_mod._load_model(cpu_device)
            pipeline_mod._load_model(cuda_device)

        assert mock_model_cls.from_pretrained.call_count == 2

    def test_cache_returns_same_objects_on_third_call(self):
        """Fix 7: Cache must be persistent — third call also returns cached objects."""
        import ghostfold.core.pipeline as pipeline_mod
        import torch

        fake_model = MagicMock()
        fake_model.to.return_value = fake_model

        device = torch.device("cpu")

        with patch("transformers.T5Tokenizer") as mock_tok_cls, \
             patch("transformers.AutoModelForSeq2SeqLM") as mock_model_cls:
            mock_tok_cls.from_pretrained.return_value = MagicMock()
            mock_model_cls.from_pretrained.return_value = fake_model

            tok1, _ = pipeline_mod._load_model(device)
            pipeline_mod._load_model(device)
            tok3, _ = pipeline_mod._load_model(device)

        assert mock_tok_cls.from_pretrained.call_count == 1
        assert tok1 is tok3
