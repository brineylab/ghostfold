"""Tests for multimer support: detection, MSA writing, and ColabFold routing."""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from ghostfold.core.colabfold import _detect_multimer_from_a3m
from ghostfold.core.pipeline import write_multimer_pst_msa

try:
    from typer.testing import CliRunner
    from ghostfold.cli.app import app
    _runner: CliRunner | None = CliRunner()
except ImportError:
    _runner = None

_typer_missing = _runner is None


# ---------------------------------------------------------------------------
# _detect_multimer_from_a3m
# ---------------------------------------------------------------------------

class TestDetectMultimerFromA3m:
    def _write_a3m(self, tmp_path: str, query_seq: str) -> str:
        path = os.path.join(tmp_path, "pstMSA.a3m")
        with open(path, "w") as f:
            f.write(f">query\n{query_seq}\n")
            f.write(">seq1\nAAAA\n")
        return path

    def test_detects_multimer_with_colon(self, tmp_path):
        path = self._write_a3m(str(tmp_path), "AAAA:BBBB")
        assert _detect_multimer_from_a3m(path) is True

    def test_monomer_returns_false(self, tmp_path):
        path = self._write_a3m(str(tmp_path), "AAAABBBB")
        assert _detect_multimer_from_a3m(path) is False

    def test_missing_file_returns_false(self):
        assert _detect_multimer_from_a3m("/nonexistent/path.a3m") is False

    def test_empty_file_returns_false(self, tmp_path):
        path = str(tmp_path / "empty.a3m")
        open(path, "w").close()
        assert _detect_multimer_from_a3m(path) is False

    def test_only_header_returns_false(self, tmp_path):
        path = str(tmp_path / "header_only.a3m")
        with open(path, "w") as f:
            f.write(">query\n")
        assert _detect_multimer_from_a3m(path) is False

    def test_colon_only_in_second_record_returns_false(self, tmp_path):
        path = str(tmp_path / "second.a3m")
        with open(path, "w") as f:
            f.write(">query\nAAAABBBB\n")
            f.write(">other\nAAA:BBB\n")
        assert _detect_multimer_from_a3m(path) is False


# ---------------------------------------------------------------------------
# write_multimer_pst_msa
# ---------------------------------------------------------------------------

class TestWriteMultimerPstMsa:
    def test_query_line_has_colon(self, tmp_path):
        path = str(tmp_path / "pstMSA.fasta")
        write_multimer_pst_msa(
            output_path=path,
            query_seq="AAAA:BBBB",
            concat_seqs=["AAAABBBB"],
            per_chain_seqs=[["AAAC"], ["BBBX"]],
            chain_lengths=[4, 4],
        )
        with open(path) as f:
            lines = f.readlines()
        assert lines[0].strip() == ">query"
        assert lines[1].strip() == "AAAA:BBBB"

    def test_concat_rows_written_without_colon(self, tmp_path):
        path = str(tmp_path / "pstMSA.fasta")
        write_multimer_pst_msa(
            output_path=path,
            query_seq="AAAA:BBBB",
            concat_seqs=["AAAABBBB", "AAACBBBC"],
            per_chain_seqs=[[], []],
            chain_lengths=[4, 4],
        )
        with open(path) as f:
            content = f.read()
        assert "AAAABBBB" in content
        assert "AAACBBBC" in content
        assert "concat_0" in content
        assert "concat_1" in content

    def test_chain_rows_gap_padded(self, tmp_path):
        path = str(tmp_path / "pstMSA.fasta")
        write_multimer_pst_msa(
            output_path=path,
            query_seq="AAAA:BBBB",
            concat_seqs=[],
            per_chain_seqs=[["AAAC"], ["BBBX"]],
            chain_lengths=[4, 4],
        )
        with open(path) as f:
            content = f.read()
        # Chain 0 seq padded with 4 trailing gaps
        assert "AAAC----" in content
        # Chain 1 seq padded with 4 leading gaps
        assert "----BBBX" in content

    def test_three_chain_gap_padding(self, tmp_path):
        path = str(tmp_path / "pstMSA.fasta")
        write_multimer_pst_msa(
            output_path=path,
            query_seq="AA:BB:CC",
            concat_seqs=[],
            per_chain_seqs=[["AX"], ["BX"], ["CX"]],
            chain_lengths=[2, 2, 2],
        )
        with open(path) as f:
            content = f.read()
        assert "AX----" in content   # chain 0: no prefix, 4 gap suffix
        assert "--BX--" in content   # chain 1: 2 gap prefix, 2 gap suffix
        assert "----CX" in content   # chain 2: 4 gap prefix, no suffix

    def test_empty_concat_and_chain_seqs(self, tmp_path):
        path = str(tmp_path / "pstMSA.fasta")
        write_multimer_pst_msa(
            output_path=path,
            query_seq="AAAA:BBBB",
            concat_seqs=[],
            per_chain_seqs=[[], []],
            chain_lengths=[4, 4],
        )
        with open(path) as f:
            content = f.read()
        # Only query record present
        assert content.count(">") == 1

    def test_total_row_length_equals_concat_length(self, tmp_path):
        path = str(tmp_path / "pstMSA.fasta")
        chain_a = "ACDE"
        chain_b = "FGHI"
        write_multimer_pst_msa(
            output_path=path,
            query_seq=f"{chain_a}:{chain_b}",
            concat_seqs=["ACDEFGHI"],
            per_chain_seqs=[["ACDE"], ["FGHX"]],
            chain_lengths=[4, 4],
        )
        with open(path) as f:
            lines = f.readlines()
        seq_lines = [line.strip() for line in lines if not line.startswith(">") and line.strip()]
        # Query line contains ':' so length is 9, all others must be 8
        for seq in seq_lines[1:]:
            assert len(seq) == 8, f"Expected 8, got {len(seq)} for '{seq}'"


# ---------------------------------------------------------------------------
# CLI: fold --multimer-model-version
# ---------------------------------------------------------------------------

@pytest.mark.skipif(_typer_missing, reason="typer not installed")
class TestFoldCliMultimerFlag:
    def test_fold_help_shows_multimer_version(self):
        result = _runner.invoke(app, ["fold", "--help"])  # type: ignore[union-attr]
        assert result.exit_code == 0
        assert "--multimer-model-version" in result.output

    def test_run_help_shows_multimer_version(self):
        result = _runner.invoke(app, ["run", "--help"])  # type: ignore[union-attr]
        assert result.exit_code == 0
        assert "--multimer-model-version" in result.output


# ---------------------------------------------------------------------------
# Pipeline: multimer detection routes to process_multimer_run
# ---------------------------------------------------------------------------

class TestPipelineMultimerDetection:
    def test_multimer_sequence_triggers_multimer_run(self, tmp_path):
        """run_pipeline routes ':'-containing sequences to process_multimer_run."""
        with patch("ghostfold.core.pipeline._load_model") as mock_load, \
             patch("ghostfold.core.pipeline.process_multimer_run") as mock_multi, \
             patch("ghostfold.core.pipeline.process_sequence_run") as mock_mono, \
             patch("ghostfold.core.pipeline.write_multimer_pst_msa") as mock_write:

            mock_load.return_value = (MagicMock(), MagicMock())
            mock_multi.return_value = {
                "concat_seqs": ["AAAABBBB"],
                "per_chain_seqs": [["AAAA"], ["BBBB"]],
                "per_chain_evolved_seqs": None,
            }

            fasta_path = str(tmp_path / "input.fasta")
            with open(fasta_path, "w") as f:
                f.write(">complex\nAAAA:BBBB\n")

            from ghostfold.core.pipeline import run_pipeline
            run_pipeline(
                project=str(tmp_path / "proj"),
                fasta_path=fasta_path,
                config={"decoding_params": {"base": {"top_k": 5}, "matrix": {}}},
                coverage_list=[1.0],
                evolve_msa=False,
                mutation_rates_str="{}",
                sample_percentage=1.0,
                plot_msa=False,
                plot_coevolution=False,
                show_progress=False,
            )

            mock_multi.assert_called_once()
            mock_mono.assert_not_called()
            mock_write.assert_called_once()

    def test_monomer_sequence_skips_multimer_run(self, tmp_path):
        """run_pipeline routes plain sequences to process_sequence_run."""
        with patch("ghostfold.core.pipeline._load_model") as mock_load, \
             patch("ghostfold.core.pipeline.process_multimer_run") as mock_multi, \
             patch("ghostfold.core.pipeline.process_sequence_run") as mock_mono, \
             patch("ghostfold.core.pipeline.concatenate_fasta_files"):

            mock_load.return_value = (MagicMock(), MagicMock())
            mock_mono.return_value = {"filtered": None, "evolved": None}

            fasta_path = str(tmp_path / "input.fasta")
            with open(fasta_path, "w") as f:
                f.write(">mono\nAAAABBBB\n")

            from ghostfold.core.pipeline import run_pipeline
            run_pipeline(
                project=str(tmp_path / "proj"),
                fasta_path=fasta_path,
                config={"decoding_params": {"base": {"top_k": 5}, "matrix": {}}},
                coverage_list=[1.0],
                evolve_msa=False,
                mutation_rates_str="{}",
                sample_percentage=1.0,
                plot_msa=False,
                plot_coevolution=False,
                show_progress=False,
            )

            mock_mono.assert_called_once()
            mock_multi.assert_not_called()
