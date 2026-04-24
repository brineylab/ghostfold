# tests/test_benchmark_runner.py
"""Tests for benchmark runner dataset discovery."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_bench_dir(tmp_path: Path) -> Path:
    fasta_dir = tmp_path / "fasta"
    pdb_dir = tmp_path / "pdb"
    fasta_dir.mkdir()
    pdb_dir.mkdir()
    (fasta_dir / "1ABC.fasta").write_text(">1ABC\nACDEFGHIKLM\n")
    (fasta_dir / "2XYZ.fasta").write_text(">2XYZ\nMNPQRSTVWY\n")
    (pdb_dir / "1ABC.pdb").write_text("ATOM  ...")
    return tmp_path


class TestRunnerDiscovery:
    def test_discovers_proteins_from_fasta_dir(self, tmp_path):
        bench_dir = _make_bench_dir(tmp_path)
        from ghostfold.benchmark.runner import _discover_proteins
        proteins = _discover_proteins(bench_dir)
        ids = [p[0] for p in proteins]
        assert "1ABC" in ids
        assert "2XYZ" in ids

    def test_protein_id_is_stem(self, tmp_path):
        bench_dir = _make_bench_dir(tmp_path)
        from ghostfold.benchmark.runner import _discover_proteins
        proteins = _discover_proteins(bench_dir)
        for pid, seq in proteins:
            assert "." not in pid
            assert "/" not in pid

    def test_sequence_read_correctly(self, tmp_path):
        bench_dir = _make_bench_dir(tmp_path)
        from ghostfold.benchmark.runner import _discover_proteins
        proteins = dict(_discover_proteins(bench_dir))
        assert proteins["1ABC"] == "ACDEFGHIKLM"

    def test_pdb_matched_by_stem(self, tmp_path):
        bench_dir = _make_bench_dir(tmp_path)
        from ghostfold.benchmark.runner import _find_ref_pdb
        pdb = _find_ref_pdb(bench_dir, "1ABC")
        assert pdb is not None
        assert pdb.exists()

    def test_missing_pdb_returns_none(self, tmp_path):
        bench_dir = _make_bench_dir(tmp_path)
        from ghostfold.benchmark.runner import _find_ref_pdb
        pdb = _find_ref_pdb(bench_dir, "2XYZ")  # no pdb for 2XYZ
        assert pdb is None


class TestEncoderModelArg:
    def test_generate_msa_receives_encoder_model(self, tmp_path):
        """Strategy.generate_msa must be called with encoder_model kwarg."""
        bench_dir = _make_bench_dir(tmp_path)

        from ghostfold.msa.strategies.base import BaseStrategy

        class CapturingStrategy(BaseStrategy):
            name = "capturing"
            received_config = {}

            def generate_msa(self, query_seq, model, tokenizer, device, config):
                CapturingStrategy.received_config = config
                return []

        with patch("ghostfold.benchmark.runner.STRATEGIES", {"capturing": CapturingStrategy}):
            from ghostfold.benchmark.runner import run_benchmark
            import torch
            run_benchmark(
                bench_dir=bench_dir,
                out_dir=tmp_path / "out",
                strategy_names=["capturing"],
                strategy_configs={"capturing": {}},
                model=MagicMock(),
                tokenizer=MagicMock(),
                device=torch.device("cpu"),
                encoder_model=None,
                run_colabfold=False,
            )
        assert "encoder_model" in CapturingStrategy.received_config
