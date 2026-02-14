import os
import pytest
from pathlib import Path

from ghostfold.core.postprocess import (
    postprocess_msa_outputs,
    cleanup_colabfold_outputs,
    _is_recycle_pdb,
)


class TestPostprocessMsaOutputs:
    def test_header_rewriting(self, tmp_dir):
        msa_dir = tmp_dir / "project" / "msa" / "test_protein" / "run_1"
        msa_dir.mkdir(parents=True)
        fasta_path = msa_dir / "pstMSA.fasta"
        fasta_path.write_text(">old_header\nACDEF\n>seq2\nGHIKL\n")
        postprocess_msa_outputs(str(tmp_dir / "project"))
        content = fasta_path.read_text()
        assert content.startswith(">run_1\n")

    def test_a3m_copy_created(self, tmp_dir):
        msa_dir = tmp_dir / "project" / "msa" / "test_protein"
        msa_dir.mkdir(parents=True)
        fasta_path = msa_dir / "pstMSA.fasta"
        fasta_path.write_text(">old_header\nACDEF\n")
        postprocess_msa_outputs(str(tmp_dir / "project"))
        a3m_path = msa_dir / "pstMSA.a3m"
        assert a3m_path.exists()
        assert a3m_path.read_text() == fasta_path.read_text()


class TestCleanupColabfoldOutputs:
    def test_file_organization(self, tmp_dir):
        subsample_dir = tmp_dir / "subsample_1"
        pred_dir = subsample_dir / "preds" / "test_protein"
        pred_dir.mkdir(parents=True)

        # Create test files
        (pred_dir / "scores.json").write_text("{}")
        (pred_dir / "plot.png").write_bytes(b"")
        (pred_dir / "model.r0.pdb").write_text("ATOM")
        (pred_dir / "model_rank_001.pdb").write_text("ATOM rank1")
        (pred_dir / "done.txt").write_text("done")

        cleanup_colabfold_outputs(str(subsample_dir))

        # Check files moved to correct subdirectories
        assert (pred_dir / "scores" / "scores.json").exists()
        assert (pred_dir / "imgs" / "plot.png").exists()
        assert (pred_dir / "recycles" / "model.r0.pdb").exists()
        # done.txt should be deleted
        assert not (pred_dir / "done.txt").exists()
        # rank_001 copied to best/
        best_dir = subsample_dir / "best"
        assert (best_dir / "test_protein_ghostfold.pdb").exists()

    def test_no_preds_dir(self, tmp_dir):
        subsample_dir = tmp_dir / "subsample_1"
        subsample_dir.mkdir()
        cleanup_colabfold_outputs(str(subsample_dir))
        # Should handle missing preds directory gracefully
        assert not (subsample_dir / "preds").exists()


class TestIsRecyclePdb:
    def test_recycle_files(self):
        assert _is_recycle_pdb("model.r0.pdb") is True
        assert _is_recycle_pdb("model.r12.pdb") is True

    def test_non_recycle_files(self):
        assert _is_recycle_pdb("model_rank_001.pdb") is False
        assert _is_recycle_pdb("scores.json") is False
        assert _is_recycle_pdb("model.pdb") is False
