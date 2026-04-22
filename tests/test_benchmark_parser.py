"""Tests for ColabFold output parsing and structure metrics."""
import json
from unittest.mock import patch

import numpy as np
import pytest

from ghostfold.benchmark.colabfold_parser import parse_best_scores
from ghostfold.benchmark.structure_metrics import (
    _kabsch_rmsd,
    _tm_score,
    compute_structure_metrics,
)


# ---------------------------------------------------------------------------
# ColabFold parser
# ---------------------------------------------------------------------------

class TestParseColabfoldScores:
    def test_parses_ptm_and_plddt(self, tmp_path):
        plddt = [85.0, 90.0, 78.0, 92.0]
        score_data = {"ptm": 0.87, "plddt": plddt, "max_pae": 5.2}
        score_file = tmp_path / "query_scores_rank_001_alphafold2_ptm_model_1_seed_000.json"
        score_file.write_text(json.dumps(score_data))

        result = parse_best_scores(tmp_path)
        assert result["ptm"] == pytest.approx(0.87)
        assert result["mean_plddt"] == pytest.approx(sum(plddt) / len(plddt))
        assert result["max_pae"] == pytest.approx(5.2)

    def test_locates_unrelaxed_pdb(self, tmp_path):
        score_data = {"ptm": 0.8, "plddt": [80.0]}
        (tmp_path / "query_scores_rank_001_model_1_seed_000.json").write_text(
            json.dumps(score_data)
        )
        pdb_path = tmp_path / "query_unrelaxed_rank_001_model_1_seed_000.pdb"
        pdb_path.write_text("ATOM ...")

        result = parse_best_scores(tmp_path)
        assert result["best_pdb"] == pdb_path

    def test_prefers_relaxed_over_unrelaxed(self, tmp_path):
        score_data = {"ptm": 0.8, "plddt": [80.0]}
        (tmp_path / "query_scores_rank_001_model_1.json").write_text(json.dumps(score_data))
        relaxed = tmp_path / "query_relaxed_rank_001_model_1.pdb"
        unrelaxed = tmp_path / "query_unrelaxed_rank_001_model_1.pdb"
        relaxed.write_text("ATOM ...")
        unrelaxed.write_text("ATOM ...")

        result = parse_best_scores(tmp_path)
        assert result["best_pdb"] == relaxed

    def test_raises_when_no_score_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_best_scores(tmp_path)

    def test_missing_plddt_returns_none(self, tmp_path):
        score_data = {"ptm": 0.75}
        (tmp_path / "query_scores_rank_001.json").write_text(json.dumps(score_data))
        result = parse_best_scores(tmp_path)
        assert result["mean_plddt"] is None


# ---------------------------------------------------------------------------
# Structure metrics (pure NumPy, no PDB files needed)
# ---------------------------------------------------------------------------

class TestKabschRMSD:
    def test_identical_coords_zero_rmsd(self):
        coords = np.random.randn(20, 3)
        _, rmsd = _kabsch_rmsd(coords, coords.copy())
        assert rmsd == pytest.approx(0.0, abs=1e-6)

    def test_translated_structure_zero_rmsd_after_align(self):
        ref = np.random.randn(20, 3)
        mobile = ref + np.array([5.0, 3.0, -2.0])  # pure translation
        _, rmsd = _kabsch_rmsd(ref, mobile)
        assert rmsd == pytest.approx(0.0, abs=1e-5)

    def test_returns_float(self):
        ref = np.random.randn(10, 3)
        mob = np.random.randn(10, 3)
        _, rmsd = _kabsch_rmsd(ref, mob)
        assert isinstance(rmsd, float)

    def test_rmsd_positive(self):
        ref = np.eye(10, 3)
        mob = np.eye(10, 3) + np.random.randn(10, 3) * 0.5
        _, rmsd = _kabsch_rmsd(ref, mob)
        assert rmsd >= 0.0


class TestTMScore:
    def test_identical_structures_score_one(self):
        coords = np.random.randn(50, 3)
        score = _tm_score(coords, coords.copy(), l_ref=50)
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_score_between_zero_and_one(self):
        ref = np.random.randn(50, 3)
        pred = np.random.randn(50, 3)
        score = _tm_score(ref, pred, l_ref=50)
        assert 0.0 <= score <= 1.0

    def test_short_protein_uses_d0_floor(self):
        # l_ref < 22 → d0 clamped to 0.5, should not raise
        coords = np.random.randn(10, 3)
        score = _tm_score(coords, coords.copy(), l_ref=10)
        assert score == pytest.approx(1.0, abs=1e-5)


class TestComputeStructureMetrics:
    def test_raises_without_biotite(self, tmp_path):
        """compute_structure_metrics raises ImportError when biotite is absent."""
        with patch("ghostfold.benchmark.structure_metrics._BIOTITE_AVAILABLE", False):
            with pytest.raises(ImportError, match="biotite"):
                compute_structure_metrics(tmp_path / "pred.pdb", tmp_path / "ref.pdb")
