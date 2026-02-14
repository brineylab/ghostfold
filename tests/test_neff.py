import pytest
import math
from pathlib import Path

from ghostfold.msa.neff import parse_a3m, calculate_neff, process_single_file, run_neff_calculation_in_parallel


class TestParseA3m:
    def test_basic_parsing(self, sample_a3m):
        seqs = parse_a3m(sample_a3m)
        assert len(seqs) == 4  # query + 3 sequences

    def test_sequence_content(self, sample_a3m):
        seqs = parse_a3m(sample_a3m)
        assert seqs[0] == "MQCDGLDGADGTSNGQAGASGLAGG"

    def test_missing_file_raises(self, tmp_dir):
        with pytest.raises(FileNotFoundError):
            parse_a3m(tmp_dir / "nonexistent.a3m")

    def test_empty_returns_empty(self, tmp_dir):
        empty = tmp_dir / "empty.a3m"
        empty.write_text(">header\n")
        seqs = parse_a3m(empty)
        # Just a header with no sequence
        assert len(seqs) == 0


class TestCalculateNeff:
    def test_empty_sequences(self):
        assert calculate_neff([]) == 0.0

    def test_single_sequence(self):
        result = calculate_neff(["ACDEF"])
        # Single seq: 1/(1+0) = 1, * 1/sqrt(5) = ~0.447
        expected = 1.0 / math.sqrt(5)
        assert abs(result - expected) < 0.001

    def test_identical_sequences(self):
        seqs = ["ACDEF", "ACDEF", "ACDEF"]
        result = calculate_neff(seqs)
        # All identical, so each has 2 similar seqs
        # Each contributes 1/(1+2) = 1/3, total = 3*(1/3) = 1
        # Neff = 1/sqrt(5) * 1 = 0.447
        expected = (1.0 / math.sqrt(5)) * (3 * (1.0 / 3.0))
        assert abs(result - expected) < 0.001

    def test_zero_length_raises(self):
        with pytest.raises(ValueError, match="length of 0"):
            calculate_neff([""])

    def test_inconsistent_length_raises(self):
        with pytest.raises(ValueError, match="same length"):
            calculate_neff(["ACDEF", "AC"])

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError, match="identity_threshold"):
            calculate_neff(["ACDEF"], identity_threshold=1.5)


class TestProcessSingleFile:
    def test_basic(self, sample_a3m):
        result = process_single_file(sample_a3m)
        assert result is not None
        name, neff_val = result
        assert isinstance(name, str)
        assert isinstance(neff_val, float)
        assert neff_val > 0


class TestRunNeffParallel:
    def test_no_a3m_files(self, tmp_dir):
        """Should handle no files found gracefully."""
        msa_dir = tmp_dir / "msa" / "test"
        msa_dir.mkdir(parents=True)
        run_neff_calculation_in_parallel(str(tmp_dir))
        # No CSV should be created when no a3m files are found
        assert not (tmp_dir / "neff_results.csv").exists()

    def test_with_a3m_files(self, tmp_dir):
        """Should produce CSV output."""
        msa_dir = tmp_dir / "msa" / "test_protein"
        msa_dir.mkdir(parents=True)
        a3m = msa_dir / "test.a3m"
        a3m.write_text(">seq1\nACDEF\n>seq2\nACDEG\n")
        run_neff_calculation_in_parallel(str(tmp_dir))
        csv_path = tmp_dir / "neff_results.csv"
        assert csv_path.exists()
        content = csv_path.read_text()
        assert "test_protein" in content

    def test_missing_dir_raises(self):
        with pytest.raises(FileNotFoundError):
            run_neff_calculation_in_parallel("/nonexistent/path")
