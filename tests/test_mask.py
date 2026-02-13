import pytest
from pathlib import Path

from ghostfold.msa.mask import mask_a3m_file


class TestMaskA3mFile:
    def test_query_preserved(self, sample_a3m, tmp_dir):
        output = tmp_dir / "masked.a3m"
        mask_a3m_file(sample_a3m, output, 0.5)
        lines = output.read_text().splitlines()
        # First header is >query
        assert lines[0] == ">query"
        # Query sequence unchanged
        assert lines[1] == "MQCDGLDGADGTSNGQAGASGLAGG"

    def test_mask_fraction_zero_copies(self, sample_a3m, tmp_dir):
        output = tmp_dir / "masked.a3m"
        mask_a3m_file(sample_a3m, output, 0.0)
        assert output.read_text().strip() == sample_a3m.read_text().strip()

    def test_mask_fraction_one_masks_all(self, sample_a3m, tmp_dir):
        output = tmp_dir / "masked.a3m"
        mask_a3m_file(sample_a3m, output, 1.0)
        lines = output.read_text().splitlines()
        # Non-query sequences should be all X
        for i, line in enumerate(lines):
            if line.startswith(">") or i <= 1:
                continue
            # All uppercase letters should be X
            alpha_chars = [c for c in line if c.isalpha()]
            assert all(c == "X" for c in alpha_chars)

    def test_approximate_mask_fraction(self, tmp_dir):
        """Mask fraction should be approximately correct over many residues."""
        a3m = tmp_dir / "big.a3m"
        seq = "ACDEFGHIKLMNPQRSTVWY" * 50  # 1000 residues
        content = f">query\n{seq}\n>seq1\n{seq}\n"
        a3m.write_text(content)
        output = tmp_dir / "masked.a3m"
        mask_a3m_file(a3m, output, 0.5)
        lines = output.read_text().splitlines()
        # seq1 is on line 3 (0-indexed)
        masked_seq = lines[3]
        x_count = masked_seq.count("X")
        # Should be approximately 50% (within 10% tolerance for 1000 residues)
        assert 400 <= x_count <= 600

    def test_invalid_fraction_raises(self, sample_a3m, tmp_dir):
        output = tmp_dir / "masked.a3m"
        with pytest.raises(ValueError):
            mask_a3m_file(sample_a3m, output, 1.5)
        with pytest.raises(ValueError):
            mask_a3m_file(sample_a3m, output, -0.1)

    def test_missing_file_raises(self, tmp_dir):
        with pytest.raises(FileNotFoundError):
            mask_a3m_file(
                tmp_dir / "nonexistent.a3m", tmp_dir / "out.a3m", 0.5
            )

    def test_type_errors(self, sample_a3m, tmp_dir):
        output = tmp_dir / "masked.a3m"
        with pytest.raises(TypeError):
            mask_a3m_file("string_path", output, 0.5)
        with pytest.raises(TypeError):
            mask_a3m_file(sample_a3m, "string_path", 0.5)
        with pytest.raises(TypeError):
            mask_a3m_file(sample_a3m, output, 5)  # int, not float

    def test_empty_file_raises(self, tmp_dir):
        empty = tmp_dir / "empty.a3m"
        empty.write_text("")
        output = tmp_dir / "masked.a3m"
        with pytest.raises(ValueError, match="No FASTA headers"):
            mask_a3m_file(empty, output, 0.5)
