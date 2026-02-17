import pytest
from pathlib import Path

from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

from ghostfold.io.fasta import (
    read_fasta,
    write_fasta,
    append_fasta,
    create_project_dir,
    concatenate_fasta_files,
    collect_fasta_paths,
    read_fasta_from_path,
)


class TestReadFasta:
    def test_basic_read(self, sample_fasta):
        records = read_fasta(str(sample_fasta))
        assert len(records) == 1
        assert records[0].id == "test_seq"

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            read_fasta("/nonexistent/file.fasta")

    def test_path_object(self, sample_fasta):
        records = read_fasta(sample_fasta)
        assert len(records) == 1


class TestWriteFasta:
    def test_roundtrip(self, tmp_dir):
        records = [
            SeqRecord(Seq("ACDEF"), id="seq1", description=""),
            SeqRecord(Seq("GHIKL"), id="seq2", description=""),
        ]
        path = tmp_dir / "output.fasta"
        write_fasta(str(path), records)
        result = read_fasta(str(path))
        assert len(result) == 2
        assert str(result[0].seq) == "ACDEF"
        assert str(result[1].seq) == "GHIKL"


class TestAppendFasta:
    def test_creates_new_file(self, tmp_dir):
        path = tmp_dir / "new.fasta"
        append_fasta(["ACDEF", "GHIKL"], str(path))
        assert path.exists()
        records = read_fasta(str(path))
        assert len(records) == 2

    def test_appends_to_existing(self, tmp_dir):
        path = tmp_dir / "existing.fasta"
        append_fasta(["ACDEF"], str(path))
        append_fasta(["GHIKL"], str(path))
        records = read_fasta(str(path))
        assert len(records) == 2


class TestCreateProjectDir:
    def test_creates_directory(self, tmp_dir):
        project_dir = create_project_dir(str(tmp_dir / "project"), "test_header")
        assert Path(project_dir).exists()
        assert "msa" in project_dir

    def test_sanitizes_header(self, tmp_dir):
        project_dir = create_project_dir(
            str(tmp_dir / "project"), "weird/header:name!"
        )
        # Should not contain special characters in directory name
        dir_name = Path(project_dir).name
        assert "/" not in dir_name
        assert ":" not in dir_name

    def test_idempotent(self, tmp_dir):
        path1 = create_project_dir(str(tmp_dir / "project"), "test")
        path2 = create_project_dir(str(tmp_dir / "project"), "test")
        assert path1 == path2


class TestConcatenateFastaFiles:
    def test_concatenate(self, tmp_dir):
        f1 = tmp_dir / "a.fasta"
        f2 = tmp_dir / "b.fasta"
        write_fasta(str(f1), [SeqRecord(Seq("ACDEF"), id="s1", description="")])
        write_fasta(str(f2), [SeqRecord(Seq("GHIKL"), id="s2", description="")])

        output = tmp_dir / "combined.fasta"
        concatenate_fasta_files([str(f1), str(f2)], str(output))
        records = read_fasta(str(output))
        assert len(records) == 2

    def test_empty_list(self, tmp_dir):
        output = tmp_dir / "empty.fasta"
        concatenate_fasta_files([], str(output))
        assert not output.exists()

    def test_skips_missing_files(self, tmp_dir):
        output = tmp_dir / "output.fasta"
        concatenate_fasta_files(["/nonexistent/file.fasta"], str(output))
        assert not output.exists()


class TestCollectFastaPaths:
    def test_single_file(self, sample_fasta):
        paths = collect_fasta_paths(sample_fasta)
        assert len(paths) == 1
        assert paths[0] == sample_fasta

    def test_file_with_recursive_warns(self, sample_fasta, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            paths = collect_fasta_paths(sample_fasta, recursive=True)
        assert len(paths) == 1
        assert "no effect" in caplog.text.lower()

    def test_directory_non_recursive(self, fasta_dir):
        paths = collect_fasta_paths(fasta_dir)
        assert len(paths) == 2
        names = {p.name for p in paths}
        assert names == {"seq1.fasta", "seq2.fa"}

    def test_directory_recursive(self, fasta_dir):
        paths = collect_fasta_paths(fasta_dir, recursive=True)
        assert len(paths) == 3
        names = {p.name for p in paths}
        assert names == {"seq1.fasta", "seq2.fa", "seq3.fasta"}

    def test_non_recursive_ignores_subdirs(self, fasta_dir):
        paths = collect_fasta_paths(fasta_dir, recursive=False)
        names = {p.name for p in paths}
        assert "seq3.fasta" not in names

    def test_empty_dir_raises(self, tmp_dir):
        empty = tmp_dir / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="No FASTA files"):
            collect_fasta_paths(empty)

    def test_no_matching_extensions_raises(self, tmp_dir):
        no_match = tmp_dir / "no_match"
        no_match.mkdir()
        (no_match / "data.txt").write_text(">seq\nACDEF\n")
        with pytest.raises(ValueError, match="No FASTA files"):
            collect_fasta_paths(no_match)

    def test_nonexistent_path_raises(self):
        with pytest.raises(FileNotFoundError):
            collect_fasta_paths("/nonexistent/path")

    def test_deterministic_ordering(self, fasta_dir):
        paths1 = collect_fasta_paths(fasta_dir, recursive=True)
        paths2 = collect_fasta_paths(fasta_dir, recursive=True)
        assert paths1 == paths2
        # Should be sorted
        assert paths1 == sorted(paths1)


class TestReadFastaFromPath:
    def test_single_file(self, sample_fasta):
        records = read_fasta_from_path(sample_fasta)
        assert len(records) == 1
        assert records[0].id == "test_seq"

    def test_directory(self, fasta_dir):
        records = read_fasta_from_path(fasta_dir)
        assert len(records) == 2
        ids = {r.id for r in records}
        assert ids == {"seq1", "seq2"}

    def test_recursive_directory(self, fasta_dir):
        records = read_fasta_from_path(fasta_dir, recursive=True)
        assert len(records) == 3
        ids = {r.id for r in records}
        assert ids == {"seq1", "seq2", "seq3"}
