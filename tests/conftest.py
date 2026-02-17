import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def tmp_dir():
    """Create a temporary directory for test output."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_fasta(tmp_dir):
    """Create a minimal FASTA file for testing."""
    fasta_path = tmp_dir / "query.fasta"
    fasta_path.write_text(">test_seq\nMQCDGLDGADGTSNGQAGASGLAGG\n")
    return fasta_path


@pytest.fixture
def fasta_dir(tmp_dir):
    """Create a directory with FASTA files for testing, including a subdirectory."""
    # Top-level files
    (tmp_dir / "seq1.fasta").write_text(">seq1\nACDEF\n")
    (tmp_dir / "seq2.fa").write_text(">seq2\nGHIKL\n")
    # Subdirectory with another FASTA file
    sub = tmp_dir / "subdir"
    sub.mkdir()
    (sub / "seq3.fasta").write_text(">seq3\nMNPQR\n")
    return tmp_dir


@pytest.fixture
def sample_a3m(tmp_dir):
    """Create a minimal A3M file with a query and MSA entries."""
    a3m_path = tmp_dir / "test.a3m"
    content = (
        ">query\n"
        "MQCDGLDGADGTSNGQAGASGLAGG\n"
        ">seq1\n"
        "MQCDGLDGADGTSNGQAGASGLAGG\n"
        ">seq2\n"
        "MKCEGLEGADGTSNGQAGASGLAGG\n"
        ">seq3\n"
        "AQCDGLDGADGTSNGQAGASGLAGA\n"
    )
    a3m_path.write_text(content)
    return a3m_path
