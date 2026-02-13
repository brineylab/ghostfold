from __future__ import annotations

from pathlib import Path

import pytest

import ghostfold.neff as neff_module
from ghostfold.config import NeffWorkflowConfig
from ghostfold.neff import calculate_neff, run_neff_calculation_in_parallel
from ghostfold.services.neff import run_neff_workflow


def test_calculate_neff_known_value() -> None:
    sequences = ["AAAA", "AAAA", "BBBB"]
    value = calculate_neff(sequences, identity_threshold=0.5)
    assert value == pytest.approx(1.0)


def test_calculate_neff_rejects_inconsistent_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        calculate_neff(["AAAA", "AAA"])


def test_run_neff_calculation_in_parallel_writes_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class InlineExecutor:
        def __enter__(self) -> "InlineExecutor":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def map(self, func, items):
            return [func(item) for item in items]

    monkeypatch.setattr(neff_module.concurrent.futures, "ProcessPoolExecutor", InlineExecutor)

    project_dir = tmp_path / "project"
    protein_dir = project_dir / "msa" / "protA"
    protein_dir.mkdir(parents=True)
    (protein_dir / "pstMSA.a3m").write_text(">q\nAAAA\n>s1\nAAAA\n>s2\nBBBB\n")

    run_neff_calculation_in_parallel(str(project_dir))

    output_csv = project_dir / "neff_results.csv"
    assert output_csv.is_file()
    lines = output_csv.read_text().splitlines()
    assert lines[0] == "pdb,NAD"
    assert lines[1].startswith("protA,")


def test_run_neff_workflow_handles_no_results(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    (project_dir / "msa").mkdir(parents=True)

    result = run_neff_workflow(NeffWorkflowConfig(project_dir=project_dir))

    assert result.success is True
    assert result.output_csv is None
    assert "no results CSV generated" in result.message
