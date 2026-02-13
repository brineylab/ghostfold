from __future__ import annotations

from pathlib import Path

import pytest

from ghostfold.config import MaskWorkflowConfig
from ghostfold.errors import GhostfoldValidationError
from ghostfold.masking import mask_a3m_file
from ghostfold.services.masking import run_mask_workflow


def test_mask_a3m_file_masks_non_query_sequences(tmp_path: Path) -> None:
    input_path = tmp_path / "input.a3m"
    output_path = tmp_path / "output.a3m"
    input_path.write_text(">query\nAAAA\n>seq1\nBBBB\n")

    mask_a3m_file(input_path=input_path, output_path=output_path, mask_fraction=1.0)

    lines = output_path.read_text().splitlines()
    assert lines == [">query", "AAAA", ">seq1", "XXXX"]


def test_run_mask_workflow_returns_typed_result(tmp_path: Path) -> None:
    input_path = tmp_path / "input.a3m"
    output_path = tmp_path / "output.a3m"
    input_path.write_text(">query\nAAAA\n>seq1\nBBBB\n")

    result = run_mask_workflow(
        MaskWorkflowConfig(
            input_path=input_path,
            output_path=output_path,
            mask_fraction=0.0,
        )
    )

    assert result.success is True
    assert result.output_path == output_path
    assert output_path.read_text() == input_path.read_text()


def test_run_mask_workflow_rejects_invalid_fraction(tmp_path: Path) -> None:
    input_path = tmp_path / "input.a3m"
    input_path.write_text(">query\nAAAA\n")
    with pytest.raises(GhostfoldValidationError, match="mask_fraction must be between 0.0 and 1.0"):
        run_mask_workflow(
            MaskWorkflowConfig(
                input_path=input_path,
                output_path=tmp_path / "output.a3m",
                mask_fraction=1.5,
            )
        )
