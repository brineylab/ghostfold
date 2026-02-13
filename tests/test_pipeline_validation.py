from __future__ import annotations

from pathlib import Path

import pytest

from ghostfold.config import PipelineWorkflowConfig
from ghostfold.errors import GhostfoldValidationError
from ghostfold.services.pipeline import _validate_pipeline_config


def test_validate_pipeline_config_requires_project_name() -> None:
    config = PipelineWorkflowConfig(project_name="", fold_only=True)
    with pytest.raises(GhostfoldValidationError, match="--project_name is a required argument"):
        _validate_pipeline_config(config)


def test_validate_pipeline_config_rejects_mutually_exclusive_mode_flags(tmp_path: Path) -> None:
    fasta = tmp_path / "query.fasta"
    fasta.write_text(">q\nAAAA\n")
    config = PipelineWorkflowConfig(
        project_name="demo",
        fasta_file=fasta,
        msa_only=True,
        fold_only=True,
    )
    with pytest.raises(GhostfoldValidationError, match="cannot be used together"):
        _validate_pipeline_config(config)


def test_validate_pipeline_config_requires_fasta_outside_fold_only() -> None:
    config = PipelineWorkflowConfig(project_name="demo")
    with pytest.raises(GhostfoldValidationError, match="--fasta_file is required"):
        _validate_pipeline_config(config)


def test_validate_pipeline_config_requires_existing_fasta(tmp_path: Path) -> None:
    config = PipelineWorkflowConfig(
        project_name="demo",
        fasta_file=tmp_path / "missing.fasta",
    )
    with pytest.raises(GhostfoldValidationError, match="FASTA file not found"):
        _validate_pipeline_config(config)


@pytest.mark.parametrize("mask_value", ["0", "0.15", "1.0"])
def test_validate_pipeline_config_accepts_shell_compatible_mask_formats(
    tmp_path: Path,
    mask_value: str,
) -> None:
    fasta = tmp_path / "query.fasta"
    fasta.write_text(">q\nAAAA\n")
    config = PipelineWorkflowConfig(
        project_name="demo",
        fasta_file=fasta,
        mask_msa=mask_value,
    )
    _validate_pipeline_config(config)


def test_validate_pipeline_config_rejects_non_shell_mask_format(tmp_path: Path) -> None:
    fasta = tmp_path / "query.fasta"
    fasta.write_text(">q\nAAAA\n")
    config = PipelineWorkflowConfig(
        project_name="demo",
        fasta_file=fasta,
        mask_msa="1",
    )
    with pytest.raises(GhostfoldValidationError, match="--mask_msa requires a fraction"):
        _validate_pipeline_config(config)
