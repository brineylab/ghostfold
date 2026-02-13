"""Typed result models for stable GhostFold API responses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass(frozen=True)
class MSAWorkflowResult:
    """Result returned by the pseudoMSA workflow service."""

    success: bool
    message: str
    project_dir: Path
    fasta_file: Path
    config_path: Path
    coverage: Tuple[float, ...]
    num_runs: int


@dataclass(frozen=True)
class MaskWorkflowResult:
    """Result returned by the mask workflow service."""

    success: bool
    message: str
    input_path: Path
    output_path: Path
    mask_fraction: float


@dataclass(frozen=True)
class NeffWorkflowResult:
    """Result returned by the Neff workflow service."""

    success: bool
    message: str
    project_dir: Path
    output_csv: Optional[Path]


@dataclass(frozen=True)
class PipelineWorkflowResult:
    """Result returned by full/msa/fold project orchestration workflow."""

    success: bool
    message: str
    project_dir: Path
    mode: str
    num_gpus: int
    zip_outputs: Tuple[Path, ...]
    warnings: Tuple[str, ...] = tuple()
