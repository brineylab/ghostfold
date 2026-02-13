"""Typed workflow configuration objects used by CLI and API layers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

DEFAULT_COVERAGE_VALUES: Tuple[float, ...] = (1.0,)
DEFAULT_MUTATION_RATES: str = '{"MEGABLAST": 5, "PAM250": 20, "BLOSUM62": 10}'


@dataclass(frozen=True)
class MSAWorkflowConfig:
    """Configuration for pseudoMSA generation workflow."""

    project_name: str
    fasta_file: Path
    config_path: Path = Path("config.yaml")
    coverage: Optional[Sequence[float]] = None
    num_runs: int = 1
    plot_msa_coverage: bool = False
    no_coevolution_maps: bool = False
    evolve_msa: bool = False
    mutation_rates: str = DEFAULT_MUTATION_RATES
    sample_percentage: float = 1.0


@dataclass(frozen=True)
class MaskWorkflowConfig:
    """Configuration for A3M masking workflow."""

    input_path: Path
    output_path: Path
    mask_fraction: float


@dataclass(frozen=True)
class NeffWorkflowConfig:
    """Configuration for Neff calculation workflow."""

    project_dir: Path


@dataclass(frozen=True)
class PipelineWorkflowConfig:
    """Configuration for full/msa/fold project orchestration workflow."""

    project_name: str
    fasta_file: Optional[Path] = None
    msa_only: bool = False
    fold_only: bool = False
    subsample: bool = False
    mask_msa: Optional[str] = None
