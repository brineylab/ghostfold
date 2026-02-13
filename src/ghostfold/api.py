"""Public API wrappers routed through the workflow service layer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

from ghostfold.config import (
    MSAWorkflowConfig,
    MaskWorkflowConfig,
    NeffWorkflowConfig,
    PipelineWorkflowConfig,
)
from ghostfold.errors import (
    GhostfoldError,
    GhostfoldExecutionError,
    GhostfoldIOError,
    GhostfoldValidationError,
)
from ghostfold.results import (
    MSAWorkflowResult,
    MaskWorkflowResult,
    NeffWorkflowResult,
    PipelineWorkflowResult,
)

__all__ = [
    "MSAWorkflowConfig",
    "MaskWorkflowConfig",
    "NeffWorkflowConfig",
    "PipelineWorkflowConfig",
    "MSAWorkflowResult",
    "MaskWorkflowResult",
    "NeffWorkflowResult",
    "PipelineWorkflowResult",
    "GhostfoldError",
    "GhostfoldValidationError",
    "GhostfoldIOError",
    "GhostfoldExecutionError",
    "run_pipeline",
    "run_pipeline_workflow",
    "run_msa_workflow",
    "mask_a3m_file",
    "run_mask_workflow",
    "run_neff_calculation",
    "run_neff_workflow",
]


def run_pipeline_workflow(config: PipelineWorkflowConfig) -> PipelineWorkflowResult:
    """Runs full/msa/fold project orchestration from a typed config object."""
    from .services import run_pipeline_workflow as _run_pipeline_workflow

    return _run_pipeline_workflow(config)


def run_msa_workflow(config: MSAWorkflowConfig) -> MSAWorkflowResult:
    """Runs pseudoMSA workflow from a typed config object."""
    from .services import run_msa_workflow as _run_msa_workflow

    return _run_msa_workflow(config)


def run_pipeline(
    project: str,
    query_fasta: str,
    config_path: str,
    coverage_list: Optional[Sequence[float]],
    evolve_msa: bool,
    mutation_rates_str: str,
    sample_percentage: float,
    plot_msa: bool,
    plot_coevolution: bool,
    hex_colors: Optional[Any] = None,
    num_runs: int = 1,
) -> MSAWorkflowResult:
    """Compatibility wrapper preserving the historical `run_pipeline` signature."""
    # `hex_colors` is accepted for compatibility with historical call sites.
    _ = hex_colors
    return run_msa_workflow(
        MSAWorkflowConfig(
            project_name=project,
            fasta_file=Path(query_fasta),
            config_path=Path(config_path),
            coverage=coverage_list,
            num_runs=num_runs,
            plot_msa_coverage=plot_msa,
            no_coevolution_maps=not plot_coevolution,
            evolve_msa=evolve_msa,
            mutation_rates=mutation_rates_str,
            sample_percentage=sample_percentage,
        )
    )


def run_mask_workflow(config: MaskWorkflowConfig) -> MaskWorkflowResult:
    """Runs masking workflow from a typed config object."""
    from .services import run_mask_workflow as _run_mask_workflow

    return _run_mask_workflow(config)


def mask_a3m_file(input_path: Path, output_path: Path, mask_fraction: float) -> MaskWorkflowResult:
    """Compatibility wrapper preserving historical `mask_a3m_file` usage."""
    return run_mask_workflow(
        MaskWorkflowConfig(
            input_path=input_path,
            output_path=output_path,
            mask_fraction=mask_fraction,
        )
    )


def run_neff_workflow(config: NeffWorkflowConfig) -> NeffWorkflowResult:
    """Runs Neff workflow from a typed config object."""
    from .services import run_neff_workflow as _run_neff_workflow

    return _run_neff_workflow(config)


def run_neff_calculation(root_dir: str) -> NeffWorkflowResult:
    """Compatibility wrapper preserving historical Neff invocation."""
    return run_neff_workflow(NeffWorkflowConfig(project_dir=Path(root_dir)))
