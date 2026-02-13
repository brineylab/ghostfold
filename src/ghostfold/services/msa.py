"""Service wrapper for pseudoMSA generation orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import List

from ghostfold.config import DEFAULT_COVERAGE_VALUES, MSAWorkflowConfig
from ghostfold.errors import GhostfoldExecutionError, GhostfoldValidationError
from ghostfold.results import MSAWorkflowResult
from ghostfold.services.common import ensure_file


def run_msa_workflow(config: MSAWorkflowConfig) -> MSAWorkflowResult:
    """Runs the pseudoMSA workflow using a typed configuration object."""
    if config.num_runs < 1:
        raise GhostfoldValidationError("num_runs must be >= 1")
    if not 0.0 <= config.sample_percentage <= 1.0:
        raise GhostfoldValidationError("sample_percentage must be between 0.0 and 1.0")

    fasta_file = ensure_file(config.fasta_file, "FASTA file")
    config_path = ensure_file(config.config_path, "Config file")

    coverage_list: List[float] = (
        list(config.coverage) if config.coverage is not None else list(DEFAULT_COVERAGE_VALUES)
    )

    from ghostfold.msa_core import run_pipeline

    try:
        run_pipeline(
            project=config.project_name,
            query_fasta=str(fasta_file),
            config_path=str(config_path),
            coverage_list=coverage_list,
            evolve_msa=config.evolve_msa,
            mutation_rates_str=config.mutation_rates,
            sample_percentage=config.sample_percentage,
            num_runs=config.num_runs,
            plot_msa=config.plot_msa_coverage,
            plot_coevolution=not config.no_coevolution_maps,
        )
    except Exception as exc:
        raise GhostfoldExecutionError(
            f"MSA workflow failed for project '{config.project_name}': {exc}"
        ) from exc

    return MSAWorkflowResult(
        success=True,
        message=f"MSA workflow completed for project '{config.project_name}'.",
        project_dir=Path(config.project_name),
        fasta_file=fasta_file,
        config_path=config_path,
        coverage=tuple(coverage_list),
        num_runs=config.num_runs,
    )
