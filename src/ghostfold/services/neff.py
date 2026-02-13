"""Service wrapper for Neff calculation orchestration."""

from __future__ import annotations

from ghostfold.config import NeffWorkflowConfig
from ghostfold.errors import GhostfoldExecutionError
from ghostfold.results import NeffWorkflowResult
from ghostfold.services.common import ensure_dir


def run_neff_workflow(config: NeffWorkflowConfig) -> NeffWorkflowResult:
    """Runs Neff calculation workflow using a typed configuration object."""
    project_dir = ensure_dir(config.project_dir, "Project directory")

    from ghostfold.neff import run_neff_calculation_in_parallel

    try:
        run_neff_calculation_in_parallel(str(project_dir))
    except Exception as exc:
        raise GhostfoldExecutionError(
            f"Neff workflow failed for '{project_dir}': {exc}"
        ) from exc

    output_csv = project_dir / "neff_results.csv"
    if output_csv.exists():
        message = f"Neff workflow completed. Results saved to '{output_csv}'."
    else:
        message = "Neff workflow completed with no results CSV generated."
        output_csv = None

    return NeffWorkflowResult(
        success=True,
        message=message,
        project_dir=project_dir,
        output_csv=output_csv,
    )
