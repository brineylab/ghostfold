"""Service wrapper for masking workflow orchestration."""

from __future__ import annotations

from pathlib import Path

from ghostfold.config import MaskWorkflowConfig
from ghostfold.errors import GhostfoldExecutionError, GhostfoldIOError, GhostfoldValidationError
from ghostfold.results import MaskWorkflowResult
from ghostfold.services.common import ensure_file


def run_mask_workflow(config: MaskWorkflowConfig) -> MaskWorkflowResult:
    """Runs masking workflow using a typed configuration object."""
    if not 0.0 <= config.mask_fraction <= 1.0:
        raise GhostfoldValidationError("mask_fraction must be between 0.0 and 1.0")

    input_path = ensure_file(config.input_path, "Input A3M file")
    output_path = Path(config.output_path)

    from ghostfold.masking import mask_a3m_file

    try:
        mask_a3m_file(
            input_path=input_path,
            output_path=output_path,
            mask_fraction=float(config.mask_fraction),
        )
    except (ValueError, TypeError) as exc:
        raise GhostfoldValidationError(str(exc)) from exc
    except OSError as exc:
        raise GhostfoldIOError(str(exc)) from exc
    except Exception as exc:
        raise GhostfoldExecutionError(f"Masking workflow failed: {exc}") from exc

    return MaskWorkflowResult(
        success=True,
        message=f"Masked alignment written to '{output_path}'.",
        input_path=input_path,
        output_path=output_path,
        mask_fraction=float(config.mask_fraction),
    )
