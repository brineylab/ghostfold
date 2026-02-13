"""Public exception hierarchy for GhostFold workflows."""

from __future__ import annotations


class GhostfoldError(RuntimeError):
    """Base class for user-facing GhostFold workflow errors."""


class GhostfoldValidationError(GhostfoldError):
    """Raised when workflow configuration or inputs are invalid."""


class GhostfoldIOError(GhostfoldError):
    """Raised when filesystem IO fails during workflow execution."""


class GhostfoldExecutionError(GhostfoldError):
    """Raised when a workflow fails during runtime execution."""
