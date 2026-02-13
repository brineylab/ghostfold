"""Shared filesystem validation helpers for workflow services."""

from __future__ import annotations

from pathlib import Path

from ghostfold.errors import GhostfoldValidationError


def ensure_file(path: Path, label: str) -> Path:
    """Returns a resolved path if it exists and is a file."""
    resolved = Path(path)
    if not resolved.is_file():
        raise GhostfoldValidationError(f"{label} not found at '{resolved}'")
    return resolved


def ensure_dir(path: Path, label: str) -> Path:
    """Returns a resolved path if it exists and is a directory."""
    resolved = Path(path)
    if not resolved.is_dir():
        raise GhostfoldValidationError(f"{label} not found at '{resolved}'")
    return resolved
