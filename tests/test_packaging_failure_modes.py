from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from test_packaging_smoke import _build_artifacts


def test_build_artifacts_skips_on_offline_backend_resolution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail_build(*args, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=["python", "-m", "build"],
            output="",
            stderr="ERROR: No matching distribution found for hatchling>=1.25.0",
        )

    monkeypatch.setattr(subprocess, "run", fail_build)

    with pytest.raises(pytest.skip.Exception, match="offline or restricted network"):
        _build_artifacts(tmp_path, tmp_path / "dist")


def test_build_artifacts_reraises_unrelated_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail_build(*args, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=["python", "-m", "build"],
            output="",
            stderr="ERROR: backend internal crash",
        )

    monkeypatch.setattr(subprocess, "run", fail_build)

    with pytest.raises(subprocess.CalledProcessError):
        _build_artifacts(tmp_path, tmp_path / "dist")
