from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Set

MAMBA_INSTALL_URL = (
    "https://mamba.readthedocs.io/en/stable/installation/mamba-installation.html"
)
DEFAULT_COLABFOLD_ENV = "colabfold"


class ColabFoldSetupError(RuntimeError):
    """Raised when a functional ColabFold runtime is not available."""


def _install_command(colabfold_env: str) -> str:
    cmd = "ghostfold install-colabfold"
    if colabfold_env != DEFAULT_COLABFOLD_ENV:
        cmd += f" --colabfold-env {colabfold_env}"
    return cmd


def _format_setup_error(reason: str, colabfold_env: str) -> ColabFoldSetupError:
    message = (
        f"{reason}\n"
        f"Run `{_install_command(colabfold_env)}` to install/configure ColabFold.\n"
        f"Mamba installation instructions: {MAMBA_INSTALL_URL}"
    )
    return ColabFoldSetupError(message)


def _list_mamba_env_names() -> Set[str]:
    try:
        result = subprocess.run(
            ["mamba", "env", "list", "--json"],
            check=True,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as exc:
        raise ColabFoldSetupError(
            "Failed to enumerate conda environments via `mamba env list --json`."
        ) from exc

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ColabFoldSetupError(
            "Failed to parse `mamba env list --json` output."
        ) from exc

    env_paths = payload.get("envs", [])
    return {Path(env_path).name for env_path in env_paths if env_path}


def ensure_colabfold_ready(colabfold_env: str = DEFAULT_COLABFOLD_ENV) -> None:
    """Ensure ColabFold is installed and runnable in the requested mamba env."""
    if not shutil.which("mamba"):
        raise _format_setup_error(
            "mamba is not installed or not available on PATH.",
            colabfold_env,
        )

    env_names = _list_mamba_env_names()
    if colabfold_env not in env_names:
        raise _format_setup_error(
            f"Conda environment `{colabfold_env}` was not found.",
            colabfold_env,
        )

    try:
        subprocess.run(
            ["mamba", "run", "-n", colabfold_env, "colabfold_batch", "--help"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as exc:
        raise _format_setup_error(
            f"`colabfold_batch` is not functional in `{colabfold_env}`.",
            colabfold_env,
        ) from exc
