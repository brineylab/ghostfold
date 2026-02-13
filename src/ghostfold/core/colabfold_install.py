from __future__ import annotations

import shutil
import stat
import subprocess
from pathlib import Path
from typing import Sequence
from urllib.request import urlretrieve

from ghostfold.core.colabfold_env import (
    DEFAULT_COLABFOLD_ENV,
    MAMBA_INSTALL_URL,
    ColabFoldSetupError,
    ensure_colabfold_ready,
)

UPDATER_SCRIPT_URL = (
    "https://raw.githubusercontent.com/YoshitakaMo/localcolabfold/main/update_linux.sh"
)


def _run_command(cmd: Sequence[str]) -> None:
    try:
        subprocess.run(
            list(cmd),
            check=True,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as exc:
        cmd_str = " ".join(cmd)
        raise ColabFoldSetupError(
            f"Command failed during ColabFold installation: `{cmd_str}`"
        ) from exc


def _run_command_capture(cmd: Sequence[str]) -> str:
    try:
        result = subprocess.run(
            list(cmd),
            check=True,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as exc:
        cmd_str = " ".join(cmd)
        raise ColabFoldSetupError(
            f"Command failed during ColabFold installation: `{cmd_str}`"
        ) from exc
    return result.stdout.strip()


def _replace_or_fail(file_path: Path, old: str, new: str) -> None:
    if not file_path.is_file():
        raise ColabFoldSetupError(f"Expected file not found: {file_path}")

    original = file_path.read_text()
    if new in original:
        return
    if old not in original:
        raise ColabFoldSetupError(
            f"Could not find expected text to patch in: {file_path}"
        )

    file_path.write_text(original.replace(old, new, 1))


def _patch_colabfold_sources(colabfold_env: str, colabfold_dir: Path) -> None:
    package_path_str = _run_command_capture(
        [
            "mamba",
            "run",
            "-n",
            colabfold_env,
            "python",
            "-c",
            "import colabfold; print(colabfold.__path__[0])",
        ]
    )
    package_path = Path(package_path_str)
    if not package_path.is_dir():
        raise ColabFoldSetupError(
            f"Could not locate colabfold package path: {package_path}"
        )

    _replace_or_fail(
        package_path / "plot.py",
        "from matplotlib import pyplot as plt",
        "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt",
    )
    _replace_or_fail(
        package_path / "download.py",
        'appdirs.user_cache_dir(__package__ or "colabfold")',
        f'"{colabfold_dir / "colabfold"}"',
    )
    _replace_or_fail(
        package_path / "batch.py",
        "from io import StringIO",
        "from io import StringIO\nfrom silence_tensorflow import silence_tensorflow\nsilence_tensorflow()",
    )

    pycache_dir = package_path / "__pycache__"
    if pycache_dir.is_dir():
        shutil.rmtree(pycache_dir)


def install_colabfold(
    colabfold_env: str = DEFAULT_COLABFOLD_ENV,
    data_dir: Path | None = None,
) -> Path:
    """Install ColabFold into a dedicated mamba environment."""
    if shutil.which("mamba") is None:
        raise ColabFoldSetupError(
            "mamba is not installed or not available on PATH.\n"
            f"Install instructions: {MAMBA_INSTALL_URL}"
        )

    colabfold_dir = (data_dir or (Path.cwd() / "localcolabfold")).resolve()
    colabfold_dir.mkdir(parents=True, exist_ok=True)

    _run_command(
        [
            "mamba",
            "create",
            "-n",
            colabfold_env,
            "-c",
            "conda-forge",
            "python=3.10",
            "-y",
        ]
    )
    _run_command(
        [
            "mamba",
            "install",
            "-n",
            colabfold_env,
            "-c",
            "conda-forge",
            "-c",
            "bioconda",
            "git",
            "openmm==8.2.0",
            "pdbfixer",
            "kalign2=2.04",
            "hhsuite=3.3.0",
            "mmseqs2",
            "-y",
        ]
    )
    _run_command(
        [
            "mamba",
            "run",
            "-n",
            colabfold_env,
            "pip",
            "install",
            "--no-warn-conflicts",
            "colabfold[alphafold] @ git+https://github.com/sokrypton/ColabFold",
        ]
    )
    _run_command(
        [
            "mamba",
            "run",
            "-n",
            colabfold_env,
            "pip",
            "install",
            "--upgrade",
            "jax[cuda12]==0.5.3",
            "tensorflow",
            "silence_tensorflow",
        ]
    )

    updater_path = colabfold_dir / "update_linux.sh"
    try:
        urlretrieve(UPDATER_SCRIPT_URL, updater_path)
    except Exception as exc:  # noqa: BLE001 - we want a clear user-facing setup error.
        raise ColabFoldSetupError(
            f"Failed to download updater script from {UPDATER_SCRIPT_URL}"
        ) from exc
    updater_path.chmod(
        updater_path.stat().st_mode
        | stat.S_IXUSR
        | stat.S_IXGRP
        | stat.S_IXOTH
    )

    _patch_colabfold_sources(colabfold_env, colabfold_dir)
    _run_command(
        ["mamba", "run", "-n", colabfold_env, "python", "-m", "colabfold.download"]
    )
    ensure_colabfold_ready(colabfold_env)

    return colabfold_dir
