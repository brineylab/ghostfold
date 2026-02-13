from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set

PIXI_INSTALL_URL = "https://pixi.prefix.dev/latest/installation/"
MAMBA_INSTALL_URL = (
    "https://mamba.readthedocs.io/en/stable/installation/mamba-installation.html"
)
DEFAULT_COLABFOLD_ENV = "colabfold"
DEFAULT_LOCALCOLABFOLD_DIR = Path("localcolabfold")


@dataclass(frozen=True)
class ColabFoldLauncher:
    """Execution settings for a functional ColabFold runtime."""

    mode: str
    command_prefix: tuple[str, ...]
    cwd: Optional[Path]


class ColabFoldSetupError(RuntimeError):
    """Raised when a functional ColabFold runtime is not available."""


def resolve_localcolabfold_dir(localcolabfold_dir: Path | str | None = None) -> Path:
    """Resolve localcolabfold directory to an absolute path."""
    if localcolabfold_dir is None:
        candidate = Path.cwd() / DEFAULT_LOCALCOLABFOLD_DIR
    else:
        candidate = Path(localcolabfold_dir)
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
    return candidate.resolve()


def _install_command(localcolabfold_dir: Path | str | None = None) -> str:
    _ = localcolabfold_dir
    return "bash scripts/install_localcolabfold.sh"


def _format_setup_error(
    reason: str,
    localcolabfold_dir: Path | str | None = None,
    include_pixi_hint: bool = True,
    include_mamba_hint: bool = True,
) -> ColabFoldSetupError:
    lines = [
        reason,
        (
            "Install/configure ColabFold by running "
            f"`{_install_command(localcolabfold_dir)}` from the GhostFold repository root."
        ),
        "This setup is required for `ghostfold run` and `ghostfold fold`.",
    ]
    if include_pixi_hint:
        lines.append(f"Pixi installation instructions: {PIXI_INSTALL_URL}")
    if include_mamba_hint:
        lines.append(f"Mamba installation instructions: {MAMBA_INSTALL_URL}")
    return ColabFoldSetupError("\n".join(lines))


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


def _validate_pixi_runtime(
    localcolabfold_dir: Path,
) -> tuple[Optional[ColabFoldLauncher], str]:
    if not shutil.which("pixi"):
        return None, "pixi is not installed or not available on PATH."

    if not (localcolabfold_dir / "pixi.toml").is_file():
        return (
            None,
            f"No localcolabfold pixi project found at `{localcolabfold_dir}`.",
        )

    try:
        subprocess.run(
            ["pixi", "run", "colabfold_batch", "--help"],
            cwd=str(localcolabfold_dir),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return (
            None,
            f"`pixi run colabfold_batch --help` failed in `{localcolabfold_dir}`.",
        )

    return (
        ColabFoldLauncher(
            mode="pixi",
            command_prefix=("pixi", "run"),
            cwd=localcolabfold_dir,
        ),
        "",
    )


def _validate_mamba_runtime(colabfold_env: str) -> tuple[Optional[ColabFoldLauncher], str]:
    if not shutil.which("mamba"):
        return None, "mamba is not installed or not available on PATH."

    try:
        env_names = _list_mamba_env_names()
    except ColabFoldSetupError as exc:
        return None, str(exc)

    if colabfold_env not in env_names:
        return None, f"Conda environment `{colabfold_env}` was not found."

    try:
        subprocess.run(
            ["mamba", "run", "-n", colabfold_env, "colabfold_batch", "--help"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return None, f"`colabfold_batch` is not functional in `{colabfold_env}`."

    return (
        ColabFoldLauncher(
            mode="mamba",
            command_prefix=("mamba", "run", "-n", colabfold_env, "--no-capture-output"),
            cwd=None,
        ),
        "",
    )


def ensure_colabfold_ready(
    colabfold_env: str = DEFAULT_COLABFOLD_ENV,
    localcolabfold_dir: Path | str | None = None,
) -> ColabFoldLauncher:
    """Resolve and validate a functional ColabFold launcher."""
    resolved_localcolabfold_dir = resolve_localcolabfold_dir(localcolabfold_dir)

    pixi_launcher, pixi_reason = _validate_pixi_runtime(resolved_localcolabfold_dir)
    if pixi_launcher is not None:
        return pixi_launcher

    mamba_launcher, mamba_reason = _validate_mamba_runtime(colabfold_env)
    if mamba_launcher is not None:
        return mamba_launcher

    reason = (
        "Could not find a functional ColabFold runtime.\n"
        f"Pixi check: {pixi_reason}\n"
        f"Mamba check: {mamba_reason}"
    )
    raise _format_setup_error(
        reason=reason,
        localcolabfold_dir=resolved_localcolabfold_dir,
        include_pixi_hint=True,
        include_mamba_hint=True,
    )
