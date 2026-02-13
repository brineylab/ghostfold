from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional, Sequence

from ghostfold.core.colabfold_env import (
    PIXI_INSTALL_URL,
    ColabFoldSetupError,
    ensure_colabfold_ready,
    resolve_localcolabfold_dir,
)

LOCALCOLABFOLD_REPO_URL = "https://github.com/YoshitakaMo/localcolabfold.git"
_MAX_ERROR_LINES = 20
ProgressCallback = Callable[[str], None]


def _short_tail(output: str) -> str:
    lines = [line for line in output.splitlines() if line.strip()]
    if not lines:
        return ""
    return "\n".join(lines[-_MAX_ERROR_LINES:])


def _run_command(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    verbose: bool = False,
) -> None:
    kwargs = {
        "check": True,
        "stdin": subprocess.DEVNULL,
    }
    if cwd is not None:
        kwargs["cwd"] = str(cwd)

    try:
        if verbose:
            subprocess.run(list(cmd), **kwargs)
            return

        result = subprocess.run(
            list(cmd),
            capture_output=True,
            text=True,
            **kwargs,
        )
        if result.returncode != 0:
            # Defensive; check=True should raise CalledProcessError.
            raise subprocess.CalledProcessError(
                returncode=result.returncode,
                cmd=list(cmd),
                output=result.stdout,
                stderr=result.stderr,
            )
    except subprocess.CalledProcessError as exc:
        cmd_str = " ".join(cmd)
        detail = _short_tail(exc.stderr or "") or _short_tail(exc.output or "")
        if detail:
            raise ColabFoldSetupError(
                f"Command failed during ColabFold installation: `{cmd_str}`\n"
                f"{detail}"
            ) from exc
        raise ColabFoldSetupError(
            f"Command failed during ColabFold installation: `{cmd_str}`"
        ) from exc


def _emit(progress_cb: Optional[ProgressCallback], message: str) -> None:
    if progress_cb is not None:
        progress_cb(message)


def _ensure_prerequisites() -> None:
    if shutil.which("pixi") is None:
        raise ColabFoldSetupError(
            "pixi is not installed or not available on PATH.\n"
            f"Install instructions: {PIXI_INSTALL_URL}"
        )
    if shutil.which("git") is None:
        raise ColabFoldSetupError("git is not installed or not available on PATH.")


def _bootstrap_or_update_repo(localcolabfold_dir: Path, verbose: bool) -> None:
    if not localcolabfold_dir.exists():
        _run_command(
            ["git", "clone", LOCALCOLABFOLD_REPO_URL, str(localcolabfold_dir)],
            verbose=verbose,
        )
        return

    if not (localcolabfold_dir / ".git").is_dir():
        raise ColabFoldSetupError(
            f"Directory exists but is not a git checkout: {localcolabfold_dir}\n"
            "Choose a different directory with `--localcolabfold-dir`."
        )

    _run_command(
        ["git", "-C", str(localcolabfold_dir), "pull", "--ff-only"],
        verbose=verbose,
    )


def install_colabfold(
    localcolabfold_dir: Path | str | None = None,
    verbose: bool = False,
    progress_cb: Optional[ProgressCallback] = None,
) -> Path:
    """Install localcolabfold using the upstream-recommended pixi workflow."""
    resolved_localcolabfold_dir = resolve_localcolabfold_dir(localcolabfold_dir)

    _emit(progress_cb, "checking installer prerequisites")
    _ensure_prerequisites()

    _emit(progress_cb, "preparing localcolabfold repository")
    _bootstrap_or_update_repo(resolved_localcolabfold_dir, verbose=verbose)

    _emit(progress_cb, "running pixi install")
    _run_command(
        ["pixi", "install"],
        cwd=resolved_localcolabfold_dir,
        verbose=verbose,
    )

    _emit(progress_cb, "running pixi setup")
    _run_command(
        ["pixi", "run", "setup"],
        cwd=resolved_localcolabfold_dir,
        verbose=verbose,
    )

    _emit(progress_cb, "verifying colabfold runtime")
    _run_command(
        ["pixi", "run", "colabfold_batch", "--help"],
        cwd=resolved_localcolabfold_dir,
        verbose=verbose,
    )

    ensure_colabfold_ready(localcolabfold_dir=resolved_localcolabfold_dir)
    return resolved_localcolabfold_dir
