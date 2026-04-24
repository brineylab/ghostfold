"""GhostFold one-shot setup: installs ColabFold env, AF2 weights, and ProstT5."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from transformers import AutoModelForSeq2SeqLM, T5Tokenizer

_PROSTT5_MODEL = "Rostlab/ProstT5"
_COLABFOLD_ENV = "colabfold"
_COLABFOLD_URL = "git+https://github.com/sokrypton/ColabFold"
_JAX_CUDA_SPEC = "jax[cuda12]==0.5.3"
_MICROMAMBA_INSTALLER_URL = "https://micro.mamba.pm/install.sh"
_MICROMAMBA_CANDIDATE_DIRS = [
    Path.home() / "micromamba" / "bin",
    Path.home() / ".local" / "bin",
    Path.home() / "bin",
]


class GhostFoldSetupError(RuntimeError):
    """Raised when a setup step cannot be completed."""


def ensure_mamba() -> tuple[str, bool]:
    """Return (runner name, freshly_installed). Bootstraps micromamba if necessary."""
    for candidate in ("mamba", "micromamba"):
        if shutil.which(candidate):
            return candidate, False

    downloader = shutil.which("curl") or shutil.which("wget")
    if downloader is None:
        raise GhostFoldSetupError(
            "Cannot install micromamba: neither curl nor wget found on PATH.\n"
            "Install micromamba manually: https://mamba.readthedocs.io/en/stable/installation/mamba-installation.html"
        )

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".sh", delete=False) as tmp:
        installer_path = tmp.name

    try:
        if shutil.which("curl"):
            subprocess.run(
                ["curl", "-L", _MICROMAMBA_INSTALLER_URL, "-o", installer_path],
                check=True,
                stdin=subprocess.DEVNULL,
            )
        else:
            subprocess.run(
                ["wget", "-qO", installer_path, _MICROMAMBA_INSTALLER_URL],
                check=True,
                stdin=subprocess.DEVNULL,
            )

        # -b = batch (non-interactive), -p = prefix dir
        prefix = Path.home() / "micromamba"
        subprocess.run(
            ["bash", installer_path, "-b", "-p", str(prefix)],
            check=True,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as exc:
        raise GhostFoldSetupError(
            f"micromamba installation failed (exit {exc.returncode}).\n"
            "Install manually: https://mamba.readthedocs.io/en/stable/installation/mamba-installation.html"
        ) from exc
    finally:
        Path(installer_path).unlink(missing_ok=True)

    extra = os.pathsep.join(str(d) for d in _MICROMAMBA_CANDIDATE_DIRS)
    os.environ["PATH"] = extra + os.pathsep + os.environ.get("PATH", "")

    runner = next((c for c in ("mamba", "micromamba") if shutil.which(c)), None)
    if runner is None:
        raise GhostFoldSetupError(
            "micromamba was installed but could not be located on PATH. "
            "Open a new terminal and re-run `ghostfold setup`."
        )
    return runner, True


def _find_mamba() -> str:
    runner, _ = ensure_mamba()
    return runner


def _run(cmd: list[str], **kwargs) -> None:
    try:
        subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL, **kwargs)
    except subprocess.CalledProcessError as exc:
        raise GhostFoldSetupError(
            f"Command failed (exit {exc.returncode}): {' '.join(cmd)}"
        ) from exc


def ensure_colabfold_env(colabfold_dir: Path, force: bool = False) -> None:
    """Create mamba ColabFold env if not already valid. Mirrors install_localcolabfold.sh."""
    runner = _find_mamba()

    if force:
        # Remove existing env so all packages are reinstalled clean
        subprocess.run(
            [runner, "env", "remove", "-n", _COLABFOLD_ENV, "-y"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
    else:
        # Skip if env already has a working colabfold_batch
        try:
            subprocess.run(
                [runner, "run", "-n", _COLABFOLD_ENV, "colabfold_batch", "--help"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
            )
            return
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    # Step 1: create base env with conda deps
    _run([
        runner, "create", "-n", _COLABFOLD_ENV,
        "-c", "conda-forge", "-c", "bioconda",
        "python=3.10", "openmm=8.2.0", "pdbfixer",
        "kalign2=2.04", "hhsuite=3.3.0", "mmseqs2",
        "-y",
    ])

    # Step 2: install colabfold (pulls CPU jax as transitive dep)
    _run([
        runner, "run", "-n", _COLABFOLD_ENV,
        "pip", "install", "--no-warn-conflicts",
        f"colabfold[alphafold] @ {_COLABFOLD_URL}",
    ])

    # Step 3: upgrade to CUDA jax — must come after colabfold install
    _run([
        runner, "run", "-n", _COLABFOLD_ENV,
        "pip", "install", "--upgrade",
        _JAX_CUDA_SPEC, "tensorflow", "silence_tensorflow",
    ])


def ensure_af2_weights(colabfold_dir: Path) -> None:
    """Download AF2 weights into colabfold_dir if not already present."""
    colabfold_dir = Path(colabfold_dir)
    params_dir = colabfold_dir / "colabfold" / "params"

    if params_dir.exists() and any(params_dir.iterdir()):
        return

    runner = _find_mamba()
    _run(
        [runner, "run", "-n", _COLABFOLD_ENV, "python", "-m", "colabfold.download"],
        env={**os.environ, "XDG_CACHE_HOME": str(colabfold_dir), "MPLBACKEND": "Agg"},
    )


def ensure_prostt5(hf_token: str | None = None) -> None:
    """Pre-download ProstT5 into HuggingFace cache."""
    try:
        T5Tokenizer.from_pretrained(
            _PROSTT5_MODEL,
            do_lower_case=False,
            legacy=True,
            token=hf_token,
        )
        AutoModelForSeq2SeqLM.from_pretrained(
            _PROSTT5_MODEL,
            device_map="cpu",
            token=hf_token,
        )
    except Exception as exc:
        raise GhostFoldSetupError(
            f"ProstT5 download failed: {exc}\n"
            "Run `huggingface-cli login` or pass `--hf-token TOKEN` to `ghostfold setup`."
        ) from exc


def run_setup(
    colabfold_dir: Path,
    skip_weights: bool = False,
    hf_token: str | None = None,
    force: bool = False,
) -> bool:
    """Run all setup steps in order. Each step is idempotent. Returns True if micromamba was freshly installed."""
    colabfold_dir = Path(colabfold_dir).resolve()

    _, mamba_fresh = ensure_mamba()
    ensure_colabfold_env(colabfold_dir, force=force)
    if not skip_weights:
        ensure_af2_weights(colabfold_dir)
    ensure_prostt5(hf_token=hf_token)
    return mamba_fresh
