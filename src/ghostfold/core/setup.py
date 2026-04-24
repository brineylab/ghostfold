"""GhostFold one-shot setup: bootstraps pixi, ColabFold env, AF2 weights, ProstT5."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from transformers import AutoModelForSeq2SeqLM, T5Tokenizer

_PROSTT5_MODEL = "Rostlab/ProstT5"

_PIXI_HOME = Path.home()
_PIXI_BIN = Path.home() / ".pixi" / "bin" / "pixi"
_PIXI_INSTALLER_URL = "https://pixi.prefix.dev/install.sh"

COLABFOLD_PIXI_TOML = """\
[project]
name = "colabfold"
version = "0.1.0"
channels = ["conda-forge", "bioconda"]
platforms = ["linux-64"]

[dependencies]
python = "3.10.*"
openmm = "8.2.0.*"
pdbfixer = "*"
kalign2 = "2.04.*"
hhsuite = "3.3.0.*"
mmseqs2 = "*"
git = "*"

[pypi-dependencies]
"colabfold[alphafold]" = { git = "https://github.com/sokrypton/ColabFold" }
"jax[cuda12]" = "==0.5.3"
tensorflow = "*"
silence_tensorflow = "*"
"""


class GhostFoldSetupError(RuntimeError):
    """Raised when a setup step cannot be completed."""


def ensure_pixi() -> str:
    """Return pixi binary path, bootstrapping if necessary."""
    path = shutil.which("pixi")
    if path:
        return path

    downloader = shutil.which("curl") or shutil.which("wget")
    if downloader is None:
        raise GhostFoldSetupError(
            "Cannot install pixi: neither curl nor wget found on PATH.\n"
            "Install pixi manually: https://pixi.prefix.dev"
        )

    with tempfile.NamedTemporaryFile(suffix=".sh", delete=False) as tmp:
        installer_path = tmp.name

    try:
        if shutil.which("curl"):
            subprocess.run(
                ["curl", "-fsSL", _PIXI_INSTALLER_URL, "-o", installer_path],
                check=True,
            )
        else:
            subprocess.run(
                ["wget", "-qO", installer_path, _PIXI_INSTALLER_URL],
                check=True,
            )

        subprocess.run(
            ["bash", installer_path, "--no-modify-path"],
            check=True,
            env={**os.environ, "PIXI_HOME": str(_PIXI_HOME)},
        )
    except subprocess.CalledProcessError as exc:
        raise GhostFoldSetupError(
            f"pixi installation failed (exit {exc.returncode}).\n"
            "Install manually: https://pixi.prefix.dev"
        ) from exc
    finally:
        Path(installer_path).unlink(missing_ok=True)

    bin_dir = str(_PIXI_BIN.parent)
    os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")

    path = shutil.which("pixi")
    if path is None:
        raise GhostFoldSetupError(
            f"pixi installed but not found on PATH. Add {bin_dir} to your ~/.bashrc:\n"
            f'  export PATH="{bin_dir}:$PATH"'
        )
    return path


def ensure_colabfold_env(colabfold_dir: Path) -> None:
    """Create pixi ColabFold env in colabfold_dir if not already valid."""
    colabfold_dir = Path(colabfold_dir)
    colabfold_dir.mkdir(parents=True, exist_ok=True)

    pixi_toml = colabfold_dir / "pixi.toml"

    if pixi_toml.exists():
        try:
            subprocess.run(
                ["pixi", "run", "colabfold_batch", "--help"],
                cwd=str(colabfold_dir),
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return
        except subprocess.CalledProcessError:
            pass

    pixi_toml.write_text(COLABFOLD_PIXI_TOML)

    try:
        subprocess.run(
            ["pixi", "install"],
            cwd=str(colabfold_dir),
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise GhostFoldSetupError(
            f"ColabFold environment creation failed (exit {exc.returncode}).\n"
            "Check: CUDA 12.x is available and you have ~10 GB free disk space."
        ) from exc


def ensure_af2_weights(colabfold_dir: Path) -> None:
    """Download AF2 weights into colabfold_dir if not already present."""
    colabfold_dir = Path(colabfold_dir)
    params_dir = colabfold_dir / "colabfold" / "params"

    if params_dir.exists() and any(params_dir.iterdir()):
        return

    try:
        subprocess.run(
            ["pixi", "run", "python", "-m", "colabfold.download"],
            cwd=str(colabfold_dir),
            check=True,
            env={**os.environ, "XDG_CACHE_HOME": str(colabfold_dir), "MPLBACKEND": "Agg"},
        )
    except subprocess.CalledProcessError as exc:
        raise GhostFoldSetupError(
            f"AF2 weight download failed (exit {exc.returncode}).\n"
            "Re-run `ghostfold setup` to resume (download is resumable)."
        ) from exc


def ensure_prostt5(hf_token: str | None = None) -> None:
    """Pre-download ProstT5 into HuggingFace cache (CPU only)."""
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
) -> None:
    """Run all setup steps in order. Each step is idempotent."""
    colabfold_dir = Path(colabfold_dir).resolve()

    ensure_pixi()
    ensure_colabfold_env(colabfold_dir)
    if not skip_weights:
        ensure_af2_weights(colabfold_dir)
    ensure_prostt5(hf_token=hf_token)
