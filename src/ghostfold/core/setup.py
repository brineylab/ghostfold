"""GhostFold one-shot setup: bootstraps pixi, ColabFold env, AF2 weights, ProstT5."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

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
