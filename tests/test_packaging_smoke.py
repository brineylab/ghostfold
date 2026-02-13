from __future__ import annotations

import os
import subprocess
import sys
import venv
import zipfile
from pathlib import Path

import pytest


RUN_PACKAGING_SMOKE = os.environ.get("GHOSTFOLD_RUN_PACKAGING_SMOKE") == "1"
pytestmark = pytest.mark.skipif(
    not RUN_PACKAGING_SMOKE,
    reason="Set GHOSTFOLD_RUN_PACKAGING_SMOKE=1 to run packaging smoke tests.",
)


def _build_artifacts(repo_root: Path, out_dir: Path) -> tuple[Path, Path]:
    try:
        subprocess.run(
            [sys.executable, "-m", "build", "--sdist", "--wheel", "--outdir", str(out_dir)],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        details = f"{exc.stdout}\n{exc.stderr}"
        offline_markers = [
            "No matching distribution found for hatchling",
            "Failed to establish a new connection",
            "Temporary failure in name resolution",
            "nodename nor servname provided",
        ]
        if any(marker in details for marker in offline_markers):
            pytest.skip("Build backend resolution failed in an offline or restricted network environment.")
        raise

    wheels = sorted(out_dir.glob("*.whl"))
    sdists = sorted(out_dir.glob("*.tar.gz"))
    assert len(wheels) == 1
    assert len(sdists) == 1
    return wheels[0], sdists[0]


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


@pytest.mark.packaging
def test_build_generates_sdist_and_wheel(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    wheel, sdist = _build_artifacts(repo_root, tmp_path / "dist")
    assert wheel.is_file()
    assert sdist.is_file()


@pytest.mark.packaging
def test_wheel_metadata_exposes_required_extras(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    wheel, _ = _build_artifacts(repo_root, tmp_path / "dist")

    with zipfile.ZipFile(wheel, "r") as handle:
        metadata_name = next(name for name in handle.namelist() if name.endswith(".dist-info/METADATA"))
        metadata_text = handle.read(metadata_name).decode("utf-8")

    assert "Provides-Extra: test" in metadata_text
    assert "Provides-Extra: fold" in metadata_text
    assert "Provides-Extra: fold-cuda12" in metadata_text
    assert "Requires-Dist: pytest" in metadata_text
    assert "Requires-Dist: colabfold[alphafold]" in metadata_text


@pytest.mark.packaging
def test_wheel_installs_in_clean_venv_and_cli_api_load(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    wheel, _ = _build_artifacts(repo_root, tmp_path / "dist")

    venv_dir = tmp_path / "venv"
    venv.EnvBuilder(with_pip=True, system_site_packages=True).create(venv_dir)
    python_bin = _venv_python(venv_dir)

    subprocess.run([str(python_bin), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run([str(python_bin), "-m", "pip", "install", str(wheel)], check=True)

    subprocess.run([str(python_bin), "-m", "ghostfold.cli", "version"], check=True)
    subprocess.run(
        [
            str(python_bin),
            "-c",
            "import ghostfold as g; assert hasattr(g, 'run_pipeline_workflow'); print(g.__version__)",
        ],
        check=True,
    )
