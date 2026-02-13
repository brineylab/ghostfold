import os
from pathlib import Path

import pytest

from ghostfold.core.colabfold_env import ColabFoldSetupError
from ghostfold.core.colabfold_install import install_colabfold


def test_install_colabfold_missing_mamba(monkeypatch, tmp_path):
    calls = {"run": 0}

    def fake_run_command(_cmd):
        calls["run"] += 1

    monkeypatch.setattr("ghostfold.core.colabfold_install.shutil.which", lambda _: None)
    monkeypatch.setattr("ghostfold.core.colabfold_install._run_command", fake_run_command)

    with pytest.raises(ColabFoldSetupError) as exc:
        install_colabfold(data_dir=tmp_path / "localcolabfold")

    assert "mamba is not installed" in str(exc.value)
    assert "mamba-installation.html" in str(exc.value)
    assert calls["run"] == 0


def test_install_colabfold_happy_path(monkeypatch, tmp_path):
    commands = []
    patch_calls = []
    ready_calls = []

    package_path = tmp_path / "fake_site" / "colabfold"
    package_path.mkdir(parents=True)

    def fake_run_command(cmd):
        commands.append(list(cmd))

    def fake_run_command_capture(cmd):
        assert cmd[:5] == ["mamba", "run", "-n", "colabfold", "python"]
        return str(package_path)

    def fake_patch(colabfold_env, colabfold_dir):
        patch_calls.append((colabfold_env, colabfold_dir))

    def fake_urlretrieve(_url, dst):
        Path(dst).write_text("#!/bin/bash\n")
        return str(dst), None

    def fake_ready(colabfold_env):
        ready_calls.append(colabfold_env)

    monkeypatch.setattr("ghostfold.core.colabfold_install.shutil.which", lambda _: "/usr/bin/mamba")
    monkeypatch.setattr("ghostfold.core.colabfold_install._run_command", fake_run_command)
    monkeypatch.setattr("ghostfold.core.colabfold_install._run_command_capture", fake_run_command_capture)
    monkeypatch.setattr("ghostfold.core.colabfold_install._patch_colabfold_sources", fake_patch)
    monkeypatch.setattr("ghostfold.core.colabfold_install.urlretrieve", fake_urlretrieve)
    monkeypatch.setattr("ghostfold.core.colabfold_install.ensure_colabfold_ready", fake_ready)

    data_dir = tmp_path / "custom_localcolabfold"
    resolved_data_dir = install_colabfold(colabfold_env="colabfold", data_dir=data_dir)

    assert resolved_data_dir == data_dir.resolve()
    assert commands == [
        ["mamba", "create", "-n", "colabfold", "-c", "conda-forge", "python=3.10", "-y"],
        [
            "mamba",
            "install",
            "-n",
            "colabfold",
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
        ],
        [
            "mamba",
            "run",
            "-n",
            "colabfold",
            "pip",
            "install",
            "--no-warn-conflicts",
            "colabfold[alphafold] @ git+https://github.com/sokrypton/ColabFold",
        ],
        [
            "mamba",
            "run",
            "-n",
            "colabfold",
            "pip",
            "install",
            "--upgrade",
            "jax[cuda12]==0.5.3",
            "tensorflow",
            "silence_tensorflow",
        ],
        ["mamba", "run", "-n", "colabfold", "python", "-m", "colabfold.download"],
    ]
    assert patch_calls == [("colabfold", data_dir.resolve())]
    assert ready_calls == ["colabfold"]
    assert (data_dir / "update_linux.sh").is_file()
    assert os.access(data_dir / "update_linux.sh", os.X_OK)
