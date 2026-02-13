import subprocess

import pytest

from ghostfold.core.colabfold_env import ColabFoldSetupError, ensure_colabfold_ready


def test_ensure_colabfold_ready_success(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd == ["mamba", "env", "list", "--json"]:
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout='{"envs": ["/opt/conda/envs/base", "/opt/conda/envs/colabfold"]}',
            )
        if cmd == ["mamba", "run", "-n", "colabfold", "colabfold_batch", "--help"]:
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr("ghostfold.core.colabfold_env.shutil.which", lambda _: "/usr/bin/mamba")
    monkeypatch.setattr("ghostfold.core.colabfold_env.subprocess.run", fake_run)

    ensure_colabfold_ready("colabfold")
    assert calls == [
        ["mamba", "env", "list", "--json"],
        ["mamba", "run", "-n", "colabfold", "colabfold_batch", "--help"],
    ]


def test_ensure_colabfold_ready_missing_mamba(monkeypatch):
    monkeypatch.setattr("ghostfold.core.colabfold_env.shutil.which", lambda _: None)

    with pytest.raises(ColabFoldSetupError) as exc:
        ensure_colabfold_ready()

    assert "mamba is not installed" in str(exc.value)
    assert "ghostfold install-colabfold" in str(exc.value)
    assert "mamba-installation.html" in str(exc.value)


def test_ensure_colabfold_ready_missing_env(monkeypatch):
    def fake_run(cmd, **kwargs):
        if cmd == ["mamba", "env", "list", "--json"]:
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout='{"envs": ["/opt/conda/envs/base"]}',
            )
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr("ghostfold.core.colabfold_env.shutil.which", lambda _: "/usr/bin/mamba")
    monkeypatch.setattr("ghostfold.core.colabfold_env.subprocess.run", fake_run)

    with pytest.raises(ColabFoldSetupError) as exc:
        ensure_colabfold_ready("custom-env")

    msg = str(exc.value)
    assert "custom-env" in msg
    assert "ghostfold install-colabfold --colabfold-env custom-env" in msg


def test_ensure_colabfold_ready_nonfunctional_colabfold(monkeypatch):
    def fake_run(cmd, **kwargs):
        if cmd == ["mamba", "env", "list", "--json"]:
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout='{"envs": ["/opt/conda/envs/colabfold"]}',
            )
        if cmd == ["mamba", "run", "-n", "colabfold", "colabfold_batch", "--help"]:
            raise subprocess.CalledProcessError(1, cmd)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr("ghostfold.core.colabfold_env.shutil.which", lambda _: "/usr/bin/mamba")
    monkeypatch.setattr("ghostfold.core.colabfold_env.subprocess.run", fake_run)

    with pytest.raises(ColabFoldSetupError) as exc:
        ensure_colabfold_ready("colabfold")

    assert "`colabfold_batch` is not functional" in str(exc.value)
