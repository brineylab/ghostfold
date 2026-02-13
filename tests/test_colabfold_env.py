import subprocess

import pytest

from ghostfold.core.colabfold_env import ColabFoldSetupError, ensure_colabfold_ready


def test_ensure_colabfold_ready_prefers_pixi(monkeypatch, tmp_path):
    local_dir = tmp_path / "localcolabfold"
    local_dir.mkdir()
    (local_dir / "pixi.toml").write_text("[project]\nname='x'\n")
    calls = []

    def fake_which(cmd):
        if cmd in {"pixi", "mamba"}:
            return f"/usr/bin/{cmd}"
        return None

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs.get("cwd")))
        if cmd == ["pixi", "run", "colabfold_batch", "--help"]:
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr("ghostfold.core.colabfold_env.shutil.which", fake_which)
    monkeypatch.setattr("ghostfold.core.colabfold_env.subprocess.run", fake_run)

    launcher = ensure_colabfold_ready(localcolabfold_dir=local_dir)
    assert launcher.mode == "pixi"
    assert launcher.command_prefix == ("pixi", "run")
    assert launcher.cwd == local_dir.resolve()
    assert calls == [(["pixi", "run", "colabfold_batch", "--help"], str(local_dir.resolve()))]


def test_ensure_colabfold_ready_falls_back_to_mamba(monkeypatch, tmp_path):
    local_dir = tmp_path / "no_localcolabfold"
    local_dir.mkdir()
    calls = []

    def fake_which(cmd):
        if cmd in {"pixi", "mamba"}:
            return f"/usr/bin/{cmd}"
        return None

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs.get("cwd")))
        if cmd == ["mamba", "env", "list", "--json"]:
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout='{"envs": ["/opt/conda/envs/base", "/opt/conda/envs/colabfold"]}',
            )
        if cmd == ["mamba", "run", "-n", "colabfold", "colabfold_batch", "--help"]:
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr("ghostfold.core.colabfold_env.shutil.which", fake_which)
    monkeypatch.setattr("ghostfold.core.colabfold_env.subprocess.run", fake_run)

    launcher = ensure_colabfold_ready(colabfold_env="colabfold", localcolabfold_dir=local_dir)
    assert launcher.mode == "mamba"
    assert launcher.command_prefix == ("mamba", "run", "-n", "colabfold", "--no-capture-output")
    assert launcher.cwd is None
    assert calls == [
        (["mamba", "env", "list", "--json"], None),
        (["mamba", "run", "-n", "colabfold", "colabfold_batch", "--help"], None),
    ]


def test_ensure_colabfold_ready_broken_pixi_uses_mamba(monkeypatch, tmp_path):
    local_dir = tmp_path / "localcolabfold"
    local_dir.mkdir()
    (local_dir / "pixi.toml").write_text("[project]\nname='x'\n")

    def fake_which(cmd):
        if cmd in {"pixi", "mamba"}:
            return f"/usr/bin/{cmd}"
        return None

    def fake_run(cmd, **kwargs):
        if cmd == ["pixi", "run", "colabfold_batch", "--help"]:
            raise subprocess.CalledProcessError(1, cmd)
        if cmd == ["mamba", "env", "list", "--json"]:
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout='{"envs": ["/opt/conda/envs/colabfold"]}',
            )
        if cmd == ["mamba", "run", "-n", "colabfold", "colabfold_batch", "--help"]:
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr("ghostfold.core.colabfold_env.shutil.which", fake_which)
    monkeypatch.setattr("ghostfold.core.colabfold_env.subprocess.run", fake_run)

    launcher = ensure_colabfold_ready(localcolabfold_dir=local_dir)
    assert launcher.mode == "mamba"


def test_ensure_colabfold_ready_total_failure(monkeypatch):
    monkeypatch.setattr("ghostfold.core.colabfold_env.shutil.which", lambda _cmd: None)

    with pytest.raises(ColabFoldSetupError) as exc:
        ensure_colabfold_ready()

    msg = str(exc.value)
    assert "bash scripts/install_localcolabfold.sh" in msg
    assert "required for `ghostfold run` and `ghostfold fold`" in msg
    assert "Pixi installation instructions" in msg
    assert "Mamba installation instructions" in msg
