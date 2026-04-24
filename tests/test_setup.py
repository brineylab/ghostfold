import shutil
import subprocess

import pytest


class TestEnsurePixi:
    def test_returns_path_when_pixi_already_installed(self, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pixi" if cmd == "pixi" else None)
        from ghostfold.core.setup import ensure_pixi
        result = ensure_pixi()
        assert result == "/usr/bin/pixi"

    def test_bootstraps_pixi_when_missing(self, monkeypatch, tmp_path):
        pixi_bin = tmp_path / ".pixi" / "bin" / "pixi"
        pixi_bin.parent.mkdir(parents=True)
        pixi_bin.touch()

        which_calls = []
        def fake_which(cmd):
            which_calls.append(cmd)
            if cmd == "pixi" and len([c for c in which_calls if c == "pixi"]) == 1:
                return None
            if cmd == "pixi":
                return str(pixi_bin)
            if cmd in ("curl", "wget"):
                return f"/usr/bin/{cmd}"
            return None

        run_calls = []
        def fake_run(cmd, **kwargs):
            run_calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(shutil, "which", fake_which)
        monkeypatch.setattr("ghostfold.core.setup.subprocess.run", fake_run)
        monkeypatch.setattr("ghostfold.core.setup._PIXI_HOME", tmp_path)
        monkeypatch.setattr("ghostfold.core.setup._PIXI_BIN", pixi_bin)

        from ghostfold.core.setup import ensure_pixi
        result = ensure_pixi()
        assert result == str(pixi_bin)
        assert any("pixi" in str(c) for c in run_calls)

    def test_raises_setup_error_when_curl_and_wget_missing(self, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda cmd: None)
        from ghostfold.core.setup import ensure_pixi, GhostFoldSetupError
        with pytest.raises(GhostFoldSetupError, match="pixi"):
            ensure_pixi()


class TestEnsureColabfoldEnv:
    def test_skips_when_env_already_valid(self, monkeypatch, tmp_path):
        local_dir = tmp_path / "localcolabfold"
        local_dir.mkdir()
        (local_dir / "pixi.toml").write_text("[project]\nname='colabfold'\n")

        run_calls = []
        def fake_run(cmd, **kwargs):
            run_calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr("ghostfold.core.setup.subprocess.run", fake_run)
        from ghostfold.core.setup import ensure_colabfold_env
        ensure_colabfold_env(local_dir)

        assert any("colabfold_batch" in str(c) for c in run_calls)
        assert not any("install" in str(c) for c in run_calls)

    def test_creates_pixi_toml_and_installs_when_missing(self, monkeypatch, tmp_path):
        local_dir = tmp_path / "localcolabfold"
        local_dir.mkdir()

        run_calls = []
        def fake_run(cmd, **kwargs):
            run_calls.append(cmd)
            if "colabfold_batch" in str(cmd):
                raise subprocess.CalledProcessError(1, cmd)
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr("ghostfold.core.setup.subprocess.run", fake_run)
        from ghostfold.core.setup import ensure_colabfold_env
        ensure_colabfold_env(local_dir)

        assert (local_dir / "pixi.toml").exists()
        assert any("install" in str(c) for c in run_calls)

    def test_raises_on_pixi_install_failure(self, monkeypatch, tmp_path):
        local_dir = tmp_path / "localcolabfold"
        local_dir.mkdir()

        def fake_run(cmd, **kwargs):
            raise subprocess.CalledProcessError(1, cmd)

        monkeypatch.setattr("ghostfold.core.setup.subprocess.run", fake_run)
        from ghostfold.core.setup import ensure_colabfold_env, GhostFoldSetupError
        with pytest.raises(GhostFoldSetupError, match="ColabFold"):
            ensure_colabfold_env(local_dir)
