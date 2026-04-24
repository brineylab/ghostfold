import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

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

        from ghostfold.core import setup as setup_mod
        import importlib
        importlib.reload(setup_mod)

        result = setup_mod.ensure_pixi()
        assert result == str(pixi_bin)
        assert any("pixi" in str(c) for c in run_calls)

    def test_raises_setup_error_when_curl_and_wget_missing(self, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda cmd: None)
        from ghostfold.core.setup import ensure_pixi, GhostFoldSetupError
        with pytest.raises(GhostFoldSetupError, match="pixi"):
            ensure_pixi()
