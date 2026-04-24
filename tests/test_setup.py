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


class TestEnsureAf2Weights:
    def test_skips_when_params_dir_populated(self, monkeypatch, tmp_path):
        local_dir = tmp_path / "localcolabfold"
        params_dir = local_dir / "colabfold" / "params"
        params_dir.mkdir(parents=True)
        (params_dir / "params_model_1.npz").touch()

        run_calls = []
        monkeypatch.setattr("ghostfold.core.setup.subprocess.run", lambda cmd, **kw: run_calls.append(cmd) or subprocess.CompletedProcess(cmd, 0))

        from ghostfold.core.setup import ensure_af2_weights
        ensure_af2_weights(local_dir)
        assert run_calls == []

    def test_downloads_when_params_dir_missing(self, monkeypatch, tmp_path):
        local_dir = tmp_path / "localcolabfold"
        local_dir.mkdir()

        run_calls = []
        def fake_run(cmd, **kwargs):
            run_calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr("ghostfold.core.setup.subprocess.run", fake_run)
        from ghostfold.core.setup import ensure_af2_weights
        ensure_af2_weights(local_dir)

        assert any("colabfold.download" in str(c) for c in run_calls)

    def test_downloads_when_params_dir_empty(self, monkeypatch, tmp_path):
        local_dir = tmp_path / "localcolabfold"
        params_dir = local_dir / "colabfold" / "params"
        params_dir.mkdir(parents=True)

        run_calls = []
        def fake_run(cmd, **kwargs):
            run_calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr("ghostfold.core.setup.subprocess.run", fake_run)
        from ghostfold.core.setup import ensure_af2_weights
        ensure_af2_weights(local_dir)

        assert any("colabfold.download" in str(c) for c in run_calls)


class TestEnsureProstt5:
    def test_calls_from_pretrained_with_no_token(self, monkeypatch):
        tokenizer_calls = []
        model_calls = []

        class FakeTokenizer:
            @classmethod
            def from_pretrained(cls, name, **kwargs):
                tokenizer_calls.append((name, kwargs))
                return cls()

        class FakeModel:
            @classmethod
            def from_pretrained(cls, name, **kwargs):
                model_calls.append((name, kwargs))
                return cls()

        monkeypatch.setattr("ghostfold.core.setup.T5Tokenizer", FakeTokenizer)
        monkeypatch.setattr("ghostfold.core.setup.AutoModelForSeq2SeqLM", FakeModel)

        from ghostfold.core.setup import ensure_prostt5
        ensure_prostt5(hf_token=None)

        assert tokenizer_calls[0][0] == "Rostlab/ProstT5"
        assert model_calls[0][1].get("device_map") == "cpu"

    def test_passes_hf_token_when_provided(self, monkeypatch):
        model_calls = []

        class FakeTokenizer:
            @classmethod
            def from_pretrained(cls, name, **kwargs):
                return cls()

        class FakeModel:
            @classmethod
            def from_pretrained(cls, name, **kwargs):
                model_calls.append(kwargs)
                return cls()

        monkeypatch.setattr("ghostfold.core.setup.T5Tokenizer", FakeTokenizer)
        monkeypatch.setattr("ghostfold.core.setup.AutoModelForSeq2SeqLM", FakeModel)

        from ghostfold.core.setup import ensure_prostt5
        ensure_prostt5(hf_token="my-token")

        assert model_calls[0].get("token") == "my-token"

    def test_raises_setup_error_on_hf_failure(self, monkeypatch):
        class FakeTokenizer:
            @classmethod
            def from_pretrained(cls, name, **kwargs):
                raise OSError("401 Unauthorized")

        monkeypatch.setattr("ghostfold.core.setup.T5Tokenizer", FakeTokenizer)

        from ghostfold.core.setup import ensure_prostt5, GhostFoldSetupError
        with pytest.raises(GhostFoldSetupError, match="ProstT5"):
            ensure_prostt5(hf_token=None)


class TestRunSetup:
    def test_calls_all_steps_in_order(self, monkeypatch, tmp_path):
        call_order = []

        monkeypatch.setattr("ghostfold.core.setup.ensure_pixi", lambda: call_order.append("pixi") or "/usr/bin/pixi")
        monkeypatch.setattr("ghostfold.core.setup.ensure_colabfold_env", lambda d: call_order.append("colabfold"))
        monkeypatch.setattr("ghostfold.core.setup.ensure_af2_weights", lambda d: call_order.append("weights"))
        monkeypatch.setattr("ghostfold.core.setup.ensure_prostt5", lambda hf_token=None: call_order.append("prostt5"))

        from ghostfold.core.setup import run_setup
        run_setup(colabfold_dir=tmp_path / "localcolabfold", skip_weights=False, hf_token=None)

        assert call_order == ["pixi", "colabfold", "weights", "prostt5"]

    def test_skips_weights_when_flag_set(self, monkeypatch, tmp_path):
        call_order = []

        monkeypatch.setattr("ghostfold.core.setup.ensure_pixi", lambda: call_order.append("pixi") or "/usr/bin/pixi")
        monkeypatch.setattr("ghostfold.core.setup.ensure_colabfold_env", lambda d: call_order.append("colabfold"))
        monkeypatch.setattr("ghostfold.core.setup.ensure_af2_weights", lambda d: call_order.append("weights"))
        monkeypatch.setattr("ghostfold.core.setup.ensure_prostt5", lambda hf_token=None: call_order.append("prostt5"))

        from ghostfold.core.setup import run_setup
        run_setup(colabfold_dir=tmp_path / "localcolabfold", skip_weights=True, hf_token=None)

        assert "weights" not in call_order
        assert "pixi" in call_order
        assert "prostt5" in call_order
