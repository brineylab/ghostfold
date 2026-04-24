import shutil
import subprocess

import pytest


class TestEnsureMamba:
    def test_returns_mamba_when_available(self, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/mamba" if cmd == "mamba" else None)
        from ghostfold.core.setup import ensure_mamba
        runner, fresh = ensure_mamba()
        assert runner == "mamba"
        assert fresh is False

    def test_falls_back_to_micromamba(self, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/micromamba" if cmd == "micromamba" else None)
        from ghostfold.core.setup import ensure_mamba
        runner, fresh = ensure_mamba()
        assert runner == "micromamba"
        assert fresh is False

    def test_bootstraps_micromamba_when_missing(self, monkeypatch, tmp_path):
        micromamba_bin = tmp_path / "micromamba" / "bin" / "micromamba"
        micromamba_bin.parent.mkdir(parents=True)
        micromamba_bin.touch()

        installed = {"done": False}
        def fake_which(cmd):
            if cmd == "curl":
                return "/usr/bin/curl"
            if cmd == "micromamba" and installed["done"]:
                return str(micromamba_bin)
            return None

        run_calls = []
        def fake_run(cmd, **kwargs):
            run_calls.append(list(cmd))
            installed["done"] = True
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(shutil, "which", fake_which)
        monkeypatch.setattr("ghostfold.core.setup.subprocess.run", fake_run)
        monkeypatch.setattr("ghostfold.core.setup._MICROMAMBA_CANDIDATE_DIRS", [micromamba_bin.parent])

        from ghostfold.core.setup import ensure_mamba
        runner, fresh = ensure_mamba()
        assert fresh is True
        assert any("install.sh" in str(c) or "micro.mamba" in str(c) for c in run_calls)

    def test_raises_when_no_downloader(self, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda cmd: None)
        from ghostfold.core.setup import ensure_mamba, GhostFoldSetupError
        with pytest.raises(GhostFoldSetupError, match="curl"):
            ensure_mamba()

    def test_raises_when_install_fails(self, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/curl" if cmd == "curl" else None)

        def fake_run(cmd, **kwargs):
            raise subprocess.CalledProcessError(1, cmd)

        monkeypatch.setattr("ghostfold.core.setup.subprocess.run", fake_run)
        from ghostfold.core.setup import ensure_mamba, GhostFoldSetupError
        with pytest.raises(GhostFoldSetupError, match="micromamba"):
            ensure_mamba()


class TestEnsureColabfoldEnv:
    def test_skips_when_env_already_valid(self, monkeypatch, tmp_path):
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/mamba" if cmd == "mamba" else None)

        run_calls = []
        def fake_run(cmd, **kwargs):
            run_calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr("ghostfold.core.setup.subprocess.run", fake_run)
        from ghostfold.core.setup import ensure_colabfold_env
        ensure_colabfold_env(tmp_path / "localcolabfold")

        assert any("colabfold_batch" in str(c) for c in run_calls)
        # no install calls when env already valid
        assert not any("create" in str(c) for c in run_calls)

    def test_creates_env_and_installs_when_missing(self, monkeypatch, tmp_path):
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/mamba" if cmd == "mamba" else None)

        run_calls = []
        def fake_run(cmd, **kwargs):
            run_calls.append(list(cmd))
            if "colabfold_batch" in str(cmd):
                raise subprocess.CalledProcessError(1, cmd)
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr("ghostfold.core.setup.subprocess.run", fake_run)
        from ghostfold.core.setup import ensure_colabfold_env
        ensure_colabfold_env(tmp_path / "localcolabfold")

        flat = [c for cmd in run_calls for c in cmd]
        assert "create" in flat
        assert "colabfold[alphafold]" in " ".join(flat) or any("colabfold" in c for c in flat)
        # jax cuda upgrade must come after colabfold install
        create_idx = next(i for i, cmd in enumerate(run_calls) if "create" in cmd)
        colabfold_idx = next(i for i, cmd in enumerate(run_calls) if any("colabfold" in c and "alphafold" in c for c in cmd))
        jax_idx = next(i for i, cmd in enumerate(run_calls) if any("jax" in c for c in cmd))
        assert create_idx < colabfold_idx < jax_idx

    def test_jax_install_uses_upgrade_flag(self, monkeypatch, tmp_path):
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/mamba" if cmd == "mamba" else None)

        run_calls = []
        def fake_run(cmd, **kwargs):
            run_calls.append(list(cmd))
            if "colabfold_batch" in str(cmd):
                raise subprocess.CalledProcessError(1, cmd)
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr("ghostfold.core.setup.subprocess.run", fake_run)
        from ghostfold.core.setup import ensure_colabfold_env
        ensure_colabfold_env(tmp_path / "localcolabfold")

        jax_cmd = next(cmd for cmd in run_calls if any("jax" in c for c in cmd))
        assert "--upgrade" in jax_cmd

    def test_jax_install_includes_cuda12(self, monkeypatch, tmp_path):
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/mamba" if cmd == "mamba" else None)

        run_calls = []
        def fake_run(cmd, **kwargs):
            run_calls.append(list(cmd))
            if "colabfold_batch" in str(cmd):
                raise subprocess.CalledProcessError(1, cmd)
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr("ghostfold.core.setup.subprocess.run", fake_run)
        from ghostfold.core.setup import ensure_colabfold_env
        ensure_colabfold_env(tmp_path / "localcolabfold")

        jax_cmd = next(cmd for cmd in run_calls if any("jax" in c for c in cmd))
        assert any("cuda12" in c for c in jax_cmd)

    def test_raises_on_install_failure(self, monkeypatch, tmp_path):
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/mamba" if cmd == "mamba" else None)

        def fake_run(cmd, **kwargs):
            raise subprocess.CalledProcessError(1, cmd)

        monkeypatch.setattr("ghostfold.core.setup.subprocess.run", fake_run)
        from ghostfold.core.setup import ensure_colabfold_env, GhostFoldSetupError
        with pytest.raises(GhostFoldSetupError):
            ensure_colabfold_env(tmp_path / "localcolabfold")

    def test_raises_when_no_mamba(self, monkeypatch, tmp_path):
        monkeypatch.setattr(shutil, "which", lambda cmd: None)
        from ghostfold.core.setup import ensure_colabfold_env, GhostFoldSetupError
        with pytest.raises(GhostFoldSetupError, match="micromamba"):
            ensure_colabfold_env(tmp_path / "localcolabfold")


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

        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/mamba" if cmd == "mamba" else None)

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

        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/mamba" if cmd == "mamba" else None)

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

        monkeypatch.setattr("ghostfold.core.setup.ensure_mamba", lambda: (call_order.append("mamba") or ("micromamba", False)))
        monkeypatch.setattr("ghostfold.core.setup.ensure_colabfold_env", lambda d: call_order.append("colabfold"))
        monkeypatch.setattr("ghostfold.core.setup.ensure_af2_weights", lambda d: call_order.append("weights"))
        monkeypatch.setattr("ghostfold.core.setup.ensure_prostt5", lambda hf_token=None: call_order.append("prostt5"))

        from ghostfold.core.setup import run_setup
        result = run_setup(colabfold_dir=tmp_path / "localcolabfold", skip_weights=False, hf_token=None)

        assert call_order == ["mamba", "colabfold", "weights", "prostt5"]
        assert result is False

    def test_skips_weights_when_flag_set(self, monkeypatch, tmp_path):
        call_order = []

        monkeypatch.setattr("ghostfold.core.setup.ensure_mamba", lambda: (call_order.append("mamba") or ("micromamba", False)))
        monkeypatch.setattr("ghostfold.core.setup.ensure_colabfold_env", lambda d: call_order.append("colabfold"))
        monkeypatch.setattr("ghostfold.core.setup.ensure_af2_weights", lambda d: call_order.append("weights"))
        monkeypatch.setattr("ghostfold.core.setup.ensure_prostt5", lambda hf_token=None: call_order.append("prostt5"))

        from ghostfold.core.setup import run_setup
        run_setup(colabfold_dir=tmp_path / "localcolabfold", skip_weights=True, hf_token=None)

        assert "weights" not in call_order
        assert call_order == ["mamba", "colabfold", "prostt5"]

    def test_returns_true_when_mamba_freshly_installed(self, monkeypatch, tmp_path):
        monkeypatch.setattr("ghostfold.core.setup.ensure_mamba", lambda: ("micromamba", True))
        monkeypatch.setattr("ghostfold.core.setup.ensure_colabfold_env", lambda d: None)
        monkeypatch.setattr("ghostfold.core.setup.ensure_af2_weights", lambda d: None)
        monkeypatch.setattr("ghostfold.core.setup.ensure_prostt5", lambda hf_token=None: None)

        from ghostfold.core.setup import run_setup
        assert run_setup(colabfold_dir=tmp_path, skip_weights=True) is True
