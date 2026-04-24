# Install Streamlining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `ghostfold setup` CLI command that bootstraps pixi, installs ColabFold, downloads AF2 weights, and pre-caches ProstT5 — reducing install to `pip install ghostfold && ghostfold setup`.

**Architecture:** Four idempotent `ensure_*` functions in `src/ghostfold/core/setup.py` handle each install step; `src/ghostfold/cli/setup.py` wraps them in a Typer subcommand with Rich progress output. All steps check state before acting; re-running after partial failure is safe.

**Tech Stack:** Python stdlib (shutil, subprocess, urllib, tempfile, os), Rich (progress/status), Typer, transformers (T5Tokenizer, AutoModelForSeq2SeqLM), pixi (bootstrapped by setup itself)

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/ghostfold/core/setup.py` | `ensure_pixi`, `ensure_colabfold_env`, `ensure_af2_weights`, `ensure_prostt5`, `run_setup` |
| Create | `src/ghostfold/cli/setup.py` | `ghostfold setup` Typer subcommand |
| Modify | `src/ghostfold/cli/app.py` | Register `setup` subcommand |
| Modify | `src/ghostfold/core/colabfold_env.py` | Append setup hint to `ColabFoldSetupError` messages |
| Create | `tests/test_setup.py` | Unit tests for all `ensure_*` functions |
| Modify | `README.md` | Replace three-step install with two-step install |

---

## Task 1: `ensure_pixi()` — pixi bootstrap

**Files:**
- Create: `src/ghostfold/core/setup.py`
- Create: `tests/test_setup.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_setup.py
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
            # First call (check): not found. Second call (after install): found.
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/nmishra/Desktop/ghostfold
pytest tests/test_setup.py::TestEnsurePixi -v
```

Expected: `ERROR` — `ModuleNotFoundError: No module named 'ghostfold.core.setup'`

- [ ] **Step 3: Implement `src/ghostfold/core/setup.py` with `ensure_pixi()`**

```python
"""GhostFold one-shot setup: bootstraps pixi, ColabFold env, AF2 weights, ProstT5."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional

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

    # Check for curl or wget
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

    pixi_path = str(_PIXI_BIN)
    # Add to PATH for current process
    bin_dir = str(_PIXI_BIN.parent)
    os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")

    path = shutil.which("pixi")
    if path is None:
        raise GhostFoldSetupError(
            f"pixi installed but not found on PATH. Add {bin_dir} to your ~/.bashrc:\n"
            f'  export PATH="{bin_dir}:$PATH"'
        )
    return path
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_setup.py::TestEnsurePixi::test_returns_path_when_pixi_already_installed -v
```

Expected: `PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/ghostfold/core/setup.py tests/test_setup.py
git commit -m "feat: add ensure_pixi() bootstrap in core/setup.py"
```

---

## Task 2: `ensure_colabfold_env()` — pixi ColabFold environment

**Files:**
- Modify: `src/ghostfold/core/setup.py`
- Modify: `tests/test_setup.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_setup.py`:

```python
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

        # Only the --help smoke test should be called, not pixi install
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_setup.py::TestEnsureColabfoldEnv -v
```

Expected: `FAILED` — `ImportError: cannot import name 'ensure_colabfold_env'`

- [ ] **Step 3: Implement `ensure_colabfold_env()` in `src/ghostfold/core/setup.py`**

Add after `ensure_pixi()`:

```python
def ensure_colabfold_env(colabfold_dir: Path) -> None:
    """Create pixi ColabFold env in colabfold_dir if not already valid."""
    colabfold_dir = Path(colabfold_dir)
    colabfold_dir.mkdir(parents=True, exist_ok=True)

    pixi_toml = colabfold_dir / "pixi.toml"

    # Check if already functional
    if pixi_toml.exists():
        try:
            subprocess.run(
                ["pixi", "run", "colabfold_batch", "--help"],
                cwd=str(colabfold_dir),
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return  # already valid
        except subprocess.CalledProcessError:
            pass  # fall through to install

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
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_setup.py::TestEnsureColabfoldEnv -v
```

Expected: all 3 `PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/ghostfold/core/setup.py tests/test_setup.py
git commit -m "feat: add ensure_colabfold_env() in core/setup.py"
```

---

## Task 3: `ensure_af2_weights()` — download AlphaFold2 weights

**Files:**
- Modify: `src/ghostfold/core/setup.py`
- Modify: `tests/test_setup.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_setup.py`:

```python
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
        params_dir.mkdir(parents=True)  # empty dir

        run_calls = []
        def fake_run(cmd, **kwargs):
            run_calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr("ghostfold.core.setup.subprocess.run", fake_run)
        from ghostfold.core.setup import ensure_af2_weights
        ensure_af2_weights(local_dir)

        assert any("colabfold.download" in str(c) for c in run_calls)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_setup.py::TestEnsureAf2Weights -v
```

Expected: `FAILED` — `ImportError: cannot import name 'ensure_af2_weights'`

- [ ] **Step 3: Implement `ensure_af2_weights()` in `src/ghostfold/core/setup.py`**

```python
def ensure_af2_weights(colabfold_dir: Path) -> None:
    """Download AF2 weights into colabfold_dir if not already present."""
    colabfold_dir = Path(colabfold_dir)
    params_dir = colabfold_dir / "colabfold" / "params"

    if params_dir.exists() and any(params_dir.iterdir()):
        return  # weights already present

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
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_setup.py::TestEnsureAf2Weights -v
```

Expected: all 3 `PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/ghostfold/core/setup.py tests/test_setup.py
git commit -m "feat: add ensure_af2_weights() in core/setup.py"
```

---

## Task 4: `ensure_prostt5()` — pre-cache ProstT5

**Files:**
- Modify: `src/ghostfold/core/setup.py`
- Modify: `tests/test_setup.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_setup.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_setup.py::TestEnsureProstt5 -v
```

Expected: `FAILED` — `ImportError: cannot import name 'ensure_prostt5'`

- [ ] **Step 3: Implement `ensure_prostt5()` in `src/ghostfold/core/setup.py`**

Add imports at top of file (after existing imports):

```python
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM
```

Add function:

```python
_PROSTT5_MODEL = "Rostlab/ProstT5"


def ensure_prostt5(hf_token: Optional[str] = None) -> None:
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
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_setup.py::TestEnsureProstt5 -v
```

Expected: all 3 `PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/ghostfold/core/setup.py tests/test_setup.py
git commit -m "feat: add ensure_prostt5() in core/setup.py"
```

---

## Task 5: `run_setup()` — top-level orchestrator

**Files:**
- Modify: `src/ghostfold/core/setup.py`
- Modify: `tests/test_setup.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_setup.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_setup.py::TestRunSetup -v
```

Expected: `FAILED` — `ImportError: cannot import name 'run_setup'`

- [ ] **Step 3: Implement `run_setup()` in `src/ghostfold/core/setup.py`**

```python
def run_setup(
    colabfold_dir: Path,
    skip_weights: bool = False,
    hf_token: Optional[str] = None,
) -> None:
    """Run all setup steps in order. Each step is idempotent."""
    colabfold_dir = Path(colabfold_dir).resolve()

    ensure_pixi()
    ensure_colabfold_env(colabfold_dir)
    if not skip_weights:
        ensure_af2_weights(colabfold_dir)
    ensure_prostt5(hf_token=hf_token)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_setup.py::TestRunSetup -v
```

Expected: both `PASSED`

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/test_setup.py -v
```

Expected: all tests `PASSED`

- [ ] **Step 6: Commit**

```bash
git add src/ghostfold/core/setup.py tests/test_setup.py
git commit -m "feat: add run_setup() orchestrator in core/setup.py"
```

---

## Task 6: `ghostfold setup` CLI subcommand

**Files:**
- Create: `src/ghostfold/cli/setup.py`
- Modify: `src/ghostfold/cli/app.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cli.py` (after existing classes):

```python
class TestSetupCommand:
    def test_help(self):
        result = runner.invoke(app, ["setup", "--help"])
        output = _plain(result.output)
        assert result.exit_code == 0
        assert "--colabfold-dir" in output
        assert "--skip-weights" in output
        assert "--hf-token" in output

    def test_setup_calls_run_setup(self, monkeypatch):
        calls = []

        def fake_run_setup(colabfold_dir, skip_weights, hf_token):
            calls.append({"colabfold_dir": colabfold_dir, "skip_weights": skip_weights, "hf_token": hf_token})

        monkeypatch.setattr("ghostfold.cli.setup.run_setup", fake_run_setup)
        result = runner.invoke(app, ["setup"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0]["skip_weights"] is False
        assert calls[0]["hf_token"] is None

    def test_setup_skip_weights_flag(self, monkeypatch):
        calls = []

        def fake_run_setup(colabfold_dir, skip_weights, hf_token):
            calls.append(skip_weights)

        monkeypatch.setattr("ghostfold.cli.setup.run_setup", fake_run_setup)
        result = runner.invoke(app, ["setup", "--skip-weights"])
        assert result.exit_code == 0
        assert calls[0] is True

    def test_setup_exits_nonzero_on_setup_error(self, monkeypatch):
        from ghostfold.core.setup import GhostFoldSetupError

        def fake_run_setup(colabfold_dir, skip_weights, hf_token):
            raise GhostFoldSetupError("pixi not found")

        monkeypatch.setattr("ghostfold.cli.setup.run_setup", fake_run_setup)
        result = runner.invoke(app, ["setup"])
        assert result.exit_code == 1
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_cli.py::TestSetupCommand -v
```

Expected: `FAILED` — no `setup` subcommand registered

- [ ] **Step 3: Create `src/ghostfold/cli/setup.py`**

```python
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="Install ColabFold, download AF2 weights, and cache ProstT5.")


@app.callback(invoke_without_command=True)
def setup(
    colabfold_dir: Path = typer.Option(
        Path("localcolabfold"),
        "--colabfold-dir",
        help="Directory to install ColabFold into (default: ./localcolabfold).",
    ),
    skip_weights: bool = typer.Option(
        False,
        "--skip-weights",
        help="Skip AlphaFold2 weight download.",
    ),
    hf_token: Optional[str] = typer.Option(
        None,
        "--hf-token",
        help="HuggingFace access token for ProstT5 (alternative to huggingface-cli login).",
        envvar="HF_TOKEN",
    ),
) -> None:
    """Bootstrap pixi, install ColabFold, download AF2 weights, and cache ProstT5."""
    from ghostfold.core.setup import run_setup, GhostFoldSetupError
    from ghostfold.core.logging import get_console

    console = get_console()

    steps = [
        "[1/4] Checking pixi...",
        "[2/4] Installing ColabFold...",
        "[3/4] Downloading AF2 weights...",
        "[4/4] Downloading ProstT5...",
    ]

    console.print(f"[bold]GhostFold Setup[/bold] — installing to [cyan]{colabfold_dir.resolve()}[/cyan]")
    if skip_weights:
        console.print("[dim]  --skip-weights: AF2 weight download skipped[/dim]")

    try:
        run_setup(
            colabfold_dir=colabfold_dir,
            skip_weights=skip_weights,
            hf_token=hf_token,
        )
    except GhostFoldSetupError as exc:
        typer.secho(f"\nSetup failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    console.print("\n[green]✅ Setup complete.[/green] Run: [bold]ghostfold run --help[/bold]")
    console.print(f"[dim]  Add ~/.pixi/bin to ~/.bashrc if pixi was freshly installed.[/dim]")
```

- [ ] **Step 4: Register in `src/ghostfold/cli/app.py`**

```python
# Add import
from ghostfold.cli import fold, mask, msa, neff, run, setup, subsample

# Add after existing app.add_typer calls
app.add_typer(setup.app, name="setup")
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_cli.py::TestSetupCommand -v
```

Expected: all 4 `PASSED`

- [ ] **Step 6: Smoke test CLI**

```bash
ghostfold setup --help
```

Expected: shows `--colabfold-dir`, `--skip-weights`, `--hf-token` options.

- [ ] **Step 7: Commit**

```bash
git add src/ghostfold/cli/setup.py src/ghostfold/cli/app.py tests/test_cli.py
git commit -m "feat: add ghostfold setup CLI subcommand"
```

---

## Task 7: Append setup hint to `ColabFoldSetupError`

**Files:**
- Modify: `src/ghostfold/core/colabfold_env.py`
- Modify: `tests/test_colabfold_env.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_colabfold_env.py`:

```python
def test_setup_error_includes_ghostfold_setup_hint(monkeypatch, tmp_path):
    """ColabFoldSetupError message should suggest running ghostfold setup."""
    local_dir = tmp_path / "no_colabfold"
    local_dir.mkdir()

    monkeypatch.setattr("ghostfold.core.colabfold_env.shutil.which", lambda cmd: None)

    with pytest.raises(ColabFoldSetupError, match="ghostfold setup"):
        ensure_colabfold_ready(localcolabfold_dir=local_dir)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_colabfold_env.py::test_setup_error_includes_ghostfold_setup_hint -v
```

Expected: `FAILED` — error message does not contain "ghostfold setup"

- [ ] **Step 3: Update `_format_setup_error` in `src/ghostfold/core/colabfold_env.py`**

Find the `_format_setup_error` function and add one line to `lines`:

```python
def _format_setup_error(
    reason: str,
    localcolabfold_dir: Path | str | None = None,
    include_pixi_hint: bool = True,
    include_mamba_hint: bool = True,
) -> ColabFoldSetupError:
    lines = [
        reason,
        (
            "Install/configure ColabFold by running "
            f"`{_install_command(localcolabfold_dir)}` from the GhostFold repository root."
        ),
        "Or run `ghostfold setup` to install everything automatically.",
        "This setup is required for `ghostfold run` and `ghostfold fold`.",
    ]
    if include_pixi_hint:
        lines.append(f"Pixi installation instructions: {PIXI_INSTALL_URL}")
    if include_mamba_hint:
        lines.append(f"Mamba/micromamba installation instructions: {MAMBA_INSTALL_URL}")
    return ColabFoldSetupError("\n".join(lines))
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_colabfold_env.py -v
```

Expected: all `PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/ghostfold/core/colabfold_env.py tests/test_colabfold_env.py
git commit -m "fix: add ghostfold setup hint to ColabFoldSetupError messages"
```

---

## Task 8: Update README install instructions

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace install section**

In `README.md`, replace the entire `## Installation` section (lines 18–64) with:

```markdown
## Installation

```bash
pip install ghostfold
ghostfold setup
```

`ghostfold setup` downloads and installs:
- **pixi** (environment manager, ~5 MB) — no root required
- **ColabFold + AlphaFold2 weights** (~3.5 GB) — into `./localcolabfold`
- **ProstT5** (~3 GB) — cached via HuggingFace

Total download: ~7 GB. Estimated time: 15–30 min on a fast connection.

> **PyTorch + CUDA:** GhostFold requires PyTorch with CUDA 12.x support. Install the appropriate version for your system **before** running `ghostfold setup`:
>
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu128
> ```
>
> See the [PyTorch installation guide](https://pytorch.org/get-started/locally/) for your specific CUDA version.

### Options

```bash
ghostfold setup --colabfold-dir /path/to/dir   # custom install location
ghostfold setup --skip-weights                  # skip AF2 weight download
ghostfold setup --hf-token YOUR_TOKEN          # HuggingFace token for ProstT5
```

### HuggingFace Authentication

If ProstT5 download fails with an authentication error, either pass `--hf-token` to `ghostfold setup` or run:

```bash
huggingface-cli login
```

<details>
<summary>Advanced: manual install with mamba/micromamba</summary>

If you prefer to manage the ColabFold environment yourself:

```bash
chmod +x scripts/install_localcolabfold.sh
./scripts/install_localcolabfold.sh
```

Requires mamba or micromamba on PATH.
</details>
```

- [ ] **Step 2: Verify README renders cleanly**

```bash
python -c "
import re
text = open('README.md').read()
assert 'ghostfold setup' in text
assert 'install_localcolabfold.sh' in text  # still present in advanced section
print('README OK')
"
```

Expected: `README OK`

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: update install instructions to use ghostfold setup"
```

---

## Task 9: Final integration check

- [ ] **Step 1: Run full test suite**

```bash
pytest --tb=short -q
```

Expected: all tests pass, no regressions

- [ ] **Step 2: Lint**

```bash
ruff check src tests
```

Expected: no errors

- [ ] **Step 3: Verify CLI help output**

```bash
ghostfold --help
ghostfold setup --help
```

Expected: `setup` appears in top-level help; `setup --help` shows all three flags.

- [ ] **Step 4: Commit if any lint fixes were needed**

```bash
git add -u
git commit -m "fix: ruff lint cleanup"
```
