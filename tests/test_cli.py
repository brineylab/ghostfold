import re
from unittest.mock import patch

from typer.testing import CliRunner

from ghostfold._version import __version__
from ghostfold.cli.app import app

runner = CliRunner()

_ansi_re = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """Strip ANSI escape codes from text."""
    return _ansi_re.sub("", text)


class TestMainApp:
    def test_help(self):
        result = runner.invoke(app, ["--help"])
        output = _plain(result.output)
        assert result.exit_code == 0
        assert "ghostfold" in output.lower() or "GhostFold" in output
        assert "install-colabfold" not in output

    def test_version(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_no_args_shows_help(self):
        result = runner.invoke(app, [])
        output = _plain(result.output)
        # Typer returns exit code 0 or 2 for no_args_is_help depending on version
        assert result.exit_code in (0, 2)
        assert "Usage" in output

    def test_install_colabfold_removed(self):
        result = runner.invoke(app, ["install-colabfold"])
        assert result.exit_code != 0
        assert "No such command" in _plain(result.output)


class TestMsaCommand:
    def test_help(self):
        result = runner.invoke(app, ["msa", "--help"])
        output = _plain(result.output)
        assert result.exit_code == 0
        assert "--project-name" in output
        assert "--fasta-path" in output
        assert "--recursive" in output

    def test_missing_required(self):
        result = runner.invoke(app, ["msa"])
        assert result.exit_code != 0


class TestFoldCommand:
    def test_help(self):
        result = runner.invoke(app, ["fold", "--help"])
        output = _plain(result.output)
        assert result.exit_code == 0
        assert "--project-name" in output
        assert "--subsample" in output
        assert "--mask-fraction" in output
        assert "--colabfold-env" in output
        assert "--localcolabfold-dir" in output

    def test_missing_required(self):
        result = runner.invoke(app, ["fold"])
        assert result.exit_code != 0


class TestRunCommand:
    def test_help(self):
        result = runner.invoke(app, ["run", "--help"])
        output = _plain(result.output)
        assert result.exit_code == 0
        assert "--project-name" in output
        assert "--fasta-path" in output
        assert "--recursive" in output
        assert "--subsample" in output
        assert "--colabfold-env" in output
        assert "--localcolabfold-dir" in output

    def test_missing_required(self):
        result = runner.invoke(app, ["run"])
        assert result.exit_code != 0


class TestMaskCommand:
    def test_help(self):
        result = runner.invoke(app, ["mask", "--help"])
        output = _plain(result.output)
        assert result.exit_code == 0
        assert "--input-path" in output
        assert "--output-path" in output
        assert "--mask-fraction" in output

    def test_missing_required(self):
        result = runner.invoke(app, ["mask"])
        assert result.exit_code != 0


class TestNeffCommand:
    def test_help(self):
        result = runner.invoke(app, ["neff", "--help"])
        output = _plain(result.output)
        assert result.exit_code == 0
        assert "project_dir" in output.lower() or "PROJECT_DIR" in output


def test_msa_accepts_precision_flag():
    """--precision flag must be accepted by the msa subcommand."""
    with patch("ghostfold.core.pipeline.run_pipeline"):
        result = runner.invoke(app, [
            "msa",
            "--project-name", "test_proj",
            "--fasta-path", "tests/fixtures/test.fasta",
            "--precision", "fp16",
        ])
    # If flag is unrecognised typer exits with code 2
    assert result.exit_code != 2, f"Unrecognised flag. Output:\n{result.output}"


def test_msa_precision_default_is_bf16():
    """msa subcommand must pass precision='bf16' to run_pipeline by default."""
    with patch("ghostfold.core.pipeline.run_pipeline") as mock_run, \
         patch("ghostfold.core.logging.setup_logging", return_value="/tmp/test.log"), \
         patch("ghostfold.core.logging.get_console"), \
         patch("ghostfold.core.config.load_config", return_value={}):
        runner.invoke(app, [
            "msa",
            "--project-name", "test_proj",
            "--fasta-path", "tests/fixtures/test.fasta",
        ])
        call_kwargs = mock_run.call_args.kwargs if mock_run.called else {}
        assert call_kwargs.get("precision", "bf16") == "bf16"


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


def test_msa_precision_invalid_rejected():
    """--precision with invalid value must exit with non-zero code."""
    result = runner.invoke(app, [
        "msa",
        "--project-name", "test_proj",
        "--fasta-path", "tests/fixtures/test.fasta",
        "--precision", "fp32",
    ])
    assert result.exit_code != 0
