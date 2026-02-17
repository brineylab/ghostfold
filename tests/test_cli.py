import re

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
