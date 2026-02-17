from typer.testing import CliRunner

from ghostfold.cli.app import app

runner = CliRunner()


class TestMainApp:
    def test_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "ghostfold" in result.output.lower() or "GhostFold" in result.output
        assert "install-colabfold" not in result.output

    def test_version(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_no_args_shows_help(self):
        result = runner.invoke(app, [])
        # Typer returns exit code 0 or 2 for no_args_is_help depending on version
        assert result.exit_code in (0, 2)
        assert "Usage" in result.output

    def test_install_colabfold_removed(self):
        result = runner.invoke(app, ["install-colabfold"])
        assert result.exit_code != 0
        assert "No such command" in result.output


class TestMsaCommand:
    def test_help(self):
        result = runner.invoke(app, ["msa", "--help"])
        assert result.exit_code == 0
        assert "--project-name" in result.output
        assert "--fasta-path" in result.output
        assert "--recursive" in result.output

    def test_missing_required(self):
        result = runner.invoke(app, ["msa"])
        assert result.exit_code != 0


class TestFoldCommand:
    def test_help(self):
        result = runner.invoke(app, ["fold", "--help"])
        assert result.exit_code == 0
        assert "--project-name" in result.output
        assert "--subsample" in result.output
        assert "--mask-fraction" in result.output
        assert "--colabfold-env" in result.output
        assert "--localcolabfold-dir" in result.output

    def test_missing_required(self):
        result = runner.invoke(app, ["fold"])
        assert result.exit_code != 0


class TestRunCommand:
    def test_help(self):
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "--project-name" in result.output
        assert "--fasta-path" in result.output
        assert "--recursive" in result.output
        assert "--subsample" in result.output
        assert "--colabfold-env" in result.output
        assert "--localcolabfold-dir" in result.output

    def test_missing_required(self):
        result = runner.invoke(app, ["run"])
        assert result.exit_code != 0


class TestMaskCommand:
    def test_help(self):
        result = runner.invoke(app, ["mask", "--help"])
        assert result.exit_code == 0
        assert "--input-path" in result.output
        assert "--output-path" in result.output
        assert "--mask-fraction" in result.output

    def test_missing_required(self):
        result = runner.invoke(app, ["mask"])
        assert result.exit_code != 0


class TestNeffCommand:
    def test_help(self):
        result = runner.invoke(app, ["neff", "--help"])
        assert result.exit_code == 0
        assert "project_dir" in result.output.lower() or "PROJECT_DIR" in result.output
