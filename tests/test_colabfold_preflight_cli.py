from typer.testing import CliRunner

from ghostfold.cli.app import app
from ghostfold.core.colabfold_env import ColabFoldSetupError

runner = CliRunner()


def test_run_fails_preflight_before_msa(monkeypatch, tmp_path):
    called = {"msa": False, "fold": False}
    fasta_file = tmp_path / "query.fasta"
    fasta_file.write_text(">q\nACDE\n")

    def fail_preflight(*_args, **_kwargs):
        raise ColabFoldSetupError(
            "ColabFold is not functional.\nInstall/configure ColabFold by running `bash scripts/install_localcolabfold.sh` from the GhostFold repository root."
        )

    def fake_run_parallel_msa(**_kwargs):
        called["msa"] = True

    def fake_run_colabfold(**_kwargs):
        called["fold"] = True

    monkeypatch.setattr("ghostfold.core.gpu.detect_gpus", lambda: 1)
    monkeypatch.setattr("ghostfold.core.colabfold_env.ensure_colabfold_ready", fail_preflight)
    monkeypatch.setattr("ghostfold.core.gpu.run_parallel_msa", fake_run_parallel_msa)
    monkeypatch.setattr("ghostfold.core.colabfold.run_colabfold", fake_run_colabfold)

    result = runner.invoke(
        app,
        [
            "run",
            "--project-name",
            "my_project",
            "--fasta-file",
            str(fasta_file),
        ],
    )

    assert result.exit_code == 1
    assert "bash scripts/install_localcolabfold.sh" in result.output
    assert called["msa"] is False
    assert called["fold"] is False


def test_fold_fails_preflight_before_colabfold_dispatch(monkeypatch):
    called = {"fold": False}

    def fail_preflight(*_args, **_kwargs):
        raise ColabFoldSetupError(
            "ColabFold is not functional.\nInstall/configure ColabFold by running `bash scripts/install_localcolabfold.sh` from the GhostFold repository root."
        )

    def fake_run_colabfold(**_kwargs):
        called["fold"] = True

    monkeypatch.setattr("ghostfold.core.gpu.detect_gpus", lambda: 1)
    monkeypatch.setattr("ghostfold.core.colabfold_env.ensure_colabfold_ready", fail_preflight)
    monkeypatch.setattr("ghostfold.core.colabfold.run_colabfold", fake_run_colabfold)

    result = runner.invoke(
        app,
        [
            "fold",
            "--project-name",
            "my_project",
        ],
    )

    assert result.exit_code == 1
    assert "bash scripts/install_localcolabfold.sh" in result.output
    assert called["fold"] is False
