from __future__ import annotations

import os
import subprocess
from pathlib import Path

from typer.testing import CliRunner

from ghostfold.cli import app


def test_mask_alias_commands_are_available_and_work(tmp_path: Path) -> None:
    runner = CliRunner()
    input_path = tmp_path / "input.a3m"
    input_path.write_text(">query\nAAAA\n>seq1\nBBBB\n")

    out_alias = tmp_path / "alias.a3m"
    out_primary = tmp_path / "primary.a3m"

    alias_result = runner.invoke(
        app,
        [
            "mask_msa",
            "--input_path",
            str(input_path),
            "--output_path",
            str(out_alias),
            "--mask_fraction",
            "0",
        ],
    )
    primary_result = runner.invoke(
        app,
        [
            "mask",
            "--input_path",
            str(input_path),
            "--output_path",
            str(out_primary),
            "--mask_fraction",
            "0",
        ],
    )

    assert alias_result.exit_code == 0
    assert primary_result.exit_code == 0
    assert out_alias.read_text() == input_path.read_text()
    assert out_primary.read_text() == input_path.read_text()


def test_calculate_neff_alias_runs_with_project_argument(tmp_path: Path) -> None:
    runner = CliRunner()
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    result = runner.invoke(app, ["calculate_neff", str(project_dir)])

    assert result.exit_code == 0


def test_ghostfold_sh_option_only_invocation_routes_to_run(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture_file = tmp_path / "captured_args.txt"
    fake_python = fake_bin / "python3"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' \"$@\" > \"${CAPTURE_PATH}\"\n"
    )
    fake_python.chmod(0o755)

    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    env["CAPTURE_PATH"] = str(capture_file)

    subprocess.run(
        [str(repo_root / "ghostfold.sh"), "--project_name", "demo", "--fold-only"],
        cwd=repo_root,
        env=env,
        check=True,
    )

    captured = capture_file.read_text().splitlines()
    assert captured[:3] == ["-m", "ghostfold.cli", "run"]
    assert "--project_name" in captured
    assert "--fold-only" in captured


def test_ghostfold_sh_explicit_subcommand_passthrough(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture_file = tmp_path / "captured_args.txt"
    fake_python = fake_bin / "python3"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' \"$@\" > \"${CAPTURE_PATH}\"\n"
    )
    fake_python.chmod(0o755)

    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    env["CAPTURE_PATH"] = str(capture_file)

    subprocess.run(
        [str(repo_root / "ghostfold.sh"), "mask_msa", "--help"],
        cwd=repo_root,
        env=env,
        check=True,
    )

    captured = capture_file.read_text().splitlines()
    assert captured[:4] == ["-m", "ghostfold.cli", "mask_msa", "--help"]
