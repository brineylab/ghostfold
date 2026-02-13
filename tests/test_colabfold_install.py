import pytest

from ghostfold.core.colabfold_env import ColabFoldSetupError
from ghostfold.core.colabfold_install import install_colabfold


def test_install_colabfold_missing_pixi(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "ghostfold.core.colabfold_install.shutil.which",
        lambda cmd: None if cmd == "pixi" else "/usr/bin/git",
    )

    with pytest.raises(ColabFoldSetupError) as exc:
        install_colabfold(localcolabfold_dir=tmp_path / "localcolabfold")

    msg = str(exc.value)
    assert "pixi is not installed" in msg
    assert "pixi.prefix.dev" in msg


def test_install_colabfold_happy_path_clone_quiet(monkeypatch, tmp_path):
    commands = []
    steps = []

    def fake_which(_cmd):
        return "/usr/bin/mock"

    def fake_run_command(cmd, *, cwd=None, verbose=False):
        commands.append((list(cmd), cwd, verbose))

    monkeypatch.setattr("ghostfold.core.colabfold_install.shutil.which", fake_which)
    monkeypatch.setattr("ghostfold.core.colabfold_install._run_command", fake_run_command)
    monkeypatch.setattr(
        "ghostfold.core.colabfold_install.ensure_colabfold_ready",
        lambda **_kwargs: None,
    )

    local_dir = tmp_path / "localcolabfold"
    resolved = install_colabfold(
        localcolabfold_dir=local_dir,
        verbose=False,
        progress_cb=steps.append,
    )

    assert resolved == local_dir.resolve()
    assert commands == [
        (
            [
                "git",
                "clone",
                "https://github.com/YoshitakaMo/localcolabfold.git",
                str(local_dir.resolve()),
            ],
            None,
            False,
        ),
        (["pixi", "install"], local_dir.resolve(), False),
        (["pixi", "run", "setup"], local_dir.resolve(), False),
        (["pixi", "run", "colabfold_batch", "--help"], local_dir.resolve(), False),
    ]
    assert steps == [
        "checking installer prerequisites",
        "preparing localcolabfold repository",
        "running pixi install",
        "running pixi setup",
        "verifying colabfold runtime",
    ]


def test_install_colabfold_existing_repo_pull_verbose(monkeypatch, tmp_path):
    commands = []
    local_dir = tmp_path / "localcolabfold"
    local_dir.mkdir()
    (local_dir / ".git").mkdir()

    monkeypatch.setattr("ghostfold.core.colabfold_install.shutil.which", lambda _cmd: "/usr/bin/mock")

    def fake_run_command(cmd, *, cwd=None, verbose=False):
        commands.append((list(cmd), cwd, verbose))

    monkeypatch.setattr("ghostfold.core.colabfold_install._run_command", fake_run_command)
    monkeypatch.setattr(
        "ghostfold.core.colabfold_install.ensure_colabfold_ready",
        lambda **_kwargs: None,
    )

    install_colabfold(localcolabfold_dir=local_dir, verbose=True)

    assert commands[0] == (
        ["git", "-C", str(local_dir.resolve()), "pull", "--ff-only"],
        None,
        True,
    )
    assert commands[1:] == [
        (["pixi", "install"], local_dir.resolve(), True),
        (["pixi", "run", "setup"], local_dir.resolve(), True),
        (["pixi", "run", "colabfold_batch", "--help"], local_dir.resolve(), True),
    ]


def test_install_colabfold_existing_non_git_dir_fails(monkeypatch, tmp_path):
    local_dir = tmp_path / "localcolabfold"
    local_dir.mkdir()
    (local_dir / "random.txt").write_text("x")

    monkeypatch.setattr("ghostfold.core.colabfold_install.shutil.which", lambda _cmd: "/usr/bin/mock")

    with pytest.raises(ColabFoldSetupError) as exc:
        install_colabfold(localcolabfold_dir=local_dir)

    assert "is not a git checkout" in str(exc.value)
