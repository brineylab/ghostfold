from ghostfold.core.colabfold import _build_colabfold_cmd


def test_build_colabfold_cmd_defaults():
    """Command uses hardcoded defaults when no overrides provided."""
    cmd = _build_colabfold_cmd(
        msa_file="test.a3m",
        output_dir="/out",
        max_seq=32,
        max_extra_seq=64,
        launcher_prefix=["python", "-m"],
        extra_colabfold_params=None,
    )
    assert "--num-models" in cmd
    idx = cmd.index("--num-models")
    assert cmd[idx + 1] == "5"


def test_build_colabfold_cmd_overrides():
    """extra_colabfold_params overrides individual default params."""
    cmd = _build_colabfold_cmd(
        msa_file="test.a3m",
        output_dir="/out",
        max_seq=32,
        max_extra_seq=64,
        launcher_prefix=[],
        extra_colabfold_params={"--num-models": "1", "--num-seeds": "1"},
    )
    idx = cmd.index("--num-models")
    assert cmd[idx + 1] == "1"
    idx2 = cmd.index("--num-seeds")
    assert cmd[idx2 + 1] == "1"
    # --num-recycle should still be present from defaults
    assert "--num-recycle" in cmd
