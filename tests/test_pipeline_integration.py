from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from ghostfold.config import PipelineWorkflowConfig
from ghostfold.errors import GhostfoldExecutionError, GhostfoldValidationError
import ghostfold.services.pipeline as pipeline


def test_detect_num_gpus_parses_nvidia_smi_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline.shutil, "which", lambda command: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        pipeline.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="0\n1\n"),
    )

    assert pipeline._detect_num_gpus() == 2


def test_detect_num_gpus_requires_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline.shutil, "which", lambda command: None)
    with pytest.raises(GhostfoldValidationError, match="nvidia-smi command not found"):
        pipeline._detect_num_gpus()


def test_detect_num_gpus_reports_subprocess_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline.shutil, "which", lambda command: "/usr/bin/nvidia-smi")

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ["nvidia-smi"], stderr="boom")

    monkeypatch.setattr(pipeline.subprocess, "run", fail)

    with pytest.raises(GhostfoldExecutionError, match="Failed to query GPUs with nvidia-smi: boom"):
        pipeline._detect_num_gpus()


def test_run_colabfold_masks_cleans_and_zips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_dir = tmp_path / "project"
    msa_a = project_dir / "msa" / "A"
    msa_b = project_dir / "msa" / "B"
    msa_a.mkdir(parents=True)
    msa_b.mkdir(parents=True)
    (msa_a / "pstMSA.a3m").write_text(">A\nAAAA\n")
    (msa_b / "pstMSA.a3m").write_text(">B\nBBBB\n")

    monkeypatch.setattr(
        pipeline.shutil,
        "which",
        lambda command: "/usr/bin/mamba" if command == "mamba" else None,
    )

    masked_outputs: list[Path] = []

    def fake_run_mask_workflow(config):
        config.output_path.write_text(config.input_path.read_text())
        masked_outputs.append(config.output_path)
        return SimpleNamespace(success=True)

    def fake_run_colabfold_jobs(a3m_files, preds_dir, num_gpus, max_seq, max_extra_seq):
        for msa_file in a3m_files:
            pred_dir = preds_dir / msa_file.parent.name
            pred_dir.mkdir(parents=True, exist_ok=True)
            (pred_dir / "scores.json").write_text("{}")
            (pred_dir / "plot.png").write_bytes(b"png")
            (pred_dir / "model.r1.pdb").write_text("recycle")
            (pred_dir / "model_rank_001_unrelaxed.pdb").write_text("best")
            (pred_dir / "job.done.txt").write_text("done")

    monkeypatch.setattr(pipeline, "run_mask_workflow", fake_run_mask_workflow)
    monkeypatch.setattr(pipeline, "_run_colabfold_jobs", fake_run_colabfold_jobs)

    zip_outputs = pipeline._run_colabfold(
        project_dir=project_dir,
        num_gpus=2,
        subsample=True,
        mask_msa="0.5",
    )

    assert len(zip_outputs) == 4
    for archive_path in zip_outputs:
        assert archive_path.is_file()

    for temp_file in masked_outputs:
        assert not temp_file.exists()

    subsample_dir = project_dir / "subsample_1"
    assert (subsample_dir / "preds" / "A" / "scores" / "scores.json").is_file()
    assert (subsample_dir / "preds" / "A" / "imgs" / "plot.png").is_file()
    assert (subsample_dir / "preds" / "A" / "recycles" / "model.r1.pdb").is_file()
    assert (subsample_dir / "best" / "A_ghostfold.pdb").is_file()
    assert not (subsample_dir / "preds" / "A" / "job.done.txt").exists()


def test_run_colabfold_jobs_dispatches_round_robin_gpu_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    msa_files = []
    for index in range(3):
        parent = tmp_path / f"seq{index}"
        parent.mkdir(parents=True, exist_ok=True)
        msa_file = parent / "pstMSA.a3m"
        msa_file.write_text(">q\nAAAA\n")
        msa_files.append(msa_file)

    calls = []

    class FakeProcess:
        def poll(self):
            return None

    def fake_popen(command, cwd=None, env=None):
        calls.append((command, cwd, env))
        return FakeProcess()

    def fake_reap(active, block):
        active.clear()

    monkeypatch.setattr(pipeline.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(pipeline, "_reap_processes", fake_reap)

    preds_dir = tmp_path / "preds"
    pipeline._run_colabfold_jobs(
        a3m_files=msa_files,
        preds_dir=preds_dir,
        num_gpus=2,
        max_seq=32,
        max_extra_seq=64,
    )

    assert len(calls) == 3
    gpu_ids = [call[2]["CUDA_VISIBLE_DEVICES"] for call in calls]
    assert gpu_ids == ["0", "1", "0"]
    assert all(call[0][:6] == ["mamba", "run", "-n", "colabfold", "--no-capture-output", "colabfold_batch"] for call in calls)


def test_run_pipeline_workflow_full_mode_calls_msa_and_fold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fasta_file = tmp_path / "query.fasta"
    fasta_file.write_text(">q\nAAAA\n")
    call_order: list[str] = []

    def fake_detect_num_gpus():
        return 2

    def fake_run_parallel_msa(project_dir, fasta_file, num_gpus):
        call_order.append("msa")
        return 1

    def fake_run_colabfold(project_dir, num_gpus, subsample, mask_msa):
        call_order.append("fold")
        return (project_dir / "subsample_1.zip",)

    monkeypatch.setattr(pipeline, "_detect_num_gpus", fake_detect_num_gpus)
    monkeypatch.setattr(pipeline, "_run_parallel_msa", fake_run_parallel_msa)
    monkeypatch.setattr(pipeline, "_run_colabfold", fake_run_colabfold)

    result = pipeline.run_pipeline_workflow(
        PipelineWorkflowConfig(
            project_name=str(tmp_path / "project"),
            fasta_file=fasta_file,
        )
    )

    assert result.success is True
    assert result.mode == "full"
    assert call_order == ["msa", "fold"]
    assert result.num_gpus == 2
    assert len(result.zip_outputs) == 1
    assert result.warnings == tuple()


def test_run_pipeline_workflow_requires_gpu_detection_even_for_msa_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fasta_file = tmp_path / "query.fasta"
    fasta_file.write_text(">q\nAAAA\n")
    called = {"msa": False}

    def fail_gpu_detection():
        raise GhostfoldValidationError("nvidia-smi command not found. Cannot detect GPUs.")

    def fake_run_parallel_msa(project_dir, fasta_file, num_gpus):
        called["msa"] = True
        return 1

    monkeypatch.setattr(pipeline, "_detect_num_gpus", fail_gpu_detection)
    monkeypatch.setattr(pipeline, "_run_parallel_msa", fake_run_parallel_msa)

    with pytest.raises(GhostfoldValidationError, match="nvidia-smi command not found"):
        pipeline.run_pipeline_workflow(
            PipelineWorkflowConfig(
                project_name=str(tmp_path / "project"),
                fasta_file=fasta_file,
                msa_only=True,
            )
        )

    assert called["msa"] is False


def test_run_colabfold_no_a3m_warns_and_returns_success_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project_dir = tmp_path / "project"
    (project_dir / "msa").mkdir(parents=True)
    monkeypatch.setattr(
        pipeline.shutil,
        "which",
        lambda command: "/usr/bin/mamba" if command == "mamba" else None,
    )

    zip_outputs = pipeline._run_colabfold(
        project_dir=project_dir,
        num_gpus=1,
        subsample=False,
        mask_msa=None,
    )

    captured = capsys.readouterr()
    assert zip_outputs == tuple()
    assert "No 'pstMSA.a3m' files found" in captured.err


def test_run_pipeline_workflow_preserves_warning_success_on_missing_msa_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fasta_file = tmp_path / "query.fasta"
    fasta_file.write_text(">q\nAAAA\n")

    monkeypatch.setattr(pipeline, "_detect_num_gpus", lambda: 1)
    monkeypatch.setattr(pipeline, "_run_parallel_msa", lambda *args, **kwargs: 0)
    monkeypatch.setattr(pipeline, "_run_colabfold", lambda *args, **kwargs: tuple())

    result = pipeline.run_pipeline_workflow(
        PipelineWorkflowConfig(
            project_name=str(tmp_path / "project"),
            fasta_file=fasta_file,
        )
    )

    assert result.success is True
    assert result.mode == "full"
    assert len(result.zip_outputs) == 0
    assert len(result.warnings) == 2
    assert "No pstMSA outputs were generated" in result.warnings[0]
    assert "No pstMSA.a3m inputs were found" in result.warnings[1]
