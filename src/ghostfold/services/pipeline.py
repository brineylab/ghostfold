"""Native Python orchestration for full/msa/fold GhostFold pipeline modes."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from Bio import SeqIO

from ghostfold.config import MaskWorkflowConfig, PipelineWorkflowConfig
from ghostfold.errors import GhostfoldExecutionError, GhostfoldValidationError
from ghostfold.results import PipelineWorkflowResult
from ghostfold.services.masking import run_mask_workflow

REPO_ROOT = Path(__file__).resolve().parents[3]
PSEUDOMSA_SCRIPT = REPO_ROOT / "pseudomsa.py"
MASK_MSA_PATTERN = re.compile(r"^(0\.[0-9]+|1\.0|0)$")
RECYCLE_PDB_PATTERN = re.compile(r".*\.r\d{1,2}\.pdb$")


def _emit_warning(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


def _validate_pipeline_config(config: PipelineWorkflowConfig) -> None:
    if not config.project_name:
        raise GhostfoldValidationError("--project_name is a required argument.")
    if config.msa_only and config.fold_only:
        raise GhostfoldValidationError("--msa-only and --fold-only cannot be used together.")
    if not config.fold_only and config.fasta_file is None:
        raise GhostfoldValidationError("--fasta_file is required unless in --fold-only mode.")
    if not config.fold_only and config.fasta_file is not None and not Path(config.fasta_file).is_file():
        raise GhostfoldValidationError(f"FASTA file not found at '{config.fasta_file}'")
    if config.mask_msa is not None and not MASK_MSA_PATTERN.fullmatch(config.mask_msa):
        raise GhostfoldValidationError(
            "--mask_msa requires a fraction between 0.0 and 1.0 (e.g., 0.15)."
        )


def _detect_num_gpus() -> int:
    if shutil.which("nvidia-smi") is None:
        raise GhostfoldValidationError("nvidia-smi command not found. Cannot detect GPUs.")

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=True,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or "").strip()
        if details:
            raise GhostfoldExecutionError(f"Failed to query GPUs with nvidia-smi: {details}") from exc
        raise GhostfoldExecutionError("Failed to query GPUs with nvidia-smi.") from exc

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise GhostfoldValidationError("No GPUs detected by nvidia-smi.")
    return len(lines)


def _reap_processes(active: List[subprocess.Popen], *, block: bool) -> None:
    while True:
        finished_any = False
        for proc in list(active):
            return_code = proc.poll()
            if return_code is None:
                continue
            active.remove(proc)
            finished_any = True
            if return_code != 0:
                for remaining in active:
                    if remaining.poll() is None:
                        remaining.terminate()
                raise GhostfoldExecutionError(
                    f"A subprocess exited with status code {return_code}."
                )

        if not block or finished_any:
            return
        time.sleep(0.1)


def _run_pseudomsa_job(project_name: str, fasta_file: Path, gpu_id: int) -> None:
    if not PSEUDOMSA_SCRIPT.is_file():
        raise GhostfoldExecutionError(f"pseudomsa.py not found at '{PSEUDOMSA_SCRIPT}'.")

    command = [
        sys.executable,
        str(PSEUDOMSA_SCRIPT),
        "--project_name",
        project_name,
        "--fasta_file",
        str(fasta_file),
        "--config",
        str(REPO_ROOT / "config.yaml"),
        "--evolve_msa",
        "--no_coevolution_maps",
        "--num_runs",
        "1",
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        subprocess.run(command, check=True, cwd=REPO_ROOT, env=env)
    except subprocess.CalledProcessError as exc:
        raise GhostfoldExecutionError(
            f"pseudoMSA job failed for '{fasta_file}' with status code {exc.returncode}."
        ) from exc


def _run_parallel_msa(project_dir: Path, fasta_file: Path, num_gpus: int) -> int:
    project_dir.mkdir(parents=True, exist_ok=True)
    records = list(SeqIO.parse(str(fasta_file), "fasta"))
    num_sequences = len(records)

    if num_sequences == 1:
        _run_pseudomsa_job(str(project_dir), fasta_file, 0)
    elif num_sequences > 1:
        with tempfile.TemporaryDirectory(prefix="pseudoMSA_splits_") as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            split_files: List[Path] = []
            for index, record in enumerate(records, start=1):
                split_file = temp_dir / f"split_seq_{index:04d}.fasta"
                with split_file.open("w") as handle:
                    SeqIO.write([record], handle, "fasta")
                split_files.append(split_file)

            max_jobs = min(num_gpus, num_sequences)
            active: List[subprocess.Popen] = []
            gpu_counter = 0

            for split_file in split_files:
                while len(active) >= max_jobs:
                    _reap_processes(active, block=True)

                command = [
                    sys.executable,
                    str(PSEUDOMSA_SCRIPT),
                    "--project_name",
                    str(project_dir),
                    "--fasta_file",
                    str(split_file),
                    "--config",
                    str(REPO_ROOT / "config.yaml"),
                    "--evolve_msa",
                    "--no_coevolution_maps",
                    "--num_runs",
                    "1",
                ]
                env = dict(os.environ)
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_counter % max_jobs)
                active.append(subprocess.Popen(command, cwd=REPO_ROOT, env=env))
                gpu_counter += 1

            while active:
                _reap_processes(active, block=True)

    generated_outputs = 0
    for fasta_path in sorted(project_dir.rglob("pstMSA.fasta")):
        header_id = fasta_path.parent.name
        lines = fasta_path.read_text().splitlines()
        if lines:
            lines[0] = f">{header_id}"
            fasta_path.write_text("\n".join(lines) + "\n")
        a3m_path = fasta_path.with_suffix(".a3m")
        shutil.copy2(fasta_path, a3m_path)
        generated_outputs += 1

    if generated_outputs == 0:
        _emit_warning(
            f"No 'pstMSA.fasta' outputs were detected under '{project_dir}'. "
            "This can occur when pseudoMSA exits early (including OOM scenarios)."
        )

    return generated_outputs


def _cleanup_colabfold_outputs(subsample_dir: Path) -> None:
    preds_dir = subsample_dir / "preds"
    best_dir = subsample_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)

    if not preds_dir.is_dir():
        return

    for pred_dir in sorted(path for path in preds_dir.iterdir() if path.is_dir()):
        scores_dir = pred_dir / "scores"
        imgs_dir = pred_dir / "imgs"
        recycles_dir = pred_dir / "recycles"
        scores_dir.mkdir(exist_ok=True)
        imgs_dir.mkdir(exist_ok=True)
        recycles_dir.mkdir(exist_ok=True)

        for json_file in pred_dir.glob("*.json"):
            json_file.rename(scores_dir / json_file.name)
        for image_file in pred_dir.glob("*.png"):
            image_file.rename(imgs_dir / image_file.name)

        for pdb_file in list(pred_dir.glob("*.pdb")):
            if RECYCLE_PDB_PATTERN.fullmatch(pdb_file.name):
                pdb_file.rename(recycles_dir / pdb_file.name)

        rank_candidates = sorted(
            pdb_file
            for pdb_file in pred_dir.glob("*rank_001*.pdb")
            if not RECYCLE_PDB_PATTERN.fullmatch(pdb_file.name)
        )
        if rank_candidates:
            best_dest = best_dir / f"{pred_dir.name}_ghostfold.pdb"
            shutil.copy2(rank_candidates[0], best_dest)

        for done_file in pred_dir.glob("*done.txt"):
            done_file.unlink(missing_ok=True)


def _run_colabfold_jobs(
    a3m_files: Sequence[Path],
    preds_dir: Path,
    num_gpus: int,
    max_seq: int,
    max_extra_seq: int,
) -> None:
    preds_dir.mkdir(parents=True, exist_ok=True)
    active: List[subprocess.Popen] = []
    gpu_counter = 0

    colabfold_params = [
        "--num-recycle",
        "10",
        "--num-models",
        "5",
        "--rank",
        "ptm",
        "--recycle-early-stop-tolerance",
        "0.5",
        "--use-dropout",
        "--max-seq",
        str(max_seq),
        "--max-extra-seq",
        str(max_extra_seq),
        "--num-seeds",
        "5",
        "--save-recycles",
        "--model-type",
        "auto",
    ]

    for msa_file in a3m_files:
        while len(active) >= num_gpus:
            _reap_processes(active, block=True)

        output_folder_name = msa_file.parent.name
        current_pred_dir = preds_dir / output_folder_name
        current_pred_dir.mkdir(parents=True, exist_ok=True)

        command = [
            "mamba",
            "run",
            "-n",
            "colabfold",
            "--no-capture-output",
            "colabfold_batch",
            str(msa_file),
            str(current_pred_dir),
            *colabfold_params,
        ]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_counter % num_gpus)
        active.append(subprocess.Popen(command, cwd=REPO_ROOT, env=env))
        gpu_counter += 1

    while active:
        _reap_processes(active, block=True)


def _run_colabfold(
    project_dir: Path,
    num_gpus: int,
    subsample: bool,
    mask_msa: Optional[str],
) -> Tuple[Path, ...]:
    if shutil.which("mamba") is None:
        raise GhostfoldValidationError("mamba command not found.")

    msa_root_dir = project_dir / "msa"
    if not msa_root_dir.is_dir():
        raise GhostfoldValidationError(f"MSA directory not found at '{msa_root_dir}'")

    original_a3m_files = sorted(msa_root_dir.rglob("pstMSA.a3m"))
    if not original_a3m_files:
        _emit_warning(f"No 'pstMSA.a3m' files found in '{msa_root_dir}'. Skipping fold stage.")
        return tuple()

    a3m_files_to_process: List[Path] = list(original_a3m_files)
    temp_masked_files: List[Path] = []

    try:
        if mask_msa is not None and mask_msa != "0":
            mask_fraction = float(mask_msa)
            a3m_files_to_process = []
            for msa_file in original_a3m_files:
                temp_masked_file = msa_file.with_name(f"{msa_file.stem}_masked_temp.a3m")
                run_mask_workflow(
                    MaskWorkflowConfig(
                        input_path=msa_file,
                        output_path=temp_masked_file,
                        mask_fraction=mask_fraction,
                    )
                )
                a3m_files_to_process.append(temp_masked_file)
                temp_masked_files.append(temp_masked_file)

        if subsample:
            subsample_levels = [(16, 32), (32, 64), (64, 128), (128, 256)]
        else:
            subsample_levels = [(32, 64)]

        zip_outputs: List[Path] = []
        for level_index, (max_seq, max_extra_seq) in enumerate(subsample_levels, start=1):
            subsample_dir = project_dir / f"subsample_{level_index}"
            preds_dir = subsample_dir / "preds"

            _run_colabfold_jobs(
                a3m_files=a3m_files_to_process,
                preds_dir=preds_dir,
                num_gpus=num_gpus,
                max_seq=max_seq,
                max_extra_seq=max_extra_seq,
            )

            _cleanup_colabfold_outputs(subsample_dir)

            archive_path = shutil.make_archive(
                base_name=str(subsample_dir),
                format="zip",
                root_dir=str(project_dir),
                base_dir=subsample_dir.name,
            )
            zip_outputs.append(Path(archive_path))

        return tuple(zip_outputs)
    finally:
        for temp_file in temp_masked_files:
            temp_file.unlink(missing_ok=True)


def run_pipeline_workflow(config: PipelineWorkflowConfig) -> PipelineWorkflowResult:
    """Runs full/msa/fold project orchestration in native Python."""
    _validate_pipeline_config(config)
    num_gpus = _detect_num_gpus()

    project_dir = Path(config.project_name)

    warnings: List[str] = []

    try:
        if config.msa_only:
            generated_outputs = _run_parallel_msa(
                project_dir=project_dir,
                fasta_file=Path(config.fasta_file),
                num_gpus=num_gpus,
            )
            message = f"MSA generation finished for project '{config.project_name}'."
            if generated_outputs == 0:
                warnings.append(
                    "No pstMSA outputs were generated; downstream fold mode would have no A3M inputs."
                )
            return PipelineWorkflowResult(
                success=True,
                message=message,
                project_dir=project_dir,
                mode="msa-only",
                num_gpus=num_gpus,
                zip_outputs=tuple(),
                warnings=tuple(warnings),
            )

        if config.fold_only:
            zip_outputs = _run_colabfold(
                project_dir=project_dir,
                num_gpus=num_gpus,
                subsample=config.subsample,
                mask_msa=config.mask_msa,
            )
            message = f"ColabFold prediction finished for project '{config.project_name}'."
            if not zip_outputs:
                warnings.append("No pstMSA.a3m inputs were found, so no fold jobs were dispatched.")
            return PipelineWorkflowResult(
                success=True,
                message=message,
                project_dir=project_dir,
                mode="fold-only",
                num_gpus=num_gpus,
                zip_outputs=zip_outputs,
                warnings=tuple(warnings),
            )

        generated_outputs = _run_parallel_msa(
            project_dir=project_dir,
            fasta_file=Path(config.fasta_file),
            num_gpus=num_gpus,
        )
        if generated_outputs == 0:
            warnings.append(
                "No pstMSA outputs were generated; fold stage may skip due to missing A3M inputs."
            )

        zip_outputs = _run_colabfold(
            project_dir=project_dir,
            num_gpus=num_gpus,
            subsample=config.subsample,
            mask_msa=config.mask_msa,
        )
        if not zip_outputs:
            warnings.append("No pstMSA.a3m inputs were found, so no fold jobs were dispatched.")

        message = f"All tasks completed successfully for project '{config.project_name}'."
        if warnings:
            message = f"Pipeline completed for project '{config.project_name}' with warnings."

        return PipelineWorkflowResult(
            success=True,
            message=message,
            project_dir=project_dir,
            mode="full",
            num_gpus=num_gpus,
            zip_outputs=zip_outputs,
            warnings=tuple(warnings),
        )
    except GhostfoldValidationError:
        raise
    except GhostfoldExecutionError:
        raise
    except Exception as exc:
        raise GhostfoldExecutionError(f"Pipeline workflow failed: {exc}") from exc
