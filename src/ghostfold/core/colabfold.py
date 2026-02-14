from __future__ import annotations

import glob
import os
import re
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Sequence

from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TaskID,
)

from ghostfold.core.colabfold_env import (
    DEFAULT_COLABFOLD_ENV,
    ColabFoldSetupError,
    ensure_colabfold_ready,
)
from ghostfold.core.logging import get_console, get_logger
from ghostfold.core.postprocess import cleanup_colabfold_outputs
from ghostfold.msa.mask import mask_a3m_file

logger = get_logger("colabfold")

# Regex to detect a completed model evaluation line from ColabFold
_MODEL_DONE_RE = re.compile(r"_seed_\d+\s+took\s+[\d.]+s")

# Default ColabFold parameters
COLABFOLD_PARAMS = [
    "--num-recycle", "10",
    "--num-models", "5",
    "--rank", "ptm",
    "--recycle-early-stop-tolerance", "0.5",
    "--use-dropout",
    "--num-seeds", "5",
    "--save-recycles",
    "--model-type", "auto",
]

# Subsampling levels
SUBSAMPLE_MAX_SEQ = [16, 32, 64, 128]
SUBSAMPLE_MAX_EXTRA_SEQ = [32, 64, 128, 256]
DEFAULT_MAX_SEQ = [32]
DEFAULT_MAX_EXTRA_SEQ = [64]


def _get_colabfold_total_models(params: list) -> int:
    """Extract --num-models * --num-seeds from the ColabFold param list."""
    num_models = 5
    num_seeds = 1
    for i, p in enumerate(params):
        if p == "--num-models" and i + 1 < len(params):
            num_models = int(params[i + 1])
        elif p == "--num-seeds" and i + 1 < len(params):
            num_seeds = int(params[i + 1])
    return num_models * num_seeds


def _run_colabfold_subprocess(
    gpu_id: int,
    msa_file: str,
    output_dir: str,
    max_seq: int,
    max_extra_seq: int,
    launcher_prefix: Sequence[str],
    launcher_cwd: Optional[str],
    progress: Optional[Progress] = None,
    task_id: Optional[TaskID] = None,
) -> None:
    """Run a single colabfold_batch subprocess on a specific GPU.

    Captures stdout/stderr line-by-line, writes everything to the log file,
    and parses model-completion lines to advance the progress bar.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [
        *launcher_prefix,
        "colabfold_batch",
        msa_file,
        output_dir,
        "--max-seq", str(max_seq),
        "--max-extra-seq", str(max_extra_seq),
        *COLABFOLD_PARAMS,
    ]

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        cwd=launcher_cwd,
        text=True,
        bufsize=1,
    )

    for line in proc.stdout:  # type: ignore[union-attr]
        line = line.rstrip()
        logger.info(line)
        if progress is not None and task_id is not None:
            if _MODEL_DONE_RE.search(line):
                progress.update(task_id, advance=1)

    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def run_colabfold(
    project_name: str,
    num_gpus: int,
    subsample: bool = False,
    mask_fraction: Optional[float] = None,
    colabfold_env: str = DEFAULT_COLABFOLD_ENV,
    localcolabfold_dir: Path | str | None = None,
) -> None:
    """Run ColabFold structure prediction on generated MSAs.

    Args:
        project_name: Project directory containing MSA files.
        num_gpus: Number of GPUs to distribute jobs across.
        subsample: If True, run multiple subsampling levels.
        mask_fraction: Optional fraction (0.0-1.0) of residues to mask.
        colabfold_env: Legacy mamba environment used as fallback.
        localcolabfold_dir: Optional path to localcolabfold pixi checkout.
    """
    logger.info("Starting ColabFold Structure Prediction...")

    try:
        launcher = ensure_colabfold_ready(
            colabfold_env=colabfold_env,
            localcolabfold_dir=localcolabfold_dir,
        )
    except ColabFoldSetupError as exc:
        raise RuntimeError(str(exc)) from exc

    msa_root_dir = os.path.join(project_name, "msa")
    if not os.path.isdir(msa_root_dir):
        raise FileNotFoundError(f"MSA directory not found at '{msa_root_dir}'")

    original_a3m_files = sorted(
        glob.glob(os.path.join(msa_root_dir, "*", "pstMSA.a3m"))
    )
    if not original_a3m_files:
        logger.warning(f"No 'pstMSA.a3m' files found in '{msa_root_dir}'.")
        return

    # Handle masking
    a3m_files_to_process = list(original_a3m_files)
    temp_masked_files: List[str] = []

    if mask_fraction is not None and mask_fraction > 0.0:
        logger.info(f"Creating temporary masked MSAs with fraction: {mask_fraction}")
        a3m_files_to_process = []

        for msa_file in original_a3m_files:
            temp_masked_file = msa_file.replace(".a3m", "_masked_temp.a3m")
            logger.info(f"     [Masking] {msa_file} -> {temp_masked_file}")
            mask_a3m_file(
                Path(msa_file), Path(temp_masked_file), mask_fraction
            )
            a3m_files_to_process.append(temp_masked_file)
            temp_masked_files.append(temp_masked_file)

        logger.info("Temporary masked MSAs created.")

    # Define subsampling parameters
    if subsample:
        logger.info("Subsampling mode enabled.")
        max_seq_vals = SUBSAMPLE_MAX_SEQ
        max_extra_seq_vals = SUBSAMPLE_MAX_EXTRA_SEQ
    else:
        logger.info("Standard mode 32:64 (no subsampling).")
        max_seq_vals = DEFAULT_MAX_SEQ
        max_extra_seq_vals = DEFAULT_MAX_EXTRA_SEQ

    models_per_file = _get_colabfold_total_models(COLABFOLD_PARAMS)
    console = get_console()

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        ) as progress:
            for i, (max_seq, max_extra_seq) in enumerate(
                zip(max_seq_vals, max_extra_seq_vals)
            ):
                subsample_dir = os.path.join(project_name, f"subsample_{i + 1}")
                preds_dir = os.path.join(subsample_dir, "preds")

                logger.info(f"Running subsample level {i + 1} / {len(max_seq_vals)}")
                logger.info(f"   Parameters: --max-seq {max_seq} --max-extra-seq {max_extra_seq}")
                logger.info(f"   Output Dir: {preds_dir}")

                os.makedirs(preds_dir, exist_ok=True)

                total_models = len(a3m_files_to_process) * models_per_file
                level_task = progress.add_task(
                    f"Folding (level {i + 1}/{len(max_seq_vals)})",
                    total=total_models,
                )

                # Use ThreadPoolExecutor so threads can share the Progress instance
                with ThreadPoolExecutor(max_workers=num_gpus) as executor:
                    futures = []
                    for j, msa_file in enumerate(a3m_files_to_process):
                        output_folder_name = os.path.basename(os.path.dirname(msa_file))
                        current_pred_dir = os.path.join(preds_dir, output_folder_name)
                        os.makedirs(current_pred_dir, exist_ok=True)
                        gpu_id = j % num_gpus

                        logger.info(
                            f"   [Dispatching Colabfold] Input: {output_folder_name} -> GPU: {gpu_id}"
                        )
                        future = executor.submit(
                            _run_colabfold_subprocess,
                            gpu_id,
                            msa_file,
                            current_pred_dir,
                            max_seq,
                            max_extra_seq,
                            launcher.command_prefix,
                            str(launcher.cwd) if launcher.cwd is not None else None,
                            progress,
                            level_task,
                        )
                        futures.append(future)

                    logger.info("Waiting for ColabFold jobs at this level to complete...")
                    for future in as_completed(futures):
                        future.result()

                logger.info(f"ColabFold jobs finished for subsample level {i + 1}.")

                cleanup_colabfold_outputs(subsample_dir)

                # Create zip archive
                logger.info("Zipping results for this level...")
                zip_path = f"{subsample_dir}.zip"
                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    for root, _dirs, files in os.walk(subsample_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, project_name)
                            zf.write(file_path, arcname)
                logger.info(f"Zip file created at: {zip_path}")

    finally:
        # Clean up temporary masked files
        if temp_masked_files:
            logger.info("Cleaning up temporary masked MSA files...")
            for temp_file in temp_masked_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            logger.info("Cleanup of temporary files complete.")
