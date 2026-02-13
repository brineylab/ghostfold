from __future__ import annotations

import glob
import os
import subprocess
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

from ghostfold.core.colabfold_env import (
    DEFAULT_COLABFOLD_ENV,
    ColabFoldSetupError,
    ensure_colabfold_ready,
)
from ghostfold.core.postprocess import cleanup_colabfold_outputs
from ghostfold.msa.mask import mask_a3m_file


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


def _run_colabfold_subprocess(
    gpu_id: int,
    msa_file: str,
    output_dir: str,
    max_seq: int,
    max_extra_seq: int,
    colabfold_env: str,
) -> None:
    """Run a single colabfold_batch subprocess on a specific GPU."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        "mamba", "run", "-n", colabfold_env, "--no-capture-output",
        "colabfold_batch",
        msa_file,
        output_dir,
        "--max-seq", str(max_seq),
        "--max-extra-seq", str(max_extra_seq),
        *COLABFOLD_PARAMS,
    ]

    subprocess.run(cmd, env=env, check=True, stdin=subprocess.DEVNULL)


def run_colabfold(
    project_name: str,
    num_gpus: int,
    subsample: bool = False,
    mask_fraction: Optional[float] = None,
    colabfold_env: str = DEFAULT_COLABFOLD_ENV,
) -> None:
    """Run ColabFold structure prediction on generated MSAs.

    Args:
        project_name: Project directory containing MSA files.
        num_gpus: Number of GPUs to distribute jobs across.
        subsample: If True, run multiple subsampling levels.
        mask_fraction: Optional fraction (0.0-1.0) of residues to mask.
        colabfold_env: Mamba environment containing ColabFold.
    """
    print("---\nStarting ColabFold Structure Prediction...")

    try:
        ensure_colabfold_ready(colabfold_env)
    except ColabFoldSetupError as exc:
        raise RuntimeError(str(exc)) from exc

    msa_root_dir = os.path.join(project_name, "msa")
    if not os.path.isdir(msa_root_dir):
        raise FileNotFoundError(f"MSA directory not found at '{msa_root_dir}'")

    original_a3m_files = sorted(
        glob.glob(os.path.join(msa_root_dir, "*", "pstMSA.a3m"))
    )
    if not original_a3m_files:
        print(f"Warning: No 'pstMSA.a3m' files found in '{msa_root_dir}'.")
        return

    # Handle masking
    a3m_files_to_process = list(original_a3m_files)
    temp_masked_files: List[str] = []

    if mask_fraction is not None and mask_fraction > 0.0:
        print(f"---\nCreating temporary masked MSAs with fraction: {mask_fraction}")
        a3m_files_to_process = []

        for msa_file in original_a3m_files:
            temp_masked_file = msa_file.replace(".a3m", "_masked_temp.a3m")
            print(f"     [Masking] {msa_file} -> {temp_masked_file}")
            mask_a3m_file(
                Path(msa_file), Path(temp_masked_file), mask_fraction
            )
            a3m_files_to_process.append(temp_masked_file)
            temp_masked_files.append(temp_masked_file)

        print("Temporary masked MSAs created.")

    # Define subsampling parameters
    if subsample:
        print("Subsampling mode enabled.")
        max_seq_vals = SUBSAMPLE_MAX_SEQ
        max_extra_seq_vals = SUBSAMPLE_MAX_EXTRA_SEQ
    else:
        print("Standard mode 32:64 (no subsampling).")
        max_seq_vals = DEFAULT_MAX_SEQ
        max_extra_seq_vals = DEFAULT_MAX_EXTRA_SEQ

    try:
        for i, (max_seq, max_extra_seq) in enumerate(
            zip(max_seq_vals, max_extra_seq_vals)
        ):
            subsample_dir = os.path.join(project_name, f"subsample_{i + 1}")
            preds_dir = os.path.join(subsample_dir, "preds")

            print(f"---\nRunning subsample level {i + 1} / {len(max_seq_vals)}")
            print(f"   Parameters: --max-seq {max_seq} --max-extra-seq {max_extra_seq}")
            print(f"   Output Dir: {preds_dir}")

            os.makedirs(preds_dir, exist_ok=True)

            # Dispatch ColabFold jobs across GPUs
            with ProcessPoolExecutor(max_workers=num_gpus) as executor:
                futures = []
                for j, msa_file in enumerate(a3m_files_to_process):
                    output_folder_name = os.path.basename(os.path.dirname(msa_file))
                    current_pred_dir = os.path.join(preds_dir, output_folder_name)
                    os.makedirs(current_pred_dir, exist_ok=True)
                    gpu_id = j % num_gpus

                    print(
                        f"   [Dispatching Colabfold] Input: {output_folder_name} -> GPU: {gpu_id}"
                    )
                    future = executor.submit(
                        _run_colabfold_subprocess,
                        gpu_id,
                        msa_file,
                        current_pred_dir,
                        max_seq,
                        max_extra_seq,
                        colabfold_env,
                    )
                    futures.append(future)

                print("Waiting for ColabFold jobs at this level to complete...")
                for future in as_completed(futures):
                    future.result()

            print(f"ColabFold jobs finished for subsample level {i + 1}.")

            cleanup_colabfold_outputs(subsample_dir)

            # Create zip archive
            print("Zipping results for this level...")
            zip_path = f"{subsample_dir}.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for root, _dirs, files in os.walk(subsample_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, project_name)
                        zf.write(file_path, arcname)
            print(f"Zip file created at: {zip_path}")

    finally:
        # Clean up temporary masked files
        if temp_masked_files:
            print("---\nCleaning up temporary masked MSA files...")
            for temp_file in temp_masked_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            print("Cleanup of temporary files complete.")
