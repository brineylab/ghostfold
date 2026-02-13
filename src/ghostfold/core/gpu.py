from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Optional

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord


def detect_gpus() -> int:
    """Detect the number of available NVIDIA GPUs.

    Returns:
        The number of GPUs detected.

    Raises:
        RuntimeError: If nvidia-smi is not found or no GPUs are detected.
    """
    if not shutil.which("nvidia-smi"):
        raise RuntimeError("nvidia-smi command not found. Cannot detect GPUs.")

    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")

    lines = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("No GPUs detected by nvidia-smi.")

    return len(lines)


def split_fasta(records: List[SeqRecord], output_dir: Path) -> List[Path]:
    """Write each SeqRecord to its own FASTA file.

    Args:
        records: List of SeqRecord objects to split.
        output_dir: Directory to write individual FASTA files.

    Returns:
        List of paths to the created FASTA files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for i, record in enumerate(records):
        path = output_dir / f"split_seq_{i:04d}.fasta"
        SeqIO.write([record], str(path), "fasta")
        paths.append(path)
    return paths


def _msa_worker(
    gpu_id: int,
    project_name: str,
    fasta_file: str,
    config_path: Optional[str],
) -> None:
    """Worker function that runs the MSA pipeline on a specific GPU.

    Must be called in a separate process. Sets CUDA_VISIBLE_DEVICES
    before importing torch to ensure proper GPU isolation.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from ghostfold.core.config import load_config
    from ghostfold.core.pipeline import run_pipeline, DEFAULT_MUTATION_RATES_STR

    config = load_config(config_path)

    run_pipeline(
        project=project_name,
        query_fasta=fasta_file,
        config=config,
        coverage_list=[1.0],
        evolve_msa=True,
        mutation_rates_str=DEFAULT_MUTATION_RATES_STR,
        sample_percentage=1.0,
        plot_msa=False,
        plot_coevolution=False,
        num_runs=1,
    )


def run_parallel_msa(
    project_name: str,
    fasta_file: str,
    num_gpus: int,
    config_path: Optional[str] = None,
) -> None:
    """Generate MSAs in parallel across multiple GPUs, then post-process.

    Args:
        project_name: Name of the project directory.
        fasta_file: Path to the input FASTA file.
        num_gpus: Number of GPUs to use.
        config_path: Optional path to a user config YAML.
    """
    from ghostfold.core.postprocess import postprocess_msa_outputs

    records = list(SeqIO.parse(fasta_file, "fasta"))
    num_seqs = len(records)

    print(f"Detected {num_gpus} GPUs and {num_seqs} sequences for project '{project_name}'.")
    os.makedirs(project_name, exist_ok=True)

    if num_seqs == 1:
        print("Only one sequence found. Running on a single GPU.")
        _msa_worker(0, project_name, fasta_file, config_path)
    else:
        temp_dir = Path(tempfile.mkdtemp(prefix="ghostfold_splits_"))
        print(f"Splitting FASTA file into temporary directory: {temp_dir}")
        split_paths = split_fasta(records, temp_dir)

        max_jobs = min(num_gpus, num_seqs)
        print(f"Starting parallel MSA generation on {max_jobs} GPUs...")

        with ProcessPoolExecutor(max_workers=max_jobs) as executor:
            futures = []
            for i, split_path in enumerate(split_paths):
                gpu_id = i % max_jobs
                future = executor.submit(
                    _msa_worker, gpu_id, project_name, str(split_path), config_path
                )
                futures.append(future)

            # Wait for all to complete and propagate exceptions
            for future in futures:
                future.result()

        print("Cleaning up temporary files...")
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("Starting post-processing of MSA files...")
    postprocess_msa_outputs(project_name)
    print("MSA generation and processing complete.")
