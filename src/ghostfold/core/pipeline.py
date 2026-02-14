import os
import itertools
import json
import time
from typing import List, Dict, Any, Optional

from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from Bio.Seq import Seq
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from ghostfold.core.logging import get_console, get_logger
from ghostfold.mutator import MSA_Mutator
from ghostfold.io.fasta import write_fasta, create_project_dir, concatenate_fasta_files
from ghostfold.msa.filters import filter_sequences
from ghostfold.viz.plotting import generate_optional_plots
from ghostfold.msa.generation import generate_sequences_for_coverage

logger = get_logger("pipeline")

# --- Constants ---
MSA_COLORS: List[str] = [
    "#FFFFFF", "#90E0EF", "#48CAE4", "#00B4D8",
    "#219EBC", "#023047", "#FFB703", "#FB8500",
]
DEFAULT_COVERAGE_VALUES: List[float] = [1.0]
DEFAULT_MUTATION_RATES_STR: str = '{"MEGABLAST": 5, "PAM250": 20, "BLOSUM62": 10}'
MODEL_NAME: str = "Rostlab/ProstT5"


def process_sequence_run(
    query_seq: str,
    header: str,
    full_len: int,
    run_idx: int,
    base_project_dir: str,
    decoding_configs: List[Dict[str, Any]],
    num_return_sequences: int,
    multiplier: int,
    coverage_list: List[float],
    model: Any,
    tokenizer: Any,
    device: Any,
    evolve_msa: bool,
    mutation_rates_str: str,
    sample_percentage: float,
    hex_colors: List[str],
    plot_msa: bool,
    plot_coevolution: bool,
    inference_batch_size: int,
    progress: Optional[Progress] = None,
) -> Dict[str, Optional[str]]:
    """Processes a single run for a given query sequence, with OOM handling."""
    import torch

    run_dir_name = f"run_{run_idx}"
    project_dir = os.path.join(base_project_dir, run_dir_name)
    os.makedirs(project_dir, exist_ok=True)
    logger.info(
        f"Starting independent run {run_idx} for sequence '{header}' in '{project_dir}'"
    )

    img_dir = os.path.join(project_dir, "img")
    os.makedirs(img_dir, exist_ok=True)

    # Add a coverage-level progress sub-task if progress bar is available
    coverage_task = None
    if progress is not None:
        coverage_task = progress.add_task(
            f"  {header} run {run_idx}",
            total=len(coverage_list),
        )

    total_backtranslated = [query_seq]
    try:
        for coverage in coverage_list:
            start_time_coverage = time.time()
            generated_sequences = generate_sequences_for_coverage(
                query_seq=query_seq,
                full_len=full_len,
                decoding_configs=decoding_configs,
                num_return_sequences=num_return_sequences,
                multiplier=multiplier,
                coverage=coverage,
                model=model,
                tokenizer=tokenizer,
                device=device,
                project_dir=project_dir,
                inference_batch_size=inference_batch_size,
            )
            total_backtranslated.extend(generated_sequences)
            end_time_coverage = time.time()
            logger.info(
                f"Generated sequences for coverage {coverage} in "
                f"{end_time_coverage - start_time_coverage:.2f} seconds."
            )
            if progress is not None and coverage_task is not None:
                progress.update(coverage_task, advance=1)

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error(
                f"CUDA out of memory during sequence generation for run {run_idx}!"
            )
            logger.error(
                "This is common for large sequences or high batch sizes. "
                "Try reducing 'inference_batch_size' in your config file."
            )
            torch.cuda.empty_cache()
            if progress is not None and coverage_task is not None:
                progress.remove_task(coverage_task)
            return {"filtered": None, "evolved": None}
        else:
            logger.error(
                f"A runtime error occurred during generation: {e}"
            )
            if progress is not None and coverage_task is not None:
                progress.remove_task(coverage_task)
            return {"filtered": None, "evolved": None}

    if progress is not None and coverage_task is not None:
        progress.remove_task(coverage_task)

    unfiltered_path = os.path.join(project_dir, "unfiltered.fasta")
    write_fasta(
        unfiltered_path,
        [
            SeqRecord(Seq(s), id=f"unfiltered_{i}", description="")
            for i, s in enumerate(total_backtranslated)
        ],
    )
    logger.info(f"Unfiltered sequences saved to {unfiltered_path}")

    sequences_for_unfiltered_plot = [
        s for s in total_backtranslated if len(s) == full_len
    ]
    generate_optional_plots(
        sequences_for_unfiltered_plot,
        full_len,
        img_dir,
        "unfiltered",
        hex_colors,
        plot_msa,
        plot_coevolution,
    )

    logger.info("Filtering generated sequences...")
    filtered_sequences = filter_sequences(total_backtranslated, full_len)
    filtered_path = os.path.join(project_dir, "filtered.fasta")
    if not filtered_sequences:
        logger.warning(
            "No sequences passed the filter. Skipping coevolution "
            "and mutation steps for this run."
        )
        return {"filtered": None, "evolved": None}

    write_fasta(
        filtered_path,
        [
            SeqRecord(Seq(s), id=f"filtered_{i}", description="")
            for i, s in enumerate(filtered_sequences)
        ],
    )
    logger.info(
        f"Filtered sequences saved to {filtered_path}. "
        f"{len(filtered_sequences)} sequences passed the filter."
    )
    generate_optional_plots(
        filtered_sequences,
        full_len,
        img_dir,
        "filtered",
        hex_colors,
        plot_msa,
        plot_coevolution,
    )

    evolved_path: Optional[str] = None
    if evolve_msa:
        logger.info("Attempting to evolve MSA...")
        try:
            mutation_rates = json.loads(mutation_rates_str)
            mutator = MSA_Mutator(mutation_rates=mutation_rates)
            evolved_path = os.path.join(project_dir, "filtered_evolved.fasta")
            mutator.evolve_msa(
                filtered_path, evolved_path, sample_percentage=sample_percentage
            )

            if os.path.exists(evolved_path) and os.path.getsize(evolved_path) > 0:
                evolved_sequences = [
                    str(record.seq) for record in SeqIO.parse(evolved_path, "fasta")
                ]
                logger.info(
                    f"Evolved sequences saved to {evolved_path}."
                )
                generate_optional_plots(
                    evolved_sequences,
                    full_len,
                    img_dir,
                    "filtered_evolved",
                    hex_colors,
                    plot_msa,
                    plot_coevolution,
                )
            else:
                evolved_path = None
        except Exception as e:
            logger.error(
                f"An error occurred during MSA evolution: {e}."
            )
            evolved_path = None

    logger.info(f"Finished run {run_idx} for sequence '{header}'")
    return {"filtered": filtered_path, "evolved": evolved_path}


def generate_decoding_configs(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generates a list of decoding configurations from a parameter matrix."""
    base_config = params.get("base", {})
    matrix_params = params.get("matrix", {})

    if not matrix_params:
        return [base_config] if base_config else []

    param_names = list(matrix_params.keys())
    param_values = list(matrix_params.values())

    combinations = list(itertools.product(*param_values))

    config_list = [
        {**base_config, **dict(zip(param_names, combo))} for combo in combinations
    ]

    return config_list


def run_pipeline(
    project: str,
    query_fasta: str,
    config: dict,
    coverage_list: List[float],
    evolve_msa: bool,
    mutation_rates_str: str,
    sample_percentage: float,
    plot_msa: bool,
    plot_coevolution: bool,
    hex_colors: List[str] = MSA_COLORS,
    num_runs: int = 1,
) -> None:
    """Runs the pseudoMSA generation pipeline with OOM handling for model loading."""
    import torch
    from transformers import T5Tokenizer, AutoModelForSeq2SeqLM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    try:
        logger.info(
            f"Loading T5 model and tokenizer from '{MODEL_NAME}' on {device.type.upper()}..."
        )
        tokenizer = T5Tokenizer.from_pretrained(
            MODEL_NAME, do_lower_case=False, legacy=True
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
        model.eval()
        if device.type == "cuda":
            model.half()
        logger.info("Model and tokenizer loaded successfully.")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error("CUDA out of memory while loading the model!")
            logger.error(
                f"The model '{MODEL_NAME}' is too large to fit on your GPU. "
                "The pipeline cannot continue."
            )
            return
        else:
            logger.error(
                f"A runtime error occurred during model loading: {e}"
            )
            return

    decoding_params = config.get("decoding_params", {})
    decoding_configs = generate_decoding_configs(decoding_params)

    if not decoding_configs:
        logger.warning(
            "No decoding configurations were generated from the config file."
        )

    num_return_sequences: int = config.get("num_return_sequences", 5)
    multiplier: int = config.get("multiplier", 1)
    inference_batch_size: int = config.get("inference_batch_size", 4)

    query_records = list(SeqIO.parse(query_fasta, "fasta"))

    console = get_console()

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
        overall_task = progress.add_task(
            "MSA Generation",
            total=len(query_records),
        )

        for record in query_records:
            header, query_seq = record.id, str(record.seq)
            progress.update(
                overall_task,
                description=f"MSA Generation ({header})",
            )

            base_project_dir = create_project_dir(project, header)
            all_run_filtered_paths, all_run_evolved_paths = [], []

            for run_idx in range(1, num_runs + 1):
                run_results = process_sequence_run(
                    query_seq=query_seq,
                    header=header,
                    full_len=len(query_seq),
                    run_idx=run_idx,
                    base_project_dir=base_project_dir,
                    decoding_configs=decoding_configs,
                    num_return_sequences=num_return_sequences,
                    multiplier=multiplier,
                    coverage_list=coverage_list,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    evolve_msa=evolve_msa,
                    mutation_rates_str=mutation_rates_str,
                    sample_percentage=sample_percentage,
                    hex_colors=hex_colors,
                    plot_msa=plot_msa,
                    plot_coevolution=plot_coevolution,
                    inference_batch_size=inference_batch_size,
                    progress=progress,
                )
                if run_results["filtered"]:
                    all_run_filtered_paths.append(run_results["filtered"])
                if run_results["evolved"]:
                    all_run_evolved_paths.append(run_results["evolved"])

            logger.info(f"Concatenating all FASTA files for '{header}'")
            pst_msa_path = os.path.join(base_project_dir, "pstMSA.fasta")
            files_to_concat = all_run_filtered_paths + all_run_evolved_paths
            concatenate_fasta_files(files_to_concat, pst_msa_path)
            progress.update(overall_task, advance=1)

    logger.info("All sequences processed. Pipeline finished!")
