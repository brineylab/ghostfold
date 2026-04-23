import os
import importlib.util
import itertools
import json
from contextlib import nullcontext
from typing import List, Dict, Any, Optional, Tuple

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

from transformers import T5Tokenizer, AutoModelForSeq2SeqLM

from ghostfold.core.logging import get_console, get_logger
from ghostfold.mutator import MSA_Mutator
from ghostfold.io.fasta import write_fasta, create_project_dir, concatenate_fasta_files, read_fasta_from_path
from ghostfold.msa.filters import filter_sequences
from ghostfold.viz.plotting import generate_optional_plots
from ghostfold.msa.generation import generate_sequences_for_coverages_batched, generate_multimer_sequences

logger = get_logger("pipeline")

# --- Constants ---
MSA_COLORS: List[str] = [
    "#FFFFFF", "#90E0EF", "#48CAE4", "#00B4D8",
    "#219EBC", "#023047", "#FFB703", "#FB8500",
]
DEFAULT_COVERAGE_VALUES: List[float] = [1.0]
DEFAULT_MUTATION_RATES_STR: str = '{"MEGABLAST": 5, "PAM250": 20, "BLOSUM62": 10}'
MODEL_NAME: str = "Rostlab/ProstT5"

# Module-level model cache: key is "{model_name}:{device_type}:{precision}"
_MODEL_CACHE: Dict[str, Tuple[Any, Any]] = {}


def _load_model(
    device: Any,
    model_name: str = MODEL_NAME,
    precision: str = "bf16",
) -> Tuple[Any, Any]:
    """Return (tokenizer, model) for the given device and precision, loading only on first call.

    Precision options:
      bf16  — bfloat16 (default, Ampere+)
      fp16  — float16
      int8  — bitsandbytes 8-bit quantization (requires pip install -e '.[quant]')
      int4  — bitsandbytes NF4 4-bit quantization (requires pip install -e '.[quant]')

    Attention backend: Flash Attention 2 if available, else SDPA (PyTorch built-in).
    torch.compile: max-autotune for bf16/fp16 only (skipped for quantized models).
    """
    import torch

    _VALID_PRECISIONS = ("bf16", "fp16", "int8", "int4")
    if precision not in _VALID_PRECISIONS:
        raise ValueError(
            f"Invalid precision '{precision}'. Must be one of: {_VALID_PRECISIONS}"
        )

    key = f"{model_name}:{device.type}:{precision}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    logger.info(f"Loading T5 model '{model_name}' on {device.type.upper()} with precision={precision}...")

    # Resolve attention backend
    if importlib.util.find_spec("flash_attn") is not None:
        attn_impl = "flash_attention_2"
        logger.info("Flash Attention 2 detected — using flash_attention_2 backend.")
    else:
        attn_impl = "eager"
        logger.info("flash-attn not found — falling back to eager attention backend.")

    if precision in ("int8", "int4") and importlib.util.find_spec("bitsandbytes") is None:
        raise ImportError(
            f"precision='{precision}' requires bitsandbytes. "
            "Install with: pip install -e '.[quant]'"
        )

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False, legacy=True, token=hf_token)

    if precision in ("int8", "int4"):
        from transformers import BitsAndBytesConfig
        if precision == "int8":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        else:  # int4
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            attn_implementation=attn_impl,
            device_map="auto",
            token=hf_token,
        )
        logger.info(f"Model loaded with {precision} quantization (bitsandbytes).")
        logger.debug("torch.compile skipped for quantized models (bitsandbytes incompatibility).")
    else:
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            dtype=dtype,
            attn_implementation=attn_impl,
            token=hf_token,
        ).to(device)
        logger.info(f"Model loaded with {precision} precision.")
        try:
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            model = torch.compile(model, mode="max-autotune")
            logger.info("Model compiled with torch.compile (max-autotune).")
        except Exception as e:
            logger.warning(f"torch.compile unavailable, using eager mode: {e}")

    model.eval()
    logger.info("Model and tokenizer loaded successfully.")
    _MODEL_CACHE[key] = (tokenizer, model)
    return _MODEL_CACHE[key]


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
        generated = generate_sequences_for_coverages_batched(
            query_seq=query_seq,
            full_len=full_len,
            decoding_configs=decoding_configs,
            num_return_sequences=num_return_sequences,
            multiplier=multiplier,
            coverage_list=coverage_list,
            model=model,
            tokenizer=tokenizer,
            device=device,
            project_dir=project_dir,
            inference_batch_size=inference_batch_size,
        )
        total_backtranslated.extend(generated)
        if progress is not None and coverage_task is not None:
            progress.update(coverage_task, advance=len(coverage_list))

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error(
                f"CUDA out of memory during sequence generation for run {run_idx}!"
            )
            logger.error(
                "Try reducing 'inference_batch_size' in your config file."
            )
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            if progress is not None and coverage_task is not None:
                progress.remove_task(coverage_task)
            return {"filtered": None, "evolved": None}
        else:
            logger.error(f"A runtime error occurred during generation: {e}")
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


def write_multimer_pst_msa(
    output_path: str,
    query_seq: str,
    concat_seqs: List[str],
    per_chain_seqs: List[List[str]],
    chain_lengths: List[int],
) -> None:
    """Write a multimer pseudoMSA file in ColabFold A3M/FASTA format.

    Layout:
      1. Query record — sequence contains ':' chain separators (ColabFold convention)
      2. Concat block — full-length sequences generated from the joined complex
      3. Per-chain blocks — per-chain sequences gap-padded to full complex length

    All MSA rows are written without ':'; only the query line carries it.
    """
    lengths_str = ",".join(str(n) for n in chain_lengths)
    cardinality_str = ",".join("1" for _ in chain_lengths)
    chain_header = "\t".join(str(i + 1) for i in range(len(chain_lengths)))
    clean_query = query_seq.replace(":", "")
    with open(output_path, "w") as fh:
        fh.write(f"#{lengths_str}\t{cardinality_str}\n")
        fh.write(f">{chain_header}\n{clean_query}\n")
        for i, seq in enumerate(concat_seqs):
            fh.write(f">concat_{i}\n{seq}\n")
        for chain_idx, chain_seqs in enumerate(per_chain_seqs):
            prefix = "-" * sum(chain_lengths[:chain_idx])
            suffix = "-" * sum(chain_lengths[chain_idx + 1:])
            for seq_idx, seq in enumerate(chain_seqs):
                fh.write(f">chain{chain_idx}_{seq_idx}\n{prefix}{seq}{suffix}\n")


def process_multimer_run(
    chains: List[str],
    header: str,
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
    inference_batch_size: int,
    multimer_msa_mode: str = "concat+per_chain",
    progress: Optional[Progress] = None,
) -> Dict[str, Any]:
    """Run one MSA-generation pass for a multimer complex.

    Returns a dict with keys:
      - "concat_seqs": filtered concat-block sequences (or None on error)
      - "per_chain_seqs": list of filtered per-chain sequence lists (or None)
      - "per_chain_evolved_seqs": list of evolved per-chain lists (or None)
    """
    import torch

    run_dir = os.path.join(base_project_dir, f"run_{run_idx}")
    os.makedirs(run_dir, exist_ok=True)
    logger.info(f"Starting multimer run {run_idx} for '{header}'")

    coverage_task = None
    if progress is not None:
        coverage_task = progress.add_task(
            f"  {header} run {run_idx} (multimer)", total=1
        )

    concat_query = "".join(chains)
    chain_lengths = [len(c) for c in chains]

    try:
        concat_generated, per_chain_generated = generate_multimer_sequences(
            chains=chains,
            coverage_list=coverage_list,
            decoding_configs=decoding_configs,
            num_return_sequences=num_return_sequences,
            multiplier=multiplier,
            model=model,
            tokenizer=tokenizer,
            device=device,
            project_dir=run_dir,
            inference_batch_size=inference_batch_size,
            multimer_msa_mode=multimer_msa_mode,
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error(f"CUDA OOM during multimer generation for run {run_idx}!")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        else:
            logger.error(f"Runtime error during multimer generation for run {run_idx}: {e}")
        if progress is not None and coverage_task is not None:
            progress.remove_task(coverage_task)
        return {"concat_seqs": None, "per_chain_seqs": None, "per_chain_evolved_seqs": None}

    if progress is not None and coverage_task is not None:
        progress.update(coverage_task, advance=1)
        progress.remove_task(coverage_task)

    # Filter concat block at full complex length (include query as first entry)
    concat_filtered = filter_sequences([concat_query] + concat_generated, len(concat_query))

    # Filter each chain block at its own length (include chain query as first entry)
    per_chain_filtered: List[List[str]] = (
        [
            filter_sequences([chain] + chain_seqs, chain_lengths[i])
            for i, (chain, chain_seqs) in enumerate(zip(chains, per_chain_generated))
        ]
        if multimer_msa_mode != "concat"
        else [[] for _ in chains]
    )

    # Optional per-chain MSA evolution
    per_chain_evolved: Optional[List[List[str]]] = None
    if evolve_msa and multimer_msa_mode != "concat":
        per_chain_evolved = []
        for chain_idx, chain_filtered in enumerate(per_chain_filtered):
            if not chain_filtered:
                per_chain_evolved.append([])
                continue
            chain_fasta_path = os.path.join(run_dir, f"filtered_chain_{chain_idx}.fasta")
            write_fasta(
                chain_fasta_path,
                [SeqRecord(Seq(s), id=f"filtered_{i}", description="")
                 for i, s in enumerate(chain_filtered)],
            )
            try:
                mutation_rates = json.loads(mutation_rates_str)
                mutator = MSA_Mutator(mutation_rates=mutation_rates)
                evolved_path = os.path.join(run_dir, f"evolved_chain_{chain_idx}.fasta")
                mutator.evolve_msa(chain_fasta_path, evolved_path, sample_percentage=sample_percentage)
                if os.path.exists(evolved_path) and os.path.getsize(evolved_path) > 0:
                    per_chain_evolved.append(
                        [str(r.seq) for r in SeqIO.parse(evolved_path, "fasta")]
                    )
                else:
                    per_chain_evolved.append([])
            except Exception as e:
                logger.error(f"MSA evolution error for chain {chain_idx}: {e}")
                per_chain_evolved.append([])

    logger.info(f"Finished multimer run {run_idx} for '{header}'")
    return {
        "concat_seqs": concat_filtered,
        "per_chain_seqs": per_chain_filtered,
        "per_chain_evolved_seqs": per_chain_evolved,
    }


def run_pipeline(
    project: str,
    fasta_path: str,
    config: dict,
    coverage_list: List[float],
    evolve_msa: bool,
    mutation_rates_str: str,
    sample_percentage: float,
    plot_msa: bool,
    plot_coevolution: bool,
    hex_colors: List[str] = MSA_COLORS,
    num_runs: int = 1,
    recursive: bool = False,
    show_progress: bool = True,
    precision: str = "bf16",
    multimer_msa_mode: str = "concat+per_chain",
) -> None:
    """Runs the pseudoMSA generation pipeline with OOM handling for model loading."""
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    try:
        tokenizer, model = _load_model(device, precision=precision)
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

    query_records = read_fasta_from_path(fasta_path, recursive=recursive)

    console = get_console()

    if show_progress:
        progress_ctx = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        )
    else:
        progress_ctx = nullcontext()

    with progress_ctx as progress:
        if progress is not None:
            overall_task = progress.add_task(
                "MSA Generation",
                total=len(query_records),
            )

        for record in query_records:
            header, query_seq = record.id, str(record.seq)
            if progress is not None:
                progress.update(
                    overall_task,
                    description=f"MSA Generation ({header})",
                )

            base_project_dir = create_project_dir(project, header)
            pst_msa_path = os.path.join(base_project_dir, "pstMSA.fasta")

            if ":" in query_seq:
                # --- Multimer path ---
                chains = query_seq.split(":")
                chain_lengths = [len(c) for c in chains]

                all_concat_seqs: List[str] = []
                all_per_chain_seqs: List[List[str]] = [[] for _ in chains]

                for run_idx in range(1, num_runs + 1):
                    run_results = process_multimer_run(
                        chains=chains,
                        header=header,
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
                        inference_batch_size=inference_batch_size,
                        multimer_msa_mode=multimer_msa_mode,
                        progress=progress,
                    )
                    if run_results["concat_seqs"]:
                        all_concat_seqs.extend(run_results["concat_seqs"])
                    per_chain = run_results["per_chain_seqs"] or [[] for _ in chains]
                    for i, seqs in enumerate(per_chain):
                        all_per_chain_seqs[i].extend(seqs)
                    if run_results["per_chain_evolved_seqs"]:
                        for i, evolved in enumerate(run_results["per_chain_evolved_seqs"]):
                            all_per_chain_seqs[i].extend(evolved)

                per_chain_to_write = (
                    all_per_chain_seqs if multimer_msa_mode == "concat+per_chain"
                    else [[] for _ in chains]
                )
                write_multimer_pst_msa(
                    output_path=pst_msa_path,
                    query_seq=query_seq,
                    concat_seqs=all_concat_seqs,
                    per_chain_seqs=per_chain_to_write,
                    chain_lengths=chain_lengths,
                )
                logger.info(
                    f"Multimer pseudoMSA written to {pst_msa_path} "
                    f"({len(all_concat_seqs)} concat rows, "
                    + ", ".join(
                        f"{len(s)} chain-{i} rows"
                        for i, s in enumerate(all_per_chain_seqs)
                    )
                    + ")"
                )

            else:
                # --- Monomer path ---
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
                files_to_concat = all_run_filtered_paths + all_run_evolved_paths
                concatenate_fasta_files(files_to_concat, pst_msa_path)

            if progress is not None:
                progress.update(overall_task, advance=1)

    logger.info("All sequences processed. Pipeline finished!")
