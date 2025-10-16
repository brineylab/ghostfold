#!/usr/bin/python3
# pseudoMSA.py

import os
import itertools
import yaml
import torch
import argparse
import time
import json
from typing import List, Dict, Any, Optional
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from Bio.Seq import Seq
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn

# --- Custom Utility/Pipeline Imports ---
from pseudomsa.mutator.mutator import MSA_Mutator
from pseudomsa.utils.io import write_fasta, create_project_dir, concatenate_fasta_files
from pseudomsa.utils.filters import filter_sequences
from pseudomsa.utils.plotting import generate_optional_plots
from pseudomsa.utils.generation import generate_sequences_for_coverage

# --- Constants ---
MSA_COLORS: List[str] = ['#FFFFFF', '#90E0EF', '#48CAE4', '#00B4D8', '#219EBC', '#023047', '#FFB703', '#FB8500']
DEFAULT_COVERAGE_VALUES: List[float] = [1.0]
DEFAULT_MUTATION_RATES_STR: str = '{"MEGABLAST": 5, "PAM250": 20, "BLOSUM62": 10}'
MODEL_NAME: str = 'Rostlab/ProstT5'

# Initialize Rich Console
console = Console()

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
    model: AutoModelForSeq2SeqLM,
    tokenizer: T5Tokenizer,
    device: torch.device,
    evolve_msa: bool,
    mutation_rates_str: str,
    sample_percentage: float,
    hex_colors: List[str],
    plot_msa: bool,
    plot_coevolution: bool,
    inference_batch_size: int,
    console: Console
) -> Dict[str, Optional[str]]:
    """
    Processes a single run for a given query sequence, with OOM handling.

    Args:
        query_seq: The amino acid sequence to process.
        header: The header/ID of the sequence.
        full_len: The full length of the query sequence.
        run_idx: The index of the current independent run.
        base_project_dir: The base directory for project output.
        decoding_configs: A list of decoding configurations for the model.
        num_return_sequences: The number of sequences to return per generation call.
        multiplier: A factor to increase the number of generated sequences.
        coverage_list: A list of coverage values to generate sequences for.
        model: The pre-trained language model.
        tokenizer: The tokenizer for the model.
        device: The torch device (CPU or CUDA).
        evolve_msa: Flag to enable/disable MSA evolution.
        mutation_rates_str: JSON string of mutation rates.
        sample_percentage: Percentage of sequences to sample for evolution.
        hex_colors: List of hex colors for plotting.
        plot_msa: Flag to enable/disable MSA plotting.
        plot_coevolution: Flag to enable/disable co-evolution plotting.
        inference_batch_size: The batch size used for inference.
        console: The Rich console instance for printing.

    Returns:
        A dictionary containing paths to the filtered and evolved FASTA files.
        Returns {'filtered': None, 'evolved': None} on failure.
    """
    run_dir_name = f"run_{run_idx}"
    project_dir = os.path.join(base_project_dir, run_dir_name)
    os.makedirs(project_dir, exist_ok=True)
    console.print(Panel(f"[bold blue]Starting independent run {run_idx} for sequence [green]'{header}'[/green] in '[yellow]{project_dir}[/yellow]'[/bold blue]", expand=False))

    img_dir = os.path.join(project_dir, 'img')
    os.makedirs(img_dir, exist_ok=True)

    total_backtranslated = [query_seq]
    try:
        for coverage in coverage_list:
            start_time_coverage = time.time()
            generated_sequences = generate_sequences_for_coverage(
                query_seq=query_seq, full_len=full_len, decoding_configs=decoding_configs,
                num_return_sequences=num_return_sequences,
                multiplier=multiplier,
                coverage=coverage,
                model=model, tokenizer=tokenizer, device=device, project_dir=project_dir,
                inference_batch_size=inference_batch_size, console=console
            )
            total_backtranslated.extend(generated_sequences)
            end_time_coverage = time.time()
            console.print(f"Generated sequences for coverage [cyan]{coverage}[/cyan] in [bold green]{end_time_coverage - start_time_coverage:.2f} seconds[/bold green].")

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            console.print(f"[bold red]CUDA out of memory during sequence generation for run {run_idx}![/bold red]")
            console.print(f"[yellow]This is common for large sequences or high batch sizes. Try reducing 'inference_batch_size' in your config file.[/yellow]")
            torch.cuda.empty_cache()
            return {'filtered': None, 'evolved': None}
        else:
            console.print(f"[bold red]A runtime error occurred during generation:[/bold red] {e}")
            return {'filtered': None, 'evolved': None}

    unfiltered_path = os.path.join(project_dir, 'unfiltered.fasta')
    write_fasta(unfiltered_path, [SeqRecord(Seq(s), id=f"unfiltered_{i}", description="") for i, s in enumerate(total_backtranslated)])
    console.print(f"Unfiltered sequences saved to [link={unfiltered_path}]{unfiltered_path}[/link]")

    sequences_for_unfiltered_plot = [s for s in total_backtranslated if len(s) == full_len]
    generate_optional_plots(sequences_for_unfiltered_plot, full_len, img_dir, 'unfiltered', hex_colors, plot_msa, plot_coevolution, console)

    console.print("Filtering generated sequences...")
    filtered_sequences = filter_sequences(total_backtranslated, full_len)
    filtered_path = os.path.join(project_dir, 'filtered.fasta')
    if not filtered_sequences:
        console.print("[bold yellow]No sequences passed the filter. Skipping coevolution and mutation steps for this run.[/bold yellow]")
        return {'filtered': None, 'evolved': None}

    write_fasta(filtered_path, [SeqRecord(Seq(s), id=f"filtered_{i}", description="") for i, s in enumerate(filtered_sequences)])
    console.print(f"Filtered sequences saved to [link={filtered_path}]{filtered_path}[/link]. [bold green]{len(filtered_sequences)}[/bold green] sequences passed the filter.")
    generate_optional_plots(filtered_sequences, full_len, img_dir, 'filtered', hex_colors, plot_msa, plot_coevolution, console)

    evolved_path: Optional[str] = None
    if evolve_msa:
        console.print("Attempting to evolve MSA...")
        try:
            mutation_rates = json.loads(mutation_rates_str)
            mutator = MSA_Mutator(mutation_rates=mutation_rates)
            evolved_path = os.path.join(project_dir, 'filtered_evolved.fasta')
            mutator.evolve_msa(filtered_path, evolved_path, sample_percentage=sample_percentage)

            if os.path.exists(evolved_path) and os.path.getsize(evolved_path) > 0:
                evolved_sequences = [str(record.seq) for record in SeqIO.parse(evolved_path, "fasta")]
                console.print(f"Evolved sequences saved to [link={evolved_path}]{evolved_path}[/link].")
                generate_optional_plots(evolved_sequences, full_len, img_dir, 'filtered_evolved', hex_colors, plot_msa, plot_coevolution, console)
            else:
                evolved_path = None
        except Exception as e:
            console.print(f"[bold red]An error occurred during MSA evolution:[/bold red] [red]{e}[/red].")
            evolved_path = None

    console.print(Panel(f"[bold green]Finished run {run_idx} for sequence [green]'{header}'[/green][/bold green]\n", expand=False))
    return {'filtered': filtered_path, 'evolved': evolved_path}


def generate_decoding_configs(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generates a list of decoding configurations from a parameter matrix.

    Args:
        params: A dictionary containing a 'base' config and a 'matrix'
                of parameter lists to be combined.

    Returns:
        A list of dictionaries, where each dictionary is a unique
        decoding configuration.
    """
    base_config = params.get('base', {})
    matrix_params = params.get('matrix', {})

    if not matrix_params:
        return [base_config] if base_config else []

    param_names = list(matrix_params.keys())
    param_values = list(matrix_params.values())

    # Generate the Cartesian product of all parameter values
    combinations = list(itertools.product(*param_values))

    # Create the list of configuration dictionaries
    config_list = [
        {**base_config, **dict(zip(param_names, combo))}
        for combo in combinations
    ]

    return config_list


def run_pipeline(
    project: str,
    query_fasta: str,
    config_path: str,
    coverage_list: List[float],
    evolve_msa: bool,
    mutation_rates_str: str,
    sample_percentage: float,
    plot_msa: bool,
    plot_coevolution: bool,
    hex_colors: List[str] = MSA_COLORS,
    num_runs: int = 1
) -> None:
    """
    Runs the pseudoMSA generation pipeline with OOM handling for model loading.

    Args:
        project: The name of the main project directory.
        query_fasta: Path to the query FASTA file.
        config_path: Path to the YAML configuration file.
        coverage_list: List of coverage values.
        evolve_msa: Flag to enable MSA evolution.
        mutation_rates_str: JSON string for mutation rates.
        sample_percentage: Percentage of sequences to sample for evolution.
        plot_msa: Flag to enable MSA plotting.
        plot_coevolution: Flag to enable co-evolution plotting.
        hex_colors: List of hex colors for plotting.
        num_runs: Number of independent runs for each query sequence.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    try:
        console.print(f"Loading T5 model and tokenizer from '[bold cyan]{MODEL_NAME}[/bold cyan]' on [magenta]{device.type.upper()}[/magenta]...", style="bold")
        tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, do_lower_case=False, legacy=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
        model.eval()
        if device.type == 'cuda':
            model.half()
        console.print("[bold green]Model and tokenizer loaded successfully.[/bold green]")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            console.print(f"[bold red]CUDA out of memory while loading the model![/bold red]")
            console.print(f"[yellow]The model '{MODEL_NAME}' is too large to fit on your GPU. The pipeline cannot continue.[/yellow]")
            return
        else:
            console.print(f"[bold red]A runtime error occurred during model loading:[/bold red] {e}")
            return
            
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    decoding_params = config.get('decoding_params', {})
    decoding_configs = generate_decoding_configs(decoding_params)

    if not decoding_configs:
        console.print("[bold red]Warning: No decoding configurations were generated from the config file.[/bold red]")

    num_return_sequences: int = config.get('num_return_sequences', 5)
    multiplier: int = config.get('multiplier', 1)
    inference_batch_size: int = config.get('inference_batch_size', 4)

    query_records = list(SeqIO.parse(query_fasta, "fasta"))

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(), TimeElapsedColumn(), console=console, transient=False
    ) as progress:
        overall_task = progress.add_task("[bold magenta]Overall Pipeline Progress[/bold magenta]", total=len(query_records))

        for record in query_records:
            header, query_seq = record.id, str(record.seq)
            progress.update(overall_task, description=f"[bold magenta]Processing:[/bold magenta] [green]{header}[/green]")
            
            base_project_dir = create_project_dir(project, header)
            all_run_filtered_paths, all_run_evolved_paths = [], []
            
            run_task = progress.add_task(f"[bold yellow]Runs for {header}[/bold yellow]", total=num_runs)
            for run_idx in range(1, num_runs + 1):
                run_results = process_sequence_run(
                    query_seq=query_seq, header=header, full_len=len(query_seq), run_idx=run_idx,
                    base_project_dir=base_project_dir, decoding_configs=decoding_configs,
                    num_return_sequences=num_return_sequences, 
                    multiplier=multiplier, 
                    coverage_list=coverage_list,
                    model=model, tokenizer=tokenizer, device=device, evolve_msa=evolve_msa,
                    mutation_rates_str=mutation_rates_str, sample_percentage=sample_percentage,
                    hex_colors=hex_colors, plot_msa=plot_msa,
                    plot_coevolution=plot_coevolution, inference_batch_size=inference_batch_size,
                    console=console
                )
                if run_results['filtered']: all_run_filtered_paths.append(run_results['filtered'])
                if run_results['evolved']: all_run_evolved_paths.append(run_results['evolved'])
                progress.update(run_task, advance=1)
            progress.remove_task(run_task)

            console.print(f"\n--- [bold purple]Concatenating all FASTA files for '[green]{header}[/green]'[/bold purple] ---")
            pst_msa_path = os.path.join(base_project_dir, "pstMSA.fasta")
            files_to_concat = all_run_filtered_paths + all_run_evolved_paths
            concatenate_fasta_files(files_to_concat, pst_msa_path, console)
            progress.update(overall_task, advance=1)
        progress.remove_task(overall_task)
    
    console.print(Panel("[bold green]All sequences processed. Pipeline finished![/bold green]", expand=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run pseudoMSA generation pipeline.')
    parser.add_argument('--project_name', type=str, required=True, help='The name of the main project directory.')
    parser.add_argument('--fasta_file', required=True, help='Path to query FASTA file.')
    parser.add_argument('--config', required=True, help='Path to YAML config.')
    parser.add_argument('--coverage', nargs='+', type=float, default=DEFAULT_COVERAGE_VALUES, help=f'List of coverage values. Default: {DEFAULT_COVERAGE_VALUES}')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of independent runs for each query sequence.')
    # Plotting arguments
    parser.add_argument('--plot_msa_coverage', action='store_true', help='Generate MSA coverage plots. Off by default.')
    parser.add_argument('--no_coevolution_maps', action='store_true', help='Do not generate coevolution maps. On by default.')
    # Evolution arguments
    parser.add_argument('--evolve_msa', action='store_true', help='Enable MSA evolution using the mutator module.')
    parser.add_argument('--mutation_rates', type=str, default=DEFAULT_MUTATION_RATES_STR, help=f'JSON string for mutation rates. Default: \'{DEFAULT_MUTATION_RATES_STR}\'')
    parser.add_argument('--sample_percentage', type=float, default=1.0, help='Percentage of sequences to sample for evolution (e.g., 0.5 for 50%).')
    
    args = parser.parse_args()

    run_pipeline(
        project=args.project_name,
        query_fasta=args.fasta_file,
        config_path=args.config,
        coverage_list=args.coverage,
        evolve_msa=args.evolve_msa,
        mutation_rates_str=args.mutation_rates,
        sample_percentage=args.sample_percentage,
        num_runs=args.num_runs,
        plot_msa=args.plot_msa_coverage, # Off by default
        plot_coevolution=not args.no_coevolution_maps # On by default
    )
  
