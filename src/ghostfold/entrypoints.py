"""Script compatibility entrypoints routed through service layer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from ghostfold.config import MSAWorkflowConfig, MaskWorkflowConfig, NeffWorkflowConfig
from ghostfold.errors import GhostfoldError
from ghostfold.services import run_mask_workflow, run_msa_workflow, run_neff_workflow


def pseudomsa_main(argv: Optional[List[str]] = None) -> None:
    """Compatibility entrypoint for the legacy `pseudomsa.py` script."""
    from ghostfold.msa_core import build_parser

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_msa_workflow(
            MSAWorkflowConfig(
                project_name=args.project_name,
                fasta_file=Path(args.fasta_file),
                config_path=Path(args.config),
                coverage=args.coverage,
                num_runs=args.num_runs,
                plot_msa_coverage=args.plot_msa_coverage,
                no_coevolution_maps=args.no_coevolution_maps,
                evolve_msa=args.evolve_msa,
                mutation_rates=args.mutation_rates,
                sample_percentage=args.sample_percentage,
            )
        )
    except GhostfoldError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


def mask_msa_main(argv: Optional[List[str]] = None) -> None:
    """Compatibility entrypoint for the legacy `mask_msa.py` script."""
    from ghostfold.masking import build_parser

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_mask_workflow(
            MaskWorkflowConfig(
                input_path=args.input_path,
                output_path=args.output_path,
                mask_fraction=args.mask_fraction,
            )
        )
        print(f"✅ Successfully created masked file '{args.output_path}'.")
    except GhostfoldError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


def neff_main(argv: Optional[List[str]] = None) -> None:
    """Compatibility entrypoint for the legacy `calculate_neff.py` script."""
    args = list(argv) if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Calculates Neff score in parallel for MSAs in a directory."
    )
    parser.add_argument("project_dir", help="Path to project directory containing msa/")
    parsed = parser.parse_args(args)
    try:
        run_neff_workflow(NeffWorkflowConfig(project_dir=Path(parsed.project_dir)))
    except GhostfoldError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
