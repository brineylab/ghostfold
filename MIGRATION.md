# Migration Guide

This document maps legacy GhostFold script usage to the modern pip-installed CLI and Python API.

## Command mapping

| Legacy command | New command | Notes |
| --- | --- | --- |
| `./ghostfold.sh --project_name P --fasta_file q.fasta` | `ghostfold run --project_name P --fasta_file q.fasta` | Full pipeline mode |
| `./ghostfold.sh --project_name P --fasta_file q.fasta --msa-only` | `ghostfold run --project_name P --fasta_file q.fasta --msa-only` | MSA-only mode |
| `./ghostfold.sh --project_name P --fold-only` | `ghostfold run --project_name P --fold-only` | Fold-only mode |
| `./ghostfold.sh --project_name P --fasta_file q.fasta --subsample` | `ghostfold run --project_name P --fasta_file q.fasta --subsample` | Multi-level fold subsampling |
| `python pseudomsa.py ...` | `ghostfold msa ...` | `pseudomsa` entrypoint is still installed for compatibility |
| `python mask_msa.py --input_path in.a3m --output_path out.a3m --mask_fraction 0.15` | `ghostfold mask_msa --input_path in.a3m --output_path out.a3m --mask_fraction 0.15` | `ghostfold mask ...` is also available |
| `python calculate_neff.py <project_dir>` | `ghostfold calculate_neff <project_dir>` | `ghostfold neff <project_dir>` is also available |

## Compatibility shims still available

- `./ghostfold.sh` remains available as a thin wrapper.
- Root scripts `pseudomsa.py`, `mask_msa.py`, and `calculate_neff.py` still work.
- Pip installs expose compatibility console scripts: `pseudomsa`, `mask_msa`, and `calculate_neff`.

## Behavioral notes retained for parity

- GPU detection via `nvidia-smi` is still required for all pipeline modes, including `--msa-only`.
- Fold mode returns success with warnings when no `pstMSA.a3m` files are present.
- Partial/OOM-style MSA outcomes may complete with warnings rather than hard failure.

## API warning semantics

`run_pipeline_workflow(...)` returns `PipelineWorkflowResult` with:

- `success`: `True` for successful completion, including warning-success paths.
- `warnings`: tuple of warning messages for soft-failure conditions (for example no generated MSA outputs or no fold inputs).

Callers should inspect `warnings` in addition to `success` when deciding whether to continue downstream automation.
