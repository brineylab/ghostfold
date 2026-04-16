# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pip install -e ".[dev]"          # editable install with pytest + ruff
pytest                            # full test suite
pytest tests/test_cli.py -q      # focused test file
ruff check src tests              # lint
ghostfold --help                  # inspect CLI
python -m build                   # build distributions
```

CI runs pytest on Python 3.10/3.11/3.12 with `--tb=short -q`.

## Architecture

GhostFold generates **synthetic MSAs from single sequences** (no database search) via ProstT5, then runs structure prediction through local ColabFold.

### Pipeline flow

```
CLI (typer) → core/pipeline.py::run_pipeline()
  → msa/generation.py   # ProstT5: AA→3Di→AA round-trip to generate pseudosequences
  → msa/filters.py      # filter generated sequences
  → mutator/            # optional MSA evolution (BLOSUM62, PAM250, MEGABLAST matrices)
  → viz/plotting.py     # coverage heatmaps, coevolution plots
  → core/colabfold.py   # run local ColabFold on resulting .a3m files
```

### Key modules

| Path | Purpose |
|------|---------|
| `src/ghostfold/cli/app.py` | Typer app; subcommands: `msa`, `fold`, `run`, `mask`, `neff` |
| `src/ghostfold/core/pipeline.py` | Top-level orchestration for multi-sequence, multi-run jobs |
| `src/ghostfold/msa/generation.py` | Batched ProstT5 inference, writes `.a3m` files |
| `src/ghostfold/msa/model.py` | ProstT5 model loading (`Rostlab/ProstT5`) |
| `src/ghostfold/core/colabfold.py` | Subprocess wrapper for local ColabFold |
| `src/ghostfold/core/colabfold_env.py` | Detects/sets up ColabFold (pixi or mamba) |
| `src/ghostfold/core/config.py` | Loads/merges YAML config with CLI overrides |
| `src/ghostfold/data/default_config.yaml` | Runtime defaults: `num_return_sequences`, `inference_batch_size`, decoding params |
| `src/ghostfold/__init__.py` | Public API: `run_pipeline`, `read_fasta_from_path`, `mask_a3m_file`, etc. |

### Public API exports

`run_pipeline`, `read_fasta_from_path`, `collect_fasta_paths`, `mask_a3m_file`, `calculate_neff`, `MSA_Mutator`

## Notes

- Structure prediction requires local ColabFold + CUDA PyTorch. Install via `scripts/install_localcolabfold.sh`.
- Use `typer.testing.CliRunner` for CLI tests (see `tests/test_cli.py`).
- `inference_batch_size` in config controls OOM behavior on large sequences.
- Rich is used for all progress/logging output (`core/logging.py`).
