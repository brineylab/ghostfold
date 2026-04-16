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
  → msa/generation.py   # ProstT5: AA→3Di→AA batched round-trip per coverage level
  → msa/filters.py      # filter + vectorized dedup of generated sequences
  → mutator/            # optional MSA evolution (BLOSUM62, PAM250, MEGABLAST matrices)
  → viz/plotting.py     # coverage heatmaps, coevolution plots
  → core/colabfold.py   # run local ColabFold on resulting .a3m files
```

### Key modules

| Path | Purpose |
|------|---------|
| `src/ghostfold/cli/app.py` | Typer app; subcommands: `msa`, `fold`, `run`, `mask`, `neff` |
| `src/ghostfold/cli/fold.py` | `fold` subcommand; exposes `--num-models`, `--num-seeds`, `--num-recycles` |
| `src/ghostfold/core/pipeline.py` | Top-level orchestration; module-level model cache keyed by `"{model}:{device}"` |
| `src/ghostfold/msa/generation.py` | Batched ProstT5 inference; `generate_sequences_for_coverages_batched` batches all coverage levels into one GPU call per decode_conf |
| `src/ghostfold/msa/model.py` | ProstT5 model loading (`Rostlab/ProstT5`) |
| `src/ghostfold/msa/filters.py` | Sequence filtering; `deduplicate` uses vectorized Hamming identity (NumPy) |
| `src/ghostfold/msa/neff.py` | Neff calculation; chunked vectorized pairwise identity (256-row blocks) to cap peak memory |
| `src/ghostfold/core/colabfold.py` | Subprocess wrapper; `_build_colabfold_cmd` merges `extra_colabfold_params` into base params |
| `src/ghostfold/core/colabfold_env.py` | Detects/sets up ColabFold (pixi or mamba) |
| `src/ghostfold/core/config.py` | Loads/merges YAML config with CLI overrides |
| `src/ghostfold/data/default_config.yaml` | Runtime defaults: `num_return_sequences`, `inference_batch_size`, decoding params |
| `src/ghostfold/__init__.py` | Public API: `run_pipeline`, `read_fasta_from_path`, `mask_a3m_file`, etc. |

### Public API exports

`run_pipeline`, `read_fasta_from_path`, `collect_fasta_paths`, `mask_a3m_file`, `calculate_neff`, `MSA_Mutator`

## Coding Style & Naming Conventions

4-space indent, type hints where useful, small focused modules. `snake_case` for functions/variables/modules; `PascalCase` for classes and test groupings like `TestMainApp`. Keep CLI command behavior in `src/ghostfold/cli/` and reusable logic in non-CLI modules. Run `ruff check src tests` before opening a PR.

## Testing Guidelines

Pytest. Name new files `tests/test_<feature>.py`, new tests `test_<behavior>`. Prefer fast unit tests; use `typer.testing.CliRunner` for CLI behavior (see `tests/test_cli.py`). Cover new flags, config changes, and failure paths especially around ColabFold environment handling.

## Commit & Pull Request Guidelines

Short imperative subjects (e.g. `fix CLI test failures in CI`). Keep commits focused. PRs need: concise summary, linked issue if applicable, test evidence (`pytest`, `ruff check`), sample CLI output for user-facing changes.

## Performance Notes

- **Batched MSA generation**: `generate_sequences_for_coverages_batched` replaces per-coverage sequential calls; all coverage chunks batched into one `generate` call per decode_conf for higher GPU utilization.
- **Vectorized dedup**: `deduplicate` uses NumPy Hamming identity matrix; falls back to `SequenceMatcher` for unequal-length inputs.
- **Chunked Neff**: `calculate_neff` computes pairwise identity in 256-row blocks to avoid OOM on large MSAs.
- **Model precision**: bfloat16 on Ampere+ CUDA, float16 on older GPUs. `torch.compile(mode="reduce-overhead")` applied when available.
- **OOM recovery**: `generate_sequences_for_coverages_batched` halves `inference_batch_size` on OOM and retries.

## Notes

- Structure prediction requires local ColabFold + CUDA PyTorch. Install via `scripts/install_localcolabfold.sh`.
- Use `typer.testing.CliRunner` for CLI tests (see `tests/test_cli.py`).
- `inference_batch_size` in config controls OOM behavior on large sequences.
- Rich is used for all progress/logging output (`core/logging.py`).
- Do not hardcode machine-specific paths or tokens; keep secrets out of the repo.
- `fold` subcommand passes `--num-models`, `--num-seeds`, `--num-recycles` as `extra_colabfold_params` to `run_colabfold`; these override defaults in `COLABFOLD_PARAMS`.
