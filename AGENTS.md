# Repository Guidelines

## Project Structure & Module Organization
GhostFold is a `src`-layout Python package. Core library code lives in `src/ghostfold/`, split by concern: `cli/` for Typer entrypoints, `core/` for pipeline and ColabFold integration, `msa/` for pseudoMSA generation and filtering, `io/` for FASTA handling, `viz/` for plots, and `mutator/` for mutation logic. Tests live in `tests/` and generally mirror module names, for example `tests/test_cli.py` and `tests/test_filters.py`. Runtime defaults are in `src/ghostfold/data/default_config.yaml`; helper scripts such as `scripts/install_localcolabfold.sh` support local setup.

## Build, Test, and Development Commands
Use Python 3.10+.

- `pip install -e ".[dev]"` installs GhostFold in editable mode with pytest and Ruff.
- `pytest` runs the full test suite.
- `pytest tests/test_cli.py -q` runs a focused test file during CLI work.
- `ruff check src tests` lints the package and tests.
- `python -m ghostfold.cli.app --help` or `ghostfold --help` inspects the CLI locally.
- `python -m build` builds source and wheel distributions if `build` is installed.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, type hints where useful, and small focused modules. Use `snake_case` for functions, variables, and module names; use `PascalCase` for classes and test groupings like `TestMainApp`. Keep CLI command behavior in `src/ghostfold/cli/` and reusable logic in non-CLI modules. Run Ruff before opening a PR.

## Testing Guidelines
Pytest is the test framework. Name new files `tests/test_<feature>.py` and new tests `test_<behavior>`. Prefer fast unit tests over end-to-end shelling out unless CLI behavior is the target; `typer.testing.CliRunner` is already used in `tests/test_cli.py`. Cover new flags, config changes, and failure paths, especially around ColabFold environment handling.

## Commit & Pull Request Guidelines
Recent history favors short, imperative commit subjects such as `Fix CLI test failures in CI` and `bump version to v0.1.3`. Keep commits focused and descriptive. Pull requests should include a concise summary, linked issue if applicable, test evidence (`pytest`, `ruff check`), and screenshots or sample CLI output when user-facing behavior changes.

## Environment & Configuration Notes
Structure prediction commands depend on a working local ColabFold install and CUDA-enabled PyTorch. Do not hardcode machine-specific paths or tokens; keep secrets out of the repo and document required environment setup in the PR when changing install or authentication flows.
