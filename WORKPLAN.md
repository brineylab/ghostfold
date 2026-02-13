# Plan

Refactor GhostFold from a script-first repository into a modern, pip-installable Python package using `pyproject.toml` + Hatchling, while preserving current behavior and outputs. The implementation will separate stable library APIs from CLI/process orchestration, then replace `ghostfold.sh` with a Typer CLI that supports identical modes (`full`, `msa-only`, `fold-only`) and options (`--subsample`, `--mask_msa`, project/fasta inputs). The rollout will prioritize parity, test coverage, and packaging correctness so `pip install ghostfold` installs all required runtime dependencies for the supported platform.

## Scope
- In: package restructuring to `src/` layout; Hatchling packaging metadata; runtime dependencies in `pyproject.toml`; public Python API for core GhostFold workflows; Typer CLI parity with current shell workflow; process orchestration for ColabFold; tests; docs; migration/deprecation path for existing shell entrypoint.
- Out: model architecture or algorithm redesign; performance tuning unrelated to parity; non-Python installers (Conda-only workflows) beyond optional documentation; cloud/distributed execution changes.

## Phase 0 results

### Baseline entrypoints and responsibilities
| Entrypoint | Current invocation | Responsibility | Notes to preserve |
| --- | --- | --- | --- |
| `ghostfold.sh` | `./ghostfold.sh --project_name <name> --fasta_file <path> [--msa-only\|--fold-only] [--subsample] [--mask_msa <frac>]` | Top-level orchestration for MSA generation, optional masking, ColabFold folding, cleanup, and zip packaging | Canonical user workflow today; new Typer CLI must be behavior-compatible |
| `pseudomsa.py` | `python pseudomsa.py --project_name ... --fasta_file ... --config config.yaml ...` | Core synthetic MSA generation pipeline around ProstT5 + filtering + optional mutation | Called by `ghostfold.sh`; contains core logic to move into Python package API |
| `mask_msa.py` | `python mask_msa.py --input_path ... --output_path ... --mask_fraction ...` | Masks non-query sequences in A3M; preserves first entry unchanged | Invoked only when `ghostfold.sh --mask_msa` is set |
| `calculate_neff.py` | `python calculate_neff.py <project_dir>` | Parallel Neff/NAD calculation over `msa/*/*.a3m`, writes `neff_results.csv` | Utility behavior must stay available via API + CLI subcommand |

### CLI parity matrix for the new Typer interface
| Current behavior | Required inputs | Output and side effects | Exit behavior |
| --- | --- | --- | --- |
| Full mode (default) runs MSA then fold | `--project_name`, `--fasta_file` | Creates `<project>/msa/...`, then `<project>/subsample_1/...` (or subsample levels 1-4), zip archives, and best PDB copies | Fails fast on validation or command errors (`set -e`) |
| `--msa-only` | `--project_name`, `--fasta_file` | Only MSA generation and MSA post-processing (`pstMSA.fasta` -> `pstMSA.a3m`) | Fails on validation or subprocess/script errors |
| `--fold-only` | `--project_name` | Only folding stage from existing `<project>/msa/*/pstMSA.a3m` | If no A3M files are found, prints warning and returns success |
| `--subsample` | Optional boolean flag for fold/full | Uses `(max_seq,max_extra_seq)` presets: `(16,32)`, `(32,64)`, `(64,128)`, `(128,256)` | Without this flag, uses only `(32,64)` |
| `--mask_msa <frac>` | Float-like string accepted by shell regex | Creates temporary `*_masked_temp.a3m` files, uses them for fold jobs, then removes temp files | Validation currently accepts `0`, `0.x`, `1.0`; rejects `1` |

### Current filesystem output contract (must remain stable)
| Path pattern | Producer | Contract details |
| --- | --- | --- |
| `<project>/msa/<safe_header>/run_<n>/unfiltered.fasta` | `pseudomsa.py` | Raw generated sequences for each run |
| `<project>/msa/<safe_header>/run_<n>/filtered.fasta` | `pseudomsa.py` | Filtered sequences for each run |
| `<project>/msa/<safe_header>/run_<n>/filtered_evolved.fasta` | `pseudomsa.py` (when enabled) | Evolved sequences sampled from filtered set |
| `<project>/msa/<safe_header>/pstMSA.fasta` | `pseudomsa.py` concat | Concatenation of all run-level filtered/evolved outputs |
| `<project>/msa/<safe_header>/pstMSA.a3m` | `ghostfold.sh` post-process | Direct copy of `pstMSA.fasta`; first header rewritten to `<safe_header>` |
| `<project>/subsample_<i>/preds/<safe_header>/...` | ColabFold stage | Raw fold artifacts before and after cleanup |
| `<project>/subsample_<i>/preds/<safe_header>/scores/*.json` | cleanup step | JSON files moved into `scores/` |
| `<project>/subsample_<i>/preds/<safe_header>/imgs/*.png` | cleanup step | PNG files moved into `imgs/` |
| `<project>/subsample_<i>/preds/<safe_header>/recycles/*.r*.pdb` | cleanup step | Recycle PDB files moved into `recycles/` |
| `<project>/subsample_<i>/best/<safe_header>_ghostfold.pdb` | cleanup step | Copy of top-ranked `*rank_001*.pdb` if present |
| `<project>/subsample_<i>.zip` | packaging step | Zip archive of each `subsample_<i>` folder |
| `<project>/neff_results.csv` | `calculate_neff.py` | CSV header currently `pdb,NAD` and values formatted to 2 decimals |

### Current validation, error, and runtime assumptions
- GPU requirement is hard in `ghostfold.sh` for all modes, including `--msa-only`: `nvidia-smi` must exist and report `>0` GPUs before any work starts.
- `mamba` must be available for fold/full modes because ColabFold is invoked via `mamba run -n colabfold ... colabfold_batch`.
- `run_colabfold` returns success when no `pstMSA.a3m` files exist; this is a warning state, not a failure.
- MSA generation dispatches one `pseudomsa.py` process per FASTA record when multi-sequence input is provided, with round-robin `CUDA_VISIBLE_DEVICES`.
- Fold jobs are dispatched in parallel up to `num_gpus` with round-robin `CUDA_VISIBLE_DEVICES`, then synchronized with `wait`.
- `pseudomsa.py` has internal OOM handling and may return early without raising process-level errors in some cases (legacy behavior to preserve unless intentionally changed in later phases).
- `mask_msa.py` preserves query sequence (first FASTA entry) and masks only subsequent entries.

### Target Python module map locked for refactor
| Current unit | Target module | Planned public API |
| --- | --- | --- |
| `ghostfold.sh` argument parsing and mode routing | `src/ghostfold/cli.py` | Typer app with `run`, `msa`, `fold` commands and parity flags |
| `run_parallel_msa` shell function | `src/ghostfold/msa_pipeline.py` | `run_msa_only(config: MSAWorkflowConfig) -> WorkflowResult` |
| `run_colabfold` shell function | `src/ghostfold/folding_pipeline.py` | `run_fold_only(config: FoldWorkflowConfig) -> WorkflowResult` |
| cleanup and zip shell logic | `src/ghostfold/postprocess.py` | `cleanup_colabfold_outputs(...)`, `zip_subsample_outputs(...)` |
| `pseudomsa.py` orchestration (`run_pipeline`) | `src/ghostfold/msa_core.py` | `run_pseudomsa_pipeline(...) -> MSAResult` |
| `mask_msa.py` | `src/ghostfold/masking.py` | `mask_a3m_file(input_path, output_path, mask_fraction)` |
| `calculate_neff.py` | `src/ghostfold/neff.py` | `run_neff_calculation(root_dir)` and helpers |
| shared options and YAML loading | `src/ghostfold/config.py` | Typed dataclasses for CLI/API configs |
| package-facing API | `src/ghostfold/api.py` and `src/ghostfold/__init__.py` | `run_full_pipeline`, `run_msa_only`, `run_fold_only`, `mask_msa_file`, `calculate_neff_for_project` |

## Phase 1 results

### Packaging and layout changes completed
- Added `/Users/bryanbriney/git/ghostfold/pyproject.toml` with PEP 621 metadata, Hatchling build backend, Python floor `>=3.10`, package version `0.1.0`, and console script entrypoint `ghostfold=ghostfold.cli:app`.
- Migrated to `src/` layout by creating `/Users/bryanbriney/git/ghostfold/src/ghostfold` and moving the legacy `pseudomsa` package to `/Users/bryanbriney/git/ghostfold/src/pseudomsa`.
- Added initial package modules: `/Users/bryanbriney/git/ghostfold/src/ghostfold/__init__.py`, `/Users/bryanbriney/git/ghostfold/src/ghostfold/cli.py`, `/Users/bryanbriney/git/ghostfold/src/ghostfold/api.py`, `/Users/bryanbriney/git/ghostfold/src/ghostfold/msa_core.py`, `/Users/bryanbriney/git/ghostfold/src/ghostfold/masking.py`, and `/Users/bryanbriney/git/ghostfold/src/ghostfold/neff.py`.

### Compatibility shims completed
- Replaced root scripts with thin wrappers that forward to package modules while adding `src/` to `sys.path` for in-repo execution: `/Users/bryanbriney/git/ghostfold/pseudomsa.py` -> `ghostfold.msa_core.main`, `/Users/bryanbriney/git/ghostfold/mask_msa.py` -> `ghostfold.masking.main`, and `/Users/bryanbriney/git/ghostfold/calculate_neff.py` -> `ghostfold.neff.main`.
- This preserves existing shell-script-based calls used by `/Users/bryanbriney/git/ghostfold/ghostfold.sh` while relocating implementation code to installable package paths.

### Validation completed
- Confirmed package import/version from `src/` path (`ghostfold.__version__ == 0.1.0`).
- Confirmed Typer entrypoint works via module invocation (`python -m ghostfold.cli --help`, `python -m ghostfold.cli version`).
- Confirmed compatibility wrappers execute (`mask_msa.py --help`, `calculate_neff.py` usage path, `pseudomsa.py --help` after parser fix).
- Build artifact generation was not executed in this environment because `hatchling` is not installed in the active interpreter; wheel/sdist validation remains in Phase 8 packaging smoke tests.

## Phase 2 results

### Dependency metadata changes completed
- Replaced temporary single dependency metadata with full package runtime dependencies in `/Users/bryanbriney/git/ghostfold/pyproject.toml`: `biopython`, `matplotlib`, `numpy`, `pyyaml`, `rich`, `scikit-learn`, `sentencepiece`, `torch`, `transformers`, `typer`.
- Added `fold` extra for Linux-targeted ColabFold installation (`colabfold[alphafold]` from upstream GitHub).
- Added `fold-cuda12` extra for Linux CUDA12 stack alignment (`jax[cuda12]`, `tensorflow`, `silence-tensorflow`).
- Added `test` extra (`pytest`, `pytest-cov`, `pytest-mock`).
- Added `dev` extra (`build`, `mypy`, `ruff`).
- Added `docs` extra (`mkdocs`, `mkdocs-material`).

### Phase 2 decision outcomes
- Chose a split-install strategy.
- Default `pip install ghostfold` now installs the full Python runtime needed by migrated GhostFold MSA/masking/Neff modules.
- Fold-specific heavy or OS-constrained dependencies are isolated in extras to avoid breaking non-Linux or non-CUDA installs.
- Encoded initial platform expectations via Linux markers for fold-related extras in `pyproject.toml`.

### Validation completed
- Verified TOML structure and dependency-group presence via `tomllib` parsing.
- Verified requirement-string syntax for all base and optional dependencies via `packaging.requirements.Requirement`.

### Remaining Phase 2 gaps carried forward
- Packaging install smoke tests (wheel build and clean-env install) are still pending and remain scheduled under Phase 8.
- Non-pip/system dependencies required by full fold execution (`mamba`, `hhsuite`, `mmseqs2`, GPU driver/toolchain binaries) remain an orchestration/runtime concern for later phases.

## Phase 3 results

### Service-layer refactor completed
- Added typed workflow config objects in `/Users/bryanbriney/git/ghostfold/src/ghostfold/config.py`:
- `MSAWorkflowConfig`, `MaskWorkflowConfig`, `NeffWorkflowConfig`.
- Added shared service-layer filesystem validation helpers in `/Users/bryanbriney/git/ghostfold/src/ghostfold/services/common.py`.
- Added workflow services in `/Users/bryanbriney/git/ghostfold/src/ghostfold/services/`:
- `/Users/bryanbriney/git/ghostfold/src/ghostfold/services/msa.py` -> `run_msa_workflow(...)`
- `/Users/bryanbriney/git/ghostfold/src/ghostfold/services/masking.py` -> `run_mask_workflow(...)`
- `/Users/bryanbriney/git/ghostfold/src/ghostfold/services/neff.py` -> `run_neff_workflow(...)`
- Added service exports in `/Users/bryanbriney/git/ghostfold/src/ghostfold/services/__init__.py`.

### Entrypoint unification completed
- Added compatibility entrypoint router `/Users/bryanbriney/git/ghostfold/src/ghostfold/entrypoints.py` so legacy script entrypoints now parse args then delegate into service-layer orchestration.
- Updated root wrappers to call unified entrypoints:
- `/Users/bryanbriney/git/ghostfold/pseudomsa.py` -> `ghostfold.entrypoints.pseudomsa_main`
- `/Users/bryanbriney/git/ghostfold/mask_msa.py` -> `ghostfold.entrypoints.mask_msa_main`
- `/Users/bryanbriney/git/ghostfold/calculate_neff.py` -> `ghostfold.entrypoints.neff_main`

### API/CLI routing updates completed
- Updated `/Users/bryanbriney/git/ghostfold/src/ghostfold/cli.py` so all commands construct typed config objects and invoke service functions.
- Updated `/Users/bryanbriney/git/ghostfold/src/ghostfold/api.py` to expose service-oriented API wrappers and typed config classes while preserving compatibility helpers (`run_pipeline`, `mask_a3m_file`, `run_neff_calculation`).
- Updated legacy module mains (`/Users/bryanbriney/git/ghostfold/src/ghostfold/msa_core.py`, `/Users/bryanbriney/git/ghostfold/src/ghostfold/masking.py`, `/Users/bryanbriney/git/ghostfold/src/ghostfold/neff.py`) to delegate through service-layer orchestration.

### Validation completed
- Confirmed Typer CLI is still functional after refactor (`python -m ghostfold.cli --help`, `python -m ghostfold.cli version` with `PYTHONPATH=src`).
- Confirmed legacy wrappers still parse and dispatch (`pseudomsa.py --help`, `mask_msa.py --help`, `calculate_neff.py` argument validation path).
- Confirmed API imports expose service-layer surface (`ghostfold.api` exports typed configs and workflow helpers).

## Phase 4 results

### Public API contract stabilization completed
- Added explicit public exception hierarchy in `/Users/bryanbriney/git/ghostfold/src/ghostfold/errors.py`:
- `GhostfoldError`, `GhostfoldValidationError`, `GhostfoldIOError`, `GhostfoldExecutionError`.
- Added typed workflow result models in `/Users/bryanbriney/git/ghostfold/src/ghostfold/results.py`:
- `MSAWorkflowResult`, `MaskWorkflowResult`, `NeffWorkflowResult`.
- Updated service-layer contracts to return typed results and raise API-level exceptions:
- `/Users/bryanbriney/git/ghostfold/src/ghostfold/services/msa.py`
- `/Users/bryanbriney/git/ghostfold/src/ghostfold/services/masking.py`
- `/Users/bryanbriney/git/ghostfold/src/ghostfold/services/neff.py`
- Updated shared validation helpers (`/Users/bryanbriney/git/ghostfold/src/ghostfold/services/common.py`) to raise `GhostfoldValidationError`.

### Export curation completed
- Curated public exports in `/Users/bryanbriney/git/ghostfold/src/ghostfold/api.py` to include:
- typed config classes
- typed result classes
- public exception classes
- stable workflow entrypoints and compatibility wrappers.
- Curated package-root exports in `/Users/bryanbriney/git/ghostfold/src/ghostfold/__init__.py` so users can import the stable API surface directly from `ghostfold`.

### Compatibility and behavior updates completed
- Updated workflow wrappers to preserve legacy entrypoint behavior while using the standardized exception/result contracts:
- `/Users/bryanbriney/git/ghostfold/src/ghostfold/entrypoints.py`
- `/Users/bryanbriney/git/ghostfold/src/ghostfold/msa_core.py`
- `/Users/bryanbriney/git/ghostfold/src/ghostfold/masking.py`
- `/Users/bryanbriney/git/ghostfold/src/ghostfold/neff.py`
- Improved mask workflow robustness by normalizing `output_path` inputs to `Path` in `/Users/bryanbriney/git/ghostfold/src/ghostfold/services/masking.py` so typed config callers can pass either `Path` or path-like strings.

### Validation completed
- Verified package-root curated exports and API symbols load correctly (`ghostfold.__version__`, result/config/exception exports).
- Verified service calls now return typed result objects:
- `run_mask_workflow(...)` -> `MaskWorkflowResult`
- `run_neff_workflow(...)` -> `NeffWorkflowResult`
- Verified CLI and script entrypoints remain functional after the contract changes (`ghostfold.cli version`, wrapper help/validation paths).

## Phase 5 results

### Typer mode-routing parity completed
- Extended `/Users/bryanbriney/git/ghostfold/src/ghostfold/cli.py` with shell-parity pipeline commands:
- `run`: supports full/msa-only/fold-only mode routing via flags.
- `full`: explicit full pipeline convenience command.
- `fold`: explicit fold-only convenience command.
- Preserved existing advanced commands (`msa`, `mask`, `neff`) so prior Phase 3/4 functionality remains available.

### Shell-compatible option and validation parity completed
- Added shell-compatible option aliases in Typer for key pipeline args:
- `--project-name` and `--project_name`
- `--fasta-file` and `--fasta_file`
- `--mask-msa` and `--mask_msa`
- Implemented validation behavior matching `ghostfold.sh`:
- Rejects `--msa-only` with `--fold-only` together.
- Requires `--fasta_file` unless `--fold-only`.
- Validates `--mask_msa` with shell-equivalent accepted formats (`0`, `0.x`, `1.0`) and rejects `1`.
- Enforces pre-run GPU detection parity through `nvidia-smi` checks for all run modes.

### Execution routing completed for Phase 5 scope
- Added internal legacy backend runner in `/Users/bryanbriney/git/ghostfold/src/ghostfold/cli.py` that executes `/Users/bryanbriney/git/ghostfold/ghostfold.sh` via `bash` with parity flags.
- This keeps behavior aligned with the existing shell orchestration while Phase 6 ports orchestration into Python services.
- Error reporting from CLI run/full/fold now uses the standardized Phase 4 exception model (`GhostfoldValidationError`, `GhostfoldExecutionError`).

### Validation completed
- Verified command discovery/help now includes `run`, `full`, and `fold`.
- Verified parity validation errors for:
- `--msa-only` + `--fold-only`
- missing `--fasta_file` outside fold-only mode
- invalid `--mask_msa` format (`1`)
- missing `nvidia-smi` path
- Verified existing commands remain callable (`ghostfold.cli msa --help`) and module syntax remains valid (`py_compile`).

## Phase 6 results

### Native Python orchestration completed
- Added `/Users/bryanbriney/git/ghostfold/src/ghostfold/services/pipeline.py` implementing native full/msa/fold orchestration previously handled by `ghostfold.sh`.
- Implemented shell-parity runtime behavior in Python:
- GPU detection via `nvidia-smi` with parity error messages.
- MSA mode orchestration with multi-sequence FASTA splitting and parallel GPU-dispatched `pseudomsa.py` subprocess jobs.
- MSA post-processing parity (`pstMSA.fasta` header normalization and `.a3m` copy generation).
- Fold mode orchestration using `mamba run -n colabfold ... colabfold_batch` per MSA, GPU job dispatch, and subsampling presets.
- Optional temporary MSA masking flow with cleanup of `*_masked_temp.a3m`.
- ColabFold output cleanup parity (`scores/`, `imgs/`, `recycles/`, best-rank copy, done-file deletion).
- Zip packaging parity for each subsample output directory.

### Service contract integration completed
- Added `PipelineWorkflowConfig` in `/Users/bryanbriney/git/ghostfold/src/ghostfold/config.py`.
- Added `PipelineWorkflowResult` in `/Users/bryanbriney/git/ghostfold/src/ghostfold/results.py`.
- Exported pipeline service via `/Users/bryanbriney/git/ghostfold/src/ghostfold/services/__init__.py`.
- Curated new public API exports for pipeline config/result and runner:
- `/Users/bryanbriney/git/ghostfold/src/ghostfold/api.py`
- `/Users/bryanbriney/git/ghostfold/src/ghostfold/__init__.py`

### CLI backend migration completed
- Updated `/Users/bryanbriney/git/ghostfold/src/ghostfold/cli.py` so `run`, `full`, and `fold` now call `run_pipeline_workflow(...)` directly.
- Removed direct `ghostfold.sh` subprocess backend usage from these Typer commands.
- Preserved Phase 5 flag and validation parity semantics while switching execution to native Python service orchestration.

### Validation completed
- Verified syntax integrity for updated modules via `py_compile`.
- Verified CLI command surface and parity validation paths still behave as expected after backend swap (`--msa-only`/`--fold-only` conflict, missing `--fasta_file`, invalid `--mask_msa`, missing `nvidia-smi`).
- Verified package exports include new pipeline API surface (`PipelineWorkflowConfig`, `run_pipeline_workflow`).
- End-to-end full/fold execution was not run in this environment due lack of available `nvidia-smi`/GPU+ColabFold runtime.

## Phase 7 results

### Shell migration shim completed
- Converted `/Users/bryanbriney/git/ghostfold/ghostfold.sh` into a thin compatibility wrapper that routes into the packaged Typer CLI instead of embedding orchestration logic.
- Preserved legacy option-only invocation semantics (`./ghostfold.sh --project_name ... --fasta_file ...`) by forwarding such calls to `python3 -m ghostfold.cli run ...`.
- Added passthrough behavior for explicit CLI subcommands (`run`, `full`, `fold`, etc.) and local-repo `src/` bootstrapping via `PYTHONPATH` for in-place execution.

### Legacy utility CLI compatibility completed
- Added compatibility subcommands to `/Users/bryanbriney/git/ghostfold/src/ghostfold/cli.py`:
- `mask_msa` (legacy script-name alias)
- `calculate_neff` (legacy script-name alias)
- Extended `mask` command options to accept both dashed and underscore forms (`--input-path`/`--input_path`, `--output-path`/`--output_path`, `--mask-fraction`/`--mask_fraction`) for parity with legacy script flags.
- Standardized error handling across all CLI commands to surface consistent `ERROR: ...` messages and exit code `1` on `GhostfoldError`.

### Installed-entrypoint migration shims completed
- Added legacy console entrypoints in `/Users/bryanbriney/git/ghostfold/pyproject.toml` so pip installs expose compatibility commands:
- `pseudomsa`
- `mask_msa`
- `calculate_neff`
- This preserves script-driven workflows while steering primary usage toward the unified `ghostfold` CLI.

### Dependency-extra guidance documentation completed
- Updated `/Users/bryanbriney/git/ghostfold/README.md` with pip-first install guidance and explicit fold extra usage:
- Base install: `pip install .`
- Fold dependencies: `pip install \".[fold]\"`
- Fold + CUDA12 stack: `pip install \".[fold,fold-cuda12]\"`
- Documented Linux-only marker expectations for fold extras and clarified runtime prerequisites (`nvidia-smi`, `mamba`) for fold execution.

### Validation completed
- Verified CLI command discovery now includes compatibility aliases (`mask_msa`, `calculate_neff`).
- Verified `ghostfold.sh --help` and option-forwarding path route into Typer `run` command help.
- Verified updated modules and wrapper syntax via `py_compile`.

## Phase 8 results

### Unit test suite completed
- Added `/Users/bryanbriney/git/ghostfold/tests/test_pipeline_validation.py` covering pipeline validation logic:
- required `--project_name`
- mutual exclusion for `--msa-only` and `--fold-only`
- `--fasta_file` requirement outside fold-only mode
- shell-compatible `--mask_msa` format acceptance/rejection
- Added `/Users/bryanbriney/git/ghostfold/tests/test_masking.py` covering masking behavior and service validation:
- query preservation + non-query masking behavior
- typed result contract for `run_mask_workflow(...)`
- mask-fraction range validation
- Added `/Users/bryanbriney/git/ghostfold/tests/test_neff.py` covering Neff computation and workflow behavior:
- deterministic Neff value check
- invalid-input validation
- CSV generation flow
- no-results path behavior through `run_neff_workflow(...)`

### Native pipeline integration tests completed (mocked subprocess/system)
- Added `/Users/bryanbriney/git/ghostfold/tests/test_pipeline_integration.py` with mocked integration coverage for:
- GPU detection success/failure paths (`nvidia-smi` discovery + subprocess failures)
- ColabFold orchestration with masking lifecycle, output cleanup, best-structure copy, temp-file removal, and zip output creation
- parallel dispatch semantics in `_run_colabfold_jobs(...)` (round-robin `CUDA_VISIBLE_DEVICES`)
- top-level `run_pipeline_workflow(...)` full-mode orchestration ordering and typed result contract

### Compatibility command tests completed
- Added `/Users/bryanbriney/git/ghostfold/tests/test_compatibility_commands.py` covering:
- CLI alias parity for `mask_msa` and `calculate_neff`
- `ghostfold.sh` thin-wrapper routing for option-only mode (auto-inserts `run`)
- `ghostfold.sh` explicit-subcommand passthrough behavior

### Packaging smoke coverage and CI wiring completed
- Added `/Users/bryanbriney/git/ghostfold/tests/test_packaging_smoke.py` with packaging smoke checks for:
- wheel/sdist generation via `python -m build`
- wheel metadata extra exposure (`test`, `fold`, `fold-cuda12`)
- wheel install in clean venv and basic CLI/API load checks
- Packaging smoke tests are env-gated via `GHOSTFOLD_RUN_PACKAGING_SMOKE=1` to keep default local test runs fast.
- Added Linux CI workflow `/Users/bryanbriney/git/ghostfold/.github/workflows/ci.yml` with:
- `.[test]` installation and pytest run
- artifact build and wheel install smoke
- extra resolver checks for `.[test]` and at least one fold path (`.[fold]`) using pip dry-run

### Pytest configuration updates completed
- Added pytest configuration in `/Users/bryanbriney/git/ghostfold/pyproject.toml`:
- `testpaths = ["tests"]`
- custom marker registration for `packaging`

### Validation completed
- Ran `python -m pytest -q`: `25 passed, 3 skipped`.
- Default run skips packaging smoke tests unless `GHOSTFOLD_RUN_PACKAGING_SMOKE=1` is set.
- Attempted `python -m build` and packaging smoke execution in this environment; both were blocked because build isolation could not fetch `hatchling>=1.25.0` due restricted network. CI workflow now covers this path where networked package resolution is available.

## Phase 9 results

### Parity-sensitive behavior decisions completed
- Finalized compatibility decisions for Phase 0 edge cases:
- **Retain** mandatory GPU detection behavior for all pipeline modes (including `--msa-only`) to preserve shell-parity semantics.
- **Retain** warning-success behavior when no `pstMSA.a3m` inputs exist for fold stage (no failure, but explicit warning).
- **Retain** non-fatal pipeline completion path for partial/OOM-like MSA outcomes where no `pstMSA` outputs are produced.
- **Evolve deliberately** by surfacing warnings in structured API output and emitted runtime warnings, instead of silent success.

### Pipeline hardening implementation completed
- Updated `/Users/bryanbriney/git/ghostfold/src/ghostfold/services/pipeline.py`:
- Added explicit warning emission for missing `pstMSA.fasta` outputs after MSA stage.
- Added explicit warning emission for missing `pstMSA.a3m` inputs in fold stage.
- Changed `_run_parallel_msa(...)` to return generated output count for downstream risk-aware messaging.
- Added warning-aware result messaging in full/msa/fold modes while preserving success semantics.
- Updated `/Users/bryanbriney/git/ghostfold/src/ghostfold/results.py`:
- Extended `PipelineWorkflowResult` with `warnings: Tuple[str, ...]` so API callers can programmatically detect soft-failure/partial-output conditions.

### Offline/dependency-resolution failure-mode handling completed
- Updated `/Users/bryanbriney/git/ghostfold/tests/test_packaging_smoke.py`:
- Packaging build helper now detects offline/dependency-resolution failure signatures (for example backend `hatchling` resolution failures) and marks tests as skipped with explicit environment reason instead of opaque failure.
- Added `/Users/bryanbriney/git/ghostfold/tests/test_packaging_failure_modes.py`:
- codifies skip behavior for offline backend-resolution failures
- codifies re-raise behavior for unrelated build failures

### Edge-case regression tests completed
- Expanded `/Users/bryanbriney/git/ghostfold/tests/test_pipeline_integration.py` coverage for:
- mandatory GPU detection in `msa-only` mode
- no-A3M warning-success fold behavior
- full-mode warning-success behavior for missing MSA outputs + missing fold inputs
- warning-free success path when outputs are generated

### Validation completed
- Ran `python -m pytest -q`: `30 passed, 3 skipped`.
- Ran `GHOSTFOLD_RUN_PACKAGING_SMOKE=1 python -m pytest -q tests/test_packaging_smoke.py`: `3 skipped` in this offline-restricted environment (explicitly expected after Phase 9 hardening).
- Verified updated modules/tests compile via `py_compile`.

## Phase 10 results

### Documentation updates completed
- Updated `/Users/bryanbriney/git/ghostfold/README.md` with final pip-first usage guidance including:
- Python API quickstart example using `PipelineWorkflowConfig` + `run_pipeline_workflow(...)`
- explicit warning-handling guidance for `PipelineWorkflowResult.warnings`
- links to migration and release references
- explicit build isolation/network guidance for `python -m build` and offline `--no-isolation` usage
- Added dedicated migration reference `/Users/bryanbriney/git/ghostfold/MIGRATION.md`:
- legacy shell/script command mapping to modern CLI
- compatibility shim inventory
- parity-behavior notes and warning semantics
- Added release process reference `/Users/bryanbriney/git/ghostfold/RELEASING.md`:
- local release checklist (lint/test/build/twine/extras/smoke)
- build-backend/network prerequisites
- tagged release workflow behavior summary

### CI and release gating completed
- Updated `/Users/bryanbriney/git/ghostfold/.github/workflows/ci.yml` to enforce staged gates:
- `lint` job with `ruff` on `src` and `tests`
- `tests` job gated on lint
- `packaging-smoke` job gated on tests
- Added tag-driven release gate workflow `/Users/bryanbriney/git/ghostfold/.github/workflows/release.yml`:
- triggers on `v*` tags and manual dispatch
- enforces lint, tests, build, artifact validation (`twine check`), extra resolver checks, and packaging smoke
- uploads built `dist/` artifacts only when gates pass

### Lint/readiness hardening completed
- Resolved pre-existing lint issues blocking enforceable CI gates in:
- `/Users/bryanbriney/git/ghostfold/src/ghostfold/msa_core.py`
- `/Users/bryanbriney/git/ghostfold/src/pseudomsa/mutator/mutator.py`
- `/Users/bryanbriney/git/ghostfold/src/pseudomsa/utils/filters.py`
- `/Users/bryanbriney/git/ghostfold/src/pseudomsa/utils/generation.py`
- `/Users/bryanbriney/git/ghostfold/src/pseudomsa/utils/plotting.py`

### Validation completed
- Ran `python -m ruff check src tests`: passed.
- Ran `python -m pytest -q`: `30 passed, 3 skipped`.
- Re-ran `python -m build`: still fails in this environment due isolated backend (`hatchling`) resolution under restricted network (expected and now explicitly documented).
- Re-ran `python -m build --no-isolation`: fails here because `hatchling` is not installed in the active interpreter; this prerequisite is now documented in release notes.

## Action items
[x] Phase 0 - Define parity contract and target module map: completed in this document via baseline inventory, parity matrix, filesystem contract, error/runtime assumptions, and target module mapping.
[x] Phase 1 - Create package skeleton and build backend: completed with Hatchling-based `pyproject.toml`, `src/` layout migration, package module scaffolding, CLI entrypoint wiring, and legacy root-script wrappers.
[x] Phase 2 - Normalize dependency management for pip installs: completed by adding full runtime dependencies, optional groups (`fold`, `fold-cuda12`, `test`, `dev`, `docs`), and Linux platform markers for fold-related extras.
[x] Phase 3 - Refactor script logic into import-safe services: completed by adding typed configs, shared service utilities, dedicated workflow services, and unified CLI/API/script routing through that service layer.
[x] Phase 4 - Define public Python API surface: completed with explicit public exceptions, typed result models, stabilized service return/raise contracts, and curated exports via `ghostfold.api` and `ghostfold.__init__`.
[x] Phase 5 - Implement Typer CLI with shell parity: completed by adding `run`/`full`/`fold` parity commands, shell-compatible option aliases, validation semantics matching `ghostfold.sh`, and legacy shell execution routing behind the new CLI interface.
[x] Phase 6 - Replace shell process orchestration with Python execution layer: completed by implementing native pipeline orchestration service for GPU dispatch, ColabFold subprocess jobs, masking temp files, cleanup, and zip packaging, and wiring Typer run/full/fold commands to it.
[x] Phase 7 - Add compatibility and migration shims: completed by replacing `ghostfold.sh` with a Typer wrapper, preserving option-only legacy invocation semantics, adding `mask_msa`/`calculate_neff` compatibility subcommands and option aliases, wiring legacy console entrypoints in package metadata, and documenting fold dependency extras.
[x] Phase 8 - Build validation and test suite: completed by adding a full pytest suite for validation/masking/Neff logic, mocked native pipeline integration tests, compatibility command tests, env-gated packaging smoke tests, pytest marker/testpath configuration, and Linux CI checks for build/wheel install plus `test`/`fold` extra resolver paths.
[x] Phase 9 - Harden for edge cases and operational risks: completed by preserving parity-critical warning-success behaviors (with explicit warning emission), adding structured pipeline warnings in API results, and codifying offline/dependency-resolution packaging failure handling with targeted tests.
[x] Phase 10 - Update docs and release workflow: completed by publishing final pip-first and API warning-semantics docs, adding dedicated migration/release guides, and enforcing lint/test/build/package gates across CI and tag-triggered release workflows with explicit build-network/isolation guidance.

## Open questions
- What minimum CUDA/toolchain matrix must be officially supported for fold parity (for example CUDA 12 only vs broader compatibility)?
- Should the package name remain `ghostfold` while keeping `pseudomsa` import compatibility aliases, or should `pseudomsa` be a separately versioned compatibility package?
- Should legacy quirks (`nvidia-smi` required even for MSA-only, CSV header `NAD`, mask-fraction parser rejecting `1`) be preserved exactly or normalized behind an explicit compatibility mode?
- Should fold-related extras pin ColabFold to a specific commit/tag for reproducibility instead of tracking GitHub HEAD?
