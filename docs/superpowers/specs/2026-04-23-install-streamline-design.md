# Design: Streamlined Installation via `ghostfold setup`

**Date:** 2026-04-23  
**Status:** Approved

---

## Problem

Users currently face three manual install steps:

1. `pip install ghostfold`
2. ProstT5 (~3GB) downloads silently on first `ghostfold msa` run
3. ColabFold requires mamba/micromamba as a prerequisite, then `./scripts/install_localcolabfold.sh` run from repo root + AF2 weights download (~3.5GB)

Target users are computational biologists / wet-lab adjacent on Linux + CUDA who may not know conda. Goal: reduce to two commands with full guidance.

---

## Target Install Flow

```bash
pip install ghostfold
ghostfold setup
```

---

## Architecture

### New files

| Path | Purpose |
|------|---------|
| `src/ghostfold/core/setup.py` | Orchestration logic: pixi bootstrap, ColabFold env, AF2 weights, ProstT5 cache |
| `src/ghostfold/cli/setup.py` | `ghostfold setup` Typer subcommand |

### Unchanged

- `src/ghostfold/core/colabfold_env.py` — `ensure_colabfold_ready()` unchanged; pipeline runtime detection untouched
- `scripts/install_localcolabfold.sh` — kept for mamba users; setup.py is additive

---

## `ghostfold setup` CLI

### User-facing output

```
[1/4] Checking pixi...         ✅ found at /home/user/.pixi/bin/pixi
[2/4] Installing ColabFold...  ⠸ creating pixi environment (this may take 10-15 min)
[3/4] Downloading AF2 weights... ⠸ ~3.5 GB (estimated 5-10 min on fast connection)
[4/4] Downloading ProstT5...   ⠸ ~3 GB
      ✅ Setup complete. Run: ghostfold run --help
```

### Flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--colabfold-dir PATH` | `./localcolabfold` | Custom ColabFold install path |
| `--skip-weights` | false | Skip AF2 weight download |
| `--hf-token TOKEN` | None | HuggingFace token for ProstT5 |

---

## `src/ghostfold/core/setup.py` — Four idempotent functions

### `ensure_pixi() -> str`

- Checks `shutil.which("pixi")`
- If missing: downloads pixi installer (curl/wget) to tempfile, runs with `--no-modify-path`, adds `~/.pixi/bin` to `os.environ["PATH"]`
- Prints note to add `~/.pixi/bin` to `~/.bashrc`
- Returns pixi binary path

### `ensure_colabfold_env(colabfold_dir: Path) -> None`

- Checks `colabfold_dir/pixi.toml` exists and `pixi run colabfold_batch --help` passes
- If not: writes `pixi.toml` into `colabfold_dir` with pinned deps matching current `install_localcolabfold.sh`:
  - JAX 0.5.3 (cuda12)
  - colabfold from `git+https://github.com/sokrypton/ColabFold`
  - openmm 8.2.0, pdbfixer, kalign2, hhsuite, mmseqs2
  - tensorflow, silence_tensorflow
- Runs `pixi install` in `colabfold_dir`

### `ensure_af2_weights(colabfold_dir: Path) -> None`

- Checks `colabfold_dir/colabfold/params/` exists and non-empty
- If not: runs `pixi run python -m colabfold.download` in `colabfold_dir`
- Resumable — safe to re-run after interruption

### `ensure_prostt5(hf_token: str | None) -> None`

- Calls `T5Tokenizer.from_pretrained("Rostlab/ProstT5", ...)` + `AutoModelForSeq2SeqLM.from_pretrained(...)`
- Loads to CPU only (no GPU needed for cache warm)
- HuggingFace caches automatically; subsequent calls are no-ops

### `run_setup(colabfold_dir, skip_weights, hf_token)` — top-level

Calls all four in order, wraps each in try/except with Rich status updates.

---

## Error Handling

| Failure | Message |
|---------|---------|
| Pixi download fails | "pixi installation failed. Install manually: https://pixi.prefix.dev" |
| `pixi install` fails | Surface raw stderr + "Check CUDA 12.x available, ~10GB free disk space" |
| AF2 weight interrupted | Re-run `ghostfold setup` — colabfold.download is resumable |
| ProstT5 HF auth error | "Run `huggingface-cli login` or pass `--hf-token TOKEN`" |
| `ghostfold run` before setup | Append to existing `ColabFoldSetupError`: "Run `ghostfold setup` to complete installation." |
| Custom `--colabfold-dir` mismatch | Documented — user must pass same dir to both `setup` and `run` |

All `ensure_*` functions check state before acting. Re-running after partial failure is safe.

---

## Dependency isolation

ProstT5 (PyTorch) and ColabFold/AF2 (JAX) never share a Python runtime:

- **ghostfold pip env** — PyTorch + ProstT5
- **pixi colabfold env** — JAX + AF2 + colabfold_batch (subprocess only)

`ensure_colabfold_ready()` already handles this via `ColabFoldLauncher` subprocess prefix. No changes needed to pipeline runtime.

---

## README update

Replace current three-step install section with:

```bash
pip install ghostfold
ghostfold setup  # downloads ColabFold, AF2 weights, and ProstT5 (~7GB total, ~15-25 min)
```

Keep mamba/manual path as a collapsible "Advanced" section.
