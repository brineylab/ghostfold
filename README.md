![](https://img.shields.io/pypi/v/ghostfold.svg?colorB=blue)
[![tests](https://github.com/brineylab/ghostfold/actions/workflows/pytest.yaml/badge.svg)](https://github.com/brineylab/ghostfold/actions/workflows/pytest.yaml)
![](https://img.shields.io/badge/license-MIT-blue.svg)

# GhostFold

**Accurate, database-free protein folding from single sequences using structure-aware synthetic MSAs**

---

## Overview

**GhostFold** is a next-generation protein folding framework that predicts 3D structures directly from single sequences — without relying on large evolutionary databases. By generating **synthetic, structure-aware multiple sequence alignments (MSAs)**, GhostFold achieves high accuracy while remaining lightweight and portable.

---

## Installation

```bash
pip install ghostfold
ghostfold setup
```

`ghostfold setup` downloads and installs:
- **pixi** (environment manager, ~5 MB) — no root required
- **ColabFold + AlphaFold2 weights** (~3.5 GB) — into `./localcolabfold`
- **ProstT5** (~3 GB) — cached via HuggingFace

Total download: ~7 GB. Estimated time: 15–30 min on a fast connection.

> **PyTorch + CUDA:** GhostFold requires PyTorch with CUDA 12.x support. Install the appropriate version for your system **before** running `ghostfold setup`:
>
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu128
> ```
>
> See the [PyTorch installation guide](https://pytorch.org/get-started/locally/) for your specific CUDA version.

### Options

```bash
ghostfold setup --colabfold-dir /path/to/dir   # custom install location
ghostfold setup --skip-weights                  # skip AF2 weight download
ghostfold setup --hf-token YOUR_TOKEN          # HuggingFace token for ProstT5
```

### HuggingFace Authentication

If ProstT5 download fails with an authentication error, either pass `--hf-token` to `ghostfold setup` or run:

```bash
huggingface-cli login
```

<details>
<summary>Advanced: manual install with mamba/micromamba</summary>

If you prefer to manage the ColabFold environment yourself:

```bash
chmod +x scripts/install_localcolabfold.sh
./scripts/install_localcolabfold.sh
```

Requires mamba or micromamba on PATH.
</details>

---

## CLI Usage

GhostFold provides a single command-line tool with five subcommands:

### Full pipeline (pseudoMSA + structure prediction)

```bash
ghostfold run --project-name my_project --fasta-path query.fasta
```

Combines all options from the `msa` and `fold` commands (see below).

### Generate pseudoMSAs

```bash
ghostfold msa --project-name my_project --fasta-path query.fasta
```

Options:
- `--config PATH` — Custom YAML config (overrides bundled defaults)
- `--recursive` — Recursively search directories for FASTA files
- `--coverage FLOAT` — Coverage values (repeatable, default: 1.0)
- `--num-runs INT` — Independent runs per sequence (default: 1)
- `--evolve-msa` — Enable MSA evolution with substitution matrices
- `--mutation-rates JSON` — Mutation rates per matrix
- `--sample-percentage FLOAT` — Fraction of sequences to evolve (default: 1.0)
- `--plot-msa-coverage` — Generate MSA coverage heatmaps
- `--no-coevolution-maps` — Skip coevolution map generation

### Run structure prediction

```bash
ghostfold fold --project-name my_project
```

Options:
- `--subsample` — Enable MSA subsampling (multiple depth levels)
- `--mask-fraction FLOAT` — Mask a fraction of MSA residues (0.0-1.0)
- `--num-gpus INT` — Override auto-detected GPU count
- `--num-models INT` — Number of AlphaFold2 models to run (default: 5)
- `--num-seeds INT` — Seeds per model (default: 5); total predictions = models × seeds
- `--num-recycles INT` — Recycling iterations per prediction (default: 10)
- `--localcolabfold-dir PATH` — Path to localcolabfold pixi checkout (default: `./localcolabfold`)
- `--colabfold-env TEXT` — Legacy mamba env name for ColabFold fallback (default: `colabfold`)


### Mask MSA files

```bash
ghostfold mask --input-path input.a3m --output-path masked.a3m --mask-fraction 0.15
```

### Calculate Neff scores

```bash
ghostfold neff my_project/
```

### Version

```bash
ghostfold --version
```

---

## Python API

GhostFold can also be used as a Python library:

```python
from ghostfold import run_pipeline, calculate_neff, run_neff_calculation_in_parallel
from ghostfold.core.config import load_config

# Load config with optional overrides
config = load_config("my_config.yaml")

# Run pseudoMSA generation pipeline
run_pipeline(
    project="my_project",
    fasta_path="query.fasta",
    config=config,
    coverage_list=[1.0],
    evolve_msa=True,
    mutation_rates_str='{"MEGABLAST": 5, "PAM250": 20, "BLOSUM62": 10}',
    sample_percentage=1.0,
    plot_msa=False,
    plot_coevolution=False,
)
```

---

## References

* [ProstT5: Protein Language Modeling](https://github.com/mheinzinger/ProstT5?tab=readme-ov-file)
* [ColabFold: AlphaFold Simplified](https://github.com/sokrypton/ColabFold)
