# GhostFold

**Accurate, database-free protein folding from single sequences using structure-aware synthetic MSAs**

---

## Overview

**GhostFold** is a next-generation protein folding framework that predicts 3D structures directly from single sequences — without relying on large evolutionary databases. By generating **synthetic, structure-aware multiple sequence alignments (MSAs)**, GhostFold achieves high accuracy while remaining lightweight and portable.

---

## Installation

### 1. Install PyTorch with CUDA

GhostFold requires PyTorch with CUDA support. Install the appropriate version for your system **before** installing GhostFold:

```bash
# Example for CUDA 12.1 (adjust for your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Refer to the [PyTorch installation guide](https://pytorch.org/get-started/locally/) for platform-specific instructions.

### 2. Install GhostFold

```bash
pip install ghostfold
```

For development:

```bash
git clone https://github.com/brineylab/ghostfold.git
cd ghostfold
pip install -e ".[dev]"
```

### 3. Install localcolabfold (required for `ghostfold run` and `ghostfold fold`)

`ghostfold run` and `ghostfold fold` require a working local ColabFold runtime.
From the GhostFold repository root:

```bash
chmod +x scripts/install_localcolabfold.sh
./scripts/install_localcolabfold.sh
```

---

## Hugging Face Authentication

GhostFold uses ProstT5 from the Hugging Face Hub. You may need to configure a Hugging Face access token:

```bash
huggingface-cli login
```

See the [Hugging Face documentation](https://huggingface.co/docs/hub/security-tokens) for details.

---

## CLI Usage

GhostFold provides a single command-line tool with five subcommands:

### Generate pseudoMSAs

```bash
ghostfold msa --project-name my_project --fasta-file query.fasta
```

Options:
- `--config PATH` — Custom YAML config (overrides bundled defaults)
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
- `--localcolabfold-dir PATH` — Path to localcolabfold pixi checkout (default: `./localcolabfold`)
- `--colabfold-env TEXT` — Legacy mamba env name for ColabFold fallback (default: `colabfold`)

### Full pipeline (MSA + folding)

```bash
ghostfold run --project-name my_project --fasta-file query.fasta
```

Combines all options from `msa` and `fold` commands.

Both `ghostfold run` and `ghostfold fold` perform a ColabFold preflight check before
starting work. If ColabFold is not installed or not functional, the command exits
with setup instructions. Runtime resolution is pixi-first (localcolabfold), with
legacy mamba fallback when available.

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
from ghostfold import run_pipeline, mask_a3m_file, calculate_neff, MSA_Mutator
from ghostfold.core.config import load_config

# Load config with optional overrides
config = load_config("my_config.yaml")

# Run MSA generation pipeline
run_pipeline(
    project="my_project",
    query_fasta="query.fasta",
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

## Local ColabFold Setup

GhostFold supports both a pixi-based localcolabfold runtime and a legacy mamba runtime.
The bundled `scripts/install_localcolabfold.sh` installer sets up the legacy mamba runtime.

- Localcolabfold upstream setup reference: [localcolabfold README](https://github.com/YoshitakaMo/localcolabfold/blob/main/README.md)
- Pixi installation instructions (for pixi-based runtime): [Pixi installation instructions](https://pixi.prefix.dev/latest/installation/)
- Mamba installation guide (required by the bundled installer script): [Mamba installation guide](https://mamba.readthedocs.io/en/stable/installation/mamba-installation.html)

Bundled installer script:

```bash
chmod +x scripts/install_localcolabfold.sh
./scripts/install_localcolabfold.sh
```

### Troubleshooting ColabFold setup

- If you plan to use a pixi-based localcolabfold checkout, install pixi first: [Pixi installation instructions](https://pixi.prefix.dev/latest/installation/)
- If you use the bundled installer script, install mamba first: [Mamba installation guide](https://mamba.readthedocs.io/en/stable/installation/mamba-installation.html)
- If `ghostfold run` or `ghostfold fold` reports ColabFold is not functional, rerun:

```bash
bash scripts/install_localcolabfold.sh
```

If you prefer cloud-based prediction, you can use the generated pseudoMSAs directly in [ColabFold](https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb) by selecting **"custom_msa"** under MSA settings.

---

## References

* [ProstT5: Protein Language Modeling](https://github.com/mheinzinger/ProstT5?tab=readme-ov-file)
* [ColabFold: AlphaFold Simplified](https://github.com/sokrypton/ColabFold)
