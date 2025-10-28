# GhostFold

**Accurate, database-free protein folding from single sequences using structure-aware synthetic MSAs**

---

## Overview

**GhostFold** is a next-generation protein folding framework that predicts 3D structures directly from single sequences — without relying on large evolutionary databases. By generating **synthetic, structure-aware multiple sequence alignments (MSAs)**, GhostFold achieves high accuracy while remaining lightweight and portable.

This repository provides scripts and configurations to set up GhostFold locally using **Mamba** and **ColabFold** environments, along with integration support for **Hugging Face Transformers**.

---

## Installation

### 1. Install Mamba (if not already installed)

GhostFold uses **Mamba** for virtual environment management due to its speed and reproducibility advantages over Conda.

You can install Mamba using one of the following methods:

#### **Using Conda (recommended if Conda is already installed)**

```bash
conda install -n base -c conda-forge mamba
```

#### **Using Miniforge (standalone installation)**

```bash
# For Linux or macOS (ARM/x86)
wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh
bash Mambaforge-Linux-x86_64.sh
```

Follow the on-screen prompts to complete installation, then restart your terminal or run:

```bash
source ~/.bashrc
```

---

### 2. Clone the repository

```bash
git clone https://github.com/brineylab/ghostfold.git
cd ghostfold
```

---

### 3. Create and activate a new Mamba environment

```bash
mamba create -n ghostfold python=3.10
mamba activate ghostfold
```

If you receive an error indicating Mamba isn’t initialized, activate it manually:

```bash
source ~/.bashrc
```

---

### 4. Install dependencies

GhostFold relies on the following core libraries:

```
torch
transformers
sentencepiece
```

Install them using:

```bash
mamba install pytorch torchvision torchaudio -c pytorch
mamba install transformers sentencepiece -c conda-forge
```

Install the appropriate CUDA drivers for PyTorch.

Refer to the [Transformers Installation Guide](https://huggingface.co/docs/transformers/installation) for platform-specific setup details.

---

## Hugging Face Authentication

When using `from_pretrained()` to load models, GhostFold automatically fetches pretrained weights from the Hugging Face Hub.
You may need to create and configure a Hugging Face access token. See the instructions [here](https://huggingface.co/docs/hub/security-tokens).

```bash
huggingface-cli login
```

---

## Local ColabFold Setup

To enable local structure prediction, first ensure the setup scripts are executable:

```bash
chmod +x install_localcolabfold.sh ghostfold.sh
```

Then run the installation script:

```bash
./install_localcolabfold
```

This script will:

* Configure a compatible ColabFold environment
* Install all required dependencies
* Download model weights automatically

If you prefer to run predictions via **Google Colab**, you can use the generated **pseudoMSAs** directly in [ColabFold](https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb) by selecting **“custom_msa”** under [MSA settings](https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb#scrollTo=C2_sh2uAonJH).

---

## Running GhostFold

Once setup is complete, ensure all scripts are executable:

```bash
chmod +x ghostfold.sh
```

Then, launch a prediction using:

```bash
./ghostfold.sh --project_name <your_project_name> --fasta_file <path/to/your/fasta_file.fasta>
```

### Example run

GhostFold includes a sample FASTA file for demonstration:

```bash
./ghostfold.sh --project_name 7JJV --fasta_file query.fasta
```

---

## Modes of Operation

GhostFold can operate in two primary modes:

1. **MSA Generation Mode** – Generates structure-aware synthetic MSAs only.
2. **Full Mode** – Runs both synthetic MSA generation **and** structure prediction.

You may run folding separately after generating MSAs. To explore the available options:

```bash
./ghostfold.sh --help
```

---

## References

* [ProstT5: Protein Language Modeling](https://github.com/mheinzinger/ProstT5?tab=readme-ov-file)
* [ColabFold: AlphaFold Simplified](https://github.com/sokrypton/ColabFold)
