#!/bin/bash -e

ENV_NAME="colabfold"

type wget >/dev/null 2>&1 || {
    echo "Error: wget is not installed. Please install it using your system's package manager."
    exit 1
}

if type mamba >/dev/null 2>&1; then
    CONDA_RUNNER="mamba"
elif type micromamba >/dev/null 2>&1; then
    CONDA_RUNNER="micromamba"
else
    echo "Error: neither mamba nor micromamba is installed or available on PATH."
    exit 1
fi

CURRENTPATH=$(pwd)
COLABFOLDDIR="${CURRENTPATH}/localcolabfold"

echo "Using ${CONDA_RUNNER} to manage the '${ENV_NAME}' environment."
echo "ColabFold data will be stored in ${COLABFOLDDIR}"
mkdir -p "${COLABFOLDDIR}"

echo "Creating conda environment named '${ENV_NAME}'..."
"${CONDA_RUNNER}" create -n "${ENV_NAME}" -c conda-forge python=3.10 -y

echo "Installing dependencies into '${ENV_NAME}'..."
"${CONDA_RUNNER}" install -n "${ENV_NAME}" -c conda-forge -c bioconda \
    git openmm==8.2.0 pdbfixer \
    kalign2=2.04 hhsuite=3.3.0 mmseqs2 -y

echo "Installing ColabFold and its Python dependencies..."
"${CONDA_RUNNER}" run -n "${ENV_NAME}" pip install --no-warn-conflicts \
    "colabfold[alphafold] @ git+https://github.com/sokrypton/ColabFold"

"${CONDA_RUNNER}" run -n "${ENV_NAME}" pip install --upgrade \
    "jax[cuda12]==0.5.3" \
    tensorflow \
    silence_tensorflow

echo "Downloading updater script..."
wget -qnc -O "$COLABFOLDDIR/update_linux.sh" \
    https://raw.githubusercontent.com/YoshitakaMo/localcolabfold/main/update_linux.sh
chmod +x "$COLABFOLDDIR/update_linux.sh"

echo "Downloading AlphaFold2 weights (this may take a while)..."
XDG_CACHE_HOME="${COLABFOLDDIR}" MPLBACKEND=Agg \
    "${CONDA_RUNNER}" run -n "${ENV_NAME}" python -m colabfold.download
echo "Download of AlphaFold2 weights finished."
echo "-----------------------------------------"
echo "✅ Installation of ColabFold finished successfully."
echo ""
echo "To run ColabFold, you first need to activate the conda environment:"
echo "  ${CONDA_RUNNER} activate ${ENV_NAME}"
echo ""
echo "Weights are stored under: ${COLABFOLDDIR}/colabfold"
echo "Once the environment is active, run 'colabfold_batch --help' for more details."
