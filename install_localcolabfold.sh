#!/bin/bash -e

# ---
# Refactored script to install localcolabfold using an existing mamba installation.
# This version creates a named conda environment for easier management.
# ---

# 1. Configuration
# Define the name for the conda environment.
ENV_NAME="colabfold"

# 2. Dependency checks
# Ensure wget and mamba are available in the system's PATH.
type wget 2>/dev/null || { echo "Error: wget is not installed. Please install it using your system's package manager." ; exit 1 ; }
type mamba 2>/dev/null || { echo "Error: mamba is not installed or not in your PATH. Please install mamba first." ; exit 1 ; }

# 3. Setup paths and data directory
# The COLABFOLDDIR will store weights and other data, while the environment
# is managed separately by mamba.
CURRENTPATH=$(pwd)
COLABFOLDDIR="${CURRENTPATH}/localcolabfold"

echo "ColabFold data will be stored in ${COLABFOLDDIR}"
mkdir -p "${COLABFOLDDIR}"

# 4. Create the named environment using mamba
# We create the environment first, then install packages into it.
# This is more robust than installing everything in one command.
echo "Creating conda environment named '${ENV_NAME}'..."
mamba create -n "${ENV_NAME}" -c conda-forge python=3.10 -y

echo "Installing dependencies into '${ENV_NAME}'..."
mamba install -n "${ENV_NAME}" -c conda-forge -c bioconda \
    git openmm==8.2.0 pdbfixer \
    kalign2=2.04 hhsuite=3.3.0 mmseqs2 -y

# 5. Install ColabFold and Python dependencies using pip
# Use `mamba run` to execute commands within the correct environment
# without needing to activate it in the script's shell.
echo "Installing ColabFold and its Python dependencies..."

# Install the latest version of ColabFold from GitHub with the [alphafold] extra.
mamba run -n "${ENV_NAME}" pip install --no-warn-conflicts \
    "colabfold[alphafold] @ git+https://github.com/sokrypton/ColabFold"

# Upgrade JAX to a specific version with CUDA 12 support and install other helpers.
mamba run -n "${ENV_NAME}" pip install --upgrade \
    "jax[cuda12]==0.5.3" \
    tensorflow \
    silence_tensorflow

# 6. Download the updater script
echo "Downloading updater script..."
wget -qnc -O "$COLABFOLDDIR/update_linux.sh" \
    https://raw.githubusercontent.com/YoshitakaMo/localcolabfold/main/update_linux.sh
chmod +x "$COLABFOLDDIR/update_linux.sh"

# 7. Apply patches to the ColabFold source code
# These modifications are necessary for running in a non-interactive environment
# and for ensuring data and cache are stored locally.
echo "Applying patches to ColabFold source..."

# Dynamically find the site-packages directory for the colabfold installation.
# This is more robust than hardcoding the path.
COLABFOLD_PKG_PATH=$(mamba run -n "${ENV_NAME}" python -c "import colabfold; print(colabfold.__path__[0])")

pushd "${COLABFOLD_PKG_PATH}" > /dev/null

# Use 'Agg' for non-GUI matplotlib backend
sed -i -e "s#from matplotlib import pyplot as plt#import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt#g" plot.py
# Modify the default params directory to be inside our installation folder
sed -i -e "s#appdirs.user_cache_dir(__package__ or \"colabfold\")#\"${COLABFOLDDIR}/colabfold\"#g" download.py
# Suppress excessive TensorFlow warnings
sed -i -e "s#from io import StringIO#from io import StringIO\nfrom silence_tensorflow import silence_tensorflow\nsilence_tensorflow()#g" batch.py
# Remove __pycache__ directory in case it was created
rm -rf __pycache__

popd > /dev/null

# 8. Download AlphaFold2 model weights
echo "Downloading AlphaFold2 weights (this may take a while)..."
mamba run -n "${ENV_NAME}" python -m colabfold.download
echo "Download of AlphaFold2 weights finished."
echo "-----------------------------------------"
echo "✅ Installation of ColabFold finished successfully."
echo ""
echo "To run ColabFold, you first need to activate the conda environment:"
echo "  mamba activate ${ENV_NAME}"
echo ""
echo "Once the environment is active, run 'colabfold_batch --help' for more details."
