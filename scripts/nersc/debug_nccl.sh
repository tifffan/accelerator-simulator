#!/bin/bash
#SBATCH -A m669
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 0:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --output=logs/debug_nccl_%j.out
#SBATCH --error=logs/debug_nccl_%j.err


# =============================================================================
# Environment Configuration
# =============================================================================

# Bind CPUs to cores for optimal performance
export SLURM_CPU_BIND="cores"

# Load necessary modules
module load conda
module load cudatoolkit
module load pytorch/2.3.1

# Activate the conda environment
source activate ignn

# Set the PYTHONPATH to include your project directory
export PYTHONPATH=/global/homes/t/tiffan/repo/accelerator-simulator
echo "PYTHONPATH is set to: $PYTHONPATH"

# ----------------------
# Set NCCL Debug Variables
# ----------------------
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_TIMEOUT=600    # Timeout in seconds (adjust as needed)
# Optionally, set the network interface if needed (e.g., eth0)
export NCCL_SOCKET_IFNAME=eth0

# =============================================================================
# Setup and Debugging
# =============================================================================

# Navigate to the project directory
cd /global/homes/t/tiffan/repo/accelerator-simulator

python -c "import torch; print(torch.version.cuda)"

# Record the start time
start_time=$(date +%s)
echo "Start time: $(date)"
