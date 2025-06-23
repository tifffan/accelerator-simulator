#!/bin/bash
#SBATCH --account=ad:default
#SBATCH --partition=ampere
#SBATCH --job-name=test_dataloader_rollout
#SBATCH --output=logs/20250606/test_dataloader_rollout_%j.out
#SBATCH --error=logs/20250606/test_dataloader_rollout_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=0:05:00

# Set the PYTHONPATH to include your project directory
export PYTHONPATH=/sdf/home/t/tiffan/repo/accelerator-simulator

# Print the PYTHONPATH for debugging purposes
echo "PYTHONPATH is set to: $PYTHONPATH"

# Navigate to the project directory
cd /sdf/home/t/tiffan/repo/accelerator-simulator

# Set the graph data directory (update as needed)
export GRAPH_DATA_DIR=/sdf/data/ad/ard/u/tiffan/data/sequence_graph_data_archive_4_test/knn_k5_weighted_graphs

# Print the command for verification
echo "Running test_dataloaders_rollout.py with GRAPH_DATA_DIR=$GRAPH_DATA_DIR"

# Note: The following output will include debug batching information from dataloaders_rollout.py
# Look for the separator: '---- The following lines are debug output from dataloaders_rollout.py batching ----'

# Run the test script
python src/graph_simulators/test_dataloaders_rollout.py 