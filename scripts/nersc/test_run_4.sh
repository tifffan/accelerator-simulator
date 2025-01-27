#!/bin/bash
#SBATCH -A m669
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --output=logs/train_scgn_%j.out
#SBATCH --error=logs/train_scgn_%j.err

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

# Print the PYTHONPATH for debugging purposes
echo "PYTHONPATH is set to: $PYTHONPATH"

# =============================================================================
# Setup and Debugging
# =============================================================================

# Navigate to the project directory
cd /global/homes/t/tiffan/repo/accelerator-simulator

# Record the start time
start_time=$(date +%s)
echo "Start time: $(date)"

# =============================================================================
# Training Configuration
# =============================================================================

# Define the training configuration
MODEL="scgn"
DATASET="sequence_graph_data_archive_4"
DATA_KEYWORD="knn_k5_weighted"
BASE_DATA_DIR="/pscratch/sd/t/tiffan/data"
BASE_RESULTS_DIR="/pscratch/sd/t/tiffan/sequence_results"
INITIAL_STEP=0
FINAL_STEP=76
NTRAIN=80
NVAL=10
NTEST=10
BATCH_SIZE=16
NOISE_LEVEL=0.0 #0.01
LAMBDA_RATIO=1.0
NEPOCHS=1000
LR=0.001
HIDDEN_DIM=128
NUM_LAYERS=4
DISCOUNT_FACTOR=1.0
HORIZON=1
SCALING_FACTORS_FILE="data/sequence_particles_data_archive_4_global_statistics.txt"
VERBOSE="--verbose"
# VERBOSE=""
RANDOM_SEED=63

# Additional Parameters
LR=1e-3
LR_SCHEDULER="lin"
LIN_START_EPOCH=10
LIN_END_EPOCH=1000
LIN_FINAL_LR=1e-5

# Define the Python command with the updated config
python_command="src/graph_simulators/train.py \
    --model $MODEL \
    --dataset $DATASET \
    --data_keyword $DATA_KEYWORD \
    --base_data_dir $BASE_DATA_DIR \
    --base_results_dir $BASE_RESULTS_DIR \
    --initial_step $INITIAL_STEP \
    --final_step $FINAL_STEP \
    --include_settings \
    --identical_settings \
    --include_position_index \
    --include_scaling_factors \
    --scaling_factors_file /global/homes/t/tiffan/repo/accelerator-simulator/data/sequence_particles_data_archive_4_global_statistics.txt \
    --use_edge_attr \
    --ntrain $NTRAIN \
    --nval $NVAL \
    --ntest $NTEST \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --hidden_dim 256 \
    --num_layers 6 \
    --discount_factor 0.9 \
    --horizon 1 \
    --nepochs $NEPOCHS \
    --noise_level $NOISE_LEVEL \
    --lambda_ratio $LAMBDA_RATIO \
    --lr_scheduler $LR_SCHEDULER \
    --lin_start_epoch $((LIN_START_EPOCH * SLURM_JOB_NUM_NODES * SLURM_GPUS_PER_NODE)) \
    --lin_end_epoch $((LIN_END_EPOCH * SLURM_JOB_NUM_NODES * SLURM_GPUS_PER_NODE)) \
    --lin_final_lr $LIN_FINAL_LR \
    --random_seed $RANDOM_SEED \
    $VERBOSE"

# =============================================================================

# Print the command for verification
echo "Running command: $python_command"

# =============================================================================
# Distributed Training Configuration
# =============================================================================

# Set master address and port for distributed training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500  # You can choose any free port
export OMP_NUM_THREADS=4  # Adjust as needed

# =============================================================================
# Training Execution
# =============================================================================

# Launch the training using Accelerate
srun -l bash -c "
    accelerate launch \
    --num_machines $SLURM_JOB_NUM_NODES \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $SLURM_PROCID \
    --num_processes $((SLURM_JOB_NUM_NODES * SLURM_GPUS_PER_NODE)) \
    --multi_gpu \
    $python_command
"

# =============================================================================
# Duration Logging
# =============================================================================

# Record the end time
end_time=$(date +%s)
echo "End time: $(date)"

# Calculate the duration in seconds
duration=$((end_time - start_time))

# Convert duration to hours, minutes, and seconds
hours=$((duration / 3600))
minutes=$(( (duration % 3600) / 60 ))
seconds=$((duration % 60))

# Display the total time taken
echo "Time taken: ${hours}h ${minutes}m ${seconds}s"
