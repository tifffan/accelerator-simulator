#!/bin/bash
#SBATCH --account=ad:beamphysics
#SBATCH --partition=ampere
#SBATCH --job-name=seq_1_4
#SBATCH --output=logs/seq_1_4_%j.out
#SBATCH --error=logs/seq_1_4_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=33:30:00

# =============================================================================
# Environment Configuration
# =============================================================================

# Set the PYTHONPATH to include your project directory
export PYTHONPATH=/sdf/home/t/tiffan/repo/accelerator-simulator

# Print the PYTHONPATH for debugging purposes
echo "PYTHONPATH is set to: $PYTHONPATH"

# Navigate to the project directory
cd /sdf/home/t/tiffan/repo/accelerator-simulator

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
BASE_DATA_DIR="/sdf/data/ad/ard/u/tiffan/data"
BASE_RESULTS_DIR="/sdf/data/ad/ard/u/tiffan/sequence_results"
INITIAL_STEP=0
FINAL_STEP=76
NTRAIN=80
NVAL=10
NTEST=10
BATCH_SIZE=16
NOISE_LEVEL=0.01
LAMBDA_RATIO=1.0
NEPOCHS=1000
LR=0.001
HIDDEN_DIM=256
NUM_LAYERS=4
DISCOUNT_FACTOR=0.9
HORIZON=1
SCALING_FACTORS_FILE="data/sequence_particles_data_archive_4_global_statistics.txt"
VERBOSE="--verbose"
# VERBOSE=""

# Define the Python command with the updated config
python_command="src/graph_simulators/train.py \
    --model $MODEL \
    --dataset $DATASET \
    --data_keyword $DATA_KEYWORD \
    --base_data_dir $BASE_DATA_DIR \
    --base_results_dir $BASE_RESULTS_DIR \
    --initial_step $INITIAL_STEP \
    --final_step $FINAL_STEP \
    --ntrain $NTRAIN \
    --nval $NVAL \
    --ntest $NTEST \
    --include_settings \
    --identical_settings \
    --include_position_index \
    --include_scaling_factors \
    --scaling_factors_file $SCALING_FACTORS_FILE \
    --use_edge_attr \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --hidden_dim $HIDDEN_DIM \
    --num_layers $NUM_LAYERS \
    --discount_factor $DISCOUNT_FACTOR \
    --horizon $HORIZON \
    --nepochs $NEPOCHS \
    --noise_level $NOISE_LEVEL \
    --lambda_ratio $LAMBDA_RATIO \
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
    --num_machines \$SLURM_JOB_NUM_NODES \
    --main_process_ip \$MASTER_ADDR \
    --main_process_port \$MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --num_processes \$SLURM_GPUS_PER_NODE \
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
