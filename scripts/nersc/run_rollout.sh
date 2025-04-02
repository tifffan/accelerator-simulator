#!/bin/bash
#SBATCH -A m669
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 3:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --output=logs/train_rollout_%j.out
#SBATCH --error=logs/train_rollout_%j.err


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

# =============================================================================
# Setup and Debugging
# =============================================================================

# Navigate to the project directory
cd /global/homes/t/tiffan/repo/accelerator-simulator

python -c "import torch; print(torch.version.cuda)"

# Record the start time
start_time=$(date +%s)
echo "Start time: $(date)"

# =============================================================================
# Training Configuration
# =============================================================================

MODEL="scgn"
DATASET="sequence_graph_data_archive_4"
DATA_KEYWORD="knn_k5_weighted"
BASE_DATA_DIR="/pscratch/sd/t/tiffan/data"
BASE_RESULTS_DIR="/pscratch/sd/t/tiffan/sequence_results"
INITIAL_STEP=20
FINAL_STEP=76
NTRAIN=80
NVAL=10
NTEST=10
BATCH_SIZE=16
NOISE_LEVEL=0.0   # For test runs, set to 0.0
LAMBDA_RATIO=1.0
NEPOCHS=500
HIDDEN_DIM=128
NUM_LAYERS=6
DISCOUNT_FACTOR=1.0
HORIZON=3
SCALING_FACTORS_FILE="/global/homes/t/tiffan/repo/accelerator-simulator/data/sequence_particles_data_archive_4_global_statistics.txt"
VERBOSE="--verbose"
RANDOM_SEED=63

# LR Parameters
LR=1e-3
LR_SCHEDULER="lin"
LIN_START_EPOCH=10
LIN_END_EPOCH=100
LIN_FINAL_LR=1e-4

# Construct the training command.
# Note that we now use the correct path to the training script and include the additional parameters.
python_command="src/graph_simulators/train_rollout.py \
    --model ${MODEL} \
    --dataset ${DATASET} \
    --data_keyword ${DATA_KEYWORD} \
    --base_data_dir ${BASE_DATA_DIR} \
    --base_results_dir ${BASE_RESULTS_DIR} \
    --initial_step ${INITIAL_STEP} \
    --final_step ${FINAL_STEP} \
    --include_settings \
    --identical_settings \
    --include_position_index \
    --include_scaling_factors \
    --scaling_factors_file ${SCALING_FACTORS_FILE} \
    --use_edge_attr \
    --ntrain ${NTRAIN} \
    --nval ${NVAL} \
    --ntest ${NTEST} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --hidden_dim ${HIDDEN_DIM} \
    --num_layers ${NUM_LAYERS} \
    --discount_factor ${DISCOUNT_FACTOR} \
    --horizon ${HORIZON} \
    --nepochs ${NEPOCHS} \
    --noise_level ${NOISE_LEVEL} \
    --lambda_ratio ${LAMBDA_RATIO} \
    --lr_scheduler ${LR_SCHEDULER} \
    --lin_start_epoch $((LIN_START_EPOCH * SLURM_JOB_NUM_NODES * SLURM_GPUS_PER_NODE)) \
    --lin_end_epoch $((LIN_END_EPOCH * SLURM_JOB_NUM_NODES * SLURM_GPUS_PER_NODE)) \
    --lin_final_lr ${LIN_FINAL_LR} \
    --random_seed ${RANDOM_SEED} \
    ${VERBOSE}"

echo "Running command: accelerate launch ${python_command}"

# =============================================================================
# Distributed Training Configuration
# =============================================================================

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500  # Choose an available port
export OMP_NUM_THREADS=4  # Adjust if needed

# =============================================================================
# Training Execution
# =============================================================================

# Directly pass the training script and its arguments to accelerate launch.
srun accelerate launch \
    --mixed_precision no \
    --dynamo_backend no \
    --num_machines ${SLURM_JOB_NUM_NODES} \
    --main_process_ip ${MASTER_ADDR} \
    --main_process_port ${MASTER_PORT} \
    --machine_rank ${SLURM_PROCID} \
    --num_processes $((SLURM_JOB_NUM_NODES * SLURM_GPUS_PER_NODE)) \
    --multi_gpu \
    ${python_command}

# =============================================================================
# Duration Logging
# =============================================================================

end_time=$(date +%s)
echo "End time: $(date)"
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(( (duration % 3600) / 60 ))
seconds=$((duration % 60))
echo "Time taken: ${hours}h ${minutes}m ${seconds}s"
