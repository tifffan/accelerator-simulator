#!/bin/bash
#SBATCH --account=ad:beamphysics
#SBATCH --partition=ampere
#SBATCH --job-name=rollout_1_1
#SBATCH --output=logs/rollout_1_1_%j.out
#SBATCH --error=logs/rollout_1_1_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=0:30:00

# =============================================================================
# Environment Configuration
# =============================================================================

# Set the PYTHONPATH to include your project directory
export PYTHONPATH=/sdf/home/t/tiffan/repo/accelerator-simulator

# Print the PYTHONPATH for debugging purposes
echo "PYTHONPATH is set to: $PYTHONPATH"

# Navigate to the project directory
cd /sdf/home/t/tiffan/repo/accelerator-simulator
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
BASE_DATA_DIR="/sdf/data/ad/ard/u/tiffan/data"
BASE_RESULTS_DIR="/sdf/data/ad/ard/u/tiffan/sequence_results"
# BASE_DATA_DIR="/pscratch/sd/t/tiffan/data"
# BASE_RESULTS_DIR="/pscratch/sd/t/tiffan/sequence_results"
INITIAL_STEP=0
FINAL_STEP=76
NTRAIN=80
NVAL=10
NTEST=10
BATCH_SIZE=16
NOISE_LEVEL=0.0   # For test runs, set to 0.0
LAMBDA_RATIO=1.0
NEPOCHS=100
HIDDEN_DIM=128
NUM_LAYERS=6
DISCOUNT_FACTOR=1.0
HORIZON=5
SCALING_FACTORS_FILE="data/sequence_particles_data_archive_4_global_statistics.txt"
# SCALING_FACTORS_FILE="/global/homes/t/tiffan/repo/accelerator-simulator/data/sequence_particles_data_archive_4_global_statistics.txt"
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
    # --multi_gpu \
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
