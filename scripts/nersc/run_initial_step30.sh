#!/bin/bash
#SBATCH -A m669
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 3:00:00
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

# Define the Python command with the updated config
python_command="src/graph_simulators/train.py     --model scgn     --dataset sequence_graph_data_archive_4     --data_keyword knn_k5_weighted     --base_data_dir /pscratch/sd/t/tiffan/data     --base_results_dir /pscratch/sd/t/tiffan/sequence_results     --initial_step 30     --final_step 76     --include_settings     --identical_settings     --include_position_index     --include_scaling_factors     --scaling_factors_file /global/homes/t/tiffan/repo/accelerator-simulator/data/sequence_particles_data_archive_4_global_statistics.txt     --use_edge_attr     --ntrain 80     --nval 10     --ntest 10     --batch_size 16     --lr 1e-3     --hidden_dim 128     --num_layers 6     --discount_factor 1.0     --horizon 1     --nepochs 1000     --noise_level 0.0     --lambda_ratio 1.0     --lr_scheduler lin     --lin_start_epoch 40     --lin_end_epoch 400     --lin_final_lr 1e-4     --random_seed 63 "

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
