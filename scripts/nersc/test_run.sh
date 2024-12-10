#!/bin/bash
#SBATCH -A m669
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH --output=logs/train_mgn_accelerate_1_1_%j.out
#SBATCH --error=logs/train_mgn_accelerate_1_1_%j.err

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

# Navigate to the project directory
cd /global/homes/t/tiffan/repo/accelerator-simulator

# Record the start time
start_time=$(date +%s)
echo "Start time: $(date)"

python_command="src/graph_simulators/train_accelerate.py   --model gcn   --dataset sequence_graph_data_archive_4   --initial_step 0   --final_step 2   --data_keyword knn_k5_weighted   --base_data_dir /pscratch/sd/t/tiffan/data   --base_results_dir ./results   --ntrain 10   --batch_size 4   --lr 0.001   --hidden_dim 64   --num_layers 3   --discount_factor 0.9   --horizon 2   --nepochs 5   --verbose"


# Print the command for verification
echo "Running command: $python_command"

# Set master address and port for distributed training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500  # You can choose any free port
export OMP_NUM_THREADS=4  # Adjust as needed

# Check if sequence_train.py supports accelerate
# If it does, use accelerate launch; otherwise, run the script directly

# For this example, let's assume sequence_train.py supports accelerate
srun -l bash -c "
    accelerate launch \
    --num_machines \$SLURM_JOB_NUM_NODES \
    --main_process_ip \$MASTER_ADDR \
    --main_process_port \$MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --num_processes \$((SLURM_JOB_NUM_NODES * SLURM_GPUS_PER_NODE)) \
    --multi_gpu \
    $python_command
"

# If sequence_train.py does NOT support accelerate, use the following instead:
# srun -l bash -c "$python_command"

# =============================================================================
# Record and Display the Duration
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
