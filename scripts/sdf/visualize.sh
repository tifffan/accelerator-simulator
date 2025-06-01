#!/bin/bash
#SBATCH --account=ad:beamphysics
#SBATCH --partition=ampere
#SBATCH --job-name=vis_seq_1_4
#SBATCH --output=logs/vis_seq_1_4_%j.out
#SBATCH --error=logs/vis_seq_1_4_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=1:30:00

# Set the PYTHONPATH to include your project directory
export PYTHONPATH=/sdf/home/t/tiffan/repo/accelerator-simulator

# Print the PYTHONPATH for debugging purposes
echo "PYTHONPATH is set to: $PYTHONPATH"

# Navigate to the project directory
cd /sdf/home/t/tiffan/repo/accelerator-simulator

# Record the start time
start_time=$(date +%s)
echo "Start time: $(date)"

# Define the Python command with the updated config
python_command="src/graph_simulators/visualize_data.py --model scgn     --dataset sequence_graph_data_archive_4     --data_keyword knn_k5_weighted     --base_data_dir /sdf/data/ad/ard/u/tiffan/data     --initial_step 0 --final_step 76     --ntrain 80 --nval 10 --ntest 10     --batch_size 16     --random_seed 63
"


# Print the command for verification
echo "Running command: $python_command"

# Set master address and port for distributed training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500  # You can choose any free port
export OMP_NUM_THREADS=4  # Adjust as needed

# Launch the training using Accelerate
srun -l bash -c "
    accelerate launch \
    --num_machines \$SLURM_JOB_NUM_NODES \
    --main_process_ip \$MASTER_ADDR \
    --main_process_port \$MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --num_processes \$((SLURM_GPUS_PER_NODE)) \
    --multi_gpu \
    $python_command
"

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
