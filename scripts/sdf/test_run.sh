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
#SBATCH --time=2:30:00

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
python_command="src/graph_simulators/train.py \
    --model scgn \
    --dataset sequence_graph_data_archive_4 \
    --data_keyword knn_k5_weighted \
    --base_data_dir /sdf/data/ad/ard/u/tiffan/data \
    --base_results_dir /sdf/data/ad/ard/u/tiffan/sequence_results \
    --initial_step 1 \
    --final_step 21 \
    --ntrain 10 \
    --include_settings \
    --identical_settings \
    --include_position_index \
    --include_scaling_factors \
    --scaling_factors_file data/sequence_particles_data_archive_4_global_statistics.txt \
    --use_edge_attr \
    --batch_size 32 \
    --lr 0.001 \
    --hidden_dim 256 \
    --num_layers 4 \
    --discount_factor 0.9 \
    --horizon 1 \
    --nepochs 100 \
    --noise_level 1e-2 \
    --lambda_ratio 1.0 \
    --verbose \
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