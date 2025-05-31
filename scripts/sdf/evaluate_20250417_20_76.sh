#!/bin/bash
#SBATCH --account=ad:default
#SBATCH --partition=ampere
#SBATCH --job-name=eval_seq_20_76_20250417
#SBATCH --output=logs/eval_seq_20_76_20250417_%j.out
#SBATCH --error=logs/eval_seq_20_76_20250417_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=0:30:00

# Set the PYTHONPATH to include your project directory
export PYTHONPATH=/sdf/home/t/tiffan/repo/accelerator-simulator

# Print the PYTHONPATH for debugging purposes
echo "PYTHONPATH is set to: $PYTHONPATH"

# Navigate to the project directory
cd /sdf/home/t/tiffan/repo/accelerator-simulator

# Record the start time
start_time=$(date +%s)
echo "Start time: $(date)"

# Define the Python command with the updated script name
python_command="src/graph_simulators/evaluate_kde_filtering.py \
    --model scgn \
    --dataset sequence_graph_data_archive_4 \
    --data_keyword knn_k5_weighted \
    --base_data_dir /sdf/data/ad/ard/u/tiffan/data \
    --base_results_dir /sdf/data/ad/ard/u/tiffan/sequence_results \
    --initial_step 20 \
    --final_step 76 \
    --ntrain 80 \
    --nval 10 \
    --ntest 10 \
    --include_settings \
    --identical_settings \
    --include_position_index \
    --include_scaling_factors \
    --scaling_factors_file data/sequence_particles_data_archive_4_global_statistics.txt \
    --use_edge_attr \
    --batch_size 16 \
    --hidden_dim 128 \
    --num_layers 6 \
    --horizon 1 \
    --lambda_ratio 1.0 \
    --random_seed 63 \
    --checkpoint /sdf/home/t/tiffan/repo/accelerator-simulator/5_76_best_model.pth \
    --results_folder /sdf/home/t/tiffan/repo/accelerator-simulator/results/20250417_rollout_20_76
"

# Print the command for verification
echo "Running command: $python_command"

# Set master address and port for distributed training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export OMP_NUM_THREADS=4  # match --cpus-per-task

# Launch the evaluation using Accelerate
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

end_time=$(date +%s)
echo "End time: $(date)"

duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(((duration % 3600) / 60))
seconds=$((duration % 60))

echo "Time taken: ${hours}h ${minutes}m ${seconds}s"
