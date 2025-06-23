#!/bin/bash
#SBATCH --account=ad:default
#SBATCH --partition=ampere
#SBATCH --mail-user=tiffan@slac.stanford.edu
#SBATCH --mail-type=ALL            # Notifications
#SBATCH --job-name=eval_cgnv0
#SBATCH --output=logs/20250606/eval_cgnv0_%j.out
#SBATCH --error=logs/20250606/eval_cgnv0_%j.err
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

# ===== Configuration =====
BASE_DIR="$HOME/results_cgnv0_20250606"
EVAL_SCRIPT="src/graph_simulators/evaluate_20250606.py"

# Prepare folders
mkdir -p logs
mkdir -p results

# Loop over each training run folder
find "$BASE_DIR" -type f -path "*/checkpoints/best_model.pth" | while read checkpoint; do
  run_dir=$(dirname "$(dirname "$checkpoint")")
  run_name=$(basename "$run_dir")
  echo "[$run_name] Starting evaluation..."

  # Parse hyperparameters from folder name
  # Horizon: extract digits after 'hor'
  horizon=$(echo "$run_name" | sed -n 's/.*hor\([0-9]\+\).*/\1/p')
  [ -z "$horizon" ] && horizon=3

  # Noise level: extract number (including decimal) after 'nl'
  noise=$(echo "$run_name" | sed -n 's/.*nl\([0-9.]\+\).*/\1/p')
  [ -z "$noise" ] && noise=0.0

  # Initial and final step: extract from seq_initX_finalY
  initial_step=$(echo "$run_dir" | sed -n 's/.*seq_init\([0-9]\+\)_final[0-9]\+.*/\1/p')
  final_step=$(echo "$run_dir" | sed -n 's/.*seq_init[0-9]\+_final\([0-9]\+\).*/\1/p')
  [ -z "$initial_step" ] && initial_step=30
  [ -z "$final_step" ] && final_step=35

  # Only process if initial_step is 5 and final_step is 76
  if [ "$initial_step" != "5" ] || [ "$final_step" != "76" ]; then
    echo "[$run_name] Skipping: initial_step ($initial_step) != 5 or final_step ($final_step) != 76."
    continue
  fi

  # Batch size: extract digits after 'bs'
  batch_size=$(echo "$run_name" | sed -n 's/.*b\([0-9]\+\).*/\1/p')
  [ -z "$batch_size" ] && batch_size=16

  # Hidden dim: extract digits after 'hd'
  hidden_dim=$(echo "$run_name" | sed -n 's/.*h\([0-9]\+\).*/\1/p')
  [ -z "$hidden_dim" ] && hidden_dim=256

  echo "[INFO] dirname: $run_dir"
  echo "[INFO] run_name: $run_name"
  echo "[INFO] batch_size: $batch_size"
  echo "[INFO] hidden_dim: $hidden_dim"
  echo "[INFO] horizon: $horizon"
  echo "[INFO] noise: $noise"
  echo "[INFO] initial_step: $initial_step"
  echo "[INFO] final_step: $final_step"

  # Position encoding method
  echo "[DEBUG] run_name for pos_method extraction: >$run_name<"
  pos_method=$(echo "$run_name" | sed -n 's/.*pos_\([a-zA-Z]*\)\([^a-zA-Z].*\|$\)/\1/p')
  echo "[DEBUG] extracted pos_method: >$pos_method<"
  if [ -z "$pos_method" ]; then
    echo "[$run_name] Unknown position encoding method, skipping."
    continue
  fi

  # Use the full directory path (relative to BASE_DIR) for results_folder
  rel_dirname="${run_dir#$BASE_DIR/}"
  results_folder="results/$rel_dirname"
  mkdir -p "$results_folder"

  # Run evaluation
  python "$EVAL_SCRIPT" \
    --model cgnv0 \
    --dataset sequence_graph_data_archive_4 \
    --data_keyword knn_k5_weighted \
    --base_data_dir /sdf/data/ad/ard/u/tiffan/data \
    --base_results_dir /sdf/data/ad/ard/u/tiffan/sequence_results \
    --initial_step "$initial_step" \
    --final_step "$final_step" \
    --ntrain 0 \
    --nval 0 \
    --ntest 10 \
    --batch_size "$batch_size" \
    --hidden_dim "$hidden_dim" \
    --num_layers 6 \
    --lr 1e-4 \
    --discount_factor 1.0 \
    --lambda_ratio 1.0 \
    --horizon "$horizon" \
    --noise_level "$noise" \
    --position_encoding_method "$pos_method" \
    --checkpoint "$checkpoint" \
    --results_folder "$results_folder" \
    --include_settings \
    --identical_settings \
    --use_edge_attr \
    --include_position_index \
    --include_scaling_factors \
    --scaling_factors_file data/sequence_particles_data_archive_4_global_statistics.txt 

  echo "[$run_name] Done. Results in $results_folder"
done 