#!/bin/bash
#SBATCH --account=ad:default
#SBATCH --partition=ampere
#SBATCH --mail-user=tiffan@slac.stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=eval_cgnv0_65
#SBATCH --output=logs/20250608/eval_cgnv0_65_%j.out
#SBATCH --error=logs/20250608/eval_cgnv0_65_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=1:30:00

export PYTHONPATH=/sdf/home/t/tiffan/repo/accelerator-simulator
echo "PYTHONPATH is set to: $PYTHONPATH"
cd /sdf/home/t/tiffan/repo/accelerator-simulator

# ===== Configuration =====
BASE_DIR="/sdf/data/ad/ard/u/tiffan/sequence_results/cgnv0"
EVAL_SCRIPT="src/graph_simulators/evaluate_20250606.py"

mkdir -p logs
mkdir -p results

# Only evaluate the specific run/folder
run_name="sequence_graph_data_archive_4_160_train/seq_init5_final76/knn_k5_weighted_r0_nt8832_nv1104_b16_lr1e-03_h128_ly6_df1.00_hor3_nl0.0_ep1000_pr1.00_sch_exp_0.0001_100_pos_sinusoidal"
run_dir="$BASE_DIR/$run_name"
checkpoint="$run_dir/checkpoints/best_model.pth"

# Check if checkpoint exists
if [ ! -f "$checkpoint" ]; then
  echo "Checkpoint not found: $checkpoint"
  exit 1
fi

# Set parameters explicitly for this run
initial_step=65
final_step=76
horizon=3
noise=0.0
batch_size=8
hidden_dim=128
num_layers=6
lr=1e-3
discount_factor=1.0
lambda_ratio=1.0
position_encoding_method="sinusoidal"
results_folder="results/$run_name/eval_at_65"
mkdir -p "$results_folder"

echo "[INFO] run_name: $run_name"
echo "[INFO] batch_size: $batch_size"
echo "[INFO] hidden_dim: $hidden_dim"
echo "[INFO] horizon: $horizon"
echo "[INFO] noise: $noise"
echo "[INFO] initial_step: $initial_step"
echo "[INFO] final_step: $final_step"
echo "[INFO] position_encoding_method: $position_encoding_method"

python "$EVAL_SCRIPT" \
  --model cgnv0 \
  --dataset sequence_graph_data_archive_4_160_train \
  --data_keyword knn_k5_weighted \
  --base_data_dir /sdf/data/ad/ard/u/tiffan/data \
  --base_results_dir /sdf/data/ad/ard/u/tiffan/sequence_results \
  --initial_step 65 \
  --final_step 70 \
  --ntrain 0 \
  --nval 0 \
  --ntest 10 \
  --batch_size 1 \
  --hidden_dim "$hidden_dim" \
  --num_layers "$num_layers" \
  --lr "$lr" \
  --discount_factor "$discount_factor" \
  --lambda_ratio "$lambda_ratio" \
  --horizon 5 \
  --noise_level "$noise" \
  --position_encoding_method "$position_encoding_method" \
  --checkpoint "$checkpoint" \
  --results_folder "$results_folder" \
  --include_settings \
  --identical_settings \
  --use_edge_attr \
  --include_position_index \
  --include_scaling_factors \
  --scaling_factors_file data/sequence_particles_data_archive_4_global_statistics.txt 

echo "[$run_name] Done. Results in $results_folder" 