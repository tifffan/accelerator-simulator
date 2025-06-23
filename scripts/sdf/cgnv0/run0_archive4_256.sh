#!/bin/bash
#SBATCH --account=ad:default
#SBATCH --partition=ampere
#SBATCH --mail-user=tiffan@slac.stanford.edu
#SBATCH --mail-type=ALL            # Notifications
#SBATCH --job-name=run_cgnv0
#SBATCH --output=logs/cgnv0_archive4/cgnv0_h256_hor3_nl0.0_%j.out
#SBATCH --error=logs/cgnv0_archive4/cgnv0_h256_hor3_nl0.0_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=12:00:00

export PYTHONPATH=/sdf/home/t/tiffan/repo/accelerator-simulator
echo "PYTHONPATH is set to: $PYTHONPATH"
cd /sdf/home/t/tiffan/repo/accelerator-simulator


start_time=$(date +%s)
echo "Start time: $(date)"

MODEL_LIST=(
    # "scgn"
    # "cdgn"
    "cgnv0"
)

DATASET="sequence_graph_data_archive_4_160_train"
DATA_KEYWORD="knn_k5_weighted"
BASE_DATA_DIR="/sdf/data/ad/ard/u/tiffan/data"
BASE_RESULTS_DIR="/sdf/data/ad/ard/u/tiffan/sequence_results"
BATCH_SIZE=8
LR=1e-4
HIDDEN_DIM=256
NUM_LAYERS=6
NEPOCHS=1000
HORIZON=3
RANDOM_SEED=0

INITIAL_STEP=5
FINAL_STEP=76
CHECKPOINT_PATH="/sdf/data/ad/ard/u/tiffan/sequence_results/cgnv0/sequence_graph_data_archive_4_160_train/seq_init5_final76/knn_k5_weighted_r0_nt8832_nv1104_b8_lr1e-04_h256_ly6_df1.00_hor3_nl0.0_ep1000_pr1.00_sch_exp_0.0001_100_pos_sinusoidal/checkpoints/best_model.pth"   


export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export OMP_NUM_THREADS=4

for MODEL_NAME in "${MODEL_LIST[@]}"; do
    echo "Testing model: $MODEL_NAME"
    python_command="src/graph_simulators/train_rollout.py \
        --model $MODEL_NAME \
        --dataset $DATASET \
        --data_keyword $DATA_KEYWORD \
        --base_data_dir $BASE_DATA_DIR \
        --base_results_dir $BASE_RESULTS_DIR \
        --initial_step $INITIAL_STEP \
        --final_step $FINAL_STEP \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --lr_scheduler exp \
        --exp_decay_rate 0.0001 \
        --exp_start_epoch 100 \
        --hidden_dim $HIDDEN_DIM \
        --num_layers $NUM_LAYERS \
        --nepochs $NEPOCHS \
        --horizon $HORIZON \
        --random_seed $RANDOM_SEED \
        --include_settings \
        --identical_settings \
        --use_edge_attr \
        --ntrain 8832 \
        --nval 1104 \
        --ntest 0 \
        --include_position_index \
        --position_encoding_method sinusoidal \
        --checkpoint $CHECKPOINT_PATH"
        
    if [[ "$MODEL_NAME" == "scgn" ]]; then
        python_command+=" \
        --include_scaling_factors \
        --scaling_factors_file data/sequence_particles_data_archive_4_global_statistics.txt"
    fi

    echo "Running command: $python_command"

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
done

end_time=$(date +%s)
echo "End time: $(date)"

duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(( (duration % 3600) / 60 ))
seconds=$((duration % 60))
echo "Total time taken: ${hours}h ${minutes}m ${seconds}s" 