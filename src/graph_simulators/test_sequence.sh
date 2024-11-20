# python src/sequence_graph_models/sequence_train_accelerate.py \
#     --model gcn \
#     --dataset sequence_graph_data_archive_4\
#     --data_keyword knn_edges_k5_weighted \
#     --task predict_n6d \
#     --initial_step 0 \
#     --final_step 1 \
#     --identical_settings \
#     --settings_file /sdf/data/ad/ard/u/tiffan/data/sequence_particles_data_archive_4/settings.pt \
#     --ntrain 100 \
#     --nepochs 10 \
#     --lr 0.001 \
#     --batch_size 8 \
#     --hidden_dim 64 \
#     --num_layers 3 \
#     --pool_ratios 1.0 \
#     --verbose


#!/bin/bash

# =============================================================================
# Configuration
# =============================================================================

# Set the Python path to include the project directory
export PYTHONPATH=$(pwd):$PYTHONPATH

# Define the script and test file paths
SEQUENCE_TRAIN_SCRIPT="src/sequence_graph_models/sequence_train_accelerate.py"
TEST_LOG="test_sequence_model.log"

# Python executable (modify if necessary, e.g., python3)
PYTHON_EXEC="python"

# Arguments for testing (modify as needed)
ARGS="--model gcn \
      --dataset sequence_particles_data_archive_4 \
      --task predict_n6d \
      --initial_step 0 \
      --final_step 2 \
      --data_keyword knn_k5_weighted \
      --base_data_dir /pscratch/sd/t/tiffan/data \
      --base_results_dir ./results \
      --ntrain 10 \
      --batch_size 4 \
      --lr 0.001 \
      --hidden_dim 64 \
      --num_layers 3 \
      --discount_factor 0.9 \
      --horizon 2 \
      --nepochs 5 \
      --verbose"

# =============================================================================
# Run the Test
# =============================================================================

echo "Testing the sequence model training script: $SEQUENCE_TRAIN_SCRIPT"
echo "Logging output to: $TEST_LOG"

# Remove the previous log file if it exists
if [ -f "$TEST_LOG" ]; then
    rm "$TEST_LOG"
fi

# Run the test and capture both stdout and stderr
$PYTHON_EXEC $SEQUENCE_TRAIN_SCRIPT $ARGS > "$TEST_LOG" 2>&1

# Check if the script executed successfully
if [ $? -eq 0 ]; then
    echo "Test completed successfully. See $TEST_LOG for details."
else
    echo "Test failed. Check $TEST_LOG for error details."
    exit 1
fi
