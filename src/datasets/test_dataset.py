# # test_sequence_graph_dataset_real.py

# import os
# import torch
# from torch_geometric.data import Data
# from torch.utils.data import DataLoader
# import re

# # Import the modified dataset class
# from sequence_graph_position_scale_datasets import SequenceGraphSettingsPositionScaleDataset

# def test_real_dataset():
#     """
#     Tests the SequenceGraphSettingsPositionScaleDataset with real data
#     by printing out the dimensions and values of a sample output from __getitem__.
#     """
#     # Define parameters based on the provided command-line arguments
#     graph_data_dir = '/pscratch/sd/t/tiffan/data/sequence_graph_data_archive_4/knn_k5_weighted_graphs/'
#     scaling_factors_file = '/global/homes/t/tiffan/repo/accelerator-simulator/data/sequence_particles_data_archive_4_global_statistics.txt'
#     initial_step = 0
#     final_step = 2
#     max_prediction_horizon = 2
#     include_settings = True
#     identical_settings = True
#     use_edge_attr = False
#     subsample_size = None  # Use the entire dataset
#     include_position_index = True
#     include_scaling_factors = True

#     # Verify that the dataset directory exists
#     if not os.path.isdir(graph_data_dir):
#         raise FileNotFoundError(f"Dataset directory not found: {graph_data_dir}")

#     # Verify that the scaling factors file exists
#     if include_scaling_factors and not os.path.isfile(scaling_factors_file):
#         raise FileNotFoundError(f"Scaling factors file not found: {scaling_factors_file}")

#     # Initialize the dataset
#     dataset = SequenceGraphSettingsPositionScaleDataset(
#         graph_data_dir=graph_data_dir,
#         initial_step=initial_step,
#         final_step=final_step,
#         max_prediction_horizon=max_prediction_horizon,
#         include_settings=include_settings,
#         identical_settings=identical_settings,
#         use_edge_attr=use_edge_attr,
#         subsample_size=subsample_size,
#         include_position_index=include_position_index,
#         include_scaling_factors=include_scaling_factors,
#         scaling_factors_file=scaling_factors_file
#     )

#     print(f"\nDataset initialized with {len(dataset)} samples.")

#     # Choose a sample index to test
#     sample_idx = 0  # You can change this to test different samples

#     # Retrieve the sample
#     try:
#         sample = dataset[sample_idx]
#     except IndexError:
#         raise IndexError(f"Sample index {sample_idx} is out of range for the dataset.")

#     print(f"\nSample at index {sample_idx}:")

#     # Iterate through the sequences in the sample
#     for seq_idx, data_tuple in enumerate(sample):
#         if include_settings:
#             if len(data_tuple) == 4:
#                 initial_graph, target_graph, seq_length, settings_tensor = data_tuple
#             else:
#                 raise ValueError("Expected 4 elements in the tuple when include_settings is True.")
#         else:
#             if len(data_tuple) == 3:
#                 initial_graph, target_graph, seq_length = data_tuple
#             else:
#                 raise ValueError("Expected 3 elements in the tuple when include_settings is False.")

#         print(f"\nSequence {seq_idx + 1}:")
#         print(f"  Initial Graph:")
#         print(f"    Node Features Shape: {initial_graph.x.shape}")
#         print(f"    Node Features:\n{initial_graph.x}")
#         print(f"    Edge Index Shape: {initial_graph.edge_index.shape}")
#         print(f"    Edge Index:\n{initial_graph.edge_index}")
#         print(f"  Target Graph:")
#         print(f"    Node Features Shape: {target_graph.x.shape}")
#         print(f"    Node Features:\n{target_graph.x}")
#         print(f"    Edge Index Shape: {target_graph.edge_index.shape}")
#         print(f"    Edge Index:\n{target_graph.edge_index}")
#         print(f"  Sequence Length: {seq_length}")

#         if include_settings:
#             print(f"  Settings Tensor Shape: {settings_tensor.shape}")
#             print(f"  Settings Tensor Values:\n{settings_tensor}")

# if __name__ == "__main__":
#     test_real_dataset()


# test_sequence_graph_dataset_prediction.py

import os
import torch
from torch_geometric.data import Data
from torch.utils.data import DataLoader
import re

# Import the modified dataset class
from sequence_graph_position_scale_datasets import SequenceGraphSettingsPositionScaleDataset

def test_prediction_dataset():
    """
    Tests the SequenceGraphSettingsPositionScaleDataset by retrieving a sample
    and printing out the dimensions and values of its components, including scaling factors.
    """
    # Define parameters based on your setup
    graph_data_dir = '/pscratch/sd/t/tiffan/data/sequence_graph_data_archive_4/knn_k5_weighted_graphs/'
    scaling_factors_file = '/global/homes/t/tiffan/repo/accelerator-simulator/data/sequence_particles_data_archive_4_global_statistics.txt'
    initial_step = 50
    final_step = 51
    max_prediction_horizon = 2
    include_settings = True
    identical_settings = True
    use_edge_attr = False
    subsample_size = 10  # Example value
    include_position_index = True
    include_scaling_factors = True

    # Initialize the dataset
    dataset = SequenceGraphSettingsPositionScaleDataset(
        graph_data_dir=graph_data_dir,
        initial_step=initial_step,
        final_step=final_step,
        max_prediction_horizon=max_prediction_horizon,
        include_settings=include_settings,
        identical_settings=identical_settings,
        use_edge_attr=use_edge_attr,
        subsample_size=subsample_size,
        include_position_index=include_position_index,
        include_scaling_factors=include_scaling_factors,
        scaling_factors_file=scaling_factors_file
    )

    print(f"Dataset initialized with {len(dataset)} samples.")

    # Choose a sample index to test
    sample_idx = 0  # Change as needed

    # Retrieve the sample
    try:
        sample = dataset[sample_idx]
    except IndexError:
        print(f"Sample index {sample_idx} is out of range.")
        return

    print(f"\nSample at index {sample_idx}:")

    # Iterate through the sequences in the sample
    for seq_num, data_tuple in enumerate(sample):
        if include_settings:
            if len(data_tuple) == 4:
                initial_graph, target_graph, seq_length, settings_tensor = data_tuple
            else:
                print("Expected 4 elements in the tuple when include_settings is True.")
                continue
        else:
            if len(data_tuple) == 3:
                initial_graph, target_graph, seq_length = data_tuple
            else:
                print("Expected 3 elements in the tuple when include_settings is False.")
                continue

        print(f"\nSequence {seq_num + 1}:")
        print(f"  Initial Graph:")
        print(f"    Node Features Shape: {initial_graph.x.shape}")
        print(f"    Node Features:\n{initial_graph.x}")
        print(f"    Edge Index Shape: {initial_graph.edge_index.shape}")
        print(f"    Edge Index:\n{initial_graph.edge_index}")

        # Check if scaling factors are present in initial_graph
        if hasattr(initial_graph, 'scale'):
            print(f"    Initial Scaling Factors Shape: {initial_graph.scale.shape}")
            print(f"    Initial Scaling Factors:\n{initial_graph.scale}")
        else:
            print("    Initial Scaling Factors: Not Found")

        print(f"  Target Graph:")
        print(f"    Node Features Shape: {target_graph.x.shape}")
        print(f"    Node Features:\n{target_graph.x}")
        print(f"    Edge Index Shape: {target_graph.edge_index.shape}")
        print(f"    Edge Index:\n{target_graph.edge_index}")

        # Check if scaling factors are present in target_graph
        if hasattr(target_graph, 'scale'):
            print(f"    Target Scaling Factors Shape: {target_graph.scale.shape}")
            print(f"    Target Scaling Factors:\n{target_graph.scale}")
        else:
            print("    Target Scaling Factors: Not Found")

        print(f"  Sequence Length: {seq_length}")

        print(f"  Settings Tensor Shape: {settings_tensor.shape}")
        print(f"  Settings Tensor Values:\n{settings_tensor}")

if __name__ == "__main__":
    test_prediction_dataset()
