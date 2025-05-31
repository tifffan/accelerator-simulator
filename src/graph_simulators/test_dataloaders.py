# test_sequence_context_dataloaders.py

import pytest
import torch
import os
from dataloaders import SequenceGraphSettingsDataLoaders

@pytest.fixture
def dummy_graph_data_dir(tmp_path):
    """
    Creates a dummy graph_data_dir with mock .pt files to emulate a dataset.
    For real-world tests, replace this with a path to your actual data.
    """
    # Example structure: We'll create 50 files in 5 sequences (10 steps each).
    #   graph_0_step_0.pt, graph_0_step_1.pt, ..., graph_0_step_9.pt
    #   graph_1_step_0.pt, ...
    #   ...
    #   graph_4_step_9.pt
    num_sequences = 5
    steps_per_sequence = 10

    for seq_id in range(num_sequences):
        for step in range(steps_per_sequence):
            # Dummy torch file creation
            file_name = f"graph_{seq_id}_step_{step}.pt"
            file_path = tmp_path / file_name
            
            # Save a dummy tensor to emulate a .pt file
            dummy_tensor = torch.rand(2, 2)  # Example shape
            torch.save(dummy_tensor, file_path)
    
    # return str(tmp_path)
    return "/sdf/data/ad/ard/u/tiffan/data/sequence_graph_data_archive_4/knn_k5_weighted_graphs/"


@pytest.mark.parametrize("batch_size", [1, 10])
def test_sequence_graph_settings_dataloaders(dummy_graph_data_dir, batch_size):
    """
    Test the SequenceGraphSettingsDataLoaders with two batch sizes: 1 and 10.
    Verifies that train, val, and test DataLoader objects can be created and iterated.
    """

    # Instantiate the data loaders
    # - n_train=20, n_val=5, n_test=5 for this example (total 30). 
    # - We created 5 sequences x 10 steps each = 50 .pt files in the dummy fixture.
    data_loaders = SequenceGraphSettingsDataLoaders(
        graph_data_dir=dummy_graph_data_dir,
        initial_step=0,
        final_step=9,               # Our dummy data has steps 0..9
        max_prediction_horizon=3,
        include_settings=False,
        identical_settings=False,
        use_edge_attr=False,
        subsample_size=None,
        include_position_index=False,
        include_scaling_factors=False,
        scaling_factors_file=None,
        batch_size=batch_size,
        n_train=20,
        n_val=5,
        n_test=5
    )

    # Train Loader
    train_loader = data_loaders.get_train_loader()
    assert train_loader is not None, "Train DataLoader should not be None."

    # Check iteration
    train_batches = list(train_loader)
    assert len(train_batches) > 0, "Train DataLoader should yield at least one batch."
    # Each batch is a list of sequences: each item in the batch is [initial_graph, target_graph, seq_length, (maybe settings)]
    first_train_batch = train_batches[0]
    assert isinstance(first_train_batch, list), "Each batch should be a list of samples."

    # Val Loader
    val_loader = data_loaders.get_val_loader()
    assert val_loader is not None, "Val DataLoader should not be None."
    val_batches = list(val_loader)
    assert len(val_batches) > 0, "Val DataLoader should yield at least one batch."

    # Test Loader
    test_loader = data_loaders.get_test_loader()
    assert test_loader is not None, "Test DataLoader should not be None."
    test_batches = list(test_loader)
    assert len(test_batches) > 0, "Test DataLoader should yield at least one batch."

    # Print some debug info (not strictly necessary, but useful to visualize)
    print(f"\n[Batch Size {batch_size}]")
    print(f"  - Num train batches: {len(train_batches)}")
    print(f"  - Num val batches:   {len(val_batches)}")
    print(f"  - Num test batches:  {len(test_batches)}")

    # Additional checks (optional)
    # For example, you can check that each item in a batch has the correct structure
    for sample in first_train_batch:
        assert isinstance(sample, list), "Sample should be a list with [initial_graph, target_graph, seq_length, ...]."
        # If you included settings, this length might be 4. Without settings, it might be 3.
        assert len(sample) in (3, 4), "Sample should have 3 or 4 elements."
        
        # sample[0] (initial_graph) and sample[1] (target_graph) are usually PyG Data objects.
        # sample[2] is the seq_length (an integer).
        # sample[3] might be settings if include_settings=True.
        # For brevity, we won't test their internals here, but you could go deeper.

