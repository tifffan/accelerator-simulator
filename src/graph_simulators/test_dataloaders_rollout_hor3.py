import os
from datasets_rollout import SequenceGraphSettingsDataset
from dataloaders_rollout import SequenceGraphSettingsDataLoaders

def print_data_shapes(data, prefix=""):
    if isinstance(data, (list, tuple)):
        for i, d in enumerate(data):
            print_data_shapes(d, prefix=f"{prefix}[{i}]")
    elif hasattr(data, 'shape'):
        print(f"{prefix} shape: {data.shape}")
    elif hasattr(data, 'x') and hasattr(data, 'edge_index'):
        print(f"{prefix} (Graph): x: {data.x.shape}, edge_index: {data.edge_index.shape}")
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            print(f"{prefix} (Graph): edge_attr: {data.edge_attr.shape}")
        if hasattr(data, 'scale') and data.scale is not None:
            print(f"{prefix} (Graph): scale: {data.scale.shape}")
    else:
        print(f"{prefix} type: {type(data)}")

def test_dataset_rollout(graph_data_dir, **kwargs):
    print("Testing SequenceGraphSettingsDataset...")
    dataset_kwargs = dict(kwargs)
    for k in ["batch_size", "n_train", "n_val", "n_test"]:
        dataset_kwargs.pop(k, None)
    dataset = SequenceGraphSettingsDataset(graph_data_dir, **dataset_kwargs)
    print(f"Dataset length: {len(dataset)}")
    print(f"[INFO] Total dataset size: {len(dataset)} samples")
    for i in range(min(3, len(dataset))):
        print(f"Sample {i}:")
        sample = dataset[i]
        # Try to print graph id and step id if possible
        input_graph = sample[0]
        if hasattr(input_graph, 'graph_id') and hasattr(input_graph, 'step_id'):
            print(f"  Input graph: graph_id={input_graph.graph_id}, step_id={input_graph.step_id}")
        else:
            print(f"  Input graph: (graph_id/step_id not available)")
        target_graphs = sample[1]
        for j, tg in enumerate(target_graphs):
            if hasattr(tg, 'graph_id') and hasattr(tg, 'step_id'):
                print(f"  Target {j}: graph_id={tg.graph_id}, step_id={tg.step_id}")
            else:
                print(f"  Target {j}: (graph_id/step_id not available)")
        print_data_shapes(sample, prefix="  sample")

def test_dataloader_rollout(graph_data_dir, **kwargs):
    print("Testing SequenceGraphSettingsDataLoaders...")
    loaders = SequenceGraphSettingsDataLoaders(graph_data_dir=graph_data_dir, **kwargs)
    test_loader = loaders.get_test_loader()
    print(f"Test DataLoader: {len(test_loader)} batches.")
    print("---- The following lines are debug output from dataloaders_rollout.py batching ----")
    for i, batch in enumerate(test_loader):
        print(f"Batch {i}:")
        print_data_shapes(batch, prefix="  batch")
        # Print total data size after batching for input graph (batch[0])
        input_graph = batch[0]
        num_nodes = input_graph.x.shape[0] if hasattr(input_graph, 'x') else None
        num_edges = input_graph.edge_index.shape[1] if hasattr(input_graph, 'edge_index') else None
        print(f"  [INFO] Total nodes in batched input graph: {num_nodes}")
        print(f"  [INFO] Total edges in batched input graph: {num_edges}")

        # Print which step and which graph index is for each batch's input and target lists
        if hasattr(input_graph, 'batch'):
            print("  [INFO] Input graph batch indices (graph_idx for each node):", input_graph.batch.tolist())
        if hasattr(input_graph, 'batch'):
            unique_graph_indices = input_graph.batch.unique(sorted=True).tolist()
            print(f"  [INFO] Input graph contains {len(unique_graph_indices)} graphs with indices: {unique_graph_indices}")

        # For each horizon step, print batch indices for targets
        batched_targets = batch[1]
        for h, target_batch in enumerate(batched_targets):
            if hasattr(target_batch, 'batch'):
                print(f"  [INFO] Target step {h} batch indices (graph_idx for each node):", target_batch.batch.tolist())
                unique_target_indices = target_batch.batch.unique(sorted=True).tolist()
                print(f"  [INFO] Target step {h} contains {len(unique_target_indices)} graphs with indices: {unique_target_indices}")
        if i >= 2:
            break

if __name__ == "__main__":
    # Example usage: update these paths/params as needed
    graph_data_dir = os.environ.get("GRAPH_DATA_DIR", "./data/sequence_graphs")
    kwargs = dict(
        initial_step=0,
        final_step=10,
        max_prediction_horizon=3,
        include_settings=True,
        identical_settings=True,
        use_edge_attr=True,
        subsample_size=None,
        include_position_index=True,
        position_encoding_method="sinusoidal",
        sinusoidal_encoding_dim=64,
        include_scaling_factors=False,
        scaling_factors_file=None,
        batch_size=16,
        n_train=1024,
        n_val=128,
        n_test=0
    )
    test_dataset_rollout(graph_data_dir, **kwargs)
    test_dataloader_rollout(graph_data_dir, **kwargs) 