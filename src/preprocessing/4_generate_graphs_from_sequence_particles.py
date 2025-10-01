#!/usr/bin/env python3

import argparse
import json
import pandas as pd
import torch
import os
import logging
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from torch_geometric.utils import dense_to_sparse
from pathlib import Path
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_global_statistics(statistics_file):
    """Load global mean and standard deviation from a file."""
    with open(statistics_file, 'r') as f:
        lines = f.readlines()

    global_mean = []
    global_std = []
    settings_mean = None
    settings_std = None

    mode = None  # Tracks whether we're reading means, stds, or settings

    for line in lines:
        line = line.strip()
        if line == "Per-Step Global Mean:":
            mode = 'mean'
            continue
        elif line == "Per-Step Global Std:":
            mode = 'std'
            continue
        elif line.startswith("Settings Global Mean:"):
            settings_mean_str = line.split(":", 1)[1].strip()
            settings_mean = [float(x) for x in settings_mean_str.split(",")]
            mode = None
            continue
        elif line.startswith("Settings Global Std:"):
            settings_std_str = line.split(":", 1)[1].strip()
            settings_std = [float(x) for x in settings_std_str.split(",")]
            mode = None
            continue
        elif line.startswith("Step"):
            if mode == 'mean' or mode == 'std':
                # Parse the step line
                parts = line.split(":")
                if len(parts) != 2:
                    continue
                step_str, values_str = parts
                values = [float(x) for x in values_str.strip().split(",")]
                if mode == 'mean':
                    global_mean.append(values)
                elif mode == 'std':
                    global_std.append(values)
            else:
                # Unknown mode; skip
                continue
        else:
            # Skip any other lines
            continue

    # Convert lists to tensors
    global_mean = torch.tensor(global_mean, dtype=torch.float32)
    global_std = torch.tensor(global_std, dtype=torch.float32)
    if settings_mean is not None:
        settings_mean = torch.tensor(settings_mean, dtype=torch.float32)
    if settings_std is not None:
        settings_std = torch.tensor(settings_std, dtype=torch.float32)

    return global_mean, global_std, settings_mean, settings_std


def process_data_catalog(
    data_catalog,
    output_base_dir,
    k=5,
    distance_threshold=float('inf'),
    edge_method='knn',
    weighted_edge=False,
    global_mean=None,
    global_std=None,
    settings_mean=None,
    settings_std=None,
    identical_settings=False,
    settings_file=None,
    subsample_size=None,
):
    """Process the data catalog and save the graph data to files."""
    data = pd.read_csv(data_catalog)

    if subsample_size is not None:
        # Subsample the data to only process the first 'subsample_size' rows
        data = data.head(subsample_size)

    # Ensure the output directories exist
    graph_data_dir = generate_graph_data_dir(
        edge_method=edge_method,
        weighted_edge=weighted_edge,
        k=k,
        distance_threshold=distance_threshold,
        base_dir=output_base_dir,
    )
    Path(graph_data_dir).mkdir(parents=True, exist_ok=True)

    # Load identical settings if applicable
    if identical_settings:
        if settings_file is None:
            raise ValueError("Settings file must be provided when identical_settings is True.")
        settings = torch.load(settings_file)
        settings_tensor = settings_dict_to_tensor(settings)
        # Normalize settings if mean and std are provided
        if settings_mean is not None and settings_std is not None:
            epsilon = 1e-6
            settings_tensor = (settings_tensor - settings_mean) / (settings_std + epsilon)

    # Process each sample in the data catalog
    for idx, row in tqdm(data.iterrows(), total=len(data), desc="Processing data catalog"):
        filepath = row['filepath']

        # Use the row index from the CSV as the unique identifier
        unique_id = idx

        # Load the time-series particle data
        try:
            particle_data = torch.load(filepath)  # Shape: (num_time_steps, num_particles, num_features)
        except Exception as e:
            logging.error(f"Error loading particle data from file {filepath}: {e}. Skipping this file.")
            continue

        num_time_steps, num_particles, num_features = particle_data.shape

        # Load settings
        if identical_settings:
            settings_tensor_sample = settings_tensor  # Use the preloaded settings tensor
        else:
            # Load settings for the current sample
            settings_filepath = filepath.replace('_particle_data.pt', '_settings.pt')
            if not os.path.isfile(settings_filepath):
                logging.error(f"Settings file {settings_filepath} not found. Skipping this file.")
                continue
            settings = torch.load(settings_filepath)
            settings_tensor_sample = settings_dict_to_tensor(settings)
            # Normalize settings if mean and std are provided
            if settings_mean is not None and settings_std is not None:
                epsilon = 1e-6
                settings_tensor_sample = (settings_tensor_sample - settings_mean) / (settings_std + epsilon)

        # Process each time step
        for t in range(num_time_steps):
            # Get particle data at time step t
            particle_data_t = particle_data[t]  # Shape: (num_particles, num_features)
            particle_data_t = particle_data_t.T  # Shape: (num_features, num_particles)

            # Apply normalization if global mean and std are provided
            if global_mean is not None and global_std is not None:
                epsilon = 1e-6
                mean_t = global_mean[t]
                std_t = global_std[t]
                particle_data_t = (particle_data_t - mean_t[:, None]) / (std_t[:, None] + epsilon)

            # Construct edges
            if edge_method == 'dist':
                edge_index, edge_weight = _build_edges_by_distance(particle_data_t, distance_threshold, weighted_edge)
            else:
                edge_index, edge_weight = _build_edges_knn(particle_data_t, k, weighted_edge)

            # Create graph data object
            graph_data = Data(x=particle_data_t.T, edge_index=edge_index, edge_weight=edge_weight)

            # Optionally, you can include settings in the graph data object
            graph_data.settings = settings_tensor_sample

            # Save the graph data to a file
            save_dir = os.path.join(graph_data_dir, f"step_{t}")
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            save_path = os.path.join(save_dir, f"graph_{unique_id}.pt")
            torch.save(graph_data, save_path)

    # Save metadata
    metadata = {
        'edge_method': edge_method,
        'weighted_edge': weighted_edge,
        'k': k,
        'distance_threshold': distance_threshold,
    }
    if global_mean is not None and global_std is not None:
        metadata['global_mean'] = global_mean.tolist()
        metadata['global_std'] = global_std.tolist()
    if settings_mean is not None and settings_std is not None:
        metadata['settings_mean'] = settings_mean.tolist()
        metadata['settings_std'] = settings_std.tolist()
    metadata_path = os.path.join(graph_data_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    logging.info(f"Graph data saved to: {graph_data_dir}")


def settings_dict_to_tensor(settings_dict):
    """
    Converts a settings dictionary to a tensor.

    Args:
        settings_dict (dict): Dictionary of settings.

    Returns:
        torch.Tensor: Tensor of settings values.
    """
    # Sort settings by key to maintain consistent order
    keys = sorted(settings_dict.keys())
    values = []
    for key in keys:
        value = settings_dict[key]
        if isinstance(value, torch.Tensor):
            value = value.squeeze().float()
        else:
            value = torch.tensor(float(value)).float()
        values.append(value)
    settings_tensor = torch.stack(values)
    return settings_tensor


def _build_edges_knn(node_features, k, weighted_edge):
    """Build k-NN graph edges and weight by applying Gaussian kernel on Euclidean distance."""
    # Create k-NN graph
    edge_index = knn_graph(node_features.T, k=k)

    if weighted_edge:
        # Calculate Euclidean distances between connected nodes
        position_coords = node_features[:3, :].T  # Use only the x, y, z coordinates
        dist_matrix = torch.cdist(position_coords, position_coords, p=2)  # Pairwise distances

        # Extract distances for the selected edges and apply Gaussian kernel
        edge_distances = dist_matrix[edge_index[0], edge_index[1]]
        edge_weight = gaussian_kernel(edge_distances)
    else:
        edge_weight = None  # No edge weights if weighted_edge is False

    return edge_index, edge_weight


def _build_edges_by_distance(node_features, distance_threshold, weighted_edge):
    """Build graph edges based on Euclidean distance and weight by applying Gaussian kernel."""
    position_coords = node_features[:3, :].T  # Use only the x, y, z coordinates
    dist_matrix = torch.cdist(position_coords, position_coords, p=2)  # Compute pairwise distances

    # Create adjacency matrix (0 if distance > threshold, 1 otherwise)
    adj_matrix = (dist_matrix < distance_threshold).float()
    edge_index, _ = dense_to_sparse(adj_matrix)

    if weighted_edge:
        # Apply Gaussian kernel to the distances for the edges
        edge_distances = dist_matrix[edge_index[0], edge_index[1]]
        edge_weight = gaussian_kernel(edge_distances)
    else:
        # Binary edge weights (1 for connected edges)
        edge_weight = torch.ones(edge_index.size(1))

    return edge_index, edge_weight


def gaussian_kernel(distances, sigma=1.0):
    """Gaussian kernel function for weighting edges."""
    return torch.exp(-distances ** 2 / (2 * sigma ** 2))


def generate_graph_data_dir(edge_method, weighted_edge, k=None, distance_threshold=None, base_dir=None):
    """Helper function to generate graph_data_dir based on arguments."""
    weighted_str = "weighted" if weighted_edge else "unweighted"

    if base_dir is None:
        base_dir = "./graph_data"

    if edge_method == 'knn':
        dir_name = "{}_{}_k{}_{}_graphs".format(edge_method, 'edges', k, weighted_str)
    elif edge_method == 'dist':
        if distance_threshold is not None and not np.isinf(distance_threshold):
            dir_name = "{}_{}_d{}_{}_graphs".format(edge_method, 'edges', distance_threshold, weighted_str)
        else:
            dir_name = "{}_{}_{}_graphs".format(edge_method, 'edges', weighted_str)
    else:
        dir_name = "{}_{}_{}_graphs".format(edge_method, 'edges', weighted_str)

    return os.path.join(base_dir, dir_name)


def parse_args():
    """Argument parser for command-line inputs."""
    parser = argparse.ArgumentParser(description="Process data catalog and generate graph data")
    parser.add_argument('--data_catalog', type=str, required=True, help="Path to the data catalog CSV (required)")
    parser.add_argument('--statistics_file', type=str, required=True, help="Path to the global statistics file (required)")
    parser.add_argument('--k', type=int, default=5, help="Number of nearest neighbors for graph construction")
    parser.add_argument('--distance_threshold', type=float, default=float('inf'),
                        help="Euclidean distance threshold for connecting nodes (if not using k-nearest neighbors)")
    parser.add_argument('--edge_method', type=str, default='knn', choices=['knn', 'dist'],
                        help="Edge construction method: 'knn' or 'dist'")
    parser.add_argument('--weighted_edge', action='store_true',
                        help="Use Gaussian kernel on edge weights instead of binary edges")
    parser.add_argument('--output_base_dir', type=str, default="./graph_data/",
                        help="Base directory to save the processed graph data")
    parser.add_argument('--identical_settings', action='store_true',
                        help="Flag indicating whether settings are identical across samples")
    parser.add_argument('--settings_file', type=str, help="Path to the settings file when identical_settings is True")
    parser.add_argument('--subsample_size', type=int, default=None,
                        help="Number of data catalog entries to process (default: all)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Ensure the output base directory exists
    Path(args.output_base_dir).mkdir(parents=True, exist_ok=True)

    # Load global mean and std from the statistics file
    global_mean, global_std, settings_mean, settings_std = load_global_statistics(args.statistics_file)
    logging.info(f"Global mean shape: {global_mean.shape}")
    logging.info(f"Global std shape: {global_std.shape}")

    # Process the data catalog and save the graph data to files
    process_data_catalog(
        data_catalog=args.data_catalog,
        output_base_dir=args.output_base_dir,
        k=args.k,
        distance_threshold=args.distance_threshold,
        edge_method=args.edge_method,
        weighted_edge=args.weighted_edge,
        global_mean=global_mean,
        global_std=global_std,
        settings_mean=settings_mean,
        settings_std=settings_std,
        identical_settings=args.identical_settings,
        settings_file=args.settings_file,
        subsample_size=args.subsample_size,
    )

    logging.info(f"Graph data saved to: {generate_graph_data_dir(args.edge_method, args.weighted_edge, args.k, args.distance_threshold, args.output_base_dir)}")
