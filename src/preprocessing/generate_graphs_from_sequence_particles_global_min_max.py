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


def load_global_min_max(min_max_file):
    """Load global min and max from a txt file."""
    with open(min_max_file, 'r') as f:
        lines = f.readlines()
    global_min, global_max = None, None
    settings_min, settings_max = None, None
    for i, line in enumerate(lines):
        if line.startswith('Global Min:'):
            global_min = [float(x) for x in lines[i+1].strip().split(',')]
        elif line.startswith('Global Max:'):
            global_max = [float(x) for x in lines[i+1].strip().split(',')]
        elif line.startswith('Settings Global Min:'):
            settings_min = [float(x) for x in line.split(':', 1)[1].strip().split(',')]
        elif line.startswith('Settings Global Max:'):
            settings_max = [float(x) for x in line.split(':', 1)[1].strip().split(',')]
    global_min = torch.tensor(global_min, dtype=torch.float32)
    global_max = torch.tensor(global_max, dtype=torch.float32)
    if settings_min is not None:
        settings_min = torch.tensor(settings_min, dtype=torch.float32)
    if settings_max is not None:
        settings_max = torch.tensor(settings_max, dtype=torch.float32)
    return global_min, global_max, settings_min, settings_max


def process_data_catalog(
    data_catalog,
    output_base_dir,
    k=5,
    distance_threshold=float('inf'),
    edge_method='knn',
    weighted_edge=False,
    global_min=None,
    global_max=None,
    settings_min=None,
    settings_max=None,
    identical_settings=False,
    settings_file=None,
    subsample_size=None,
):
    """Process the data catalog and save the graph data to files, with min-max normalization."""
    data = pd.read_csv(data_catalog)
    if subsample_size is not None:
        data = data.head(subsample_size)
    graph_data_dir = generate_graph_data_dir(
        edge_method=edge_method,
        weighted_edge=weighted_edge,
        k=k,
        distance_threshold=distance_threshold,
        base_dir=output_base_dir,
    )
    Path(graph_data_dir).mkdir(parents=True, exist_ok=True)
    if identical_settings:
        if settings_file is None:
            raise ValueError("Settings file must be provided when identical_settings is True.")
        settings = torch.load(settings_file)
        settings_tensor = settings_dict_to_tensor(settings)
        if settings_min is not None and settings_max is not None:
            epsilon = 1e-6
            settings_tensor = (settings_tensor - settings_min) / (settings_max - settings_min + epsilon)
    for idx, row in tqdm(data.iterrows(), total=len(data), desc="Processing data catalog"):
        filepath = row['filepath']
        unique_id = idx
        try:
            particle_data = torch.load(filepath)
        except Exception as e:
            logging.error(f"Error loading particle data from file {filepath}: {e}. Skipping this file.")
            continue
        num_time_steps, num_particles, num_features = particle_data.shape
        if identical_settings:
            settings_tensor_sample = settings_tensor
        else:
            settings_filepath = filepath.replace('_particle_data.pt', '_settings.pt')
            if not os.path.isfile(settings_filepath):
                logging.error(f"Settings file {settings_filepath} not found. Skipping this file.")
                continue
            settings = torch.load(settings_filepath)
            settings_tensor_sample = settings_dict_to_tensor(settings)
            if settings_min is not None and settings_max is not None:
                epsilon = 1e-6
                settings_tensor_sample = (settings_tensor_sample - settings_min) / (settings_max - settings_min + epsilon)
        for t in range(num_time_steps):
            particle_data_t = particle_data[t]
            particle_data_t = particle_data_t.T
            if global_min is not None and global_max is not None:
                epsilon = 1e-6
                particle_data_t = (particle_data_t - global_min[:, None]) / (global_max[:, None] - global_min[:, None] + epsilon)
            if edge_method == 'dist':
                edge_index, edge_weight = _build_edges_by_distance(particle_data_t, distance_threshold, weighted_edge)
            else:
                edge_index, edge_weight = _build_edges_knn(particle_data_t, k, weighted_edge)
            graph_data = Data(x=particle_data_t.T, edge_index=edge_index, edge_weight=edge_weight)
            graph_data.settings = settings_tensor_sample
            save_dir = os.path.join(graph_data_dir, f"step_{t}")
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            save_path = os.path.join(save_dir, f"graph_{unique_id}.pt")
            torch.save(graph_data, save_path)
    metadata = {
        'edge_method': edge_method,
        'weighted_edge': weighted_edge,
        'k': k,
        'distance_threshold': distance_threshold,
        'min_max_normalization': True
    }
    if global_min is not None and global_max is not None:
        metadata['global_min'] = global_min.tolist()
        metadata['global_max'] = global_max.tolist()
    if settings_min is not None and settings_max is not None:
        metadata['settings_min'] = settings_min.tolist()
        metadata['settings_max'] = settings_max.tolist()
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
        dir_name = "{}_k{}_{}_graphs".format(edge_method, k, weighted_str)
    elif edge_method == 'dist':
        if distance_threshold is not None and not np.isinf(distance_threshold):
            dir_name = "{}_d{}_{}_graphs".format(edge_method, distance_threshold, weighted_str)
        else:
            dir_name = "{}_{}_graphs".format(edge_method, weighted_str)
    else:
        dir_name = "{}_{}_graphs".format(edge_method, weighted_str)

    return os.path.join(base_dir, dir_name)


def parse_args():
    parser = argparse.ArgumentParser(description="Process data catalog and generate graph data with min-max normalization")
    parser.add_argument('--data_catalog', type=str, required=True, help="Path to the data catalog CSV (required)")
    parser.add_argument('--min_max_file', type=str, required=True, help="Path to the global min/max txt file (required)")
    parser.add_argument('--k', type=int, default=5, help="Number of nearest neighbors for graph construction")
    parser.add_argument('--distance_threshold', type=float, default=float('inf'), help="Euclidean distance threshold for connecting nodes (if not using k-nearest neighbors)")
    parser.add_argument('--edge_method', type=str, default='knn', choices=['knn', 'dist'], help="Edge construction method: 'knn' or 'dist'")
    parser.add_argument('--weighted_edge', action='store_true', help="Use Gaussian kernel on edge weights instead of binary edges")
    parser.add_argument('--output_base_dir', type=str, default="./graph_data/", help="Base directory to save the processed graph data")
    parser.add_argument('--identical_settings', action='store_true', help="Flag indicating whether settings are identical across samples")
    parser.add_argument('--settings_file', type=str, help="Path to the settings file when identical_settings is True")
    parser.add_argument('--subsample_size', type=int, default=None, help="Number of data catalog entries to process (default: all)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    Path(args.output_base_dir).mkdir(parents=True, exist_ok=True)
    global_min, global_max, settings_min, settings_max = load_global_min_max(args.min_max_file)
    process_data_catalog(
        data_catalog=args.data_catalog,
        output_base_dir=args.output_base_dir,
        k=args.k,
        distance_threshold=args.distance_threshold,
        edge_method=args.edge_method,
        weighted_edge=args.weighted_edge,
        global_min=global_min,
        global_max=global_max,
        settings_min=settings_min,
        settings_max=settings_max,
        identical_settings=args.identical_settings,
        settings_file=args.settings_file,
        subsample_size=args.subsample_size,
    )
    logging.info(f"Graph data saved to: {generate_graph_data_dir(args.edge_method, args.weighted_edge, args.k, args.distance_threshold, args.output_base_dir)}")
