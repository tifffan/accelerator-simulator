#!/usr/bin/env python3

import os
import torch
import logging
import argparse

# Import the rollout data loader class.
from dataloaders_rollout import SequenceGraphSettingsDataLoaders

def parse_args():
    parser = argparse.ArgumentParser(description="Debug actual log ratios computed on rollout dataset")
    parser.add_argument("--base_data_dir", type=str, default="./data", help="Base directory for dataset")
    parser.add_argument("--dataset", type=str, default="my_dataset", help="Dataset identifier")
    parser.add_argument("--data_keyword", type=str, default="knn_k5_weighted", help="Keyword for graph files")
    parser.add_argument("--initial_step", type=int, default=0)
    parser.add_argument("--final_step", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=3, help="Maximum prediction horizon")
    parser.add_argument("--include_settings", action="store_true", help="Whether to include settings")
    parser.add_argument("--identical_settings", action="store_true", help="Whether settings are identical across horizons")
    parser.add_argument("--use_edge_attr", action="store_true", help="Whether to use edge attributes")
    parser.add_argument("--subsample_size", type=int, default=None)
    parser.add_argument("--include_position_index", action="store_true", help="Whether to include positional indices")
    parser.add_argument("--position_encoding_method", type=str, default="sinusoidal", help="Method for position encoding")
    parser.add_argument("--sinusoidal_encoding_dim", type=int, default=16, help="Dimensionality for sinusoidal encoding")
    parser.add_argument("--include_scaling_factors", action="store_true", help="Whether to include scaling factors")
    parser.add_argument("--scaling_factors_file", type=str, default=None, help="File for scaling factors")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--ntrain", type=int, default=80)
    parser.add_argument("--nval", type=int, default=10)
    parser.add_argument("--ntest", type=int, default=10)
    parser.add_argument("--random_seed", type=int, default=42)
    return parser.parse_args()

def compute_actual_log_ratios(target_scale, current_scale, epsilon=1e-8):
    """
    Computes actual log ratios from target and current scales.
    Formula:
       log( abs( (target_scale + epsilon) / (current_scale + epsilon) ) )
    """
    ratio = (target_scale + epsilon) / (current_scale + epsilon)
    actual_log_ratios = torch.log(torch.abs(ratio))
    return actual_log_ratios

def check_actual_log_ratios(batch, epsilon=1e-8):
    """
    For a given batch from the dataloader, compute the actual log ratios for each horizon step.
    The batch is expected to be either a 4-tuple (including settings) or a 3-tuple:
        (batch_initial, batch_targets, seq_lengths, [settings])
    The field "scale" is expected to be available on both batch_initial and each target graph.
    """
    # Unpack the batch
    if len(batch) == 4:
        batch_initial, batch_targets, seq_lengths, settings = batch
    else:
        batch_initial, batch_targets, seq_lengths = batch

    # For debugging, print out each horizon's scales and computed log ratios.
    for horizon, target_graph in enumerate(batch_targets):
        # Compute actual log ratios
        actual_log_ratios = compute_actual_log_ratios(
            target_scale=target_graph.scale,
            current_scale=batch_initial.scale,
            epsilon=epsilon
        )
        separator = "=" * 60
        print(separator)
        print(f"Horizon step {horizon}:")
        print("Batch initial scale:")
        print(batch_initial.scale)
        print("Target graph scale:")
        print(target_graph.scale)
        print("Computed actual log ratios:")
        print(actual_log_ratios)
        if not torch.all(torch.isfinite(actual_log_ratios)):
            print("WARNING: Non-finite values found in computed actual log ratios!")
        else:
            print("All computed actual log ratios are finite.")
        print(separator)
        print()

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    logging.info("Parsed command-line arguments.")

    # Generate the graph data directory
    graph_data_dir = os.path.join(args.base_data_dir, args.dataset, f"{args.data_keyword}_graphs")
    logging.info(f"Graph data directory: {graph_data_dir}")

    # Initialize the DataLoaders for the rollout dataset.
    try:
        data_loaders = SequenceGraphSettingsDataLoaders(
            graph_data_dir=graph_data_dir,
            initial_step=args.initial_step,
            final_step=args.final_step,
            max_prediction_horizon=args.horizon,
            include_settings=args.include_settings,
            identical_settings=args.identical_settings,
            use_edge_attr=args.use_edge_attr,
            subsample_size=args.subsample_size,
            include_position_index=args.include_position_index,
            position_encoding_method=args.position_encoding_method,
            sinusoidal_encoding_dim=args.sinusoidal_encoding_dim,
            include_scaling_factors=args.include_scaling_factors,
            scaling_factors_file=args.scaling_factors_file,
            batch_size=args.batch_size,
            n_train=args.ntrain,
            n_val=args.nval,
            n_test=args.ntest
        )
        logging.info("DataLoaders initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize DataLoaders: {e}")
        raise

    # Use the training DataLoader.
    train_loader = data_loaders.get_train_loader()
    logging.info(f"Train DataLoader has {len(train_loader)} batches.")

    # Fetch one batch for debugging.
    try:
        batch = next(iter(train_loader))
    except StopIteration:
        logging.error("Train DataLoader appears to be empty.")
        return
    except Exception as e:
        logging.error(f"Error retrieving batch: {e}")
        return

    # Now check actual log ratios for this batch.
    check_actual_log_ratios(batch)

if __name__ == "__main__":
    main()
