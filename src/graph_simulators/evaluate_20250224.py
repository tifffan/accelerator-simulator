#!/usr/bin/env python
"""
evaluate_full.py

This script performs evaluation of a trained SCGN model on a test dataset.
It demonstrates two evaluation procedures:
  1. One-step evaluation: Evaluate the transition from a specified step (e.g. 0->1).
  2. Rollout evaluation: Roll out predictions from a specified start index for a given number of steps.
  
The script iterates over all test samples. For each sample, it:
  - Runs a one-step evaluation (e.g. from step 0->1, or other indices as desired).
  - Runs a rollout evaluation (e.g. rollout from a specified start for a fixed length).
  
All generated figures are saved with filenames that include the sample index.

Usage:
    python evaluate_full.py --model scgn --data_keyword sequence_graph --dataset my_dataset \
        --ntrain 100 --nval 20 --ntest 20 --initial_step 0 --final_step 10 \
        --batch_size 8 --hidden_dim 64 --num_layers 3 --lr 1e-4 --discount_factor 0.9 \
        --lambda_ratio 1.0 --horizon 5 --checkpoint /path/to/checkpoint.pth [other args...]
"""

import os
import sys
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Batch, Data

# Import configuration, dataset, model and utilities.
from src.graph_simulators.config import parse_args
from src.graph_simulators.utils import set_random_seed
from src.datasets.sequence_graph_position_scale_datasets import SequenceGraphSettingsPositionScaleSequenceDataset
from src.graph_models.context_models.scale_graph_networks import ScaleAwareLogRatioConditionalGraphNetwork

# Import visualization helpers.
from src.utils.plot_particle_groups import (
    plot_particle_groups_filename as plot_particle_groups,
    transform_to_particle_group,
    compute_normalized_emittance_x,
    compute_normalized_emittance_y,
    compute_normalized_emittance_z
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def inverse_normalize_features(features, scale):
    """
    Inverse-normalizes features given a scale tensor.
    Assumes scale is a tensor of shape [1, 12] with the first 6 values as mean and next 6 as std.
    """
    mean = scale[:, :6]
    std = scale[:, 6:]
    return features * std + mean


def evaluate_one_step(sequence, step_idx, model, device, lambda_ratio, results_folder,
                      sample_idx, include_settings=False, settings_sequence=None):
    """
    Evaluates the transition from sequence[step_idx] to sequence[step_idx+1] (one-step evaluation)
    for a given sample and saves a visualization with a filename that includes the sample index.
    
    Returns a dictionary with loss and relative error metrics.
    """
    if step_idx < 0 or step_idx >= len(sequence) - 1:
        raise ValueError("step_idx must be in the range [0, len(sequence)-2].")
    
    # Extract graphs for the transition.
    initial_graph = sequence[step_idx].clone()
    target_graph = sequence[step_idx + 1]
    
    if include_settings:
        settings_tensor = settings_sequence[step_idx] if isinstance(settings_sequence, list) else settings_sequence
        settings_input = torch.stack([settings_tensor]).to(device)
    else:
        settings_input = None

    batch_input = Batch.from_data_list([initial_graph]).to(device)
    
    # Forward pass.
    predicted_node_features, predicted_log_ratios = model(
        batch_input.x,
        batch_input.edge_index,
        batch_input.edge_attr,
        settings_input,
        batch_input.scale,
        batch_input.batch
    )
    
    criterion = torch.nn.MSELoss(reduction="mean")
    epsilon = 1e-8
    gt_x = target_graph.x.to(device)
    node_loss = criterion(predicted_node_features, gt_x)
    actual_log_ratios = torch.log(torch.abs((target_graph.scale.to(device) + epsilon) /
                                              (batch_input.scale + epsilon)))
    log_ratio_loss = criterion(predicted_log_ratios, actual_log_ratios)
    total_loss = node_loss.item() + lambda_ratio * log_ratio_loss.item()
    
    # Inverse normalization.
    initial_inv = inverse_normalize_features(initial_graph.x, initial_graph.scale)
    target_inv = inverse_normalize_features(target_graph.x.to(device), target_graph.scale.to(device))
    predicted_scale = batch_input.scale * torch.exp(predicted_log_ratios)
    predicted_inv = inverse_normalize_features(predicted_node_features, predicted_scale)
    
    pred_pg = transform_to_particle_group(predicted_inv.detach().cpu())
    target_pg = transform_to_particle_group(target_inv.detach().cpu())
    
    pred_norm_emit_x = compute_normalized_emittance_x(pred_pg)
    target_norm_emit_x = compute_normalized_emittance_x(target_pg)
    rel_err_x = abs(pred_norm_emit_x - target_norm_emit_x) / abs(target_norm_emit_x)
    
    pred_norm_emit_y = compute_normalized_emittance_y(pred_pg)
    target_norm_emit_y = compute_normalized_emittance_y(target_pg)
    rel_err_y = abs(pred_norm_emit_y - target_norm_emit_y) / abs(target_norm_emit_y)
    
    pred_norm_emit_z = compute_normalized_emittance_z(pred_pg)
    target_norm_emit_z = compute_normalized_emittance_z(target_pg)
    rel_err_z = abs(pred_norm_emit_z - target_norm_emit_z) / abs(target_norm_emit_z)
    
    # Save visualization with a filename that includes the sample index.
    title_text = f"Sample {sample_idx}: One-Step Transition: {step_idx} -> {step_idx+1}"
    filename = os.path.join(results_folder, f"sample_{sample_idx}_one_step_transition_{step_idx}_{step_idx+1}.png")
    plot_particle_groups(
        pred_pg, target_pg, step_idx, title=title_text,
        mse_value=total_loss, rel_err_x=rel_err_x, rel_err_y=rel_err_y, rel_err_z=rel_err_z,
        filename=filename
    )
    
    return {
        'node_loss': node_loss.item(),
        'log_ratio_loss': log_ratio_loss.item(),
        'total_loss': total_loss,
        'rel_err_x': rel_err_x,
        'rel_err_y': rel_err_y,
        'rel_err_z': rel_err_z
    }


def rollout_evaluation(sequence, rollout_length, model, device, lambda_ratio, results_folder,
                       sample_idx, include_settings=False, settings_sequence=None, start_idx=0):
    """
    Performs a rollout evaluation starting from sequence[start_idx] for rollout_length steps for a given sample.
    At each step, the predicted state becomes the input for the next prediction (using fixed edge info).
    If ground truth is available for a step, the loss is computed and a visualization is saved.
    
    The title text for each rollout step includes both the starting and target step indices, and the filename includes the sample index.
    
    Returns a tuple: (losses_list, predicted_states)
      - losses_list: List of loss dictionaries for each rollout step.
      - predicted_states: List of predicted Data objects for each step.
    """
    if rollout_length < 1:
        raise ValueError("rollout_length must be at least 1.")
    
    predicted_states = []
    losses_list = []
    
    # Use fixed edge info from the starting state.
    initial_state = sequence[start_idx]
    fixed_edges = initial_state.edge_index
    fixed_edge_attr = initial_state.edge_attr
    
    # Initialize current state.
    current_x = initial_state.x.clone().to(device)
    current_scale = initial_state.scale.clone().to(device)
    
    criterion = torch.nn.MSELoss(reduction="mean")
    epsilon = 1e-8
    
    for t in range(rollout_length):
        if include_settings:
            if isinstance(settings_sequence, list):
                current_settings = settings_sequence[start_idx + t]
            else:
                current_settings = settings_sequence
            settings_input = torch.stack([current_settings]).to(device)
        else:
            settings_input = None
        
        current_data = Data(x=current_x, scale=current_scale,
                            edge_index=fixed_edges, edge_attr=fixed_edge_attr)
        batch_input = Batch.from_data_list([current_data]).to(device)
        
        predicted_node_features, predicted_log_ratios = model(
            batch_input.x,
            batch_input.edge_index,
            batch_input.edge_attr,
            settings_input,
            batch_input.scale,
            batch_input.batch
        )
        predicted_scale = batch_input.scale * torch.exp(predicted_log_ratios)
        new_state = Data(x=predicted_node_features, scale=predicted_scale,
                         edge_index=fixed_edges, edge_attr=fixed_edge_attr)
        predicted_states.append(new_state)
        
        if start_idx + t + 1 < len(sequence):
            gt_state = sequence[start_idx + t + 1].to(device)
            node_loss = criterion(predicted_node_features, gt_state.x)
            actual_log_ratios = torch.log(torch.abs((gt_state.scale + epsilon) / (batch_input.scale + epsilon)))
            log_ratio_loss = criterion(predicted_log_ratios, actual_log_ratios)
            total_loss = node_loss.item() + lambda_ratio * log_ratio_loss.item()
        else:
            total_loss = None
        
        predicted_inv = inverse_normalize_features(predicted_node_features, predicted_scale)
        if start_idx + t + 1 < len(sequence):
            gt_inv = inverse_normalize_features(gt_state.x, gt_state.scale)
        else:
            gt_inv = None
        
        pred_pg = transform_to_particle_group(predicted_inv.detach().cpu())
        gt_pg = transform_to_particle_group(gt_inv.detach().cpu()) if gt_inv is not None else None
        
        if total_loss is not None:
            pred_norm_emit_x = compute_normalized_emittance_x(pred_pg)
            gt_norm_emit_x = compute_normalized_emittance_x(gt_pg)
            rel_err_x = abs(pred_norm_emit_x - gt_norm_emit_x) / abs(gt_norm_emit_x)
    
            pred_norm_emit_y = compute_normalized_emittance_y(pred_pg)
            gt_norm_emit_y = compute_normalized_emittance_y(gt_pg)
            rel_err_y = abs(pred_norm_emit_y - gt_norm_emit_y) / abs(gt_norm_emit_y)
    
            pred_norm_emit_z = compute_normalized_emittance_z(pred_pg)
            gt_norm_emit_z = compute_normalized_emittance_z(gt_pg)
            rel_err_z = abs(pred_norm_emit_z - gt_norm_emit_z) / abs(gt_norm_emit_z)
    
            title_text = f"Rollout: {start_idx} -> {start_idx+t+1}"
        else:
            title_text = f"Rollout: {start_idx} -> {start_idx+t+1} (No ground truth)"
            rel_err_x = rel_err_y = rel_err_z = 0
        
        filename = os.path.join(results_folder, f"sample_{sample_idx}_rollout_transition_{start_idx}_{start_idx+t+1}.png")
        plot_particle_groups(
            pred_pg, gt_pg, start_idx+t+1, title=title_text,
            mse_value=total_loss if total_loss is not None else 0,
            rel_err_x=rel_err_x,
            rel_err_y=rel_err_y,
            rel_err_z=rel_err_z,
            filename=filename
        )
        
        losses_list.append({
            'node_loss': node_loss.item() if total_loss is not None else None,
            'log_ratio_loss': log_ratio_loss.item() if total_loss is not None else None,
            'total_loss': total_loss
        })
        
        current_x = predicted_node_features.detach()
        current_scale = predicted_scale.detach()
    
    return losses_list, predicted_states


def main():
    args = parse_args()

    # Set device.
    if args.cpu_only:
        device = torch.device("cpu")
        logging.info("Running in CPU-only mode.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")

    # Set random seed.
    set_random_seed(args.random_seed)

    # Initialize SCGN model with hard-coded dimensions.
    if args.model.lower() == 'scgn':
        node_in_dim = 6
        edge_in_dim = 4
        cond_in_dim = 7
        scale_dim = 12
        node_out_dim = 6
        
        model = ScaleAwareLogRatioConditionalGraphNetwork(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            cond_in_dim=cond_in_dim,
            scale_dim=scale_dim,
            node_out_dim=node_out_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            log_ratio_dim=scale_dim
        )
        logging.info("Initialized ScaleAwareLogRatioConditionalGraphNetwork for evaluation.")
    else:
        logging.error(f"Evaluation for model '{args.model}' is not implemented.")
        sys.exit(1)

    model.to(device)
    model.eval()

    # Load model checkpoint.
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logging.info(f"Loaded checkpoint from {args.checkpoint}")
    else:
        logging.error("Checkpoint path must be provided.")
        sys.exit(1)

    # Define graph data directory.
    graph_data_dir = os.path.join(args.base_data_dir, args.dataset, f"{args.data_keyword}_graphs")
    logging.info(f"Graph data directory: {graph_data_dir}")

    if not os.path.exists(args.results_folder):
        os.mkdir(args.results_folder)
    logging.info(f"Results will be saved to: {args.results_folder}")

    # Initialize the sequence dataset for evaluation.
    test_dataset = SequenceGraphSettingsPositionScaleSequenceDataset(
        graph_data_dir=graph_data_dir,
        initial_step=args.initial_step,
        final_step=args.final_step,
        max_prediction_horizon=args.horizon,
        include_settings=args.include_settings,
        identical_settings=args.identical_settings,
        use_edge_attr=args.use_edge_attr,
        subsample_size=10,
        include_position_index=args.include_position_index,
        include_scaling_factors=args.include_scaling_factors,
        scaling_factors_file=args.scaling_factors_file
    )
    num_sequences = len(test_dataset)
    logging.info(f"Evaluating {num_sequences} test sequences.")

    
    start_step = 0
    
    # Loop over each test sample.
    all_one_step_results = []
    all_rollout_results = []
    for sample_idx, sample in enumerate(test_dataset):
        logging.info(f"Evaluating sample {sample_idx}")
        if args.include_settings:
            sequence, settings_seq = sample
        else:
            sequence = sample
            settings_seq = None

        # Run one-step evaluation (e.g. from step 0->1).
        one_step_results = evaluate_one_step(
            sequence, start_step, model, device, args.lambda_ratio, args.results_folder,
            sample_idx=sample_idx, include_settings=args.include_settings, settings_sequence=settings_seq
        )
        all_one_step_results.append(one_step_results)

        # Run rollout evaluation (rollout 5 steps from start index 0).
        rollout_losses, _ = rollout_evaluation(
            sequence, rollout_length=5, model=model, device=device, lambda_ratio=args.lambda_ratio,
            results_folder=args.results_folder, sample_idx=sample_idx, include_settings=args.include_settings,
            settings_sequence=settings_seq, start_idx=start_step
        )
        # For reference, take the final loss from the rollout.
        final_loss = rollout_losses[-1]['total_loss']
        all_rollout_results.append(final_loss)

    # For aggregated error plots, plot one-step errors and rollout final errors vs sample index.
    sample_indices = list(range(len(all_one_step_results)))
    one_step_errors = [res['total_loss'] for res in all_one_step_results]
    rollout_errors = all_rollout_results

    plt.figure(figsize=(8, 6))
    plt.plot(sample_indices, one_step_errors, 'o-', label='One-step Evaluation (0->1)')
    plt.plot(sample_indices, rollout_errors, 's-', label='Rollout Evaluation (0->5)')
    plt.xlabel('Sample Index')
    plt.ylabel('Total Loss')
    plt.title('Error vs. Sample Index: One-step vs. Rollout Evaluation')
    plt.legend()
    
    error_plot_path = os.path.join(args.results_folder, 'error_vs_sample_index.png')
    plt.savefig(error_plot_path, dpi=150)
    plt.close()
    logging.info(f"Saved aggregated error plot to {error_plot_path}")


if __name__ == "__main__":
    main()
