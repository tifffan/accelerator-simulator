#!/usr/bin/env python
"""
evaluate_20250607.py

This script performs evaluation of a trained CGNV0 model on a test dataset.
It demonstrates two evaluation procedures:
  1. One-step evaluation: Evaluate the transition from a specified step (e.g. 0->1).
  2. Rollout evaluation: Roll out predictions from a specified start index for a given number of steps.

The script iterates over all test samples. For each sample, it:
  - Runs a one-step evaluation (e.g. from step 0->1, or other indices as desired).
  - Runs a rollout evaluation (e.g. rollout from a specified start for a fixed length).

All generated figures are saved with filenames that include the sample index.

Usage:
    python evaluate_20250607.py --model cgnv0 --data_keyword sequence_graph --dataset my_dataset \
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
from torch_scatter import scatter_mean

# Import configuration, dataset, model and utilities.
from src.graph_simulators.config import parse_args
from src.graph_simulators.utils import set_random_seed
from dataloaders_rollout import SequenceGraphSettingsDataLoaders
from src.graph_models.context_models.context_graph_networks import ContextAwareGraphNetworkV0

# Import visualization helpers.
from src.utils.plot_particle_groups import (
    plot_particle_groups_filename as plot_particle_groups,
    transform_to_particle_group,
    compute_emittance_x,
    compute_emittance_y,
    compute_emittance_z
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def inverse_normalize_features(features, scale):
    scale = scale.to(features.device)
    mean = scale[:, :6]
    std = scale[:, 6:]
    return features * std + mean

def model_forward(model, initial_graph, settings_tensor, batch, device=None):
    x = initial_graph.x.to(device) if device is not None else initial_graph.x
    edge_index = initial_graph.edge_index.to(device) if device is not None else initial_graph.edge_index
    edge_attr = initial_graph.edge_attr.to(device) if (hasattr(initial_graph, 'edge_attr') and initial_graph.edge_attr is not None and device is not None) else initial_graph.edge_attr
    batch_tensor = batch.to(device) if device is not None else batch
    if settings_tensor is not None and device is not None:
        conditions = settings_tensor.to(device)
    else:
        conditions = settings_tensor
    # For cgnv0, concatenate conditions to each node
    num_nodes = x.shape[0]
    if conditions is not None:
        if conditions.dim() == 1:
            cond_per_node = conditions.unsqueeze(0).repeat(num_nodes, 1)
        else:
            cond_per_node = conditions
        x_input = torch.cat([x, cond_per_node], dim=1)
    else:
        x_input = x
    predicted_node_features = model(
        x=x_input,
        edge_index=edge_index,
        edge_attr=edge_attr,
        conditions=conditions.unsqueeze(0) if conditions is not None and conditions.dim() == 1 else conditions,
        batch=batch_tensor
    )
    return predicted_node_features

def evaluate_one_step(sequence, step_idx, model, device, results_folder,
                      sample_idx, include_settings=False, settings_sequence=None, initial_step=0):
    if step_idx < 0 or step_idx >= len(sequence) - 1:
        raise ValueError("step_idx must be in the range [0, len(sequence)-2].")
    initial_graph = sequence[step_idx].clone()
    target_graph = sequence[step_idx + 1]
    if include_settings:
        if isinstance(settings_sequence, list):
            settings_tensor = settings_sequence[step_idx]
        else:
            settings_tensor = settings_sequence
        # Squeeze singleton batch dimension if present
        if settings_tensor.dim() == 2 and settings_tensor.shape[0] == 1:
            settings_tensor = settings_tensor.squeeze(0)
        assert settings_tensor.dim() == 1, f"settings_tensor should be 1D, got shape {settings_tensor.shape}"
    else:
        settings_tensor = None
    batch_input = Batch.from_data_list([initial_graph]).to(device)
    predicted_node_features = model_forward(
        model, initial_graph, settings_tensor, batch_input.batch, device=device
    )
    gt_x = target_graph.x.to(device)
    node_recon_loss_per_node = torch.nn.functional.mse_loss(predicted_node_features, gt_x, reduction='none')
    if node_recon_loss_per_node.dim() > 1:
        node_recon_loss_per_node = node_recon_loss_per_node.mean(dim=1)
    node_recon_loss_per_graph = scatter_mean(
        node_recon_loss_per_node,
        batch_input.batch,
        dim=0,
        dim_size=1
    )
    total_loss = node_recon_loss_per_graph.mean().item()
    # Inverse normalization and plotting
    initial_inv = inverse_normalize_features(initial_graph.x, initial_graph.scale)
    target_inv = inverse_normalize_features(target_graph.x.to(device), target_graph.scale.to(device))
    predicted_inv = inverse_normalize_features(predicted_node_features, target_graph.scale.to(device))
    pred_pg = transform_to_particle_group(predicted_inv.detach().cpu())
    target_pg = transform_to_particle_group(target_inv.detach().cpu())
    pred_norm_emit_x = compute_emittance_x(pred_pg)
    target_norm_emit_x = compute_emittance_x(target_pg)
    rel_err_x = abs(pred_norm_emit_x - target_norm_emit_x) / abs(target_norm_emit_x)
    pred_norm_emit_y = compute_emittance_y(pred_pg)
    target_norm_emit_y = compute_emittance_y(target_pg)
    rel_err_y = abs(pred_norm_emit_y - target_norm_emit_y) / abs(target_norm_emit_y)
    pred_norm_emit_z = compute_emittance_z(pred_pg)
    target_norm_emit_z = compute_emittance_z(target_pg)
    rel_err_z = abs(pred_norm_emit_z - target_norm_emit_z) / abs(target_norm_emit_z)
    plot_step_idx = step_idx + initial_step
    plot_next_idx = step_idx + 1 + initial_step
    title_text = f"Sample {sample_idx}: One-Step Transition: {plot_step_idx} -> {plot_next_idx}"
    filename = os.path.join(results_folder, f"sample_{sample_idx}_one_step_transition_{plot_step_idx}_{plot_next_idx}.png")
    plot_particle_groups(
        pred_pg, target_pg, plot_step_idx, title=title_text,
        mse_value=total_loss, rel_err_x=rel_err_x, rel_err_y=rel_err_y, rel_err_z=rel_err_z,
        filename=filename
    )
    return {
        'node_loss': node_recon_loss_per_graph.item(),
        'total_loss': total_loss,
        'rel_err_x': rel_err_x,
        'rel_err_y': rel_err_y,
        'rel_err_z': rel_err_z
    }

def rollout_evaluation(sequence, rollout_length, model, device, results_folder,
                       sample_idx, include_settings=False, settings_sequence=None, start_idx=0, discount_factor=1.0, initial_step=0):
    if rollout_length < 1:
        raise ValueError("rollout_length must be at least 1.")
    predicted_states = []
    losses_list = []
    current_graph = sequence[start_idx].clone()
    for t in range(rollout_length):
        if include_settings:
            if isinstance(settings_sequence, list):
                current_settings = settings_sequence[start_idx + t]
            else:
                current_settings = settings_sequence
            # Squeeze singleton batch dimension if present
            if current_settings.dim() == 2 and current_settings.shape[0] == 1:
                current_settings = current_settings.squeeze(0)
            assert current_settings.dim() == 1, f"current_settings should be 1D, got shape {current_settings.shape}"
        else:
            current_settings = None
        batch_input = Batch.from_data_list([current_graph]).to(device)
        predicted_node_features = model_forward(
            model, current_graph, current_settings, batch_input.batch, device=device
        )
        new_state = current_graph.clone()
        new_state.x = predicted_node_features
        predicted_states.append(new_state)
        if start_idx + t + 1 < len(sequence):
            gt_state = sequence[start_idx + t + 1].to(device)
            node_recon_loss_per_node = torch.nn.functional.mse_loss(predicted_node_features, gt_state.x, reduction='none')
            if node_recon_loss_per_node.dim() > 1:
                node_recon_loss_per_node = node_recon_loss_per_node.mean(dim=1)
            node_recon_loss_per_graph = scatter_mean(
                node_recon_loss_per_node,
                batch_input.batch,
                dim=0,
                dim_size=1
            )
            discount = discount_factor ** t
            total_loss = (discount * node_recon_loss_per_graph.mean()).item()
        else:
            total_loss = None
        predicted_inv = inverse_normalize_features(predicted_node_features, current_graph.scale)
        if start_idx + t + 1 < len(sequence):
            gt_inv = inverse_normalize_features(gt_state.x, gt_state.scale)
            predicted_inv = inverse_normalize_features(predicted_node_features, gt_state.scale)
        else:
            gt_inv = None
        pred_pg = transform_to_particle_group(predicted_inv.detach().cpu())
        gt_pg = transform_to_particle_group(gt_inv.detach().cpu()) if gt_inv is not None else None
        if total_loss is not None:
            pred_norm_emit_x = compute_emittance_x(pred_pg)
            gt_norm_emit_x = compute_emittance_x(gt_pg)
            rel_err_x = abs(pred_norm_emit_x - gt_norm_emit_x) / abs(gt_norm_emit_x)
            pred_norm_emit_y = compute_emittance_y(pred_pg)
            gt_norm_emit_y = compute_emittance_y(gt_pg)
            rel_err_y = abs(pred_norm_emit_y - gt_norm_emit_y) / abs(gt_norm_emit_y)
            pred_norm_emit_z = compute_emittance_z(pred_pg)
            gt_norm_emit_z = compute_emittance_z(gt_pg)
            rel_err_z = abs(pred_norm_emit_z - gt_norm_emit_z) / abs(gt_norm_emit_z)
            plot_from_idx = start_idx + initial_step
            plot_to_idx = start_idx + t + 1 + initial_step
            title_text = f"Rollout: {plot_from_idx} -> {plot_to_idx}"
        else:
            title_text = f"Rollout: {start_idx + initial_step} -> {start_idx + t + 1 + initial_step} (No ground truth)"
            rel_err_x = rel_err_y = rel_err_z = 0
        filename = os.path.join(results_folder, f"sample_{sample_idx}_rollout_transition_{plot_from_idx}_{plot_to_idx}.png")
        plot_particle_groups(
            pred_pg, gt_pg, plot_to_idx, title=title_text,
            mse_value=total_loss if total_loss is not None else 0,
            rel_err_x=rel_err_x,
            rel_err_y=rel_err_y,
            rel_err_z=rel_err_z,
            filename=filename
        )
        losses_list.append({
            'node_loss': node_recon_loss_per_graph.item() if total_loss is not None else None,
            'total_loss': total_loss
        })
        current_graph = new_state
    return losses_list, predicted_states

def main():
    args = parse_args()
    if args.cpu_only:
        device = torch.device("cpu")
        logging.info("Running in CPU-only mode.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")
    set_random_seed(args.random_seed)

    # Use SequenceGraphSettingsDataLoaders for test data
    subsample_size = args.ntrain + args.nval + args.ntest
    graph_data_dir = os.path.join(args.base_data_dir, args.dataset, f"{args.data_keyword}_graphs")
    data_loaders = SequenceGraphSettingsDataLoaders(
        graph_data_dir=graph_data_dir,
        initial_step=args.initial_step,
        final_step=args.final_step,
        max_prediction_horizon=args.horizon,
        include_settings=args.include_settings,
        identical_settings=args.identical_settings,
        use_edge_attr=args.use_edge_attr,
        subsample_size=subsample_size,
        include_position_index=args.include_position_index,
        position_encoding_method=getattr(args, 'position_encoding_method', None),
        sinusoidal_encoding_dim=getattr(args, 'sinusoidal_encoding_dim', None),
        include_scaling_factors=args.include_scaling_factors,
        scaling_factors_file=args.scaling_factors_file,
        batch_size=1,  # Force batch size 1 for evaluation
        n_train=args.ntrain,
        n_val=args.nval,
        n_test=args.ntest
    )
    test_loader = data_loaders.get_test_loader()
    logging.info(f"Test DataLoader: {len(test_loader)} batches.")

    # Get a sample for model initialization
    sample = next(iter(test_loader))
    if args.include_settings:
        batch_initial_graph, batch_target_list, seq_lengths, batch_settings_list = sample
        sample_initial_graph = batch_initial_graph
        sample_target_graph = batch_target_list[0][0]
        sample_settings = batch_settings_list[0][0]
        cond_in_dim = sample_settings.shape[0]
    else:
        batch_initial_graph, batch_target_list, seq_lengths = sample
        sample_initial_graph = batch_initial_graph
        sample_target_graph = batch_target_list[0]
        cond_in_dim = 0
    node_in_dim = sample_initial_graph.x.shape[1]
    edge_in_dim = sample_initial_graph.edge_attr.shape[1] if hasattr(sample_initial_graph, 'edge_attr') and sample_initial_graph.edge_attr is not None else 0
    node_out_dim = sample_target_graph.x.shape[1]

    model = ContextAwareGraphNetworkV0(
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        cond_in_dim=cond_in_dim,
        node_out_dim=node_out_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    )
    logging.info("Initialized ContextAwareGraphNetworkV0 for evaluation.")
    model.to(device)
    model.eval()
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logging.info(f"Loaded checkpoint from {args.checkpoint}")
    else:
        logging.error("Checkpoint path must be provided.")
        sys.exit(1)
    if not os.path.exists(args.results_folder):
        os.mkdir(args.results_folder)
    logging.info(f"Results will be saved to: {args.results_folder}")

    # Evaluation loop over test_loader
    num_sequences = len(test_loader.dataset)
    logging.info(f"Evaluating {num_sequences} test sequences.")
    start_step = 0
    all_one_step_results = []
    all_rollout_results = []
    for sample_idx, batch in enumerate(test_loader):
        logging.info(f"Evaluating sample {sample_idx}")
        if args.include_settings:
            batch_initial_graph, batch_target_list, seq_lengths, settings_list = batch
            sequence = [batch_initial_graph] + list(batch_target_list)
            settings_seq = list(settings_list)
        else:
            batch_initial_graph, batch_target_list, seq_lengths = batch
            sequence = [batch_initial_graph] + list(batch_target_list)
            settings_seq = None
        one_step_results = evaluate_one_step(
            sequence, start_step, model, device, args.results_folder,
            sample_idx=sample_idx, include_settings=args.include_settings, settings_sequence=settings_seq,
            initial_step=args.initial_step
        )
        all_one_step_results.append(one_step_results)
        rollout_losses, _ = rollout_evaluation(
            sequence, rollout_length=5, model=model, device=device,
            results_folder=args.results_folder, sample_idx=sample_idx, include_settings=args.include_settings,
            settings_sequence=settings_seq, start_idx=start_step,
            discount_factor=args.discount_factor, initial_step=args.initial_step
        )
        final_loss = rollout_losses[-1]['total_loss']
        all_rollout_results.append(final_loss)
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