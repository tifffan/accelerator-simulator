#!/usr/bin/env python
"""
evaluate_20250606.py

This script performs evaluation of a trained CGNV0 model on a test dataset.
It demonstrates two evaluation procedures:
  1. One-step evaluation: Evaluate the transition from a specified step (e.g. 0->1).
  2. Rollout evaluation: Roll out predictions from a specified start index for a given number of steps.

The script iterates over all test samples. For each sample, it:
  - Runs a one-step evaluation (e.g. from step 0->1, or other indices as desired).
  - Runs a rollout evaluation (e.g. rollout from a specified start for a fixed length).

All generated figures are saved with filenames that include the sample index.

Usage:
    python evaluate_20250606.py --model cgnv0 --data_keyword sequence_graph --dataset my_dataset \
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
    mean = scale[:, :6]
    std = scale[:, 6:]
    # Ensure mean and std are on the same device as features
    mean = mean.to(features.device)
    std = std.to(features.device)
    return features * std + mean

def model_forward(model, initial_graph, settings_tensor, batch, model_type, device=None):
    # Move all tensors to device if device is specified
    x = initial_graph.x.to(device) if device is not None else initial_graph.x
    edge_index = initial_graph.edge_index.to(device) if device is not None else initial_graph.edge_index
    edge_attr = initial_graph.edge_attr.to(device) if (hasattr(initial_graph, 'edge_attr') and initial_graph.edge_attr is not None and device is not None) else initial_graph.edge_attr
    batch_tensor = batch.to(device) if device is not None else batch
    if settings_tensor is not None and device is not None:
        conditions = settings_tensor.to(device)
    else:
        conditions = settings_tensor
    if model_type == 'ScaleAwareLogRatioConditionalGraphNetwork':
        scale = initial_graph.scale.to(device) if (hasattr(initial_graph, 'scale') and initial_graph.scale is not None and device is not None) else initial_graph.scale
        predicted_node_features, predicted_log_ratios = model(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            conditions=conditions,
            scale=scale,
            batch=batch_tensor
        )
    elif model_type in ['ConditionalGraphNetwork', 'ContextAwareGraphNetworkV0']:
        predicted_node_features = model(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            conditions=conditions,
            batch=batch_tensor
        )
        predicted_log_ratios = torch.zeros_like(x[:, :1])
    else:
        raise NotImplementedError(f"Model type '{model_type}' is not supported.")
    return predicted_node_features, predicted_log_ratios

def update_graph_for_next_step(current_graph, predicted_node_features, predicted_log_ratios, model_type):
    updated_graph = current_graph.clone()
    updated_graph.x = predicted_node_features
    if model_type == 'ScaleAwareLogRatioConditionalGraphNetwork':
        updated_graph.scale = current_graph.scale * torch.exp(predicted_log_ratios)
    return updated_graph

def evaluate_one_step(sequence, step_idx, model, device, results_folder,
                      sample_idx, dataset_name, include_settings=False, settings_sequence=None, model_type='ContextAwareGraphNetworkV0', lambda_ratio=1.0, initial_step=0):
    epsilon = 1e-8
    if step_idx < 0 or step_idx >= len(sequence) - 1:
        raise ValueError("step_idx must be in the range [0, len(sequence)-2].")
    initial_graph = sequence[step_idx].clone()
    target_graph = sequence[step_idx + 1]
    if include_settings:
        if isinstance(settings_sequence, list):
            settings_tensor = settings_sequence[step_idx]
        else:
            settings_tensor = settings_sequence
        if settings_tensor.dim() == 2 and settings_tensor.shape[0] == 1:
            settings_tensor = settings_tensor.squeeze(0)
        assert settings_tensor.dim() == 1, f"settings_tensor should be 1D, got shape {settings_tensor.shape}"
        num_nodes = initial_graph.x.shape[0]
        settings_input = settings_tensor.unsqueeze(0).repeat(num_nodes, 1)
        if settings_input is not None:
            settings_input = settings_input.to(device)
    else:
        settings_input = None
    batch_input = Batch.from_data_list([initial_graph]).to(device)
    predicted_node_features, predicted_log_ratios = model_forward(
        model, initial_graph, settings_input, batch_input.batch, model_type, device=device
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
    if model_type == 'ScaleAwareLogRatioConditionalGraphNetwork':
        actual_log_ratios = torch.log(torch.abs((target_graph.scale + epsilon) / (initial_graph.scale + epsilon)))
        log_ratio_loss = torch.nn.functional.mse_loss(predicted_log_ratios, actual_log_ratios, reduction='none')
        if log_ratio_loss.dim() > 1:
            log_ratio_loss = log_ratio_loss.mean(dim=1)
        loss_per_graph = node_recon_loss_per_graph + lambda_ratio * log_ratio_loss
    else:
        loss_per_graph = node_recon_loss_per_graph
    total_loss = loss_per_graph.mean().item()
    # Inverse normalization and plotting as before
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
    title_text = f"Sample {sample_idx} ({dataset_name}): One-Step: {plot_step_idx} -> {plot_next_idx}"
    filename = os.path.join(results_folder, f"sample_{sample_idx}_{dataset_name}_one_step_{plot_step_idx}_{plot_next_idx}.png")
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
                       sample_idx, dataset_name, include_settings=False, settings_sequence=None, start_idx=0, model_type='ContextAwareGraphNetworkV0', lambda_ratio=1.0, discount_factor=1.0, initial_step=0):
    epsilon = 1e-8
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
            if current_settings.dim() == 2 and current_settings.shape[0] == 1:
                current_settings = current_settings.squeeze(0)
            assert current_settings.dim() == 1, f"current_settings should be 1D, got shape {current_settings.shape}"
            num_nodes = current_graph.x.shape[0]
            settings_input = current_settings.unsqueeze(0).repeat(num_nodes, 1)
            if settings_input is not None:
                settings_input = settings_input.to(device)
        else:
            settings_input = None
        batch_input = Batch.from_data_list([current_graph]).to(device)
        predicted_node_features, predicted_log_ratios = model_forward(
            model, current_graph, settings_input, batch_input.batch, model_type, device=device
        )
        new_state = update_graph_for_next_step(current_graph, predicted_node_features, predicted_log_ratios, model_type)
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
            if model_type == 'ScaleAwareLogRatioConditionalGraphNetwork':
                actual_log_ratios = torch.log(torch.abs((gt_state.scale + epsilon) / (current_graph.scale + epsilon)))
                log_ratio_loss = torch.nn.functional.mse_loss(predicted_log_ratios, actual_log_ratios, reduction='none')
                if log_ratio_loss.dim() > 1:
                    log_ratio_loss = log_ratio_loss.mean(dim=1)
                loss_per_graph = node_recon_loss_per_graph + lambda_ratio * log_ratio_loss
            else:
                loss_per_graph = node_recon_loss_per_graph
            discount = discount_factor ** t
            total_loss = (discount * loss_per_graph.mean()).item()
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
            title_text = f"Rollout ({dataset_name}): {plot_from_idx} -> {plot_to_idx}"
        else:
            title_text = f"Rollout ({dataset_name}): {start_idx + initial_step} -> {start_idx + t + 1 + initial_step} (No GT)"
            rel_err_x = rel_err_y = rel_err_z = 0
        filename = os.path.join(results_folder, f"sample_{sample_idx}_{dataset_name}_rollout_{plot_from_idx}_{plot_to_idx}.png")
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

def run_evaluation(loader, dataset_name, model, device, args, model_type):
    """
    Runs evaluation on a given dataloader (train or test).
    """
    logging.info(f"--- Starting evaluation on {dataset_name} set ---")
    num_sequences = len(loader.dataset)
    if num_sequences == 0:
        logging.warning(f"No samples in {dataset_name} set, skipping evaluation.")
        return

    logging.info(f"Evaluating {num_sequences} {dataset_name} sequences.")
    all_one_step_results = []
    all_rollout_results = []
    rollout_length = 5
    for sample_idx, batch in enumerate(loader):
        logging.info(f"Evaluating {dataset_name} sample {sample_idx}")
        if args.include_settings:
            batch_initial_graph, batch_target_list, seq_lengths, settings_list = batch
            sequence = [batch_initial_graph] + list(batch_target_list)
            settings_seq = list(settings_list)
        else:
            batch_initial_graph, batch_target_list, seq_lengths = batch
            sequence = [batch_initial_graph] + list(batch_target_list)
            settings_seq = None
        rel_start = 0
        one_step_results = evaluate_one_step(
            sequence, rel_start, model, device, args.results_folder,
            sample_idx=sample_idx, dataset_name=dataset_name, include_settings=args.include_settings,
            settings_sequence=settings_seq, model_type=model_type,
            lambda_ratio=args.lambda_ratio, initial_step=args.initial_step
        )
        all_one_step_results.append(one_step_results['total_loss'])
        rollout_losses, _ = rollout_evaluation(
            sequence, rollout_length=rollout_length, model=model, device=device,
            results_folder=args.results_folder, sample_idx=sample_idx,
            dataset_name=dataset_name, include_settings=args.include_settings,
            settings_sequence=settings_seq, start_idx=rel_start,
            model_type=model_type, lambda_ratio=args.lambda_ratio,
            discount_factor=args.discount_factor, initial_step=args.initial_step
        )
        final_loss = rollout_losses[-1]['total_loss']
        all_rollout_results.append(final_loss)

    sample_indices = list(range(len(all_one_step_results)))
    plt.figure(figsize=(8, 6))
    plt.plot(sample_indices, all_one_step_results, 'o-', label=f'One-step Eval (start={args.initial_step})')
    plt.plot(sample_indices, all_rollout_results, 's-', label=f'Rollout Eval (start={args.initial_step})')
    plt.xlabel('Sample Index')
    plt.ylabel('Total Loss')
    plt.title(f'Error vs. Sample Index: Start Step {args.initial_step} ({dataset_name} set)')
    plt.legend()
    error_plot_path = os.path.join(args.results_folder, f'error_vs_sample_index_start_{args.initial_step}_{dataset_name}.png')
    plt.savefig(error_plot_path, dpi=150)
    plt.close()
    logging.info(f"Saved {dataset_name} error plot for start step {args.initial_step} to {error_plot_path}")
    logging.info(f"--- Finished evaluation on {dataset_name} set ---")

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
    train_loader = data_loaders.get_train_loader()
    test_loader = data_loaders.get_test_loader()
    logging.info(f"Train DataLoader: {len(train_loader)} batches.")
    logging.info(f"Test DataLoader: {len(test_loader)} batches.")

    # Get a sample for model initialization
    sample = next(iter(test_loader)) if len(test_loader) > 0 else (next(iter(train_loader)) if len(train_loader) > 0 else None)
    if sample is None:
        logging.error("Both train and test loaders are empty. Cannot initialize model.")
        sys.exit(1)

    if args.include_settings:
        batch_initial_graph, batch_target_list, seq_lengths, batch_settings_list = sample
        sample_initial_graph = batch_initial_graph
        sample_target_graph = batch_target_list[0][0]
        sample_settings = batch_settings_list[0][0]

        cond_in_dim = sample_settings.shape[0]
        if args.position_encoding_method == 'normalized' or args.position_encoding_method == 'fieldmaps':
            cond_in_dim = 6 + 1
        elif args.position_encoding_method == 'fieldmaps':
            cond_in_dim = 6 + 3
        elif args.position_encoding_method == 'sinusoidal' or args.position_encoding_method == 'learned':
            cond_in_dim = 6 + 64
        elif args.position_encoding_method == 'onehot':
            cond_in_dim = 6 + 77
        else:
            raise ValueError(f"Unknown position encoding method: {args.position_encoding_method}")
    else:
        batch_initial_graph, batch_target_list, seq_lengths = sample
        sample_initial_graph = batch_initial_graph
        sample_target_graph = batch_target_list[0]
        cond_in_dim = 0
    node_in_dim = sample_initial_graph.x.shape[1]
    edge_in_dim = sample_initial_graph.edge_attr.shape[1] if hasattr(sample_initial_graph, 'edge_attr') and sample_initial_graph.edge_attr is not None else 0
    node_out_dim = sample_target_graph.x.shape[1]

    if args.model.lower() == 'cgnv0':
        model = ContextAwareGraphNetworkV0(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            cond_in_dim=cond_in_dim,
            node_out_dim=node_out_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers
        )
        logging.info("Initialized ContextAwareGraphNetworkV0 for evaluation.")
        model_type = 'ContextAwareGraphNetworkV0'
    else:
        logging.error(f"Evaluation for model '{args.model}' is not implemented.")
        sys.exit(1)
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

    # Run evaluation on test set
    run_evaluation(test_loader, 'test', model, device, args, model_type)

    # Run evaluation on training set
    run_evaluation(train_loader, 'train', model, device, args, model_type)

if __name__ == "__main__":
    main() 