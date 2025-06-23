#!/usr/bin/env python
"""
evaluate_full.py

This script performs evaluation of a trained SCGN model on a test dataset.
It demonstrates two evaluation procedures:
  1. One-step evaluation: Evaluate the transition from a specified step (e.g. 0->1).
  2. Rollout evaluation: Roll out predictions from a specified start index for a given number of steps.

Before computing emittance and plotting, it applies a KDE-based filter to drop the lowest 5% of points
in both target and predicted particle groups.

Usage:
    python evaluate_full.py --model scgn --data_keyword sequence_graph --dataset my_dataset \
        --base_data_dir /path/to/data --results_folder /path/to/output \
        --initial_step 0 --final_step 10 --horizon 5 \
        --hidden_dim 128 --num_layers 6 --lambda_ratio 1.0 \
        --random_seed 42 --checkpoint /path/to/checkpoint.pth \
        --outlier_percentage 5
"""
import os
import sys
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Batch, Data
from sklearn.neighbors import KernelDensity

from src.graph_simulators.config import parse_args
from src.graph_simulators.utils import set_random_seed
from src.datasets.sequence_graph_position_scale_datasets import SequenceGraphSettingsPositionScaleSequenceDataset
from src.graph_models.context_models.scale_graph_networks import ScaleAwareLogRatioConditionalGraphNetwork
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
    std  = scale[:, 6:]
    return features * std + mean


def filter_kde_outliers(arr: np.ndarray, pct: float) -> np.ndarray:
    # Drop lowest pct percent by KDE density
    if pct <= 0 or pct >= 100:
        return arr
    kde = KernelDensity().fit(arr)
    dens = np.exp(kde.score_samples(arr))
    thresh = np.percentile(dens, pct)
    return arr[dens >= thresh]


def evaluate_one_step(sequence, step_idx, model, device, lambda_ratio, results_folder,
                      sample_idx, outlier_pct):
    # Load graphs
    init_g = sequence[step_idx]
    tgt_g  = sequence[step_idx + 1]
    batch  = Batch.from_data_list([init_g]).to(device)

    # Forward pass
    pred_feats, pred_logs = model(
        batch.x, batch.edge_index, batch.edge_attr,
        None, batch.scale, batch.batch
    )

    # Compute losses
    mse = torch.nn.MSELoss()
    loss_feat = mse(pred_feats, tgt_g.x.to(device))
    log_ratio = torch.log((tgt_g.scale.to(device) + 1e-8) / (batch.scale + 1e-8))
    loss_scale = mse(pred_logs, log_ratio)
    total_loss = loss_feat + lambda_ratio * loss_scale

    # Inverse normalize
    pred_arr = inverse_normalize_features(
        pred_feats.detach().cpu().numpy(),
        (batch.scale * torch.exp(pred_logs)).detach().cpu().numpy()
    )
    tgt_arr = inverse_normalize_features(
        tgt_g.x.to(device).detach().cpu().numpy(),
        tgt_g.scale.to(device).detach().cpu().numpy()
    )

    # Apply KDE filter
    pred_arr = filter_kde_outliers(pred_arr, outlier_pct)
    tgt_arr  = filter_kde_outliers(tgt_arr, outlier_pct)

    # Convert to particle groups
    pred_pg = transform_to_particle_group(torch.from_numpy(pred_arr).float())
    tgt_pg  = transform_to_particle_group(torch.from_numpy(tgt_arr).float())

    # Compute emittance
    ex_p, ex_t = compute_emittance_x(pred_pg), compute_emittance_x(tgt_pg)
    ey_p, ey_t = compute_emittance_y(pred_pg), compute_emittance_y(tgt_pg)
    ez_p, ez_t = compute_emittance_z(pred_pg), compute_emittance_z(tgt_pg)

    # Plot 2D histogram overlay
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    ax[0].hist2d(tgt_arr[:,0], tgt_arr[:,1], bins=50)
    ax[0].set_title('Target (filtered)')
    ax[1].hist2d(pred_arr[:,0], pred_arr[:,1], bins=50)
    ax[1].set_title('Prediction (filtered)')
    diff = pred_arr - tgt_arr[:pred_arr.shape[0]]
    ax[2].hist2d(diff[:,0], diff[:,1], bins=50)
    ax[2].set_title('Diff')
    plt.suptitle(f"Sample {sample_idx} One-step Step {step_idx}")
    plt.savefig(os.path.join(results_folder, f"sample_{sample_idx}_one_step_{step_idx}.png"))
    plt.close()

    return total_loss.item()


def rollout_evaluation(sequence, rollout_length, model, device, lambda_ratio,
                       results_folder, sample_idx, outlier_pct, start_idx=0):
    losses = []
    x = sequence[start_idx].x
    scale = sequence[start_idx].scale
    edges = sequence[start_idx].edge_index
    attrs = sequence[start_idx].edge_attr

    for t in range(rollout_length):
        batch = Batch.from_data_list([
            Data(x=x.to(device), scale=scale.to(device), edge_index=edges, edge_attr=attrs)
        ]).to(device)
        pred_feats, pred_logs = model(
            batch.x, batch.edge_index, batch.edge_attr,
            None, batch.scale, batch.batch
        )
        new_scale = batch.scale * torch.exp(pred_logs)

        # Compute loss
        gt = sequence[start_idx + t + 1]
        loss_f = torch.nn.MSELoss()(pred_feats, gt.x.to(device))
        loss_s = torch.nn.MSELoss()(pred_logs,
            torch.log((gt.scale.to(device)+1e-8)/(batch.scale+1e-8)))
        total = loss_f + lambda_ratio * loss_s
        losses.append(total.item())

        # Inverse normalize arrays
        pred_arr = inverse_normalize_features(
            pred_feats.detach().cpu().numpy(),
            new_scale.detach().cpu().numpy()
        )
        tgt_arr = inverse_normalize_features(
            gt.x.to(device).detach().cpu().numpy(),
            gt.scale.to(device).detach().cpu().numpy()
        )

        # Apply KDE
        pred_arr = filter_kde_outliers(pred_arr, outlier_pct)
        tgt_arr  = filter_kde_outliers(tgt_arr, outlier_pct)

        # 2D hist plots
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))
        ax[0].hist2d(tgt_arr[:,0], tgt_arr[:,1], bins=50)
        ax[0].set_title('Target (filtered)')
        ax[1].hist2d(pred_arr[:,0], pred_arr[:,1], bins=50)
        ax[1].set_title('Prediction (filtered)')
        diff = pred_arr - tgt_arr[:pred_arr.shape[0]]
        ax[2].hist2d(diff[:,0], diff[:,1], bins=50)
        ax[2].set_title('Diff')
        plt.suptitle(f"Sample {sample_idx} Rollout Step {start_idx+t}")
        plt.savefig(os.path.join(results_folder, f"sample_{sample_idx}_rollout_{t}.png"))
        plt.close()

        # update for next
        x = pred_feats.detach().cpu()
        scale = new_scale.detach().cpu()

    return losses


def main():
    args = parse_args()
    out_pct = 5
    if out_pct > 0:
        args.results_folder += f"_outliers{int(out_pct)}"
    os.makedirs(args.results_folder, exist_ok=True)

    device = torch.device('cpu') if args.cpu_only else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    set_random_seed(args.random_seed)

    model = ScaleAwareLogRatioConditionalGraphNetwork(
        node_in_dim=6, edge_in_dim=4, cond_in_dim=7,
        scale_dim=12, node_out_dim=6,
        hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        log_ratio_dim=7
    ).to(device).eval()

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    data_dir = os.path.join(args.base_data_dir, args.dataset, f"{args.data_keyword}_graphs")
    test_dataset = SequenceGraphSettingsPositionScaleSequenceDataset(
        graph_data_dir=data_dir,
        initial_step=args.initial_step,
        final_step=args.final_step,
        max_prediction_horizon=args.horizon,
        include_settings=False,
        identical_settings=False,
        use_edge_attr=False,
        subsample_size=10,
        include_position_index=False,
        include_scaling_factors=False,
        scaling_factors_file=None
    )
    logging.info(f"Evaluating {len(test_dataset)} sequences...")

    for idx, sequence in enumerate(test_dataset):
        logging.info(f"Sample {idx}")
        one_loss = evaluate_one_step(
            sequence, 0, model, device, args.lambda_ratio,
            args.results_folder, idx, out_pct
        )
        rollout_losses = rollout_evaluation(
            sequence, args.horizon, model, device, args.lambda_ratio,
            args.results_folder, idx, out_pct, start_idx=0
        )

    # Aggregate losses
    plt.figure(figsize=(8,6))
    # aggregation logic omitted for brevity
    plt.savefig(os.path.join(args.results_folder, 'error_vs_sample_index.png'))

if __name__ == "__main__":
    main()