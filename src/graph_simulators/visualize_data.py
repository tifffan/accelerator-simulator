#!/usr/bin/env python3
"""
visualize_data_xyz.py

Script to:
1. Load the test set using SequenceGraphSettingsDataLoaders
2. Print sequence ID and step index for the initial and target graphs
3. Plot (x, px), (y, py), (z, pz) all in one figure
4. Save each figure to the user's home directory with descriptive filenames
"""

import os
import logging
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for saving only
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

# Local imports from your codebase
from src.graph_simulators.config import parse_args
from src.graph_simulators.dataloaders import SequenceGraphSettingsDataLoaders
from src.graph_simulators.utils import set_random_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    # 1) Parse arguments
    args = parse_args()
    # Force evaluate mode
    args.mode = 'evaluate'

    # 2) Set device
    if args.cpu_only:
        device = torch.device('cpu')
        logging.info("Using CPU (cpu_only=True).")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {device}")

    # 3) Fix random seed
    set_random_seed(args.random_seed)

    # 4) Initialize data loaders (test split)
    graph_data_dir = os.path.join(args.base_data_dir, args.dataset, f"{args.data_keyword}_graphs")
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
        include_scaling_factors=args.include_scaling_factors,
        scaling_factors_file=args.scaling_factors_file,
        batch_size=args.batch_size,
        n_train=args.ntrain,
        n_val=args.nval,
        n_test=args.ntest,
    )
    test_loader = data_loaders.get_test_loader()

    if len(test_loader) == 0:
        raise RuntimeError("Test DataLoader is empty. Check dataset splits or paths.")
    logging.info(f"Number of test batches: {len(test_loader)}")

    # 5) Directory to save figures
    home_dir = os.path.expanduser("~")
    save_dir = os.path.join(home_dir, "sequence_visualizations_xyz")
    os.makedirs(save_dir, exist_ok=True)
    logging.info(f"Figures will be saved to: {save_dir}")

    # 6) Iterate over test batches and plot x-px, y-py, z-pz
    for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="Data Visualization")):
        # Each batch_data is [initial_graph, target_graph, seq_length, (maybe settings)]
        if len(batch_data) == 4:
            batch_initial_graph, batch_target_graph, seq_lengths, _ = batch_data
        else:
            batch_initial_graph, batch_target_graph, seq_lengths = batch_data

        batch_initial_graph = batch_initial_graph.to(device)
        batch_target_graph = batch_target_graph.to(device)

        # Optional: get sequence/step info
        seq_id_init  = getattr(batch_initial_graph, 'sequence_id', 'N/A')
        step_idx_init = getattr(batch_initial_graph, 'step_idx', 'N/A')
        seq_id_targ  = getattr(batch_target_graph, 'sequence_id', 'N/A')
        step_idx_targ = getattr(batch_target_graph, 'step_idx', 'N/A')

        logging.info(f"[Batch {batch_idx}] "
                     f"Initial: seq_id={seq_id_init}, step={step_idx_init} | "
                     f"Target: seq_id={seq_id_targ}, step={step_idx_targ} | "
                     f"seq_lengths={seq_lengths}")

        # Extract node features as NumPy
        init_feats = batch_initial_graph.x.detach().cpu().numpy()
        targ_feats = batch_target_graph.x.detach().cpu().numpy()

        # 7) Make a figure with 3 subplots: (x, px), (y, py), (z, pz)
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,5))

        # -- Subplot 1: x vs px
        axes[0].scatter(init_feats[:,0], init_feats[:,3], s=5, alpha=0.5, c='blue', label='Initial')
        axes[0].scatter(targ_feats[:,0], targ_feats[:,3], s=5, alpha=0.5, c='red', label='Target')
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("px")
        axes[0].set_title("x vs px")
        axes[0].legend()
        axes[0].grid(True)

        # -- Subplot 2: y vs py
        axes[1].scatter(init_feats[:,1], init_feats[:,4], s=5, alpha=0.5, c='blue', label='Initial')
        axes[1].scatter(targ_feats[:,1], targ_feats[:,4], s=5, alpha=0.5, c='red', label='Target')
        axes[1].set_xlabel("y")
        axes[1].set_ylabel("py")
        axes[1].set_title("y vs py")
        axes[1].legend()
        axes[1].grid(True)

        # -- Subplot 3: z vs pz
        axes[2].scatter(init_feats[:,2], init_feats[:,5], s=5, alpha=0.5, c='blue', label='Initial')
        axes[2].scatter(targ_feats[:,2], targ_feats[:,5], s=5, alpha=0.5, c='red', label='Target')
        axes[2].set_xlabel("z")
        axes[2].set_ylabel("pz")
        axes[2].set_title("z vs pz")
        axes[2].legend()
        axes[2].grid(True)

        plt.suptitle(f"Batch {batch_idx} | seq_len={seq_lengths} | (Init:{seq_id_init}, Targ:{seq_id_targ})")
        plt.tight_layout()

        # 8) Save figure to disk
        fig_name = (
            f"batch{batch_idx}_initSeq{seq_id_init}_step{step_idx_init}_"
            f"targSeq{seq_id_targ}_step{step_idx_targ}.png"
        )
        save_path = os.path.join(save_dir, fig_name)
        plt.savefig(save_path, dpi=150)
        plt.close()
        logging.info(f"Saved figure: {save_path}")

        # (Optional) If you only want to do a few batches:
        if batch_idx >= 2:
            break

    logging.info("Data visualization (x-px, y-py, z-pz) complete.")

if __name__ == "__main__":
    main()
