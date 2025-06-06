# config.py

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate graph-based models on sequence data.")

    # Common arguments
    parser.add_argument('--model', type=str, required=True, choices=[
        'intgnn', 'gcn', 'gat', 'gtr', 'mgn', 
        'gcn-ae', 'gat-ae', 'gtr-ae', 'mgn-ae', 
        'singlescale', 'multiscale', 'multiscale-topk',
        'cgn', 'acgn', 'ggn',
        'scgn',
        'cdgn', 'cgnv0', 'cgnv1'
    ], help="Model to train. Includes: intgnn, gcn, gat, gtr, mgn, gcn-ae, gat-ae, gtr-ae, mgn-ae, singlescale, multiscale, multiscale-topk, cgn, acgn, ggn, scgn, cdgn, cgnv0, cgnv1.")
    parser.add_argument('--data_keyword', type=str, required=True, 
                        help="Keyword to infer data directories (e.g., 'sequence_graph').")
    parser.add_argument('--base_data_dir', type=str, default="./data/",
                        help="Base directory where the data is stored (default: ./data/).")
    parser.add_argument('--base_results_dir', type=str, default="./results/",
                        help="Base directory where the results are stored (default: ./results/).")
    parser.add_argument('--results_folder', type=str, default=None, help="Directory to save results and checkpoints.")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset identifier.")
    parser.add_argument('--initial_step', type=int, default=0, 
                        help="Index of the initial sequence step (default: 0).")
    parser.add_argument('--final_step', type=int, default=10, 
                        help="Index of the final sequence step (default: 10).")
    parser.add_argument('--include_settings', action='store_true',
                        help="Include settings data concatenated with node features (default: False).")
    parser.add_argument('--identical_settings', action='store_true',
                        help="Flag indicating whether settings are identical across samples (default: False).")
    parser.add_argument('--settings_file', type=str, default=None,
                        help="Path to the settings file if identical_settings is True.")
    parser.add_argument('--include_position_index', action='store_true',
                        help="Include position index in settings (default: False).")
    parser.add_argument('--position_encoding_method', type=str, default="normalized",
                        choices=["normalized", "onehot", "sinusoidal", "learned", "fieldmaps"],
                        help="Method for positional encoding: 'normalized', 'onehot', 'sinusoidal', 'learned', or 'fieldmaps'.")
    parser.add_argument('--sinusoidal_encoding_dim', type=int, default=64,
                        help="Dimension of the sinusoidal (or learned) encoding for positional encoding.")
    parser.add_argument('--include_scaling_factors', action='store_true',
                        help="Include scaling factors from a file (default: False).")
    parser.add_argument('--scaling_factors_file', type=str, default=None,
                        help="Path to the scaling factors text file (required if include_scaling_factors is True).")
    parser.add_argument('--use_edge_attr', action='store_true',
                        help="Compute and include edge attributes for graphs (default: False).")
    parser.add_argument('--edge_attr_method', type=str, default='v1',
                        help="Method to compute edge attributes: 'v0', 'v1(n)', 'v2(n)', or 'v3'.")
    parser.add_argument('--preload_data', action='store_true', help="Preload data into memory before training (default: False).")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate'], help="Mode to run.")

    parser.add_argument('--ntrain', type=int, required=True, help='Number of training graphs.')
    parser.add_argument('--nval', type=int, required=True, help='Number of validation graphs.')
    parser.add_argument('--ntest', type=int, required=True, help='Number of testing graphs.')

    parser.add_argument('--nepochs', type=int, default=100, help="Number of training epochs.")
    parser.add_argument('--save_checkpoint_every', type=int, default=10, help="Save checkpoint every N epochs.")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate for training.")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for training.")
    parser.add_argument('--hidden_dim', type=int, default=64, help="Hidden layer dimension size.")
    parser.add_argument('--num_layers', type=int, default=3, help="Number of layers in the model.")
    parser.add_argument('--pool_ratios', type=float, nargs='+', default=[1.0], help="Pooling ratios for TopKPooling layers.")

    parser.add_argument('--random_seed', type=int, default=63, help="Random seed for reproducibility.")
    parser.add_argument('--verbose', action='store_true', help="Display progress bar while training.")
    parser.add_argument('--subsample_size', type=int, default=None, 
                        help="Number of samples to use from the dataset (default: all).")
    parser.add_argument('--horizon', type=int, default=5, 
                        help='Maximum prediction horizon for multi-step predictions (default: 5).')
    parser.add_argument('--discount_factor', type=float, default=1.0, help='Discount factor for multi-step loss weighting')
    parser.add_argument('--noise_level', type=float, default=0.0, help='Standard deviation of Gaussian noise to add to initial node features (default: 0.0)')
    parser.add_argument('--lambda_ratio', type=float, default=0.0, help='Weighting factor for log ratio loss (default: 1.0)')

    # Learning Rate Scheduler Arguments
    parser.add_argument('--lr_scheduler', type=str, choices=['none', 'exp', 'lin'], default='none',
                        help="Learning rate scheduler type: 'none', 'exp', or 'lin'.")
    parser.add_argument('--exp_decay_rate', type=float, default=0.001, help="Decay rate for exponential scheduler.")
    parser.add_argument('--exp_start_epoch', type=int, default=0, help="Start epoch for exponential scheduler.")
    parser.add_argument('--lin_start_epoch', type=int, default=100, help="Start epoch for linear scheduler.")
    parser.add_argument('--lin_end_epoch', type=int, default=1000, help="End epoch for linear scheduler.")
    parser.add_argument('--lin_final_lr', type=float, default=1e-5, help="Final learning rate for linear scheduler.")
    
    parser.add_argument('--cpu_only', action='store_true', help="Force the script to use CPU even if GPU is available.")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to a checkpoint to resume training.")

    return parser.parse_args()
