# utils.py

from datetime import datetime
import os
import numpy as np
import logging
import torch
import random

def generate_results_folder_name(args):
    """
    Generate the results folder name for the sequence model training.

    Args:
        args (argparse.Namespace): Parsed arguments.

    Returns:
        str: Full path to the results folder.
    """
    # Base directory for results
    base_results_dir = args.base_results_dir

    # Incorporate model name
    base_results_dir = os.path.join(base_results_dir, args.model)

    # Incorporate dataset name
    base_results_dir = os.path.join(base_results_dir, args.dataset)

    # Modify task to include "seq" and initial/final steps
    task_with_steps = f"seq_init{args.initial_step}_final{args.final_step}"
    base_results_dir = os.path.join(base_results_dir, task_with_steps)

    # Extract important arguments to create a descriptive folder name
    parts = []
    parts.append(f"{args.data_keyword}")
    parts.append(f"r{args.random_seed}")
    parts.append(f"nt{args.ntrain if args.ntrain is not None else 'all'}")
    parts.append(f"nv{args.nval if args.nval is not None else 'all'}")
    parts.append(f"b{args.batch_size}")
    parts.append(f"lr{args.lr:.0e}")  # Format learning rate in scientific notation
    parts.append(f"h{args.hidden_dim}")
    parts.append(f"ly{args.num_layers}")
    parts.append(f"df{args.discount_factor:.2f}")  # Discount factor for sequence training
    parts.append(f"hor{args.horizon}")  # Prediction horizon
    parts.append(f"nl{args.noise_level}")  # Noise level
    # Only include lambda_ratio for models that require it
    models_without_lambda = ["cgnv0", "cdgn", "cgnv1"]
    if hasattr(args, "lambda_ratio") and args.model.lower() not in models_without_lambda:
        parts.append(f"lam{args.lambda_ratio}")  # Lambda ratio
    parts.append(f"ep{args.nepochs}")

    # Append pooling ratios if present
    if hasattr(args, 'pool_ratios') and args.pool_ratios:
        parts.append(f"pr{'_'.join(map(lambda x: f'{x:.2f}', args.pool_ratios))}")

    # Append scheduler information if specified
    if args.lr_scheduler == 'exp':
        parts.append(f"sch_exp_{args.exp_decay_rate}_{args.exp_start_epoch}")
    elif args.lr_scheduler == 'lin':
        parts.append(f"sch_lin_{args.lin_start_epoch}_{args.lin_end_epoch}_{args.lin_final_lr:.0e}")

    # Append new features: positional encoding & scaling factors.
    if args.include_position_index:
        parts.append(f"pos_{args.position_encoding_method}")
        if args.position_encoding_method in ["sinu", "learned"]:
            parts.append(f"dim{args.sinusoidal_encoding_dim}")
    if args.include_scaling_factors:
        parts.append("scaling")

    # Combine parts to form the folder name
    folder_name = '_'.join(map(str, parts))
    results_folder = os.path.join(base_results_dir, folder_name)
    return results_folder

def save_metadata(args, model, results_folder):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata_path = os.path.join(results_folder, f'metadata_{timestamp}.txt')
    os.makedirs(results_folder, exist_ok=True)
    with open(metadata_path, 'w') as f:
        f.write("=== Model Hyperparameters ===\n")
        hyperparams = vars(args)
        for key, value in hyperparams.items():
            if isinstance(value, list):
                value = ', '.join(map(str, value))
            f.write(f"{key}: {value}\n")

        f.write("\n=== Model Architecture ===\n")
        f.write(str(model))

    logging.info(f"Metadata saved to {metadata_path}")

def exponential_lr_scheduler(epoch, decay_rate=0.001, decay_start_epoch=0):
    if epoch < decay_start_epoch:
        return 1.0
    else:
        return np.exp(-decay_rate * (epoch - decay_start_epoch))

def linear_lr_scheduler(epoch, start_epoch=10, end_epoch=100, initial_lr=1e-4, final_lr=1e-6):
    if epoch < start_epoch:
        return 1.0
    elif start_epoch <= epoch < end_epoch:
        proportion = (epoch - start_epoch) / (end_epoch - start_epoch)
        lr = initial_lr + proportion * (final_lr - initial_lr)
        return lr / initial_lr
    else:
        return final_lr / initial_lr

def get_scheduler(args, optimizer):
    if args.lr_scheduler == 'exp':
        scheduler_func = lambda epoch: exponential_lr_scheduler(
            epoch,
            decay_rate=args.exp_decay_rate,
            decay_start_epoch=args.exp_start_epoch
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler_func)
    elif args.lr_scheduler == 'lin':
        scheduler_func = lambda epoch: linear_lr_scheduler(
            epoch,
            start_epoch=args.lin_start_epoch,
            end_epoch=args.lin_end_epoch,
            initial_lr=args.lr,
            final_lr=args.lin_final_lr
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler_func)
    else:
        scheduler = None
    return scheduler

def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
