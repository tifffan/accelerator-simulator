#!/usr/bin/env python3

import torch
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from torch_geometric.data import Batch
from torch_geometric.data.data import DataEdgeAttr
torch.serialization.add_safe_globals([DataEdgeAttr])
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Import the rollout dataset and dataloaders
from dataloaders_rollout import SequenceGraphSettingsDataLoaders

# Import models and utilities
from src.graph_models.models.graph_networks import (
    GraphConvolutionNetwork,
    GraphAttentionNetwork,
    GraphTransformer,
    MeshGraphNet
)

from src.graph_models.context_models.context_graph_networks import *
from src.graph_models.context_models.scale_graph_networks import ScaleAwareLogRatioConditionalGraphNetwork

from src.graph_simulators.utils import (
    generate_results_folder_name,
    save_metadata,
    get_scheduler,
    set_random_seed
)
from src.graph_simulators.config import parse_args

from trainers_rollout import SequenceTrainerAccelerate

def is_autoencoder_model(model_name):
    """
    Determines if the given model name corresponds to an autoencoder.
    """
    return model_name.lower().endswith('-ae') or model_name.lower() in ['multiscale-topk']

def main():
    args = parse_args()
    logging.info("Parsed command-line arguments.")

    # Validation for dependencies
    if args.include_scaling_factors and not args.scaling_factors_file:
        logging.error("Scaling factors file not specified while include_scaling_factors is True.")
        raise ValueError("Scaling factors file must be specified when include_scaling_factors is True.")
    logging.info("Validated dependencies.")

    # Set device
    if args.cpu_only:
        device = torch.device('cpu')
        logging.info("CPU only mode enabled.")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {device}")

    # Generate data directory
    graph_data_dir = os.path.join(args.base_data_dir, args.dataset, f"{args.data_keyword}_graphs")
    logging.info(f"Graph data directory: {graph_data_dir}")

    # Generate results folder name
    if args.results_folder is not None:
        results_folder = args.results_folder
        logging.info("Using provided results folder name.")
    else:
        results_folder = generate_results_folder_name(args)
        logging.info("Generated results folder name.")
    logging.info(f"Results will be saved to {results_folder}")
    os.makedirs(results_folder, exist_ok=True)
    logging.info(f"Ensured results folder exists at: {results_folder}")

    # Set random seed
    set_random_seed(args.random_seed)
    logging.info(f"Random seed set to: {args.random_seed}")

    # Determine if the model requires edge_attr / scale
    models_requiring_edge_attr = [
        'intgnn', 'gtr', 'mgn', 'gtr-ae', 'mgn-ae', 
        'singlescale', 'multiscale', 'multiscale-topk',
        'cgn', 'acgn', 'ggn', 'scgn' 
    ]
    models_requiring_scale = ['scgn']
    
    use_edge_attr = args.model.lower() in models_requiring_edge_attr
    logging.info(f"Model '{args.model}' requires edge_attr: {use_edge_attr}")
    use_scale = args.model.lower() in models_requiring_scale
    logging.info(f"Model '{args.model}' requires scale: {use_scale}")
    
    # Initialize DataLoaders using the rollout dataset class
    logging.info("Initializing SequenceGraphSettingsDataLoaders (rollout version).")
    try:
        data_loaders = SequenceGraphSettingsDataLoaders(
            graph_data_dir=graph_data_dir,
            initial_step=args.initial_step,
            final_step=args.final_step,
            max_prediction_horizon=args.horizon,  # horizon now defines max_prediction_horizon
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
            n_test=args.ntest
        )
        logging.info("SequenceGraphSettingsDataLoaders (rollout) initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize DataLoaders: {e}")
        raise

    # Retrieve DataLoaders
    train_loader = data_loaders.get_train_loader()
    val_loader = data_loaders.get_val_loader()
    test_loader = data_loaders.get_test_loader()

    logging.info(f"Train DataLoader: {len(train_loader)} batches.")
    logging.info(f"Validation DataLoader: {len(val_loader)} batches.")
    logging.info(f"Test DataLoader: {len(test_loader)} batches.")

    # Retrieve a sample batch for model initialization.
    # Note: Each sample from the dataset is now a tuple:
    # (input_graph, target_graph_list, seq_length, [settings_list])
    logging.info("Retrieving a sample data batch for model initialization.")
    try:
        sample_batch = next(iter(train_loader))
        if args.include_settings:
            batch_initial_graph, batch_target_list, seq_lengths, settings_list = sample_batch
            logging.info("Sample Batch includes settings.")
        else:
            batch_initial_graph, batch_target_list, seq_lengths = sample_batch
            logging.info("Sample Batch does not include settings.")
    except StopIteration:
        logging.error("Train DataLoader is empty. Cannot retrieve a sample batch.")
        raise
    except Exception as e:
        logging.error(f"Failed to retrieve a sample batch: {e}")
        raise

    # For model initialization, we need a sample input and a sample target.
    # Now, batch_target_list is a list of batched target graphs for each horizon.
    # We'll take the first target (horizon 0) to determine output dimension.
    sample_initial_graph = train_loader.dataset[0][0]
    sample_target_list = train_loader.dataset[0][1]  # This is a list of target graphs.
    # Make sure there's at least one target graph.
    if len(sample_target_list) == 0:
        raise ValueError("No target graphs found in the sample.")
    sample_target_graph = sample_target_list[0]
    if args.include_settings:
        logging.info("args.include_settings is True")
        sample_settings_list = train_loader.dataset[0][3]  # This should be a list of settings tensors.
        sample_settings = sample_settings_list[0]
    else:
        logging.info("args.include_settings is False")

    in_channels = sample_initial_graph.x.shape[1]
    out_channels = sample_target_graph.x.shape[1]
    logging.info(f"in_channels: {in_channels}, out_channels: {out_channels}")

    # Model initialization
    logging.info("Initializing model.")
    if args.model.lower() == 'gcn':
        model = GraphConvolutionNetwork(
            in_channels=in_channels,
            hidden_dim=args.hidden_dim,
            out_channels=out_channels,
            num_layers=args.num_layers,
            pool_ratios=args.pool_ratios,
        )
        logging.info("Initialized GraphConvolutionNetwork model.")
    elif args.model.lower() == 'mgn':
        node_in_dim = sample_initial_graph.x.shape[1]
        edge_in_dim = sample_initial_graph.edge_attr.shape[1] if hasattr(sample_initial_graph, 'edge_attr') and sample_initial_graph.edge_attr is not None else 0
        node_out_dim = sample_target_graph.x.shape[1]
        hidden_dim = args.hidden_dim
        num_layers = args.num_layers
        from src.graph_models.models.graph_networks import MeshGraphNet
        model = MeshGraphNet(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            node_out_dim=node_out_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
        logging.info("Initialized MeshGraphNet model.")
    elif args.model.lower() == 'scgn':
        node_in_dim = sample_initial_graph.x.shape[1]
        edge_in_dim = sample_initial_graph.edge_attr.shape[1] if hasattr(sample_initial_graph, 'edge_attr') and sample_initial_graph.edge_attr is not None else 0
        node_out_dim = sample_target_graph.x.shape[1]
        # For condition input, if settings are included, use settings tensor shape.
        cond_in_dim = sample_settings.shape[0] if args.include_settings else 0
        scale_dim = sample_initial_graph.scale.shape[1]
        hidden_dim = args.hidden_dim
        num_layers = args.num_layers
        logging.info(f"cond_in_dim: {cond_in_dim}")
        logging.info(f"scale_dim: {scale_dim}")
        model = ScaleAwareLogRatioConditionalGraphNetwork(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            cond_in_dim=cond_in_dim,
            scale_dim=scale_dim,
            node_out_dim=node_out_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            log_ratio_dim=scale_dim
        )
        logging.info("Initialized ScaleAwareLogRatioConditionalGraphNetwork model.")
    else:
        logging.error(f"Model '{args.model}' is not implemented.")
        raise NotImplementedError(f"Model '{args.model}' is not implemented in this script.")

    logging.info(f"Initialized model: {args.model}")
    model.to(device)
    logging.info("Model moved to device.")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    logging.info(f"Initialized Adam optimizer with learning rate: {args.lr}")

    # Scheduler
    scheduler = get_scheduler(args, optimizer)
    if scheduler:
        logging.info("Initialized learning rate scheduler.")

    # Define the loss function
    criterion = torch.nn.MSELoss(reduction='none')
    logging.info("Defined MSE loss function.")

    # Initialize trainer with rollout horizon.
    trainer = SequenceTrainerAccelerate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        nepochs=args.nepochs,
        results_folder=results_folder,
        checkpoint=args.checkpoint,
        random_seed=args.random_seed,
        device=device,
        verbose=args.verbose,
        criterion=criterion,
        discount_factor=args.discount_factor,
        lambda_ratio=args.lambda_ratio,
        noise_level=args.noise_level,
        horizon=args.horizon  # This should match max_prediction_horizon from the dataset.
    )
    logging.info("Initialized SequenceTrainerAccelerate with rollout horizon.")

    # Save metadata
    save_metadata(args, model, results_folder)
    logging.info("Saved metadata.")

    # Run training or evaluation
    if args.mode == 'train':
        logging.info("Starting training process.")
        trainer.train()
        logging.info("Training process completed.")
    else:
        logging.info("Evaluation mode is not implemented yet.")
        pass

if __name__ == "__main__":
    logging.info("Starting main execution.")
    main()
    logging.info("Script execution finished.")
