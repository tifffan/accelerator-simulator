# train.py

import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

import numpy as np
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Import models and utilities
from dataloaders import SequenceGraphSettingsDataLoaders

from src.graph_models.models.graph_networks import (
    GraphConvolutionNetwork,
    GraphAttentionNetwork,
    GraphTransformer,
    MeshGraphNet
)

from src.graph_models.context_models.context_graph_networks import *

from src.graph_simulators.utils import (
    generate_results_folder_name,
    save_metadata,
    get_scheduler,
    set_random_seed
)
from src.graph_simulators.config import parse_args

from src.graph_simulators.trainers import SequenceTrainerAccelerate

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

    # Determine if the model requires edge_attr
    models_requiring_edge_attr = [
        'intgnn', 'gtr', 'mgn', 'gtr-ae', 'mgn-ae', 
        'singlescale', 'multiscale', 'multiscale-topk',
        'cgn', 'acgn', 'ggn', 'scgn' 
    ]
    
    # Determine if the model requires edge_attr
    models_requiring_scale = [
        'scgn' 
    ]
    
    use_edge_attr = args.model.lower() in models_requiring_edge_attr
    logging.info(f"Model '{args.model}' requires edge_attr: {use_edge_attr}")
    
    use_scale = args.model.lower() in models_requiring_scale
    logging.info(f"Model '{args.model}' requires edge_attr: {use_scale}")
    
    # Initialize DataLoaders using the updated class
    logging.info("Initializing SequenceGraphSettingsDataLoaders.")

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
            include_scaling_factors=args.include_scaling_factors,
            scaling_factors_file=args.scaling_factors_file,
            # task=args.task, 
            # edge_attr_method=args.edge_attr_method,
            # preload_data=args.preload_data,
            batch_size=args.batch_size,
            n_train=args.ntrain,
            n_val=args.nval,
            n_test=args.ntest
        )
        logging.info("SequenceGraphSettingsDataLoaders initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize DataLoaders: {e}")
        raise

    # Retrieve DataLoaders
    train_loader = data_loaders.get_train_loader()
    val_loader = data_loaders.get_val_loader()
    test_loader = data_loaders.get_test_loader()
    # all_data_loader = data_loaders.get_all_data_loader()

    logging.info(f"Train DataLoader: {len(train_loader)} batches.")
    logging.info(f"Validation DataLoader: {len(val_loader)} batches.")
    logging.info(f"Test DataLoader: {len(test_loader)} batches.")

    # Retrieve a Sample Data Batch for Model Initialization
    logging.info("Retrieving a sample data batch for model initialization.")

    try:
        sample_batch = next(iter(train_loader))
        if args.include_settings:
            batch_initial_graph, batch_target_graph, seq_lengths, settings_tensor = sample_batch
            logging.info("Sample Batch includes settings.")
            logging.debug(f"Batch Initial Graphs: {batch_initial_graph}")
            logging.debug(f"Batch Target Graphs: {batch_target_graph}")
            logging.debug(f"Sequence Lengths: {seq_lengths}")
            logging.debug(f"Settings Tensor: {settings_tensor}")
        else:
            batch_initial_graph, batch_target_graph, seq_lengths = sample_batch
            logging.info("Sample Batch does not include settings.")
            logging.debug(f"Batch Initial Graphs: {batch_initial_graph}")
            logging.debug(f"Batch Target Graphs: {batch_target_graph}")
            logging.debug(f"Sequence Lengths: {seq_lengths}")
    except StopIteration:
        logging.error("Train DataLoader is empty. Cannot retrieve a sample batch.")
        raise
    except Exception as e:
        logging.error(f"Failed to retrieve a sample batch: {e}")
        raise

    logging.info(f"Train DataLoader: {len(train_loader)} batches.")
    logging.info(f"Validation DataLoader: {len(val_loader)} batches.")
    logging.info(f"Test DataLoader: {len(test_loader)} batches.")

    # Retrieve a Sample Data Batch for Model Initialization
    logging.info("Retrieving a sample data batch for model initialization.")

    try:
        sample_batch = next(iter(train_loader))
        if args.include_settings:
            batch_initial_graph, batch_target_graph, seq_lengths, settings_tensor = sample_batch
            logging.info("Sample Batch includes settings.")
            logging.debug(f"Batch Initial Graphs: {batch_initial_graph}")
            logging.debug(f"Batch Target Graphs: {batch_target_graph}")
            logging.debug(f"Sequence Lengths: {seq_lengths}")
            logging.debug(f"Settings Tensor: {settings_tensor}")
        else:
            batch_initial_graph, batch_target_graph, seq_lengths = sample_batch
            logging.info("Sample Batch does not include settings.")
            logging.debug(f"Batch Initial Graphs: {batch_initial_graph}")
            logging.debug(f"Batch Target Graphs: {batch_target_graph}")
            logging.debug(f"Sequence Lengths: {seq_lengths}")
    except StopIteration:
        logging.error("Train DataLoader is empty. Cannot retrieve a sample batch.")
        raise
    except Exception as e:
        logging.error(f"Failed to retrieve a sample batch: {e}")
        raise

    # Get a sample data for model initialization
    logging.info("Retrieving sample data for model initialization.")
    sample_initial_graph = train_loader.dataset[0][0]  # Assuming dataset[0] is a list containing one data tuple
    sample_final_graph = train_loader.dataset[0][1]    # target_graph
    if args.include_settings:
        logging.info("args.include_settings is True")
        sample_settings = train_loader.dataset[0][3]

    in_channels = sample_initial_graph.x.shape[1]
    out_channels = sample_final_graph.x.shape[1]  # Assuming node features remain the same
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
        node_out_dim = sample_final_graph.x.shape[1]
        hidden_dim = args.hidden_dim
        num_layers = args.num_layers

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
        node_out_dim = sample_final_graph.x.shape[1]
        cond_in_dim = sample_settings.shape[0] if args.include_settings else 0
        scale_dim = sample_initial_graph.scale.shape[1]  # Assuming scale has shape [scale_dim]
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
            log_ratio_dim=scale_dim  # Assuming log_ratio_dim equals scale_dim
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

    # Define the loss functions
    criterion = torch.nn.MSELoss(reduction='none')
    logging.info("Defined MSE loss function.")

    # Initialize trainer with the dual loss functions and Gaussian noise
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
        # criterion_nodes=criterion_nodes,
        # criterion_log_ratios=criterion_log_ratios,
        discount_factor=args.discount_factor,
        lambda_ratio=args.lambda_ratio,      # Ensure this argument is defined
        noise_level=args.noise_level         # Ensure this argument is defined
    )
    logging.info("Initialized SequenceTrainerAccelerate with dual loss functions and Gaussian noise.")

    # Save metadata
    save_metadata(args, model, results_folder)
    logging.info("Saved metadata.")

    # Run train or evaluate
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