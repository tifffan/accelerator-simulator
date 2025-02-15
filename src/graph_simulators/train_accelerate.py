# train_accelerate.py

import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from torch.utils.data import DataLoader

import numpy as np
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Import models and utilities
from src.datasets.sequence_graph_datasets import SequenceGraphDataset
from src.graph_models.models.graph_networks import (
    GraphConvolutionNetwork,
    GraphAttentionNetwork,
    GraphTransformer,
    MeshGraphNet
)
from src.graph_models.models.graph_autoencoders import (
    GraphConvolutionalAutoEncoder,
    GraphAttentionAutoEncoder,
    GraphTransformerAutoEncoder,
    MeshGraphAutoEncoder
)
from src.graph_models.models.intgnn.models import GNN_TopK
from src.graph_models.models.multiscale.gnn import (
    SinglescaleGNN, 
    MultiscaleGNN, 
    TopkMultiscaleGNN
)
from src.graph_simulators.utils import (
    generate_results_folder_name,
    save_metadata,
    get_scheduler,
    set_random_seed
)
from src.graph_simulators.config import parse_args

from src.graph_simulators.trainers_accelerate import SequenceTrainerAccelerate

def is_autoencoder_model(model_name):
    """
    Determines if the given model name corresponds to an autoencoder.
    """
    return model_name.lower().endswith('-ae') or model_name.lower() in ['multiscale-topk']

def main():
    args = parse_args()

    # Set device
    if args.cpu_only:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Generate data directory
    graph_data_dir = os.path.join(args.base_data_dir, args.dataset, f"{args.data_keyword}_graphs")
    logging.info(f"Graph data directory: {graph_data_dir}")

    # Generate results folder name
    if args.results_folder is not None:
        results_folder = args.results_folder
    else:
        results_folder = generate_results_folder_name(args)
    logging.info(f"Results will be saved to {results_folder}")
    os.makedirs(results_folder, exist_ok=True)

    # Set random seed
    set_random_seed(args.random_seed)

    # Determine if the model requires edge_attr
    models_requiring_edge_attr = [
        'intgnn', 'gtr', 'mgn', 'gtr-ae', 'mgn-ae', 
        'singlescale', 'multiscale', 'multiscale-topk'
    ]
    use_edge_attr = args.model.lower() in models_requiring_edge_attr
    logging.info(f"Model '{args.model}' requires edge_attr: {use_edge_attr}")

    # Initialize dataset
    dataset = SequenceGraphDataset(
        graph_data_dir=graph_data_dir,
        initial_step=args.initial_step,
        final_step=args.final_step,
        max_prediction_horizon=args.horizon,
        # task=args.task,
        include_settings=args.include_settings,
        identical_settings=args.identical_settings,
        # settings_file=args.settings_file,
        use_edge_attr=use_edge_attr,
        subsample_size=args.subsample_size
    )

    # Subset dataset if ntrain is specified
    total_dataset_size = len(dataset)
    if args.ntrain is not None:
        np.random.seed(args.random_seed)  # For reproducibility
        indices = np.random.permutation(total_dataset_size)[:args.ntrain]
        dataset = Subset(dataset, indices)

    # Create the DataLoader with the custom collate function
    def collate_fn(batch):
        """
        Custom collate function for DataLoader.
        Processes batch into initial graphs, target graphs, and sequence lengths.
        If settings are included, also processes settings.
        """
        initial_graphs = []
        target_graphs = []
        seq_lengths = []
        settings = []  # Initialize if settings are included

        for sample in batch:
            # Each sample is a tuple: (initial_graph, target_graph, seq_length)
            if len(sample) == 4:
                initial_graph, target_graph, seq_length, setting = sample
                initial_graphs.append(initial_graph)
                target_graphs.append(target_graph)
                seq_lengths.append(seq_length)
                settings.append(setting)
            else:
                initial_graph, target_graph, seq_length = sample
                initial_graphs.append(initial_graph)
                target_graphs.append(target_graph)
                seq_lengths.append(seq_length)

        # Convert seq_lengths to a tensor
        seq_lengths = torch.tensor(seq_lengths)

        # Optionally include settings
        if settings:
            settings = torch.stack(settings)  # Assuming settings_tensor is a tensor
            return initial_graphs, target_graphs, seq_lengths, settings
        else:
            return initial_graphs, target_graphs, seq_lengths

    # Flattening the dataset
    flattened_data = []
    for data_sequences in dataset:
        flattened_data.extend(data_sequences)
        
    print("Total size of repeated data:", len(flattened_data))

    # Create the DataLoader with the custom collate function
    dataloader = DataLoader(
        flattened_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    batch = next(iter(dataloader))
    initial_graphs, target_graphs_list, seq_lengths = batch
    logging.info(f"Initial graphs type: {type(initial_graphs)}")
    logging.info(f"Target graphs list type: {type(target_graphs_list)}")
    logging.info(f"Target graphs list length: {len(target_graphs_list)}")
    logging.info(f"Sequence lengths type: {type(seq_lengths)}")
    
    logging.info(f"Initial graphs [0] type: {type(initial_graphs[0])}")
    logging.info(f"Target graphs list [0] type: {type(target_graphs_list[0])}")
    logging.info(f"Target graphs list [0] length: {len(target_graphs_list[0])}")
    logging.info(f"Sequence lengths [0] type: {type(seq_lengths[0])}")
    
    for initial_graph, target_graphs, seq_length in zip(initial_graphs, target_graphs_list, seq_lengths):
        logging.info(f"Initial graphs type: {type(initial_graphs)}")
        logging.info(f"Target graphs list type: {type(target_graphs_list)}")
        logging.info(f"Target graphs list length: {len(target_graphs_list)}")
        logging.info(f"Sequence lengths type: {type(seq_lengths)}")

    # Get a sample data for model initialization
    sample_initial_graph = dataset[0][0][0]
    sample_final_graph = dataset[0][0][1]
    
    if args.include_settings:
        sample_settings = dataset[0][0][3]
    
    in_channels = sample_initial_graph.x.shape[1]
    out_channels = sample_final_graph.x.shape[1]  # Assuming node features remain the same

    # Model initialization
    if args.model.lower() == 'gcn':
        model = GraphConvolutionNetwork(
            in_channels=in_channels,
            hidden_dim=args.hidden_dim,
            out_channels=out_channels,
            num_layers=args.num_layers,
            pool_ratios=args.pool_ratios,
        )
        
    elif args.model.lower() == 'mgn':
        # Model parameters specific to MeshGraphNet
        node_in_dim = sample.x.shape[1]
        edge_in_dim = sample.edge_attr.shape[1] if hasattr(sample, 'edge_attr') and sample.edge_attr is not None else 0
        node_out_dim = sample.y.shape[1]
        hidden_dim = args.hidden_dim
        num_layers = args.num_layers

        # Initialize MeshGraphNet model
        model = MeshGraphNet(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            node_out_dim=node_out_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
        logging.info("Initialized MeshGraphNet model.")

    else:
        raise NotImplementedError(f"Model '{args.model}' is not implemented in this script.")

    logging.info(f"Initialized model: {args.model}")
    model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    logging.info(f"Initialized Adam optimizer with learning rate: {args.lr}")

    # Scheduler
    scheduler = get_scheduler(args, optimizer)
    if scheduler:
        logging.info("Initialized learning rate scheduler.")

    # Define the loss function
    criterion = torch.nn.MSELoss()

    # Initialize trainer with the loss function
    trainer = SequenceTrainerAccelerate(
        model=model,
        train_loader=dataloader,
        val_loader=None,
        optimizer=optimizer,
        scheduler=scheduler,
        nepochs=args.nepochs,
        results_folder=results_folder,
        checkpoint=args.checkpoint,
        random_seed=args.random_seed,
        device=device,
        verbose=args.verbose,
        criterion=criterion,
        discount_factor=args.discount_factor  # Ensure this argument is in args
    )
    logging.info("Initialized SequenceTrainerAccelerate with custom loss function.")
    
    # Save metadata
    save_metadata(args, model, results_folder)

    # Run train or evaluate
    if args.mode == 'train':
        trainer.train()
    else:
        logging.info("Evaluation mode is not implemented yet.")
        pass

if __name__ == "__main__":
    main()