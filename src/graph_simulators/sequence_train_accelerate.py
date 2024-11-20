# sequence_train_accelerate.py

import torch
import torch.nn.functional as F
from torch.utils.data import Subset
# from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader

import numpy as np
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Import your models and utilities
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
from sequence_utils import (
    generate_results_folder_name,
    save_metadata,
    get_scheduler,
    set_random_seed
)
from sequence_config import parse_args

from sequence_trainers_accelerate import SequenceTrainerAccelerate

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
    
    # # Print data type and example at each level
    # print("Data type of dataset:", type(dataset))
    # print("Example of dataset[0]:", dataset[0])

    # print("Data type of dataset[0]:", type(dataset[0])) list
    # print("Example of dataset[0][0]:", dataset[0][0])

    # print("Data type of dataset[0][0]:", type(dataset[0][0])) tuple
    # print("Example of dataset[0][0][0]:", dataset[0][0][0])

    # print("Data type of dataset[0][0][0]:", type(dataset[0][0][0])) torch_geometric.data.data.Data
    # print("Example of dataset[0][0][0].x:", dataset[0][0][0].x)
    
    
    # print("Type of sample_initial_graph:", type(sample_initial_graph))
    
    in_channels = sample_initial_graph.x.shape[1]
    out_channels = sample_initial_graph.x.shape[1]  # Assuming node features remain the same

    # Model initialization
    if args.model.lower() == 'gcn':
        model = GraphConvolutionNetwork(
            in_channels=in_channels,
            hidden_dim=args.hidden_dim,
            out_channels=out_channels,
            num_layers=args.num_layers,
            pool_ratios=args.pool_ratios,
        )
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
        dataloader=dataloader,
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