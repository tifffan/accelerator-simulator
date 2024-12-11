# sequence_context_dataloaders.py: DataLoaders for the SequenceGraphSettingsPositionScaleDataset.

import os
import re
import torch
import logging
from torch.utils.data import Dataset, Subset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

from src.datasets.sequence_graph_position_scale_datasets import SequenceGraphSettingsPositionScaleDataset


class SequenceGraphSettingsDataLoaders:
    def __init__(
        self,
        graph_data_dir,
        initial_step=0,
        final_step=10,
        max_prediction_horizon=3,
        include_settings=False,
        identical_settings=False,
        use_edge_attr=False,
        subsample_size=None,
        include_position_index=False,
        include_scaling_factors=False,
        scaling_factors_file=None,
        task='predict_n6d',
        edge_attr_method="v0",
        preload_data=False,
        batch_size=32,
        n_train=100,
        n_val=20,
        n_test=20
    ):
        """
        Initializes the SequenceGraphSettingsDataLoaders.

        Args:
            graph_data_dir (str): Directory containing graph data organized by steps.
            initial_step (int, optional): Starting step index. Defaults to 0.
            final_step (int, optional): Ending step index. Defaults to 10.
            max_prediction_horizon (int, optional): Maximum number of future steps to predict. Defaults to 3.
            include_settings (bool, optional): Whether to include additional settings data. Defaults to False.
            identical_settings (bool, optional): If True, uses a single settings file for all samples. Defaults to False.
            use_edge_attr (bool, optional): Whether to compute and include edge attributes. Defaults to False.
            subsample_size (int, optional): Number of samples to include. If None, includes all. Defaults to None.
            include_position_index (bool, optional): Whether to include the normalized position index. Defaults to False.
            include_scaling_factors (bool, optional): Whether to include scaling factors from a file. Defaults to False.
            scaling_factors_file (str, optional): Path to the scaling factors text file. Defaults to None.
            task (str, optional): Prediction task. One of ['predict_n6d', 'predict_n4d', 'predict_n2d']. Defaults to 'predict_n6d'.
            edge_attr_method (str, optional): Method for edge attribute computation. Defaults to "v0".
            preload_data (bool, optional): If True, preloads all data into memory. Defaults to False.
            batch_size (int, optional): Batch size for the DataLoaders. Defaults to 32.
            n_train (int, optional): Number of training graphs. Defaults to 100.
            n_val (int, optional): Number of validation graphs. Defaults to 20.
            n_test (int, optional): Number of testing graphs. Defaults to 20.
        """
        # Initialize the dataset with provided parameters
        self.dataset = SequenceGraphSettingsPositionScaleDataset(
            graph_data_dir=graph_data_dir,
            initial_step=initial_step,
            final_step=final_step,
            max_prediction_horizon=max_prediction_horizon,
            include_settings=include_settings,
            identical_settings=identical_settings,
            use_edge_attr=use_edge_attr,
            subsample_size=subsample_size,
            include_position_index=include_position_index,
            include_scaling_factors=include_scaling_factors,
            scaling_factors_file=scaling_factors_file,
            task=task,
            edge_attr_method=edge_attr_method,
            preload_data=preload_data
        )

        logging.info(f"Dataset initialized with size: {len(self.dataset)}")

        # Sort the dataset indices based on the integer extracted from filenames
        sorted_indices = sorted(
            range(len(self.dataset)),
            key=lambda idx: self._extract_graph_sequence_id(idx)
        )

        # Total samples required
        total_samples = len(self.dataset)
        n_total = n_train + n_val + n_test

        if n_total > total_samples:
            raise ValueError(f"n_train + n_val + n_test ({n_total}) exceeds dataset size ({total_samples}).")

        # Select test indices as the last n_test samples
        test_indices = sorted_indices[-n_test:]
        remaining_indices = sorted_indices[:-n_test]

        # Split remaining_indices into train and val
        n_remaining = len(remaining_indices)
        if n_train + n_val > n_remaining:
            raise ValueError(f"n_train + n_val ({n_train + n_val}) exceeds remaining dataset size ({n_remaining}) after excluding test samples.")

        train_indices = remaining_indices[:n_train]
        val_indices = remaining_indices[n_train:n_train + n_val]

        # Create subsets
        self.train_set = Subset(self.dataset, train_indices)
        self.val_set = Subset(self.dataset, val_indices)
        self.test_set = Subset(self.dataset, test_indices)

        self.batch_size = batch_size

        # Initialize DataLoaders as None
        self._train_loader = None
        self._val_loader = None
        self._test_loader = None
        self._all_data_loader = None

        logging.info(f"Initialized SequenceGraphSettingsDataLoaders with {n_train} train, {n_val} val, and {n_test} test samples.")

        # Define the custom collate function
        self.collate_fn = self._collate_fn

    def _extract_graph_sequence_id(self, idx):
        """
        Extracts a unique identifier for each graph sequence based on its index.

        Args:
            idx (int): Index of the graph sequence.

        Returns:
            int: Extracted sequence identifier.
        """
        # Assuming that each graph sequence has a unique identifier, possibly extracted from filenames
        # Modify this method based on your actual filename patterns or sequence identifiers
        # For example, if filenames are like "graph_0_step_0.pt", "graph_0_step_1.pt", etc.,
        # you can extract the graph ID from the first step's filename
        sample = self.dataset.graph_paths[idx][0]  # First graph in the sequence
        filename = os.path.basename(sample)
        match = re.search(r'graph_(\d+)_step_\d+\.pt', filename)
        if match:
            return int(match.group(1))
        else:
            # Fallback to idx if pattern does not match
            return idx

    def _collate_fn(self, batch):
        """
        Custom collate function for DataLoader.
        This version creates batched graph objects using PyTorch Geometric's Batch class.

        Args:
            batch (list of tuples): Each tuple contains (initial_graph, target_graph, seq_length)
                                    or (initial_graph, target_graph, seq_length, settings_tensor) if settings are included.

        Returns:
            tuple: Batched initial graphs, batched target graphs, sequence lengths, and settings tensors (if included).
        """
        logging.debug("Custom collate_fn called.")

        initial_graphs = []
        target_graphs = []
        seq_lengths = []
        settings_list = []
        include_settings = False

        for sample in batch:
            if len(sample) == 4:
                initial_graph, target_graph, seq_length, setting = sample
                include_settings = True
                settings_list.append(setting)
            else:
                initial_graph, target_graph, seq_length = sample

            initial_graphs.append(initial_graph)
            target_graphs.append(target_graph)
            seq_lengths.append(seq_length)

        # Batched graphs
        batch_initial = Batch.from_data_list(initial_graphs)
        batch_target = Batch.from_data_list(target_graphs)
        logging.debug("Batched initial and target graphs using Batch.from_data_list.")

        if include_settings:
            settings_tensor = torch.stack(settings_list, dim=0)
            logging.debug(f"Settings tensor created with shape: {settings_tensor.shape}")
            return batch_initial, batch_target, torch.tensor(seq_lengths), settings_tensor
        else:
            return batch_initial, batch_target, torch.tensor(seq_lengths)

    def get_train_loader(self):
        if self._train_loader is None:
            self._train_loader = DataLoader(
                self.train_set,
                batch_size=self.batch_size,
                shuffle=True,  # Shuffle only the training data
                num_workers=4,  # Adjust based on your system
                pin_memory=True,
                collate_fn=self.collate_fn
            )
            logging.info(f"Created training DataLoader with batch size {self.batch_size}.")
        return self._train_loader

    def get_val_loader(self):
        if self._val_loader is None:
            self._val_loader = DataLoader(
                self.val_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                collate_fn=self.collate_fn
            )
            logging.info(f"Created validation DataLoader with batch size {self.batch_size}.")
        return self._val_loader

    def get_test_loader(self):
        if self._test_loader is None:
            self._test_loader = DataLoader(
                self.test_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                collate_fn=self.collate_fn
            )
            logging.info(f"Created testing DataLoader with batch size {self.batch_size}.")
        return self._test_loader

    def get_all_data_loader(self):
        """
        Returns a DataLoader for the entire dataset as a single batch, without splitting.
        Useful for evaluation or visualization.

        Returns:
            torch_geometric.loader.DataLoader: DataLoader for the entire dataset.
        """
        if self._all_data_loader is None:
            self._all_data_loader = DataLoader(
                self.dataset,
                batch_size=len(self.dataset),
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                collate_fn=self.collate_fn
            )
            logging.info(f"Created all-data DataLoader with batch size {len(self.dataset)}.")
        return self._all_data_loader
