# dataloaders_rollout_v2.py

import os
import logging
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch.utils.data import Subset
from datasets_rollout import SequenceGraphSettingsDataset

class SequenceGraphSettingsDataLoaders:
    """
    DataLoader class for rollout datasets using efficient sliding window batching.
    Supports any horizon and any number of graph sequences.
    """
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
        position_encoding_method="normalized",
        sinusoidal_encoding_dim=64,
        include_scaling_factors=False,
        scaling_factors_file=None,
        batch_size=32,
        n_train=100,
        n_val=20,
        n_test=20
    ):
        """
        Initializes the DataLoaders using the rollout dataset (sliding window version).
        Each sample from the dataset is a tuple:
           (input_graph, target_graph_list, seq_length, [settings_list])
        """
        self.dataset = SequenceGraphSettingsDataset(
            graph_data_dir=graph_data_dir,
            initial_step=initial_step,
            final_step=final_step,
            max_prediction_horizon=max_prediction_horizon,
            include_settings=include_settings,
            identical_settings=identical_settings,
            use_edge_attr=use_edge_attr,
            subsample_size=subsample_size,
            include_position_index=include_position_index,
            position_encoding_method=position_encoding_method,
            sinusoidal_encoding_dim=sinusoidal_encoding_dim,
            include_scaling_factors=include_scaling_factors,
            scaling_factors_file=scaling_factors_file,
        )

        logging.info(f"Dataset (rollout, sliding window) initialized with size: {len(self.dataset)}")

        # Create train, val, and test subsets using sorted indices.
        sorted_indices = sorted(range(len(self.dataset)))
        total_samples = len(self.dataset)
        n_total = n_train + n_val + n_test

        # print(f"[SPLIT INFO] n_train: {n_train}")
        # print(f"[SPLIT INFO] n_val: {n_val}")
        # print(f"[SPLIT INFO] n_test: {n_test}")
        # print(f"[SPLIT INFO] Total: {n_train + n_val + n_test}")
        # print(f"[SPLIT INFO] Dataset size: {len(self.dataset)}")

        if n_total > total_samples:
            raise ValueError(f"n_train + n_val + n_test ({n_total}) exceeds dataset size ({total_samples}).")
        test_indices = sorted_indices[-n_test:] if n_test > 0 else []
        remaining_indices = sorted_indices[:-n_test] if n_test > 0 else sorted_indices
        # if n_train + n_val > len(remaining_indices):
        #     raise ValueError("Not enough samples for train+val split.")
        train_indices = remaining_indices[:n_train]
        val_indices = remaining_indices[n_train:n_train + n_val]

        self.train_set = Subset(self.dataset, train_indices)
        self.val_set = Subset(self.dataset, val_indices)
        self.test_set = Subset(self.dataset, test_indices)
        self.batch_size = batch_size

        
        self._train_loader = None
        self._val_loader = None
        self._test_loader = None
        self._all_data_loader = None

        logging.info(f"Initialized DataLoaders with {n_train} train, {n_val} val, and {n_test} test samples.")

        self.collate_fn = self._collate_fn

    def _collate_fn(self, batch):
        include_settings = (len(batch[0]) == 4)
        input_graphs = [sample[0] for sample in batch]
        batched_input = Batch.from_data_list(input_graphs)
        seq_lengths = torch.tensor([sample[2] for sample in batch], dtype=torch.long)
        max_horizon = max(sample[2] for sample in batch)
        batched_targets = []
        if include_settings:
            batched_settings = []
        for h in range(max_horizon):
            targets_h = []
            if include_settings:
                settings_h = []
            for sample in batch:
                if sample[2] > h:
                    targets_h.append(sample[1][h])
                    if include_settings:
                        settings_h.append(sample[3][h])
            batched_targets.append(Batch.from_data_list(targets_h))
            if include_settings:
                batched_settings.append(torch.stack(settings_h, dim=0))
        if include_settings:
            return batched_input, batched_targets, seq_lengths, batched_settings
        else:
            return batched_input, batched_targets, seq_lengths

    def get_train_loader(self):
        if self._train_loader is None:
            self._train_loader = DataLoader(
                self.train_set,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
                collate_fn=self.collate_fn,
                persistent_workers=False
            )
            logging.info(f"Created training DataLoader with batch size {self.batch_size}.")
        return self._train_loader

    def get_val_loader(self):
        if self._val_loader is None:
            self._val_loader = DataLoader(
                self.val_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                collate_fn=self.collate_fn,
                persistent_workers=False
            )
            logging.info(f"Created validation DataLoader with batch size {self.batch_size}.")
        return self._val_loader

    def get_test_loader(self):
        if self._test_loader is None:
            self._test_loader = DataLoader(
                self.test_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                collate_fn=self.collate_fn,
                persistent_workers=False
            )
            logging.info(f"Created testing DataLoader with batch size {self.batch_size}.")
        return self._test_loader

    def get_all_data_loader(self):
        if self._all_data_loader is None:
            self._all_data_loader = DataLoader(
                self.dataset,
                batch_size=len(self.dataset),
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                collate_fn=self.collate_fn,
                persistent_workers=False
            )
            logging.info(f"Created all-data DataLoader with batch size {len(self.dataset)}.")
        return self._all_data_loader 