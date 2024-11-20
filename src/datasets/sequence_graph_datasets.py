# Filename: sequence_graph_dataset.py

import os
import torch
from torch.utils.data import Dataset

class SequenceGraphDataset(Dataset):
    def __init__(self, graph_data_dir, initial_step=0, final_step=10, max_prediction_horizon=3,
                 include_settings=False, identical_settings=False,
                 use_edge_attr=False, subsample_size=None):
        self.graph_data_dir = graph_data_dir
        self.initial_step = initial_step
        self.final_step = final_step
        self.max_prediction_horizon = max_prediction_horizon
        self.sequence_length = final_step - initial_step + 1
        self.include_settings = include_settings
        self.identical_settings = identical_settings
        self.use_edge_attr = use_edge_attr
        self.subsample_size = subsample_size  # Added subsample_size parameter
        
        self.graph_paths = self._load_graph_paths()

        # Subsample the dataset if subsample_size is specified
        if self.subsample_size is not None:
            self.graph_paths = self.graph_paths[:self.subsample_size]

        if self.include_settings:
            if self.identical_settings:
                settings_file = os.path.join(graph_data_dir, 'settings.pt')
                if not os.path.isfile(settings_file):
                    raise ValueError(f"Settings file not found: {settings_file}")
                self.settings = torch.load(settings_file)
                self.settings_tensor = self.settings_dict_to_tensor(self.settings)
            else:
                self.settings_files = self._load_settings_files()
                # Subsample settings_files to match the subsampled graph_paths
                if self.subsample_size is not None:
                    self.settings_files = self.settings_files[:self.subsample_size]
                if len(self.settings_files) != len(self.graph_paths):
                    raise ValueError("Mismatch between number of graph sequences and settings files.")

    def _load_graph_paths(self):
        # Assuming the graphs are stored in directories named 'step_0', 'step_1', etc.
        graph_dirs = [os.path.join(self.graph_data_dir, f'step_{i}') for i in range(self.initial_step, self.final_step + 1)]
        graph_paths = []

        for dir in graph_dirs:
            files = sorted(os.listdir(dir))
            files = [os.path.join(dir, f) for f in files if f.endswith('.pt') and not f.endswith('_settings.pt')]
            graph_paths.append(files)

        # Transpose to get list of sequences
        graph_paths = list(zip(*graph_paths))
        return graph_paths

    def _load_settings_files(self):
        settings_files = []
        for sequence in self.graph_paths:
            # Get base filename from the first graph in the sequence
            base_fname = os.path.basename(sequence[0]).replace('.pt', '')
            # Assuming settings files are named like '<base_fname>_settings.pt' and located in the same directory
            settings_file = os.path.join(os.path.dirname(sequence[0]), f"{base_fname}_settings.pt")
            if not os.path.isfile(settings_file):
                raise ValueError(f"Settings file not found: {settings_file}")
            settings_files.append(settings_file)
        return settings_files

    def __len__(self):
        return len(self.graph_paths)

    def __getitem__(self, idx):
        """
        Retrieve a sample from the dataset.
        
        Returns:
            list of tuples: Each tuple contains (initial_graph, target_graph, seq_length)
                            or (initial_graph, target_graph, seq_length, settings_tensor) if settings are included.
        """
        # Load the sequence of graphs for the given index
        sequence = self.graph_paths[idx]
        data_list = [torch.load(path) for path in sequence]

        # Process data_list to compute edge attributes if needed
        for data in data_list:
            if self.use_edge_attr:
                self._compute_edge_attr(data)
            else:
                data.edge_attr = None  # Explicitly set to None if not used

        # Prepare data for training: list of (initial_graph, target_graph, seq_length)
        data_sequences = []
        for t in range(len(data_list) - self.max_prediction_horizon):
            initial_graph = data_list[t]
            target_graphs = data_list[t+1:t+1+self.max_prediction_horizon]
            seq_length = len(target_graphs)
            for target_graph in target_graphs:
                data_sequences.append((initial_graph, target_graph, seq_length))  # Tuple format

        # If settings are included, attach settings to each tuple
        if self.include_settings:
            if self.identical_settings:
                settings_tensor = self.settings_tensor  # Use the preloaded settings tensor
            else:
                settings_file = self.settings_files[idx]
                settings = torch.load(settings_file)
                settings_tensor = self.settings_dict_to_tensor(settings)
            
            # Attach settings to each tuple (optional)
            data_sequences = [
                (initial_graph, target_graph, seq_length, settings_tensor)
                for initial_graph, target_graph, seq_length in data_sequences
            ]

        return data_sequences  # Each element is a tuple as per above



    def _compute_edge_attr(self, data):
        """
        Computes and assigns edge attributes to the data object.

        Args:
            data (torch_geometric.data.Data): The graph data object.
        """
        # Ensure 'pos' attribute is present
        if not hasattr(data, 'pos') or data.pos is None:
            # Assuming positions are in the first 3 features of x
            data.pos = data.x[:, :3]
        if hasattr(data, 'edge_index') and data.edge_index is not None:
            row, col = data.edge_index
            pos_diff = data.pos[row] - data.pos[col]  # Shape: [num_edges, 3]
            distance = torch.norm(pos_diff, p=2, dim=1, keepdim=True)  # Shape: [num_edges, 1]
            edge_attr = torch.cat([pos_diff, distance], dim=1)  # Shape: [num_edges, 4]

            # Standardize edge attributes
            eps = 1e-10
            edge_attr_mean = edge_attr.mean(dim=0, keepdim=True)
            edge_attr_std = edge_attr.std(dim=0, keepdim=True)
            edge_attr = (edge_attr - edge_attr_mean) / (edge_attr_std + eps)

            data.edge_attr = edge_attr  # Assign the standardized edge attributes
        else:
            raise ValueError("Data object is missing 'edge_index', cannot compute 'edge_attr'.")

    def settings_dict_to_tensor(self, settings_dict):
        """
        Converts a settings dictionary to a tensor.

        Args:
            settings_dict (dict): Dictionary of settings.

        Returns:
            torch.Tensor: Tensor of settings values.
        """
        # Sort settings by key to maintain consistent order
        keys = sorted(settings_dict.keys())
        values = []
        for key in keys:
            value = settings_dict[key]
            if isinstance(value, torch.Tensor):
                value = value.squeeze().float()
            else:
                value = torch.tensor(float(value)).float()
            values.append(value)
        settings_tensor = torch.stack(values)
        return settings_tensor
