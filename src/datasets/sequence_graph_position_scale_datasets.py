# sequence_graph_dataset.py

import os
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import re

MAX_STEP_INDEX = 76  # Predefined maximum step index for normalization

class SequenceGraphSettingsPositionScaleDataset(Dataset):
    def __init__(self, graph_data_dir, initial_step=0, final_step=10, max_prediction_horizon=3,
                 include_settings=False, identical_settings=False,
                 use_edge_attr=False, subsample_size=None,
                 include_position_index=False, include_scaling_factors=False,
                 scaling_factors_file=None):
        """
        Initializes the SequenceGraphSettingsPositionScaleDataset.

        Args:
            graph_data_dir (str): Directory containing graph data organized by steps.
            initial_step (int): Starting step index.
            final_step (int): Ending step index.
            max_prediction_horizon (int): Maximum number of future steps to predict.
            include_settings (bool): Whether to include additional settings data.
            identical_settings (bool): If True, uses a single settings file for all samples.
            use_edge_attr (bool): Whether to compute and include edge attributes.
            subsample_size (int, optional): Number of samples to include. If None, includes all.
            include_position_index (bool): Whether to include the normalized position index.
            include_scaling_factors (bool): Whether to include scaling factors from a file.
            scaling_factors_file (str, optional): Path to the scaling factors text file.
        """
        self.graph_data_dir = graph_data_dir
        self.initial_step = initial_step
        self.final_step = final_step
        self.max_prediction_horizon = max_prediction_horizon
        self.sequence_length = final_step - initial_step + 1
        self.include_settings = include_settings
        self.identical_settings = identical_settings
        self.use_edge_attr = use_edge_attr
        self.subsample_size = subsample_size
        self.include_position_index = include_position_index
        self.include_scaling_factors = include_scaling_factors
        self.scaling_factors_file = scaling_factors_file

        # Load graph paths
        self.graph_paths = self._load_graph_paths()

        # print(f"Total graph sequences loaded: {len(self.graph_paths)}")

        # Subsample the dataset if subsample_size is specified
        if self.subsample_size is not None:
            self.graph_paths = self.graph_paths[:self.subsample_size]
            # print(f"Subsampled graph sequences to: {len(self.graph_paths)}")

        # Load settings if required
        if self.include_settings:
            if self.identical_settings:
                settings_file = os.path.join(graph_data_dir, 'settings.pt')
                if not os.path.isfile(settings_file):
                    raise ValueError(f"Settings file not found: {settings_file}")
                self.settings = torch.load(settings_file)
                self.settings_tensor = self.settings_dict_to_tensor(self.settings)
                # print(f"Loaded identical settings tensor shape: {self.settings_tensor.shape}")
            else:
                self.settings_files = self._load_settings_files()
                # Subsample settings_files to match the subsampled graph_paths
                if self.subsample_size is not None:
                    self.settings_files = self.settings_files[:self.subsample_size]
                if len(self.settings_files) != len(self.graph_paths):
                    raise ValueError("Mismatch between number of graph sequences and settings files.")
                # print(f"Loaded individual settings files: {len(self.settings_files)}")

        # Load scaling factors if required
        if self.include_scaling_factors:
            if self.scaling_factors_file is None:
                raise ValueError("Scaling factors file must be provided when include_scaling_factors is True.")
            self.scaling_factors = self._load_scaling_factors(self.scaling_factors_file)
            # print(f"Loaded scaling factors for {len(self.scaling_factors)} steps.")

        # Determine the maximum step index for normalization if position index is included
        if self.include_position_index:
            self.max_x = MAX_STEP_INDEX
            # print(f"Determined maximum step index for normalization: {self.max_x}")

    def _load_graph_paths(self):
        """
        Loads and sorts graph file paths based on the integer x in filenames like "graph_x.pt".

        Returns:
            list of tuples: Each tuple contains file paths for a sequence across steps.
        """
        graph_dirs = [os.path.join(self.graph_data_dir, f'step_{i}') for i in range(self.initial_step, self.final_step + 1)]
        graph_paths_per_step = []

        for dir in graph_dirs:
            if not os.path.isdir(dir):
                raise ValueError(f"Graph directory not found: {dir}")
            files = sorted(os.listdir(dir), key=self._extract_graph_x)
            files = [os.path.join(dir, f) for f in files if f.endswith('.pt') and not f.endswith('_settings.pt')]
            graph_paths_per_step.append(files)
            # print(f"Loaded {len(files)} graph files from {dir}")

        # Transpose to get list of sequences
        graph_paths = list(zip(*graph_paths_per_step))
        return graph_paths

    def _extract_graph_x(self, filename):
        """
        Extracts the integer x from a filename like "graph_x.pt".

        Args:
            filename (str): The graph filename.

        Returns:
            int: The extracted x value. If not found, returns -1 to push it to the start.
        """
        match = re.search(r'graph_(\d+)\.pt', filename)
        if match:
            return int(match.group(1))
        return -1  # Default value if x is not found

    def _load_settings_files(self):
        """
        Loads settings file paths corresponding to each graph sequence.

        Returns:
            list of str: Paths to settings files.
        """
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

    def _load_scaling_factors(self, scaling_factors_file):
        """
        Loads scaling factors from a text file.

        Args:
            scaling_factors_file (str): Path to the scaling factors text file.

        Returns:
            dict: Mapping from step number to a list of scaling factors (mean and std).
                  Each value is a list of 12 floats [mean1, ..., mean6, std1, ..., std6].
        """
        scaling_factors = {}
        current_section = None
        with open(scaling_factors_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith("Per-Step Global Mean:"):
                    current_section = "mean"
                    continue
                elif line.startswith("Per-Step Global Std:"):
                    current_section = "std"
                    continue
                elif line.startswith("Step"):
                    # Extract step number and values
                    step_match = re.match(r'Step (\d+): (.+)', line)
                    if step_match:
                        step = int(step_match.group(1))
                        values_str = step_match.group(2)
                        values = [float(v.strip()) for v in values_str.split(',')]
                        if len(values) != 6:
                            raise ValueError(f"Expected 6 scaling values per section per step, got {len(values)} for Step {step}")
                        if step not in scaling_factors:
                            scaling_factors[step] = {}
                        if current_section == "mean":
                            scaling_factors[step]['mean'] = values
                        elif current_section == "std":
                            scaling_factors[step]['std'] = values

        # Combine mean and std into a single list per step
        combined_scaling_factors = {}
        for step, factors in scaling_factors.items():
            if 'mean' not in factors or 'std' not in factors:
                raise ValueError(f"Missing mean or std for step {step} in scaling factors file.")
            combined_scaling_factors[step] = factors['mean'] + factors['std']
            # print(f"Step {step}: Combined scaling factors shape: {len(combined_scaling_factors[step])}")

        return combined_scaling_factors

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
        for t in range(len(data_list) - 1):  # Iterate until the second last graph
            initial_graph = data_list[t]
            target_graphs = data_list[t+1:t+1+self.max_prediction_horizon]
            seq_length = len(target_graphs)
            for target_graph in target_graphs:
                data_sequences.append((initial_graph, target_graph, seq_length))  # Tuple format

        # If settings are included, attach settings to each tuple
        if self.include_settings:
            if self.identical_settings:
                settings_tensor = self.settings_tensor  # Use the preloaded settings tensor
                # print(f"Using identical settings tensor for all sequences.")
            else:
                settings_file = self.settings_files[idx]
                settings = torch.load(settings_file)
                settings_tensor = self.settings_dict_to_tensor(settings)
                # print(f"Loaded settings tensor shape for index {idx}: {settings_tensor.shape}")

            # Append additional features if flags are set
            if self.include_position_index:
                augmented_settings = []
                for seq_idx, (initial_graph, target_graph, seq_length) in enumerate(data_sequences):
                    # Start with existing settings
                    current_settings = settings_tensor.clone()

                    # Compute step indices based on position in the sequence
                    # initial_step corresponds to t=0
                    # For each initial_graph, step_index = initial_step + t
                    # For each target_graph, step_index = initial_step + t + delta (where delta >=1)

                    # Determine t based on position in data_sequences
                    # Since data_sequences is a flat list, we need to compute t based on idx
                    # Alternatively, keep track of t in the outer loop
                    # To simplify, we'll assume each data_sequence corresponds to a unique t and delta

                    # Compute step index for initial_graph
                    # Since data_sequences are appended in order, we can compute step_index based on seq_idx
                    # Number of unique t's = len(data_list) -1
                    num_t = len(data_list) -1
                    t_current = seq_idx // self.max_prediction_horizon  # Integer division to get t
                    step_index = self.initial_step + t_current

                    # Normalize step index
                    normalized_position = torch.tensor([step_index / self.max_x], dtype=torch.float)
                    current_settings = torch.cat([normalized_position, current_settings], dim=0)
                    # print(f"Appended normalized position index: {normalized_position}")

                    augmented_settings.append(current_settings)

                # Stack all augmented settings tensors
                settings_tensor = torch.stack(augmented_settings)
                # print(f"Augmented settings tensor shape after adding position index: {settings_tensor.shape}")

            # Attach settings to each tuple
            if self.include_position_index:
                data_sequences = [
                    (initial_graph, target_graph, seq_length, settings_tensor[i])
                    for i, (initial_graph, target_graph, seq_length) in enumerate(data_sequences)
                ]
            else:
                # If position index is not included, settings_tensor remains as is
                data_sequences = [
                    (initial_graph, target_graph, seq_length, settings_tensor[i])
                    for i, (initial_graph, target_graph, seq_length) in enumerate(data_sequences)
                ]

        # **Modification Starts Here**
        # Assign scaling factors to graph objects based on directory step index
        if self.include_scaling_factors:
            for i, data_tuple in enumerate(data_sequences):
                if self.include_settings:
                    if len(data_tuple) == 4:
                        initial_graph, target_graph, seq_length, settings_tensor = data_tuple
                    else:
                        raise ValueError("Expected 4 elements in the tuple when include_settings is True.")
                else:
                    if len(data_tuple) == 3:
                        initial_graph, target_graph, seq_length = data_tuple
                    else:
                        raise ValueError("Expected 3 elements in the tuple when include_settings is False.")

                # Determine t and delta based on position in the sequence
                # Since data_sequences are appended in order, we can compute t based on i
                # Number of unique t's = len(data_list) -1
                t_current = i // self.max_prediction_horizon
                delta = i % self.max_prediction_horizon + 1  # delta starts from 1

                # Compute step indices
                initial_step = self.initial_step + t_current
                target_step = initial_step + delta

                # Retrieve scaling factors
                scaling_initial = self.scaling_factors.get(initial_step, [0.0]*12)
                scaling_target = self.scaling_factors.get(target_step, [0.0]*12)

                # Assign scaling factors to the graphs as new fields
                initial_graph.scale = torch.tensor(scaling_initial, dtype=torch.float).unsqueeze(0)
                target_graph.scale = torch.tensor(scaling_target, dtype=torch.float).unsqueeze(0)

                # Replace the tuple with updated graphs
                if self.include_settings:
                    data_sequences[i] = (initial_graph, target_graph, seq_length, settings_tensor)
                else:
                    data_sequences[i] = (initial_graph, target_graph, seq_length)

        # **Modification Ends Here**

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
            if data.x.shape[1] < 3:
                raise ValueError("Node feature dimension is less than 3, cannot extract 'pos'.")
            data.pos = data.x[:, :3]
            # print(f"Assigned 'pos' from node features.")

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
            # print(f"Computed and assigned standardized edge attributes.")
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
                try:
                    value = torch.tensor(float(value)).float()
                except ValueError:
                    raise ValueError(f"Value for key '{key}' in settings_dict is not convertible to float.")
            values.append(value)
            # print(f"Converted setting '{key}' to tensor: {value}")

        settings_tensor = torch.stack(values)
        # print(f"Constructed settings tensor with shape: {settings_tensor.shape}")
        return settings_tensor


# sequence_graph_sequence_dataset.py

import os
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import re

MAX_STEP_INDEX = 76  # Predefined maximum step index for normalization

class SequenceGraphSettingsPositionScaleSequenceDataset(Dataset):
    def __init__(self, graph_data_dir, initial_step=0, final_step=10, max_prediction_horizon=3,
                 include_settings=False, identical_settings=False,
                 use_edge_attr=False, subsample_size=None,
                 include_position_index=False, include_scaling_factors=False,
                 scaling_factors_file=None):
        """
        Initializes the SequenceGraphSettingsPositionScaleSequenceDataset.

        This dataset returns full sequences of graphs (one per time step) rather than flattened pairs.
        Each sample corresponds to a sequence from initial_step to final_step.

        Args:
            graph_data_dir (str): Directory containing graph data organized by steps.
            initial_step (int): Starting step index.
            final_step (int): Ending step index.
            max_prediction_horizon (int): Maximum number of future steps to predict (unused here).
            include_settings (bool): Whether to include additional settings data.
            identical_settings (bool): If True, uses a single settings file for all samples.
            use_edge_attr (bool): Whether to compute and include edge attributes.
            subsample_size (int, optional): Number of samples to include. If None, includes all.
            include_position_index (bool): Whether to include the normalized position index in settings.
            include_scaling_factors (bool): Whether to include scaling factors from a file.
            scaling_factors_file (str, optional): Path to the scaling factors text file.
        """
        self.graph_data_dir = graph_data_dir
        self.initial_step = initial_step
        self.final_step = final_step
        self.max_prediction_horizon = max_prediction_horizon
        self.sequence_length = final_step - initial_step + 1
        self.include_settings = include_settings
        self.identical_settings = identical_settings
        self.use_edge_attr = use_edge_attr
        self.subsample_size = subsample_size
        self.include_position_index = include_position_index
        self.include_scaling_factors = include_scaling_factors
        self.scaling_factors_file = scaling_factors_file

        # Load graph paths as sequences (each element is a tuple with one file per step)
        self.graph_paths = self._load_graph_paths()

        if self.subsample_size is not None:
            self.graph_paths = self.graph_paths[:self.subsample_size]

        # Load settings if required
        if self.include_settings:
            if self.identical_settings:
                settings_file = os.path.join(graph_data_dir, 'settings.pt')
                if not os.path.isfile(settings_file):
                    raise ValueError(f"Settings file not found: {settings_file}")
                self.settings = torch.load(settings_file)
                self.settings_tensor = self.settings_dict_to_tensor(self.settings)
            else:
                self.settings_files = self._load_settings_files()
                if self.subsample_size is not None:
                    self.settings_files = self.settings_files[:self.subsample_size]
                if len(self.settings_files) != len(self.graph_paths):
                    raise ValueError("Mismatch between number of graph sequences and settings files.")

        # Load scaling factors if required
        if self.include_scaling_factors:
            if self.scaling_factors_file is None:
                raise ValueError("Scaling factors file must be provided when include_scaling_factors is True.")
            self.scaling_factors = self._load_scaling_factors(self.scaling_factors_file)

        # Determine maximum step index for normalization if position index is included
        if self.include_position_index:
            self.max_x = MAX_STEP_INDEX

    def _load_graph_paths(self):
        """
        Loads and sorts graph file paths based on filenames like "graph_x.pt" for each step.
        Returns:
            list of tuples: Each tuple contains file paths for the same sample across steps.
        """
        graph_dirs = [os.path.join(self.graph_data_dir, f'step_{i}') 
                      for i in range(self.initial_step, self.final_step + 1)]
        graph_paths_per_step = []

        for dir in graph_dirs:
            if not os.path.isdir(dir):
                raise ValueError(f"Graph directory not found: {dir}")
            files = sorted(os.listdir(dir), key=self._extract_graph_x)
            files = [os.path.join(dir, f) for f in files if f.endswith('.pt') and not f.endswith('_settings.pt')]
            graph_paths_per_step.append(files)

        # Transpose list of lists to get sequences (each tuple contains one file from each step)
        graph_paths = list(zip(*graph_paths_per_step))
        return graph_paths

    def _extract_graph_x(self, filename):
        """
        Extracts the integer x from a filename like "graph_x.pt".
        """
        match = re.search(r'graph_(\d+)\.pt', filename)
        if match:
            return int(match.group(1))
        return -1

    def _load_settings_files(self):
        """
        Loads settings file paths corresponding to each graph sequence.
        Returns:
            list of str: Paths to settings files.
        """
        settings_files = []
        for sequence in self.graph_paths:
            base_fname = os.path.basename(sequence[0]).replace('.pt', '')
            settings_file = os.path.join(os.path.dirname(sequence[0]), f"{base_fname}_settings.pt")
            if not os.path.isfile(settings_file):
                raise ValueError(f"Settings file not found: {settings_file}")
            settings_files.append(settings_file)
        return settings_files

    def _load_scaling_factors(self, scaling_factors_file):
        """
        Loads scaling factors from a text file.
        Returns:
            dict: Mapping from step number to a list of 12 floats [mean1,...,mean6, std1,...,std6].
        """
        scaling_factors = {}
        current_section = None
        with open(scaling_factors_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith("Per-Step Global Mean:"):
                    current_section = "mean"
                    continue
                elif line.startswith("Per-Step Global Std:"):
                    current_section = "std"
                    continue
                elif line.startswith("Step"):
                    step_match = re.match(r'Step (\d+): (.+)', line)
                    if step_match:
                        step = int(step_match.group(1))
                        values_str = step_match.group(2)
                        values = [float(v.strip()) for v in values_str.split(',')]
                        if len(values) != 6:
                            raise ValueError(f"Expected 6 scaling values per section per step, got {len(values)} for Step {step}")
                        if step not in scaling_factors:
                            scaling_factors[step] = {}
                        if current_section == "mean":
                            scaling_factors[step]['mean'] = values
                        elif current_section == "std":
                            scaling_factors[step]['std'] = values
        combined_scaling_factors = {}
        for step, factors in scaling_factors.items():
            if 'mean' not in factors or 'std' not in factors:
                raise ValueError(f"Missing mean or std for step {step} in scaling factors file.")
            combined_scaling_factors[step] = factors['mean'] + factors['std']
        return combined_scaling_factors

    def __len__(self):
        return len(self.graph_paths)

    def __getitem__(self, idx):
        """
        Retrieves an entire sequence of graphs for a given sample.
        
        Returns:
            If include_settings is False:
                list of Data objects (one per step)
            If include_settings is True:
                A tuple (data_list, settings_sequence) where:
                    - data_list is a list of Data objects.
                    - settings_sequence is either a single tensor (if identical_settings) or
                      a list of augmented settings tensors (if include_position_index is True).
        """
        # Load the full sequence of graphs for the given index
        sequence = self.graph_paths[idx]
        data_list = [torch.load(path) for path in sequence]

        # Process each graph in the sequence
        for i, data in enumerate(data_list):
            if self.use_edge_attr:
                self._compute_edge_attr(data)
            else:
                data.edge_attr = None
            if self.include_scaling_factors:
                step_index = self.initial_step + i
                scaling = self.scaling_factors.get(step_index, [0.0] * 12)
                data.scale = torch.tensor(scaling, dtype=torch.float).unsqueeze(0)

        # If settings are included, load and (optionally) augment them per step
        if self.include_settings:
            if self.identical_settings:
                base_settings = self.settings_tensor
            else:
                settings_file = self.settings_files[idx]
                settings = torch.load(settings_file)
                base_settings = self.settings_dict_to_tensor(settings)
            if self.include_position_index:
                settings_sequence = []
                for i in range(len(data_list)):
                    step_index = self.initial_step + i
                    normalized_position = torch.tensor([step_index / self.max_x], dtype=torch.float)
                    augmented = torch.cat([normalized_position, base_settings], dim=0)
                    settings_sequence.append(augmented)
            else:
                settings_sequence = base_settings
            return data_list, settings_sequence
        else:
            return data_list

    def _compute_edge_attr(self, data):
        """
        Computes and assigns edge attributes to the data object.
        """
        if not hasattr(data, 'pos') or data.pos is None:
            if data.x.shape[1] < 3:
                raise ValueError("Node feature dimension is less than 3, cannot extract 'pos'.")
            data.pos = data.x[:, :3]
        if hasattr(data, 'edge_index') and data.edge_index is not None:
            row, col = data.edge_index
            pos_diff = data.pos[row] - data.pos[col]
            distance = torch.norm(pos_diff, p=2, dim=1, keepdim=True)
            edge_attr = torch.cat([pos_diff, distance], dim=1)
            eps = 1e-10
            edge_attr_mean = edge_attr.mean(dim=0, keepdim=True)
            edge_attr_std = edge_attr.std(dim=0, keepdim=True)
            edge_attr = (edge_attr - edge_attr_mean) / (edge_attr_std + eps)
            data.edge_attr = edge_attr
        else:
            raise ValueError("Data object is missing 'edge_index', cannot compute 'edge_attr'.")

    def settings_dict_to_tensor(self, settings_dict):
        """
        Converts a settings dictionary to a tensor.
        """
        keys = sorted(settings_dict.keys())
        values = []
        for key in keys:
            value = settings_dict[key]
            if isinstance(value, torch.Tensor):
                value = value.squeeze().float()
            else:
                try:
                    value = torch.tensor(float(value)).float()
                except ValueError:
                    raise ValueError(f"Value for key '{key}' in settings_dict is not convertible to float.")
            values.append(value)
        settings_tensor = torch.stack(values)
        return settings_tensor
