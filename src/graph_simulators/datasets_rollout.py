# datasets_rollout.py

import os
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import re

MAX_STEP_INDEX = 76  # Used for normalizing position indices

class SequenceGraphSettingsRolloutDataset(Dataset):
    def __init__(self, graph_data_dir, initial_step=0, final_step=10, max_prediction_horizon=3,
                 include_settings=False, identical_settings=False,
                 use_edge_attr=False, subsample_size=None,
                 include_position_index=False, include_scaling_factors=False,
                 scaling_factors_file=None):
        """
        Initializes the dataset to return, for each sample, a tuple:
          (input_graph, target_graph_list, seq_length, settings_list)
        where:
          - input_graph: the graph at the first time step.
          - target_graph_list: a list of subsequent graphs (up to max_prediction_horizon).
          - seq_length: number of target graphs available.
          - settings_list: a list of settings tensors for each target (if include_settings is True).
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

        # Load graph file paths (each element is a sequence across steps)
        self.graph_paths = self._load_graph_paths()

        if self.subsample_size is not None:
            self.graph_paths = self.graph_paths[:self.subsample_size]

        # Load settings if required.
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

        # Load scaling factors if required.
        if self.include_scaling_factors:
            if self.scaling_factors_file is None:
                raise ValueError("Scaling factors file must be provided when include_scaling_factors is True.")
            self.scaling_factors = self._load_scaling_factors(self.scaling_factors_file)

        if self.include_position_index:
            self.max_x = MAX_STEP_INDEX

    def _load_graph_paths(self):
        # For each step, list graph files; then zip across steps to form sequences.
        graph_dirs = [os.path.join(self.graph_data_dir, f'step_{i}') for i in range(self.initial_step, self.final_step + 1)]
        graph_paths_per_step = []
        for dir in graph_dirs:
            if not os.path.isdir(dir):
                raise ValueError(f"Graph directory not found: {dir}")
            files = sorted(os.listdir(dir), key=self._extract_graph_x)
            files = [os.path.join(dir, f) for f in files if f.endswith('.pt') and not f.endswith('_settings.pt')]
            graph_paths_per_step.append(files)
        # Transpose: each element is a tuple containing one file per step.
        graph_paths = list(zip(*graph_paths_per_step))
        return graph_paths

    def _extract_graph_x(self, filename):
        match = re.search(r'graph_(\d+)\.pt', filename)
        if match:
            return int(match.group(1))
        return -1

    def _load_settings_files(self):
        settings_files = []
        for sequence in self.graph_paths:
            base_fname = os.path.basename(sequence[0]).replace('.pt', '')
            settings_file = os.path.join(os.path.dirname(sequence[0]), f"{base_fname}_settings.pt")
            if not os.path.isfile(settings_file):
                raise ValueError(f"Settings file not found: {settings_file}")
            settings_files.append(settings_file)
        return settings_files

    def _load_scaling_factors(self, scaling_factors_file):
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
                raise ValueError(f"Missing mean or std for step {step}")
            combined_scaling_factors[step] = factors['mean'] + factors['std']
        return combined_scaling_factors

    def settings_dict_to_tensor(self, settings_dict):
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
                    raise ValueError(f"Value for key '{key}' not convertible to float.")
            values.append(value)
        settings_tensor = torch.stack(values)
        return settings_tensor

    def _compute_edge_attr(self, data):
        if not hasattr(data, 'pos') or data.pos is None:
            if data.x.shape[1] < 3:
                raise ValueError("Node feature dimension less than 3")
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
            raise ValueError("Data object missing 'edge_index'")

    def __len__(self):
        return len(self.graph_paths)

    def __getitem__(self, idx):
        # Load the full sequence of graphs.
        sequence = self.graph_paths[idx]
        data_list = [torch.load(path) for path in sequence]
        for data in data_list:
            if self.use_edge_attr:
                self._compute_edge_attr(data)
            else:
                data.edge_attr = None

        # Use the first graph as input.
        input_graph = data_list[0]
        # Use subsequent graphs (up to max_prediction_horizon) as targets.
        target_graph_list = data_list[1 : 1 + self.max_prediction_horizon]
        seq_length = len(target_graph_list)

        # Process settings if enabled.
        if self.include_settings:
            if self.identical_settings:
                settings_tensor = self.settings_tensor
            else:
                settings_file = self.settings_files[idx]
                settings = torch.load(settings_file)
                settings_tensor = self.settings_dict_to_tensor(settings)
            if self.include_position_index:
                augmented_settings_list = []
                for delta in range(1, seq_length + 1):
                    step_index = self.initial_step + delta
                    normalized_position = torch.tensor([step_index / self.max_x], dtype=torch.float)
                    aug_setting = torch.cat([normalized_position, settings_tensor], dim=0)
                    augmented_settings_list.append(aug_setting)
                settings_list = augmented_settings_list
            else:
                settings_list = [settings_tensor for _ in range(seq_length)]
        else:
            settings_list = None

        # Assign scaling factors if enabled.
        if self.include_scaling_factors:
            scaling_initial = self.scaling_factors.get(self.initial_step, [0.0] * 12)
            input_graph.scale = torch.tensor(scaling_initial, dtype=torch.float).unsqueeze(0)
            for i, target_graph in enumerate(target_graph_list):
                target_step = self.initial_step + i + 1
                scaling_target = self.scaling_factors.get(target_step, [0.0] * 12)
                target_graph.scale = torch.tensor(scaling_target, dtype=torch.float).unsqueeze(0)

        if self.include_settings:
            return (input_graph, target_graph_list, seq_length, settings_list)
        else:
            return (input_graph, target_graph_list, seq_length)
