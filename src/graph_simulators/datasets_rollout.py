# datasets_rollout.py

import os
import re
import csv
import torch
from torch.utils.data import Dataset

MAX_STEP_INDEX = 76
FIELDMAP_FILE = "data/fieldmap_at_custom_step_values.csv"

class SequenceGraphSettingsRolloutDataset(Dataset):
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
        scaling_factors_file=None
    ):
        """
        Initializes the dataset to return, for each sample, a tuple:
          (input_graph, target_graph_list, seq_length, settings_list)
        where:
          - input_graph: the graph at the first time step.
          - target_graph_list: a list of subsequent graphs (up to max_prediction_horizon).
          - seq_length: number of target graphs available.
          - settings_list: a list of settings tensors for each target (if include_settings is True).

        The parameter `position_encoding_method` accepts:
          - "normalized"
          - "onehot"
          - "sinusoidal"
          - "learned"
          - "fieldmaps"
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
        self.position_encoding_method = position_encoding_method
        self.sinusoidal_encoding_dim = sinusoidal_encoding_dim
        self.include_scaling_factors = include_scaling_factors
        self.scaling_factors_file = scaling_factors_file

        # Load graph file paths
        self.graph_paths = self._load_graph_paths()
        if self.subsample_size is not None:
            self.graph_paths = self.graph_paths[: self.subsample_size]

        # Load settings
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
                    self.settings_files = self.settings_files[: self.subsample_size]
                if len(self.settings_files) != len(self.graph_paths):
                    raise ValueError(
                        "Mismatch between number of graph sequences and settings files."
                    )

        # Load scaling factors
        if self.include_scaling_factors:
            if self.scaling_factors_file is None:
                raise ValueError(
                    "Scaling factors file must be provided when include_scaling_factors is True."
                )
            self.scaling_factors = self._load_scaling_factors(
                self.scaling_factors_file
            )

        # Position encodings
        if self.include_position_index:
            self.max_x = MAX_STEP_INDEX
            if self.position_encoding_method == "learned":
                self.pos_embedding = torch.nn.Embedding(
                    num_embeddings=self.max_x + 1,
                    embedding_dim=self.sinusoidal_encoding_dim,
                )
            elif self.position_encoding_method == "fieldmaps":
                self.fieldmap_encoding = self._load_fieldmap(FIELDMAP_FILE)

    def _load_fieldmap(self, filepath):
        """
        Reads FIELDMAP_FILE, returns a dict:
          { step_index: [Solenoid Bz, Quadrupole Bz, RF Gun Ez], ... }
        Assumes the first CSV row is the header with columns:
          "z (m)", "Solenoid Bz (T)", "Quadrupole Bz (T)", "RF Gun Ez (V/m)"
        """
        fieldmap = {}
        with open(filepath, newline='') as f:
            reader = csv.reader(f)
            header = next(reader)
            try:
                sol_idx = header.index("Solenoid Bz (T)")
                quad_idx = header.index("Quadrupole Bz (T)")
                rf_idx = header.index("RF Gun Ez (V/m)")
            except ValueError as e:
                raise ValueError(f"Expected column not found in header: {e}")

            for step_idx, row in enumerate(reader):
                if not row or len(row) < max(sol_idx, quad_idx, rf_idx) + 1:
                    continue
                fieldmap[step_idx] = [
                    float(row[sol_idx]),
                    float(row[quad_idx]),
                    float(row[rf_idx]),
                ]
        return fieldmap

    def get_sinusoidal_encoding(self, pos):
        """
        Computes a sinusoidal positional encoding for a given position.
        """
        d = self.sinusoidal_encoding_dim
        pos_tensor = torch.tensor(pos, dtype=torch.float)
        encoding = torch.zeros(d, dtype=torch.float)
        for i in range(d):
            div_term = 10000 ** (2 * (i // 2) / d)
            if i % 2 == 0:
                encoding[i] = torch.sin(pos_tensor / div_term)
            else:
                encoding[i] = torch.cos(pos_tensor / div_term)
        return encoding

    def _load_graph_paths(self):
        graph_dirs = [
            os.path.join(self.graph_data_dir, f'step_{i}')
            for i in range(self.initial_step, self.final_step + 1)
        ]
        graph_paths_per_step = []
        for dir in graph_dirs:
            if not os.path.isdir(dir):
                raise ValueError(f"Graph directory not found: {dir}")
            files = sorted(os.listdir(dir), key=self._extract_graph_x)
            files = [os.path.join(dir, f) for f in files if f.endswith('.pt') and not f.endswith('_settings.pt')]
            graph_paths_per_step.append(files)
        return list(zip(*graph_paths_per_step))

    def _extract_graph_x(self, filename):
        match = re.search(r'graph_(\d+)\.pt', filename)
        return int(match.group(1)) if match else -1

    def _load_settings_files(self):
        settings_files = []
        for sequence in self.graph_paths:
            base_fname = os.path.basename(sequence[0]).replace('.pt', '')
            settings_file = os.path.join(
                os.path.dirname(sequence[0]), f"{base_fname}_settings.pt"
            )
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
                        values = [float(v.strip()) for v in step_match.group(2).split(',')]
                        if len(values) != 6:
                            raise ValueError(
                                f"Expected 6 scaling values per step, got {len(values)} for Step {step}"
                            )
                        if step not in scaling_factors:
                            scaling_factors[step] = {}
                        scaling_factors[step][current_section] = values
        combined = {}
        for step, vals in scaling_factors.items():
            if 'mean' not in vals or 'std' not in vals:
                raise ValueError(f"Missing mean or std for step {step}")
            combined[step] = vals['mean'] + vals['std']
        return combined

    def settings_dict_to_tensor(self, settings_dict):
        keys = sorted(settings_dict.keys())
        values = []
        for key in keys:
            value = settings_dict[key]
            if isinstance(value, torch.Tensor):
                value = value.squeeze().float()
            else:
                value = torch.tensor(float(value)).float()
            values.append(value)
        return torch.stack(values)

    def _compute_edge_attr(self, data):
        if not hasattr(data, 'pos') or data.pos is None:
            if data.x.shape[1] < 3:
                raise ValueError("Node feature dimension less than 3")
            data.pos = data.x[:, :3]
        if hasattr(data, 'edge_index') and data.edge_index is not None:
            row, col = data.edge_index
            pos_diff = data.pos[row] - data.pos[col]
            dist = torch.norm(pos_diff, p=2, dim=1, keepdim=True)
            edge_attr = torch.cat([pos_diff, dist], dim=1)
            eps = 1e-10
            m = edge_attr.mean(dim=0, keepdim=True)
            s = edge_attr.std(dim=0, keepdim=True)
            data.edge_attr = (edge_attr - m) / (s + eps)
        else:
            raise ValueError("Data object missing 'edge_index'")

    def __len__(self):
        return len(self.graph_paths)

    def __getitem__(self, idx):
        sequence = self.graph_paths[idx]
        data_list = [torch.load(p) for p in sequence]
        for d in data_list:
            if self.use_edge_attr:
                self._compute_edge_attr(d)
            else:
                d.edge_attr = None

        input_graph = data_list[0]
        target_graph_list = data_list[1 : 1 + self.max_prediction_horizon]
        seq_length = len(target_graph_list)

        if self.include_settings:
            if self.identical_settings:
                settings_tensor = self.settings_tensor
            else:
                settings = torch.load(self.settings_files[idx])
                settings_tensor = self.settings_dict_to_tensor(settings)

            if self.include_position_index:
                augmented_settings_list = []
                for delta in range(1, seq_length + 1):
                    step_index = self.initial_step + delta
                    if self.position_encoding_method == "normalized":
                        pos_encoding = torch.tensor([step_index / self.max_x], dtype=torch.float)
                    elif self.position_encoding_method == "onehot":
                        vec = torch.zeros(self.max_x + 1, dtype=torch.float)
                        vec[step_index] = 1.0
                        pos_encoding = vec
                    elif self.position_encoding_method == "sinusoidal":
                        pos_encoding = self.get_sinusoidal_encoding(step_index)
                    elif self.position_encoding_method == "learned":
                        pos_encoding = self.pos_embedding(torch.tensor(step_index, dtype=torch.long))
                    elif self.position_encoding_method == "fieldmaps":
                        vals = self.fieldmap_encoding.get(step_index)
                        if vals is None:
                            raise ValueError(f"No fieldmap entry for step {step_index}")
                        pos_encoding = torch.tensor(vals, dtype=torch.float)
                    else:
                        raise ValueError(f"Unknown position encoding: {self.position_encoding_method}")
                    augmented_settings_list.append(torch.cat([pos_encoding, settings_tensor], dim=0))
                settings_list = augmented_settings_list
            else:
                settings_list = [settings_tensor] * seq_length
        else:
            settings_list = None

        if self.include_scaling_factors:
            init_scale = self.scaling_factors.get(self.initial_step, [0.0] * 12)
            input_graph.scale = torch.tensor(init_scale, dtype=torch.float).unsqueeze(0)
            for i, tg in enumerate(target_graph_list):
                tgt_step = self.initial_step + i + 1
                tgt_scale = self.scaling_factors.get(tgt_step, [0.0] * 12)
                tg.scale = torch.tensor(tgt_scale, dtype=torch.float).unsqueeze(0)

        if self.include_settings:
            return (input_graph, target_graph_list, seq_length, settings_list)
        return (input_graph, target_graph_list, seq_length)
