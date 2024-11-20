import torch
from torch.utils.data import Dataset
import glob
import os

class SequenceParticlesDataset(Dataset):
    def __init__(self, data_dir, sequence_length=None, transform=None, identical_settings=False):
        """
        Args:
            data_dir (str): Directory containing the particle data and settings tensors.
            sequence_length (int, optional): Length of the time series sequences to return.
                                             If None, use the full sequence.
            transform (callable, optional): Optional transform to be applied on a sample.
            identical_settings (bool): Whether the settings are identical across all samples.
                                       If True, loads the settings from 'settings.pt'.
        """
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.transform = transform
        self.identical_settings = identical_settings

        # Find all particle data files
        self.particle_files = sorted(glob.glob(os.path.join(data_dir, '*_particle_data.pt')))
        if not self.particle_files:
            raise ValueError(f"No particle data files found in directory: {data_dir}")

        # Load settings
        if identical_settings:
            settings_file = os.path.join(data_dir, 'settings.pt')
            if not os.path.isfile(settings_file):
                raise ValueError(f"Settings file not found: {settings_file}")
            self.settings = torch.load(settings_file)
            # Convert settings dictionary to tensor
            self.settings_tensor = self.settings_dict_to_tensor(self.settings)
        else:
            # Find all settings files
            self.settings_files = sorted(glob.glob(os.path.join(data_dir, '*_settings.pt')))
            if not self.settings_files:
                raise ValueError(f"No settings files found in directory: {data_dir}")
            if len(self.settings_files) != len(self.particle_files):
                raise ValueError("Mismatch between number of particle data files and settings files.")
            # Map particle data files to settings files
            self.particle_to_settings = {
                os.path.basename(pf).replace('_particle_data.pt', ''): sf
                for pf, sf in zip(self.particle_files, self.settings_files)
                if os.path.basename(pf).replace('_particle_data.pt', '') == os.path.basename(sf).replace('_settings.pt', '')
            }
            if len(self.particle_to_settings) != len(self.particle_files):
                raise ValueError("Mismatch between particle data files and settings files.")

    def __len__(self):
        return len(self.particle_files)

    def __getitem__(self, idx):
        # Load particle data
        particle_file = self.particle_files[idx]
        particle_data = torch.load(particle_file)  # Shape: (num_time_steps, num_particles, num_features)
        
        # Optionally select a sequence length
        if self.sequence_length is not None:
            num_time_steps = particle_data.shape[0]
            if self.sequence_length > num_time_steps:
                raise ValueError(f"Requested sequence_length {self.sequence_length} exceeds available time steps {num_time_steps}.")
            # For simplicity, we can take the first sequence_length time steps
            particle_data = particle_data[:self.sequence_length]

        # Load settings
        if self.identical_settings:
            settings_tensor = self.settings_tensor  # Use the preloaded settings tensor
        else:
            particle_filename_base = os.path.basename(particle_file).replace('_particle_data.pt', '')
            settings_file = os.path.join(self.data_dir, f"{particle_filename_base}_settings.pt")
            if not os.path.isfile(settings_file):
                raise ValueError(f"Settings file not found: {settings_file}")
            settings = torch.load(settings_file)
            settings_tensor = self.settings_dict_to_tensor(settings)

        sample = {'particle_data': particle_data, 'settings': settings_tensor}

        if self.transform:
            sample = self.transform(sample)

        return sample

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
