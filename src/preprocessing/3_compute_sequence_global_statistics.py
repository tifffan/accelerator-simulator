import pandas as pd
import torch
import numpy as np
import os
import argparse

def compute_global_statistics_per_step(data_catalog, output_file='sequence_global_statistics.txt', epsilon=1e-6, identical_settings=False, settings_file=None):
    # Load data catalog
    df = pd.read_csv(data_catalog)
    
    # Initialize variables
    sum_channels = None  # Will be initialized after determining num_time_steps and num_features
    sum_squares_channels = None
    total_counts = None  # Counts per time step
    
    # For settings, accumulate separately
    settings_list = []  # List to store settings tensors

    # Load settings once if identical_settings is True
    if identical_settings:
        if settings_file is None:
            raise ValueError("Settings file must be provided when identical_settings is True.")
        settings = torch.load(settings_file)
        settings_tensor = settings_dict_to_tensor(settings).double()
        settings_list.append(settings_tensor)
    
    # Loop over all .pt files
    for i, row in df.iterrows():
        filepath = row['filepath']
        # Load particle data tensor
        try:
            particle_data = torch.load(filepath)  # Shape: (num_time_steps, num_particles, num_features)
        except Exception as e:
            print(f"Error loading particle data from file {filepath}: {e}. Skipping this file.")
            continue

        num_time_steps, num_particles, num_features = particle_data.shape
        
        if sum_channels is None:
            # Initialize accumulators
            sum_channels = torch.zeros((num_time_steps, num_features), dtype=torch.float64)
            sum_squares_channels = torch.zeros((num_time_steps, num_features), dtype=torch.float64)
            total_counts = torch.zeros(num_time_steps, dtype=torch.int64)
        
        # Accumulate per time step
        for t in range(num_time_steps):
            data_t = particle_data[t]  # Shape: (num_particles, num_features)
            sum_channels[t] += data_t.sum(dim=0).double()
            sum_squares_channels[t] += (data_t.double() ** 2).sum(dim=0)
            total_counts[t] += data_t.shape[0]  # num_particles at time step t

        # Load settings if not identical
        if not identical_settings:
            # Load settings for the current sample
            # Assuming settings file is named similarly to particle data file
            settings_filename = os.path.basename(filepath).replace('_particle_data.pt', '_settings.pt')
            settings_filepath = os.path.join(os.path.dirname(filepath), settings_filename)
            if not os.path.isfile(settings_filepath):
                print(f"Settings file {settings_filepath} not found. Skipping settings for this sample.")
                continue
            settings = torch.load(settings_filepath)
            settings_tensor = settings_dict_to_tensor(settings).double()
            settings_list.append(settings_tensor)

        # Optional: Print progress every 10 files
        if (i + 1) % 10 == 0:
            print("Processed {}/{} files.".format(i + 1, len(df)))
    
    # Check if any data was processed
    if total_counts.sum() == 0:
        print("No data processed. Exiting.")
        return
    
    # Compute global mean and variance per time step
    global_mean = sum_channels / total_counts.unsqueeze(1)  # Shape: (num_time_steps, num_features)
    global_var = (sum_squares_channels / total_counts.unsqueeze(1)) - (global_mean ** 2)
    
    # Handle negative variances
    global_var = torch.clamp(global_var, min=epsilon**2)
    global_std = torch.sqrt(global_var)
    
    # Compute settings global stats
    if len(settings_list) > 0:
        if identical_settings:
            settings_mean = settings_list[0]
            settings_std = torch.zeros_like(settings_mean)
        else:
            settings_stack = torch.stack(settings_list)  # Shape: (num_samples, num_settings)
            settings_mean = settings_stack.mean(dim=0)
            settings_std = settings_stack.std(dim=0, unbiased=False)
            # Handle any potential zero std deviations
            settings_std = torch.clamp(settings_std, min=epsilon)
    else:
        print("No settings data processed.")
        settings_mean = None
        settings_std = None
    
    # Save the results to a file
    with open(output_file, 'w') as f:
        f.write("Per-Step Global Mean:\n")
        for t in range(num_time_steps):
            mean_str = ", ".join([f"{m.item():.12e}" for m in global_mean[t]])
            f.write(f"Step {t}: {mean_str}\n")
        f.write("Per-Step Global Std:\n")
        for t in range(num_time_steps):
            std_str = ", ".join([f"{s.item():.12e}" for s in global_std[t]])
            f.write(f"Step {t}: {std_str}\n")
        if settings_mean is not None:
            f.write("Settings Global Mean: ")
            settings_mean_str = ", ".join([f"{m.item():.12e}" for m in settings_mean])
            f.write(settings_mean_str + "\n")
            f.write("Settings Global Std: ")
            settings_std_str = ", ".join([f"{s.item():.12e}" for s in settings_std])
            f.write(settings_std_str + "\n")
    
    print("Global statistics saved to {}".format(output_file))

def settings_dict_to_tensor(settings_dict):
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

# Usage example:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute global statistics per time step from time series particle data.")
    parser.add_argument('--data_catalog', type=str, required=True, help="Path to the data catalog CSV file.")
    parser.add_argument('--output_file', type=str, default='sequence_global_statistics.txt', help="Path to the output file.")
    parser.add_argument('--epsilon', type=float, default=1e-6, help="Small value to prevent division by zero.")
    parser.add_argument('--identical_settings', action='store_true', help="Whether the settings are identical across all samples.")
    parser.add_argument('--settings_file', type=str, help="Path to the settings file when identical_settings is True.")

    args = parser.parse_args()

    compute_global_statistics_per_step(
        data_catalog=args.data_catalog,
        output_file=args.output_file,
        epsilon=args.epsilon,
        identical_settings=args.identical_settings,
        settings_file=args.settings_file
    )
