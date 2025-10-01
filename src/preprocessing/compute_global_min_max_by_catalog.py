import pandas as pd
import torch
import h5py
import numpy as np
import os

def compute_global_mean_min_max(data_catalog, output_file='global_mean_min_max.txt'):
    # Load data catalog
    df = pd.read_csv(data_catalog)
    
    # There are 6 initial, 6 final, and 6 settings channels (as in the original script)
    num_channels = 18  # 6 initial + 6 final + 6 settings
    sum_channels = torch.zeros(num_channels, dtype=torch.float64)
    max_channels = torch.full((num_channels,), float('-inf'), dtype=torch.float64)
    min_channels = torch.full((num_channels,), float('inf'), dtype=torch.float64)
    total_count = 0

    for i, row in df.iterrows():
        filepath = row['filepath']
        with h5py.File(filepath, 'r') as f:
            # Load initial state (6 channels)
            initial_state = torch.stack([
                torch.tensor(f['initial_position_x'][()], dtype=torch.float64),
                torch.tensor(f['initial_position_y'][()], dtype=torch.float64),
                torch.tensor(f['initial_position_z'][()], dtype=torch.float64),
                torch.tensor(f['initial_momentum_px'][()], dtype=torch.float64),
                torch.tensor(f['initial_momentum_py'][()], dtype=torch.float64),
                torch.tensor(f['initial_momentum_pz'][()], dtype=torch.float64)
            ], dim=1)  # Shape: (num_particles, 6)
            
            # Load final state (6 channels)
            final_state = torch.stack([
                torch.tensor(f['pr10241_position_x'][()], dtype=torch.float64),
                torch.tensor(f['pr10241_position_y'][()], dtype=torch.float64),
                torch.tensor(f['pr10241_position_z'][()], dtype=torch.float64),
                torch.tensor(f['pr10241_momentum_px'][()], dtype=torch.float64),
                torch.tensor(f['pr10241_momentum_py'][()], dtype=torch.float64),
                torch.tensor(f['pr10241_momentum_pz'][()], dtype=torch.float64)
            ], dim=1)  # Shape: (num_particles, 6)
            
            # Load settings (6 channels)
            settings_keys = [
                'CQ10121_b1_gradient', 
                'GUNF_rf_field_scale', 
                'GUNF_theta0_deg', 
                'SOL10111_solenoid_field_scale', 
                'SQ10122_b1_gradient',
                'distgen_total_charge',
            ]
            try:
                settings = torch.tensor([f[key][()] for key in settings_keys], dtype=torch.float64)  # Shape: (6,)
            except KeyError as e:
                print(f"Missing key {e} in file {filepath}. Skipping this file.")
                continue
            if settings.ndimension() != 1 or settings.shape[0] != 6:
                print(f"Settings have unexpected shape {settings.shape} in file {filepath}. Skipping this file.")
                continue
            num_particles = initial_state.shape[0]
            if num_particles == 0:
                print(f"No particles found in file {filepath}. Skipping this file.")
                continue
            settings_expanded = settings.unsqueeze(0).expand(num_particles, -1)
            full_data = torch.cat([initial_state, final_state, settings_expanded], dim=1)  # (num_particles, 18)
            sum_channels += full_data.sum(dim=0)
            max_channels = torch.max(max_channels, full_data.max(dim=0).values)
            min_channels = torch.min(min_channels, full_data.min(dim=0).values)
            total_count += num_particles
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(df)} files.")
    if total_count == 0:
        print("No data processed. Exiting.")
        return
    global_mean = sum_channels / total_count
    # Save the results to a file using scientific notation for better precision
    with open(output_file, 'w') as f:
        f.write("Global Mean:\n")
        f.write(", ".join([f"{m.item():.12e}" for m in global_mean]) + "\n")
        f.write("Global Min:\n")
        f.write(", ".join([f"{m.item():.12e}" for m in min_channels]) + "\n")
        f.write("Global Max:\n")
        f.write(", ".join([f"{m.item():.12e}" for m in max_channels]) + "\n")
    print(f"Global mean, min, and max saved to {output_file}")

# Usage example:
if __name__ == "__main__":
    data_catalog = '/sdf/home/t/tiffan/repo/accelerator-surrogate/src/preprocessing/electrons_vary_distributions_vary_settings_filtered_total_charge_51_catalog_all_sdf_cleaned.csv'
    compute_global_mean_min_max(
        data_catalog, 
        output_file='/sdf/home/t/tiffan/repo/accelerator-surrogate/src/preprocessing/global_mean_min_max_vary_distributions_vary_settings_filtered_total_charge_51_catalog_all_sdf_cleaned.txt'
    ) 