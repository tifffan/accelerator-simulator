import pandas as pd
import numpy as np
import h5py
import os
from pmd_beamphysics import ParticleGroup
import re
import torch
import argparse
from impact import Impact

# Helper function to safely check if a value is NaN
def is_nan(value):
    return isinstance(value, (float, int, np.float64, np.int64)) and np.isnan(value)

# Function to extract and save particle data and settings
def extract_and_save_particle_data(h5_file_path, settings, output_dir='data/particles', save_settings=True, settings_filename=None):
    try:
        I = Impact.from_archive(h5_file_path)
    except Exception as e:
        print(f"Error opening file {h5_file_path}: {e}")
        return

    # List all steps available in the data, sorted to ensure correct order
    unsorted_steps = list(I.particles)

    def sort_key(step):
        if step == 'initial_particles':
            return (0, 0)
        elif step.startswith('write_beam_'):
            num = int(re.search(r'\d+', step).group())
            return (1, num)
        elif step == 'PR10241':
            return (2, 0)
        return (3, 0)

    steps = sorted(unsorted_steps, key=sort_key)

    print(f"Processing file: {h5_file_path}")
    print(f"Sorted steps ({len(steps)}): {steps}")

    expected_num_steps = 77
    if len(steps) != expected_num_steps:
        print(f"Warning: File {h5_file_path} contains {len(steps)} particle groups instead of {expected_num_steps}. Skipping this file.")
        return

    num_steps = len(steps)
    os.makedirs(output_dir, exist_ok=True)
    common_particle_ids = None
    particle_data_list = []

    for idx, step_name in enumerate(steps):
        try:
            P = I.particles[step_name]
        except Exception as e:
            print(f"Error loading particles at step {step_name} in file {h5_file_path}: {e}")
            return

        t0 = P.avg('t')
        P.drift_to_t(t0)
        particle_ids = P.id

        if common_particle_ids is None:
            common_particle_ids = particle_ids
        else:
            common_particle_ids = np.intersect1d(common_particle_ids, particle_ids)

        variables = ['x', 'y', 'z', 'px', 'py', 'pz']
        data = {var: getattr(P, var) for var in variables}
        data['id'] = particle_ids
        particle_data_list.append(data)

    if len(common_particle_ids) == 0:
        print(f"No common particles found across all steps in file {h5_file_path}.")
        return

    print(f"Number of common particles: {len(common_particle_ids)}")
    num_common_particles = len(common_particle_ids)
    num_features = 6
    num_time_steps = len(particle_data_list)

    particle_tensor = torch.zeros((num_time_steps, num_common_particles, num_features))
    var_to_idx = {'x': 0, 'y': 1, 'z': 2, 'px': 3, 'py': 4, 'pz': 5}

    for t, data in enumerate(particle_data_list):
        ids = data['id']
        idx_in_current = np.array([np.where(ids == pid)[0][0] for pid in common_particle_ids])

        for var in variables:
            values = data[var][idx_in_current]
            feature_idx = var_to_idx[var]
            particle_tensor[t, :, feature_idx] = torch.from_numpy(values)

    tensor_filename = os.path.join(output_dir, f"{os.path.basename(h5_file_path).replace('.h5', '')}_particle_data.pt")
    torch.save(particle_tensor, tensor_filename)
    print(f"Saved particle tensor to {tensor_filename}")

    if save_settings and settings_filename:
        settings_tensor = {key: torch.tensor(value) for key, value in settings.items()}
        torch.save(settings_tensor, settings_filename)
        print(f"Saved settings tensor to {settings_filename}")

# Usage example
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and save particle data and settings from catalog.")
    parser.add_argument('--catalog', type=str, default='/sdf/home/t/tiffan/repo/accelerator-surrogate/src/preprocessing/Archive_5_n241_match_cleaned.csv', help="Path to the catalog CSV file.")
    parser.add_argument('--output_dir', type=str, default='/sdf/data/ad/ard/u/tiffan/data/sequence_particles_data_archive_5', help="Directory to save the output tensors.")
    parser.add_argument('--identical_settings', action='store_true', help="Flag to save settings once for all files.")

    args = parser.parse_args()

    # Load the catalog file
    catalog = pd.read_csv(args.catalog)

    # Track if settings have been saved
    settings_saved = False

    # Iterate through catalog entries
    for idx, row in catalog.iterrows():
        h5_file_path = row['filename']
        settings = {
            'SOL10111:solenoid_field_scale': row['SOL10111:solenoid_field_scale'],
            'CQ10121:b1_gradient': row['CQ10121:b1_gradient'],
            'SQ10122:b1_gradient': row['SQ10122:b1_gradient'],
            'GUNF:rf_field_scale': row['GUNF:rf_field_scale'],
            'GUNF:theta0_deg': row['GUNF:theta0_deg'],
            'distgen:total_charge': row['distgen:total_charge']
        }

        # Handle identical settings
        if args.identical_settings:
            if not settings_saved:
                settings_filename = os.path.join(args.output_dir, "settings.pt")
                save_settings = True
                settings_saved = True
            else:
                save_settings = False
                settings_filename = None
        else:
            # Unique settings per file
            settings_filename = os.path.join(args.output_dir, f"{os.path.basename(h5_file_path).replace('.h5', '')}_settings.pt")
            save_settings = True

        extract_and_save_particle_data(h5_file_path, settings, output_dir=args.output_dir, save_settings=save_settings, settings_filename=settings_filename)
