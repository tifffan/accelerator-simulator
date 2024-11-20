import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
from pmd_beamphysics import ParticleGroup
import random
import glob
from impact import Impact
import re

# Function to plot the motion of particles between time steps
def plot_particle_motion(h5_file_path, step_size=1, subsample_size=50, output_dir='figures/example/particles', max_iterations=10):
    # Open the HDF5 file
    # h5 = h5py.File(h5_file_path, 'r')
    I = Impact.from_archive(h5_file_path)

    # List all steps available in the data, sorted to ensure correct order
    unsorted_steps = list(I.particles)

    def sort_key(step):
        if step == 'initial_particles':
            return (0, 0)  # Ensure this is first
        elif step.startswith('write_beam_'):
            num = int(re.search(r'\d+', step).group())  # Extract the number part
            return (1, num)  # Sort these in ascending order after 'initial_particles'
        elif step == 'PR10241':
            return (2, 0)  # Ensure 'PR10241' is last
        return (3, 0)  # Any other steps go to the end

    # Sort the list using the custom key
    steps = sorted(unsorted_steps, key=sort_key)

    print("sorted_steps:", steps)
    
    num_steps = len(steps)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Channels and pairs for plotting
    channels = ['x', 'y', 'z', 'px', 'py', 'pz']
    channels_as_xylabels = ['$x$ (normalized)', '$y$ (normalized)', '$z$ (normalized)', '$p_x$ (normalized)', '$p_y$ (normalized)', '$p_z$ (normalized)']
    pairs = [(0, 3), (1, 4), (2, 5)]  # Only x vs px, y vs py, z vs pz

    # Initialize iteration counter
    iteration_count = 0

    # Loop over time steps with the given step size
    for idx in range(0, num_steps - step_size, step_size):
        # Increment the iteration counter
        iteration_count += 1

        # Break the loop if max_iterations is reached
        if iteration_count > max_iterations:
            break
        
        idx_next = idx + step_size

        step_name = steps[idx]
        step_name_next = steps[idx_next]

        # Load particle data at the current and next steps
        # P_idx = ParticleGroup(h5=h5['data'][step_name]['particles/'])
        # P_next = ParticleGroup(h5=h5['data'][step_name_next]['particles/'])
        P_idx = I.particles[step_name]
        P_next = I.particles[step_name_next]
        
        # Drift particles to their average time
        t0 = P_idx.avg('t')
        P_idx.drift_to_t(t0)
        
        t0 = P_next.avg('t')
        P_next.drift_to_t(t0)

        # Get particle IDs present in both steps
        particle_ids_idx = P_idx.id
        particle_ids_next = P_next.id
        common_particle_ids = np.intersect1d(particle_ids_idx, particle_ids_next)

        if len(common_particle_ids) == 0:
            print(f"No common particles found between steps {step_name} and {step_name_next}")
            continue

        # Subsample particles
        total_common_particles = len(common_particle_ids)
        if subsample_size > total_common_particles:
            subsample_size_actual = total_common_particles
            print(f"Subsample size {subsample_size} exceeds total common particles {total_common_particles}. Using subsample size {subsample_size_actual}.")
        else:
            subsample_size_actual = subsample_size

        # Randomly select subsample_size particles from common_particle_ids
        subsampled_particle_ids = np.random.choice(common_particle_ids, size=subsample_size_actual, replace=False)

        # Ensure the particles are in the same order in both arrays
        subsampled_particle_ids.sort()

        idx_in_P_idx = np.array([np.where(P_idx.id == pid)[0][0] for pid in subsampled_particle_ids])
        idx_in_P_next = np.array([np.where(P_next.id == pid)[0][0] for pid in subsampled_particle_ids])

        # Collect variables for subsampled particles
        variables = ['x', 'y', 'z', 'px', 'py', 'pz']
        
        data_idx = {var: getattr(P_idx, var)[idx_in_P_idx] for var in variables}
        data_next = {var: getattr(P_next, var)[idx_in_P_next] for var in variables}

        # Loop over normalization options
        for norm_option in ['Unnormalized', 'Normalized Separate', 'Normalized Collective']:
            if norm_option == 'Unnormalized':
                # Use unnormalized data for plotting
                data_idx_plot = data_idx
                data_next_plot = data_next
            elif norm_option == 'Normalized Separate':
                # Normalize P_idx data separately
                means_idx = {var: np.mean(data_idx[var]) for var in variables}
                stds_idx = {var: np.std(data_idx[var]) for var in variables}
                # Avoid division by zero
                epsilon = 1e-6
                stds_idx = {var: stds_idx[var] if stds_idx[var] > epsilon else epsilon for var in variables}
                data_idx_normalized = {var: (data_idx[var] - means_idx[var]) / stds_idx[var] for var in variables}

                # Normalize P_next data separately
                means_next = {var: np.mean(data_next[var]) for var in variables}
                stds_next = {var: np.std(data_next[var]) for var in variables}
                stds_next = {var: stds_next[var] if stds_next[var] > epsilon else epsilon for var in variables}
                data_next_normalized = {var: (data_next[var] - means_next[var]) / stds_next[var] for var in variables}

                # Use normalized data for plotting
                data_idx_plot = data_idx_normalized
                data_next_plot = data_next_normalized
            elif norm_option == 'Normalized Collective':
                # Combine data from both steps for normalization
                data_combined = {var: np.concatenate((data_idx[var], data_next[var])) for var in variables}

                # Compute mean and std for each variable
                means_combined = {var: np.mean(data_combined[var]) for var in variables}
                stds_combined = {var: np.std(data_combined[var]) for var in variables}

                # Avoid division by zero in case std is zero
                epsilon = 1e-6
                stds_combined = {var: stds_combined[var] if stds_combined[var] > epsilon else epsilon for var in variables}

                # Normalize the data
                data_idx_normalized = {var: (data_idx[var] - means_combined[var]) / stds_combined[var] for var in variables}
                data_next_normalized = {var: (data_next[var] - means_combined[var]) / stds_combined[var] for var in variables}

                # Use normalized data for plotting
                data_idx_plot = data_idx_normalized
                data_next_plot = data_next_normalized
            else:
                raise ValueError(f"Unknown normalization option: {norm_option}")

            # Create a figure with 1 row and 3 columns (3 subplots)
            fig, axs = plt.subplots(1, 3, figsize=(16, 6))  # Increased height to maintain square aspect ratio
            fig_title = f"{norm_option} Particle Motion from Step {idx} to {idx_next}"
            fig.suptitle(fig_title, fontsize=20)  # Increased title font size

            # Plot each pairwise relationship
            for i, pair in enumerate(pairs):
                ax = axs[i]
                var1 = channels[pair[0]]
                var2 = channels[pair[1]]
                
                var1_label = channels_as_xylabels[pair[0]]
                var2_label = channels_as_xylabels[pair[1]]
                
                # ax.set_title(f"{var1} vs {var2}", fontsize=18)  # Increased title font size
                ax.set_xlabel(var1_label, fontsize=16)  # Increased x-label font size
                ax.set_ylabel(var2_label, fontsize=16)  # Increased y-label font size
                ax.tick_params(axis='both', which='major', labelsize=14)  # Increased tick label size
                
                # Ensure square aspect ratio
                # ax.set_aspect('equal', 'box')
                
                # Plot initial and final state for each particle
                for j in range(len(subsampled_particle_ids)):
                    # Initial state: blue
                    ax.scatter(
                        data_idx_plot[var1][j],
                        data_idx_plot[var2][j],
                        color='blue',
                        alpha=0.3,
                        s=100,  # Increased marker size
                        label='Initial' if j == 0 else None
                    )

                    # Final state: red
                    ax.scatter(
                        data_next_plot[var1][j],
                        data_next_plot[var2][j],
                        color='red',
                        alpha=0.3,
                        s=100,  # Increased marker size
                        label='Final' if j == 0 else None
                    )

                    # Line connecting initial and final states
                    ax.plot(
                        [data_idx_plot[var1][j], data_next_plot[var1][j]],
                        [data_idx_plot[var2][j], data_next_plot[var2][j]],
                        color='gray',
                        linestyle='-',
                        linewidth=1.5,  # Increased line width
                        alpha=1.0
                    )

                # Add legend only to the first subplot
                if i == 0:
                    ax.legend(loc='upper right', fontsize=14)  # Increased legend font size

            # Adjust layout to avoid overlap
            plt.tight_layout(rect=[0, 0, 1, 0.95])


            # Determine the save path
            if norm_option == 'Unnormalized':
                save_filename = f"unnormalized_particles_step_{idx}_to_{idx_next}.png"
            elif norm_option == 'Normalized Separate':
                save_filename = f"normalized_separate_particles_step_{idx}_to_{idx_next}.png"
            elif norm_option == 'Normalized Collective':
                save_filename = f"normalized_collective_particles_step_{idx}_to_{idx_next}.png"
            else:
                raise ValueError(f"Unknown normalization option: {norm_option}")

            save_path = os.path.join(output_dir, save_filename)
            plt.savefig(save_path, dpi=300)
            plt.close(fig)  # Close the figure to free memory

            print(f"Saved figure: {save_path}")

    # Close the HDF5 file
    # h5.close()

# Usage example
if __name__ == "__main__":
    archive_dir = '/sdf/data/ad/ard/u/tiffan/Archive_4'  # Do not include the last '/'
    # Search for HDF5 files in the archive directory
    h5_files = glob.glob(os.path.join(archive_dir, '*.h5'))

    # Take the first HDF5 file found
    if h5_files:
        h5_file_path = h5_files[0]
    else:
        raise FileNotFoundError(f"No HDF5 files found in directory: {archive_dir}")
    
    output_dir = f'figures/{os.path.basename(archive_dir)}/{os.path.basename(h5_file_path).replace(".h5", "")}/'
    subsample_size = 50  # Subsample size for particles

    for step_size in [1, 2, 5, 10]:
        plot_particle_motion(h5_file_path, step_size=step_size, output_dir=output_dir, subsample_size=subsample_size, max_iterations=100)
