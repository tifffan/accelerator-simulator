import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tempfile
import logging
from pmd_beamphysics import ParticleGroup  # Import ParticleGroup
import numpy as np

def plot_particle_groups(pred_pg, target_pg, idx, error_type, results_folder,
                         mse_value, rel_err_x, rel_err_y, rel_err_z, title=None):
    """
    Plots predicted and target ParticleGroups onto a single figure by:
    1. Using return_figure=True to get separate figures.
    2. Saving those figures as temporary images.
    3. Reading the images and displaying them with imshow on subplots.

    This avoids manipulating artists directly and guarantees that the plots look 
    the same as originally produced, just arranged in a grid.
    """
    vars_list = [('x', 'px'), ('y', 'py'), ('z', 'pz')]
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 18))

    # A helper function to generate and save a figure from ParticleGroup.plot()
    def save_plot_as_image(pgroup, xvar, pvar):
        # Generate figure
        fig_local = pgroup.plot(xvar, pvar, return_figure=True)
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
            temp_path = tmpfile.name
        fig_local.savefig(temp_path, dpi=150)
        plt.close(fig_local)
        return temp_path

    for row, (x_var, p_var) in enumerate(vars_list):
        # Predicted subplot (left column)
        ax_pred = axes[row, 0]
        pred_img_path = save_plot_as_image(pred_pg, x_var, p_var)
        pred_img = mpimg.imread(pred_img_path)
        ax_pred.imshow(pred_img)
        ax_pred.set_title(f'{error_type.upper()} MSE Sample {idx}: Predicted {x_var}-{p_var}')
        ax_pred.axis('off')  # Remove axis lines/ticks since it's now an image
        os.remove(pred_img_path)  # Clean up the temporary file

        # Target subplot (right column)
        ax_target = axes[row, 1]
        target_img_path = save_plot_as_image(target_pg, x_var, p_var)
        target_img = mpimg.imread(target_img_path)
        ax_target.imshow(target_img)
        ax_target.set_title(f'{error_type.upper()} MSE Sample {idx}: Target {x_var}-{p_var}')
        ax_target.axis('off')
        os.remove(target_img_path)

    # Add annotation with MSE and relative errors at the bottom of the figure
    annotation_text = (f"MSE: {mse_value:.4f}\n"
                       f"Rel. Error norm_emit_x: {rel_err_x:.4f}\n"
                       f"Rel. Error norm_emit_y: {rel_err_y:.4f}\n"
                       f"Rel. Error norm_emit_z: {rel_err_z:.4f}")
    fig.text(0.5, 0.05, annotation_text, ha='center', va='center', fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
    if title is not None:
        plt.title(title)
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(os.path.join(results_folder, f'{error_type}_mse_sample_{idx}.png'))
    plt.close(fig)
    
import os
import tempfile
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def plot_particle_groups_filename(pred_pg, target_pg, idx, error_type, results_folder,
                         mse_value, rel_err_x, rel_err_y, rel_err_z, filename, title=None):
    """
    Plots predicted and target ParticleGroups onto a single figure by:
    1. Using return_figure=True to get separate figures.
    2. Saving those figures as temporary images.
    3. Reading the images and displaying them with imshow on subplots.

    This avoids manipulating artists directly and guarantees that the plots look 
    the same as originally produced, just arranged in a grid.

    The final figure is saved using the provided filename.

    Args:
        pred_pg (ParticleGroup): Predicted ParticleGroup.
        target_pg (ParticleGroup): Target ParticleGroup.
        idx (int): Sample index.
        error_type (str): Error type (e.g., 'vis' or 'rollout').
        results_folder (str): Folder in which to save the final figure.
        mse_value (float): The MSE value to annotate.
        rel_err_x (float): Relative error for normalized emittance in x.
        rel_err_y (float): Relative error for normalized emittance in y.
        rel_err_z (float): Relative error for normalized emittance in z.
        filename (str): Full path (including filename) where the figure will be saved.
        title (str, optional): Optional title for the figure.
    """
    vars_list = [('x', 'px'), ('y', 'py'), ('z', 'pz')]
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 18))

    def save_plot_as_image(pgroup, xvar, pvar):
        fig_local = pgroup.plot(xvar, pvar, return_figure=True)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
            temp_path = tmpfile.name
        fig_local.savefig(temp_path, dpi=150)
        plt.close(fig_local)
        return temp_path

    for row, (x_var, p_var) in enumerate(vars_list):
        # Predicted subplot (left column)
        ax_pred = axes[row, 0]
        pred_img_path = save_plot_as_image(pred_pg, x_var, pvar=p_var)
        pred_img = mpimg.imread(pred_img_path)
        ax_pred.imshow(pred_img)
        ax_pred.set_title(f'Sample {idx}: Predicted {x_var}-{p_var}')
        ax_pred.axis('off')
        os.remove(pred_img_path)

        # Target subplot (right column)
        ax_target = axes[row, 1]
        target_img_path = save_plot_as_image(target_pg, x_var, pvar=p_var)
        target_img = mpimg.imread(target_img_path)
        ax_target.imshow(target_img)
        ax_target.set_title(f'Sample {idx}: Target {x_var}-{p_var}')
        ax_target.axis('off')
        os.remove(target_img_path)

    annotation_text = (f"MSE: {mse_value:.4f}\n"
                       f"Rel. Error norm_emit_x: {rel_err_x:.4f}\n"
                       f"Rel. Error norm_emit_y: {rel_err_y:.4f}\n"
                       f"Rel. Error norm_emit_z: {rel_err_z:.4f}")
    fig.text(0.5, 0.05, annotation_text, ha='center', va='center', fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
    if title is not None:
        fig.text(0.5, 0.95, title, ha='center', va='center', fontsize=16, weight='bold')
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(filename)
    plt.close(fig)

    
def transform_to_particle_group(data):
    """
    Converts data tensor to ParticleGroup.

    Args:
        data (torch.Tensor): Tensor of shape [num_nodes, 6]

    Returns:
        ParticleGroup
    """
    logging.debug(f"Transforming data to ParticleGroup with shape: {data.shape}")
    num_particles = data.shape[0]
    particle_dict = {
        'x': data[:, 0].numpy(),
        'y': data[:, 1].numpy(),
        'z': data[:, 2].numpy(),
        'px': data[:, 3].numpy(),
        'py': data[:, 4].numpy(),
        'pz': data[:, 5].numpy(),
        'species': 'electron',
        'weight': np.full(num_particles, 2.e-17),
        't': np.zeros(num_particles),
        'status': np.ones(num_particles, dtype=int)
    }
    particle_group = ParticleGroup(data=particle_dict)
    logging.debug("ParticleGroup created successfully.")
    return particle_group


def compute_normalized_emittance_x(particle_group):
    """
    Computes the normalized emittance in x direction for a ParticleGroup.

    Args:
        particle_group (ParticleGroup): The ParticleGroup for which to compute the emittance.

    Returns:
        float: The normalized emittance in x direction.
    """
    x = particle_group['x']
    px = particle_group['px']
    mean_x2 = np.mean(x**2)
    mean_px2 = np.mean(px**2)
    mean_xpx = np.mean(x * px)
    norm_emit_x = np.sqrt(mean_x2 * mean_px2 - mean_xpx**2)
    logging.debug(f"Computed norm_emit_x: {norm_emit_x}")
    return norm_emit_x


def compute_normalized_emittance_y(particle_group):
    """
    Computes the normalized emittance in y direction for a ParticleGroup.
    """
    y = particle_group['y']
    py = particle_group['py']
    mean_y2 = np.mean(y**2)
    mean_py2 = np.mean(py**2)
    mean_ypy = np.mean(y * py)
    norm_emit_y = np.sqrt(mean_y2 * mean_py2 - mean_ypy**2)
    logging.debug(f"Computed norm_emit_y: {norm_emit_y}")
    return norm_emit_y


def compute_normalized_emittance_z(particle_group):
    """
    Computes the normalized emittance in z direction for a ParticleGroup.
    """
    z = particle_group['z']
    pz = particle_group['pz']
    mean_z2 = np.mean(z**2)
    mean_pz2 = np.mean(pz**2)
    mean_zpz = np.mean(z * pz)
    norm_emit_z = np.sqrt(mean_z2 * mean_pz2 - mean_zpz**2)
    logging.debug(f"Computed norm_emit_z: {norm_emit_z}")
    return norm_emit_z

def inverse_normalize_features(features, scale):
    """
    Inverse-normalizes features given the scale data.
    Assumes that scale is a tensor of shape [1, 12] where the first 6 values are the mean
    and the next 6 values are the standard deviation.
    
    Args:
        features (torch.Tensor): Tensor of shape [num_nodes, 6] (normalized node features).
        scale (torch.Tensor): Tensor of shape [1, 12] containing mean and std values.
        
    Returns:
        torch.Tensor: The inverse-normalized node features.
    """
    # Extract mean and std from the scale tensor
    mean = scale[:, :6]
    std = scale[:, 6:]
    return features * std + mean