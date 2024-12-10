# trainers.py

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import logging
import os
from pathlib import Path
from accelerate import Accelerator
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import model classes
from src.graph_models.models.graph_networks import (
    GraphConvolutionNetwork,
    GraphAttentionNetwork,
    GraphTransformer,
    MeshGraphNet
)

from src.graph_models.models.intgnn.models import GNN_TopK
from src.graph_models.models.multiscale.gnn import (
    SinglescaleGNN, 
    MultiscaleGNN, 
    TopkMultiscaleGNN
)

from src.graph_models.context_models.context_graph_networks import *
from src.graph_models.context_models.scale_graph_networks import *


def identify_model_type(model):
    """
    Identifies the type of the model and returns a string identifier.
    
    Args:
        model (nn.Module): The PyTorch model instance to identify.
        
    Returns:
        str: A string identifier representing the model type.
        
    Raises:
        ValueError: If the model type is unrecognized.
    """
    if isinstance(model, GNN_TopK):
        return 'GNN_TopK'
    elif isinstance(model, TopkMultiscaleGNN):
        return 'TopkMultiscaleGNN'
    elif isinstance(model, SinglescaleGNN):
        return 'SinglescaleGNN'
    elif isinstance(model, MultiscaleGNN):
        return 'MultiscaleGNN'
    elif isinstance(model, MeshGraphNet):
        return 'MeshGraphNet'
    elif isinstance(model, MeshGraphAutoEncoder):
        return 'MeshGraphAutoEncoder'
    elif isinstance(model, GraphTransformer):
        return 'GraphTransformer'
    elif isinstance(model, GraphTransformerAutoEncoder):
        return 'GraphTransformerAutoEncoder'
    elif isinstance(model, ScaleAwareLogRatioConditionalGraphNetwork):
        return 'ScaleAwareLogRatioConditionalGraphNetwork'
    elif isinstance(model, GeneralGraphNetwork):
        return 'GeneralGraphNetwork'
    elif isinstance(model, ConditionalGraphNetwork):
        return 'ConditionalGraphNetwork'
    elif isinstance(model, AttentionConditionalGraphNetwork):
        return 'AttentionConditionalGraphNetwork'
    else:
        raise ValueError(f"Unrecognized model type: {type(model).__name__}")


class BaseTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler=None, device='cpu', **kwargs):
        # Initialize the accelerator
        self.accelerator = Accelerator()
        # if self.accelerator.distributed_type == DistributedType.MULTI_GPU:
        #     torch.distributed.init_process_group(backend='nccl')
        
        # Identify and store the model type
        self.model_type = identify_model_type(model)
        logging.info(f"Identified model type: {self.model_type}")
        
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = self.accelerator.device  # Use accelerator's device
        
        self.start_epoch = 0
        self.nepochs = kwargs.get('nepochs', 100)
        self.save_checkpoint_every = kwargs.get('save_checkpoint_every', 10)
        self.results_folder = Path(kwargs.get('results_folder', './results'))
        self.results_folder.mkdir(parents=True, exist_ok=True)
        self.loss_history = []
        self.val_loss_history = []
        self.best_val_loss = float('inf')
        self.best_epoch = -1
        self.verbose = kwargs.get('verbose', False)

        # Create 'checkpoints' subfolder under results_folder
        self.checkpoints_folder = self.results_folder / 'checkpoints'
        self.checkpoints_folder.mkdir(parents=True, exist_ok=True)

        self.random_seed = kwargs.get('random_seed', 42)

        # Checkpoint
        self.checkpoint = kwargs.get('checkpoint', None)
        if self.checkpoint:
            self.load_checkpoint(self.checkpoint)

        # Prepare the model, optimizer, scheduler, and dataloaders
        if self.scheduler:
            self.model, self.optimizer, self.scheduler, self.train_loader, self.val_loader = self.accelerator.prepare(
                self.model, self.optimizer, self.scheduler, self.train_loader, self.val_loader)
        else:
            self.model, self.optimizer, self.train_loader, self.val_loader = self.accelerator.prepare(
                self.model, self.optimizer, self.train_loader, self.val_loader)

    def train(self):
        logging.info("Starting training...")
        for epoch in range(self.start_epoch, self.nepochs):
            self.model.train()
            total_loss = 0
            # Adjust progress bar for distributed training
            if self.verbose and self.accelerator.is_main_process:
                progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.nepochs}")
            else:
                progress_bar = self.train_loader
            for batch_idx, batch in enumerate(progress_bar):
                # No need to move data to device; accelerator handles it
                self.optimizer.zero_grad()
                loss = self.train_step(batch)
                # Use accelerator's backward method
                self.accelerator.backward(loss)
                self.optimizer.step()
                total_loss += loss.item()
                if self.verbose and self.accelerator.is_main_process:
                    current_loss = total_loss / (batch_idx + 1)
                    progress_bar.set_postfix(loss=current_loss)
                    
            self.accelerator.wait_for_everyone()
            # Scheduler step
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                if self.verbose:
                    logging.info(f"Epoch {epoch+1}: Learning rate adjusted to {current_lr}")

            # Save loss history
            avg_loss = total_loss / len(self.train_loader)
            self.loss_history.append(avg_loss)
            
            # Validation
            if self.val_loader:
                val_loss = self.validate()
                self.val_loss_history.append(val_loss)
            else:
                val_loss = None  # No validation loss

            if self.accelerator.is_main_process:
                if val_loss is not None:
                    logging.info(f'Epoch {epoch+1}/{self.nepochs}, Loss: {avg_loss:.4e}, Val Loss: {val_loss:.4e}')
                else:
                    logging.info(f'Epoch {epoch+1}/{self.nepochs}, Loss: {avg_loss:.4e}')

            # Save checkpoint
            if (epoch + 1) % self.save_checkpoint_every == 0 or (epoch + 1) == self.nepochs:
                self.save_checkpoint(epoch)

            # Check and save the best model
            if val_loss is not None and val_loss < self.best_val_loss:
                logging.info(f"New best model found at epoch {epoch+1} with Val Loss: {val_loss:.4e}")
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
                logging.info(f"self.best_epoch is updated to epoch {epoch+1}")
                self.save_checkpoint(epoch, best=True)

        # Plot loss convergence
        if self.accelerator.is_main_process:
            self.plot_loss_convergence()
            logging.info("Training complete!")
            logging.info(f"Best Val Loss: {self.best_val_loss:.4e} at epoch {self.best_epoch}")

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for batch in self.val_loader:
                loss = self.validate_step(batch)
                val_loss += loss.item()
                num_batches += 1

        # Convert to tensors for aggregation
        total_val_loss = torch.tensor(val_loss, device=self.accelerator.device)
        total_num_batches = torch.tensor(num_batches, device=self.accelerator.device)

        # Aggregate the losses across all processes
        total_val_loss = self.accelerator.gather(total_val_loss).sum()
        total_num_batches = self.accelerator.gather(total_num_batches).sum()

        # Compute the average validation loss
        avg_val_loss = total_val_loss / total_num_batches

        return avg_val_loss.item()

    def train_step(self, data):
        raise NotImplementedError("Subclasses should implement this method.")

    def validate_step(self, data):
        # By default, use the same logic as train_step
        self.model.eval()
        with torch.no_grad():
            loss = self.train_step(data)
        return loss

    def save_checkpoint(self, epoch, best=False):
        # Unwrap the model to get the original model (not wrapped by accelerator)
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        if best:
            checkpoint_path = self.checkpoints_folder / f'best_model.pth'
        else:
            checkpoint_path = self.checkpoints_folder / f'model-{epoch}.pth'
        # Prepare checkpoint data
        checkpoint_data = {
            'model_state_dict': unwrapped_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': epoch + 1,  # Save the next epoch to resume from
            'random_seed': self.random_seed,
            'loss_history': self.loss_history,
            'val_loss_history': self.val_loss_history,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
        }
        # Use accelerator's save method
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.accelerator.save(checkpoint_data, checkpoint_path)
            logging.info(f"Model checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch']
        if 'random_seed' in checkpoint:
            self.random_seed = checkpoint['random_seed']
            logging.info(f"Using random seed from checkpoint: {self.random_seed}")
        if 'loss_history' in checkpoint:
            self.loss_history = checkpoint['loss_history']
        if 'val_loss_history' in checkpoint:
            self.val_loss_history = checkpoint['val_loss_history']
        if 'best_val_loss' in checkpoint:
            self.best_val_loss = checkpoint['best_val_loss']
        if 'best_epoch' in checkpoint:
            self.best_epoch = checkpoint['best_epoch']
        logging.info(f"Resumed training from epoch {self.start_epoch}")

    def plot_loss_convergence(self):
        if self.accelerator.is_main_process:
            plt.figure(figsize=(10, 6))
            plt.plot(self.loss_history, label="Training Loss")
            if self.val_loss_history:
                plt.plot(self.val_loss_history, label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Loss Convergence")
            plt.legend()
            plt.grid(True)
            plt.savefig(self.results_folder / "loss_convergence.png")
            plt.close()

class SequenceTrainerAccelerate(BaseTrainer):
    def __init__(self, **kwargs):
        # Pop out values from kwargs
        criterion = kwargs.pop('criterion', None)
        discount_factor = kwargs.pop('discount_factor', 0.9)
        lambda_ratio = kwargs.pop('lambda_ratio', 1.0)
        noise_level = kwargs.pop('noise_level', 0.0)
        
        # Initialize the parent class
        super().__init__(**kwargs)
        
        self.criterion = criterion if criterion is not None else torch.nn.MSELoss()
        self.discount_factor = discount_factor
        self.lambda_ratio = lambda_ratio
        self.noise_level = noise_level
        
        logging.info(f"Using loss function: {self.criterion.__class__.__name__}")
        logging.info(f"Using discount factor: {self.discount_factor}")
        logging.info(f"Using lambda ratio for loss weighting: {self.lambda_ratio}")
        logging.info(f"Using noise level: {self.noise_level}")

    def train_step(self, batch):
        """
        Perform one training step with the new model and dataset.
        """
        self.model.train()
        total_loss = 0.0
        batch_size = len(batch)
        epsilon = 1e-8  # To prevent division by zero
        
        for data_tuple in batch:
            # Unpack the data tuple
            if len(data_tuple) == 4:
                initial_graph, target_graph, seq_length, settings_tensor = data_tuple
            else:
                raise ValueError("Expected 4 elements in the data tuple.")
            
            # Add noise to initial node features if noise_level > 0
            if self.noise_level > 0:
                initial_graph.x += torch.randn_like(initial_graph.x) * self.noise_level

            # Forward pass
            predicted_node_features, predicted_log_ratios = self.model_forward(
                initial_graph=initial_graph,
                settings_tensor=settings_tensor,
                batch=initial_graph.batch,
                model_type=self.model_type
            )
            
            # Compute actual log ratios
            actual_log_ratios = torch.log(
                (target_graph.scale + epsilon) / (initial_graph.scale + epsilon)
            )

            # Compute losses
            log_ratio_loss = self.criterion(predicted_log_ratios, actual_log_ratios)
            node_reconstruction_loss = self.criterion(predicted_node_features, target_graph.x)
            
            # Weighted loss
            loss = node_reconstruction_loss + self.lambda_ratio * log_ratio_loss

            # Apply discount factor for multi-step sequences
            discounted_loss = (self.discount_factor ** (seq_length - 1)) * loss
            
            # Accumulate loss
            total_loss += discounted_loss / batch_size
        
        return total_loss

    def validate_step(self, batch):
        """
        Perform one validation step with the new model and dataset.
        """
        self.model.eval()
        total_loss = 0.0
        batch_size = len(batch)
        epsilon = 1e-8  # To prevent division by zero

        with torch.no_grad():
            for data_tuple in batch:
                # Unpack the data tuple
                if len(data_tuple) == 4:
                    initial_graph, target_graph, seq_length, settings_tensor = data_tuple
                else:
                    raise ValueError("Expected 4 elements in the data tuple.")
                
                # Forward pass
                predicted_node_features, predicted_log_ratios = self.model_forward(
                    initial_graph=initial_graph,
                    settings_tensor=settings_tensor,
                    batch=initial_graph.batch,
                    model_type=self.model_type
                )
                
                # Compute actual log ratios
                actual_log_ratios = torch.log(
                    (target_graph.scale + epsilon) / (initial_graph.scale + epsilon)
                )

                # Compute losses
                log_ratio_loss = self.criterion(predicted_log_ratios, actual_log_ratios)
                node_reconstruction_loss = self.criterion(predicted_node_features, target_graph.x)

                # Weighted loss
                loss = node_reconstruction_loss + self.lambda_ratio * log_ratio_loss

                # Apply discount factor for multi-step sequences
                discounted_loss = (self.discount_factor ** (seq_length - 1)) * loss
                
                # Accumulate loss
                total_loss += discounted_loss / batch_size
        
        return total_loss

    def model_forward(self, initial_graph, settings_tensor, batch, model_type):
        """
        Calls the model's forward method based on the identified model type.
        """
        if model_type == 'ScaleAwareLogRatioConditionalGraphNetwork':
            # Extract necessary inputs from initial_graph
            x = initial_graph.x
            edge_index = initial_graph.edge_index
            edge_attr = initial_graph.edge_attr
            scale = initial_graph.scale
            conditions = settings_tensor
            
            # Forward pass through the model
            predicted_features, predicted_log_ratios = self.model(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                conditions=conditions,
                scale=scale,
                batch=batch
            )
        elif model_type == 'GeneralGraphNetwork':
            # Example handling for GeneralGraphNetwork
            x = initial_graph.x
            edge_index = initial_graph.edge_index
            edge_attr = initial_graph.edge_attr
            u = initial_graph.global_features  # Assuming global_features exist
            # Forward pass
            x_pred, edge_pred, u_pred = self.model(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                u=u,
                batch=batch
            )
            predicted_features = x_pred
            predicted_log_ratios = u_pred  # Assuming u_pred contains log ratios
        else:
            raise NotImplementedError(f"Model type '{model_type}' is not supported.")
        
        return predicted_node_features, predicted_log_ratios