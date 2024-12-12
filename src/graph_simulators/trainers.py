# trainers.py

# import torch
# import torch.nn.functional as F
# from torch_geometric.data import Data
# from torch_scatter import scatter_mean

# import logging
# import os
# from pathlib import Path
# from accelerate import Accelerator
# import matplotlib.pyplot as plt
# from tqdm import tqdm

# # Import model classes
# from src.graph_models.models.graph_networks import (
#     GraphConvolutionNetwork,
#     GraphAttentionNetwork,
#     GraphTransformer,
#     MeshGraphNet
# )
# from src.graph_models.models.graph_autoencoders import (
#     GraphConvolutionalAutoEncoder,
#     GraphAttentionAutoEncoder,
#     GraphTransformerAutoEncoder,
#     MeshGraphAutoEncoder
# )

# from src.graph_models.models.intgnn.models import GNN_TopK
# from src.graph_models.models.multiscale.gnn import (
#     SinglescaleGNN, 
#     MultiscaleGNN, 
#     TopkMultiscaleGNN
# )

# from src.graph_models.context_models.context_graph_networks import *
# from src.graph_models.context_models.scale_graph_networks import *


# def identify_model_type(model):
#     """
#     Identifies the type of the model and returns a string identifier.
    
#     Args:
#         model (nn.Module): The PyTorch model instance to identify.
        
#     Returns:
#         str: A string identifier representing the model type.
        
#     Raises:
#         ValueError: If the model type is unrecognized.
#     """
#     if isinstance(model, GNN_TopK):
#         return 'GNN_TopK'
#     elif isinstance(model, TopkMultiscaleGNN):
#         return 'TopkMultiscaleGNN'
#     elif isinstance(model, SinglescaleGNN):
#         return 'SinglescaleGNN'
#     elif isinstance(model, MultiscaleGNN):
#         return 'MultiscaleGNN'
#     elif isinstance(model, MeshGraphNet):
#         return 'MeshGraphNet'
#     elif isinstance(model, MeshGraphAutoEncoder):
#         return 'MeshGraphAutoEncoder'
#     elif isinstance(model, GraphTransformer):
#         return 'GraphTransformer'
#     elif isinstance(model, GraphTransformerAutoEncoder):
#         return 'GraphTransformerAutoEncoder'
#     elif isinstance(model, ScaleAwareLogRatioConditionalGraphNetwork):
#         return 'ScaleAwareLogRatioConditionalGraphNetwork'
#     elif isinstance(model, GeneralGraphNetwork):
#         return 'GeneralGraphNetwork'
#     elif isinstance(model, ConditionalGraphNetwork):
#         return 'ConditionalGraphNetwork'
#     elif isinstance(model, AttentionConditionalGraphNetwork):
#         return 'AttentionConditionalGraphNetwork'
#     else:
#         raise ValueError(f"Unrecognized model type: {type(model).__name__}")


# class BaseTrainer:
#     def __init__(self, model, train_loader, val_loader, optimizer, scheduler=None, device='cpu', **kwargs):
#         # Initialize the accelerator
#         self.accelerator = Accelerator()
#         # if self.accelerator.distributed_type == DistributedType.MULTI_GPU:
#         #     torch.distributed.init_process_group(backend='nccl')
        
#         # Identify and store the model type
#         self.model_type = identify_model_type(model)
#         logging.info(f"Identified model type: {self.model_type}")
        
#         self.model = model
#         self.optimizer = optimizer
#         self.scheduler = scheduler
#         self.train_loader = train_loader
#         self.val_loader = val_loader
#         self.device = self.accelerator.device  # Use accelerator's device
        
#         self.start_epoch = 0
#         self.nepochs = kwargs.get('nepochs', 100)
#         self.save_checkpoint_every = kwargs.get('save_checkpoint_every', 10)
#         self.results_folder = Path(kwargs.get('results_folder', './results'))
#         self.results_folder.mkdir(parents=True, exist_ok=True)
#         self.loss_history = []
#         self.val_loss_history = []
#         self.best_val_loss = float('inf')
#         self.best_epoch = -1
#         self.verbose = kwargs.get('verbose', False)

#         # Create 'checkpoints' subfolder under results_folder
#         self.checkpoints_folder = self.results_folder / 'checkpoints'
#         self.checkpoints_folder.mkdir(parents=True, exist_ok=True)

#         self.random_seed = kwargs.get('random_seed', 42)

#         # Checkpoint
#         self.checkpoint = kwargs.get('checkpoint', None)
#         if self.checkpoint:
#             self.load_checkpoint(self.checkpoint)

#         # Prepare the model, optimizer, scheduler, and dataloaders
#         if self.scheduler:
#             self.model, self.optimizer, self.scheduler, self.train_loader, self.val_loader = self.accelerator.prepare(
#                 self.model, self.optimizer, self.scheduler, self.train_loader, self.val_loader)
#         else:
#             self.model, self.optimizer, self.train_loader, self.val_loader = self.accelerator.prepare(
#                 self.model, self.optimizer, self.train_loader, self.val_loader)

#     def train(self):
#         logging.info("Starting training...")
#         for epoch in range(self.start_epoch, self.nepochs):
#             self.model.train()
#             total_loss = 0
#             # Adjust progress bar for distributed training
#             if self.verbose and self.accelerator.is_main_process:
#                 progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.nepochs}")
#             else:
#                 progress_bar = self.train_loader
#             for batch_idx, batch in enumerate(progress_bar):
#                 # No need to move data to device; accelerator handles it
#                 self.optimizer.zero_grad()
#                 loss = self.train_step(batch)
#                 # Use accelerator's backward method
#                 self.accelerator.backward(loss)
#                 self.optimizer.step()
#                 total_loss += loss.item()
#                 if self.verbose and self.accelerator.is_main_process:
#                     current_loss = total_loss / (batch_idx + 1)
#                     progress_bar.set_postfix(loss=current_loss)
                    
#             self.accelerator.wait_for_everyone()
#             # Scheduler step
#             if self.scheduler:
#                 self.scheduler.step()
#                 current_lr = self.optimizer.param_groups[0]['lr']
#                 if self.verbose:
#                     logging.info(f"Epoch {epoch+1}: Learning rate adjusted to {current_lr}")

#             # Save loss history
#             avg_loss = total_loss / len(self.train_loader)
#             self.loss_history.append(avg_loss)
            
#             # Validation
#             if self.val_loader:
#                 val_loss = self.validate()
#                 self.val_loss_history.append(val_loss)
#             else:
#                 val_loss = None  # No validation loss

#             if self.accelerator.is_main_process:
#                 if val_loss is not None:
#                     logging.info(f'Epoch {epoch+1}/{self.nepochs}, Loss: {avg_loss:.4e}, Val Loss: {val_loss:.4e}')
#                 else:
#                     logging.info(f'Epoch {epoch+1}/{self.nepochs}, Loss: {avg_loss:.4e}')

#             # Save checkpoint
#             if (epoch + 1) % self.save_checkpoint_every == 0 or (epoch + 1) == self.nepochs:
#                 self.save_checkpoint(epoch)

#             # Check and save the best model
#             if val_loss is not None and val_loss < self.best_val_loss:
#                 logging.info(f"New best model found at epoch {epoch+1} with Val Loss: {val_loss:.4e}")
#                 self.best_val_loss = val_loss
#                 self.best_epoch = epoch + 1
#                 logging.info(f"self.best_epoch is updated to epoch {epoch+1}")
#                 self.save_checkpoint(epoch, best=True)

#         # Plot loss convergence
#         if self.accelerator.is_main_process:
#             self.plot_loss_convergence()
#             logging.info("Training complete!")
#             logging.info(f"Best Val Loss: {self.best_val_loss:.4e} at epoch {self.best_epoch}")

#     def validate(self):
#         self.model.eval()
#         val_loss = 0.0
#         num_batches = 0
#         with torch.no_grad():
#             for batch in self.val_loader:
#                 loss = self.validate_step(batch)
#                 val_loss += loss.item()
#                 num_batches += 1

#         # Convert to tensors for aggregation
#         total_val_loss = torch.tensor(val_loss, device=self.accelerator.device)
#         total_num_batches = torch.tensor(num_batches, device=self.accelerator.device)

#         # Aggregate the losses across all processes
#         total_val_loss = self.accelerator.gather(total_val_loss).sum()
#         total_num_batches = self.accelerator.gather(total_num_batches).sum()

#         # Compute the average validation loss
#         avg_val_loss = total_val_loss / total_num_batches

#         return avg_val_loss.item()

#     def train_step(self, data):
#         raise NotImplementedError("Subclasses should implement this method.")

#     def validate_step(self, data):
#         # By default, use the same logic as train_step
#         self.model.eval()
#         with torch.no_grad():
#             loss = self.train_step(data)
#         return loss

#     def save_checkpoint(self, epoch, best=False):
#         # Unwrap the model to get the original model (not wrapped by accelerator)
#         unwrapped_model = self.accelerator.unwrap_model(self.model)
#         if best:
#             checkpoint_path = self.checkpoints_folder / f'best_model.pth'
#         else:
#             checkpoint_path = self.checkpoints_folder / f'model-{epoch}.pth'
#         # Prepare checkpoint data
#         checkpoint_data = {
#             'model_state_dict': unwrapped_model.state_dict(),
#             'optimizer_state_dict': self.optimizer.state_dict(),
#             'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
#             'epoch': epoch + 1,  # Save the next epoch to resume from
#             'random_seed': self.random_seed,
#             'loss_history': self.loss_history,
#             'val_loss_history': self.val_loss_history,
#             'best_val_loss': self.best_val_loss,
#             'best_epoch': self.best_epoch,
#         }
#         # Use accelerator's save method
#         self.accelerator.wait_for_everyone()
#         if self.accelerator.is_main_process:
#             self.accelerator.save(checkpoint_data, checkpoint_path)
#             logging.info(f"Model checkpoint saved to {checkpoint_path}")

#     def load_checkpoint(self, checkpoint_path):
#         logging.info(f"Loading checkpoint from {checkpoint_path}")
#         checkpoint = torch.load(checkpoint_path, map_location=self.device)
#         self.model.load_state_dict(checkpoint['model_state_dict'])
#         self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         if self.scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
#             self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
#         self.start_epoch = checkpoint['epoch']
#         if 'random_seed' in checkpoint:
#             self.random_seed = checkpoint['random_seed']
#             logging.info(f"Using random seed from checkpoint: {self.random_seed}")
#         if 'loss_history' in checkpoint:
#             self.loss_history = checkpoint['loss_history']
#         if 'val_loss_history' in checkpoint:
#             self.val_loss_history = checkpoint['val_loss_history']
#         if 'best_val_loss' in checkpoint:
#             self.best_val_loss = checkpoint['best_val_loss']
#         if 'best_epoch' in checkpoint:
#             self.best_epoch = checkpoint['best_epoch']
#         logging.info(f"Resumed training from epoch {self.start_epoch}")

#     def plot_loss_convergence(self):
#         if self.accelerator.is_main_process:
#             plt.figure(figsize=(10, 6))
#             plt.plot(self.loss_history, label="Training Loss")
#             if self.val_loss_history:
#                 plt.plot(self.val_loss_history, label="Validation Loss")
#             plt.xlabel("Epoch")
#             plt.ylabel("Loss")
#             plt.title("Loss Convergence")
#             plt.legend()
#             plt.grid(True)
#             plt.savefig(self.results_folder / "loss_convergence.png")
#             plt.close()

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_scatter import scatter_mean

import logging
import os
from pathlib import Path
from accelerate import Accelerator
from torch.optim import Optimizer

import matplotlib.pyplot as plt
from tqdm import tqdm

# Import model classes
from src.graph_models.models.graph_networks import (
    GraphConvolutionNetwork,
    GraphAttentionNetwork,
    GraphTransformer,
    MeshGraphNet
)
from src.graph_models.models.graph_autoencoders import (
    GraphConvolutionalAutoEncoder,
    GraphAttentionAutoEncoder,
    GraphTransformerAutoEncoder,
    MeshGraphAutoEncoder
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


# class BaseTrainer:
#     def __init__(self, model, train_loader, val_loader, optimizer, scheduler=None, device='cpu', **kwargs):
#         self.accelerator = Accelerator()
#         self.model_type = identify_model_type(model)
#         logging.info(f"Identified model type: {self.model_type}")
        
#         self.model = model
#         self.optimizer = optimizer
#         self.scheduler = scheduler
#         self.train_loader = train_loader
#         self.val_loader = val_loader
#         self.device = self.accelerator.device
        
#         self.start_epoch = 0
#         self.nepochs = kwargs.get('nepochs', 100)
#         self.save_checkpoint_every = kwargs.get('save_checkpoint_every', 10)
#         self.results_folder = Path(kwargs.get('results_folder', './results'))
#         self.results_folder.mkdir(parents=True, exist_ok=True)
#         self.loss_history = []
#         self.val_loss_history = []
#         self.best_val_loss = float('inf')
#         self.best_epoch = -1
#         self.verbose = kwargs.get('verbose', False)

#         self.checkpoints_folder = self.results_folder / 'checkpoints'
#         self.checkpoints_folder.mkdir(parents=True, exist_ok=True)

#         self.random_seed = kwargs.get('random_seed', 42)

#         self.checkpoint = kwargs.get('checkpoint', None)
#         if self.checkpoint:
#             self.load_checkpoint(self.checkpoint)

#         if self.scheduler:
#             self.model, self.optimizer, self.scheduler, self.train_loader, self.val_loader = self.accelerator.prepare(
#                 self.model, self.optimizer, self.scheduler, self.train_loader, self.val_loader)
#         else:
#             self.model, self.optimizer, self.train_loader, self.val_loader = self.accelerator.prepare(
#                 self.model, self.optimizer, self.train_loader, self.val_loader)

#     def train(self):
#         logging.info("Starting training...")
#         for epoch in range(self.start_epoch, self.nepochs):
#             logging.info(f"Starting epoch {epoch+1}/{self.nepochs}")
#             self.model.train()
#             total_loss = 0
#             if self.verbose and self.accelerator.is_main_process:
#                 progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.nepochs}")
#             else:
#                 progress_bar = self.train_loader
#             for batch_idx, batch in enumerate(progress_bar):
#                 self.optimizer.zero_grad()
#                 loss = self.train_step(batch)
#                 self.accelerator.backward(loss)
#                 self.optimizer.step()
#                 total_loss += loss.item()
#                 if self.verbose and self.accelerator.is_main_process:
#                     current_loss = total_loss / (batch_idx + 1)
#                     progress_bar.set_postfix(loss=current_loss)
#                     logging.debug(f"Batch {batch_idx}: Loss = {current_loss:.4e}")
            
#             self.accelerator.wait_for_everyone()
#             if self.scheduler:
#                 self.scheduler.step()
#                 current_lr = self.optimizer.param_groups[0]['lr']
#                 logging.info(f"Epoch {epoch+1}: Learning rate adjusted to {current_lr}")

#             avg_loss = total_loss / len(self.train_loader)
#             self.loss_history.append(avg_loss)
#             logging.info(f"Epoch {epoch+1}: Average Training Loss = {avg_loss:.4e}")

#             if self.val_loader:
#                 val_loss = self.validate()
#                 self.val_loss_history.append(val_loss)
#                 logging.info(f"Epoch {epoch+1}: Validation Loss = {val_loss:.4e}")
#             else:
#                 val_loss = None

#             if (epoch + 1) % self.save_checkpoint_every == 0 or (epoch + 1) == self.nepochs:
#                 self.save_checkpoint(epoch)
#                 logging.info(f"Checkpoint saved at epoch {epoch+1}")

#             if val_loss is not None and val_loss < self.best_val_loss:
#                 self.best_val_loss = val_loss
#                 self.best_epoch = epoch + 1
#                 self.save_checkpoint(epoch, best=True)
#                 logging.info(f"New best model at epoch {epoch+1}: Val Loss = {val_loss:.4e}")

#         if self.accelerator.is_main_process:
#             self.plot_loss_convergence()
#             logging.info("Training complete!")
#             logging.info(f"Best Val Loss = {self.best_val_loss:.4e} at epoch {self.best_epoch}")

#     def validate(self):
#         logging.info("Starting validation...")
#         self.model.eval()
#         val_loss = 0.0
#         num_batches = 0
#         with torch.no_grad():
#             for batch in self.val_loader:
#                 loss = self.validate_step(batch)
#                 val_loss += loss.item()
#                 num_batches += 1
#                 logging.debug(f"Validation Batch: Loss = {loss.item():.4e}")
        
#         avg_val_loss = val_loss / num_batches
#         logging.info(f"Validation completed: Average Loss = {avg_val_loss:.4e}")
#         return avg_val_loss

#     def train_step(self, data):
#         raise NotImplementedError("Subclasses should implement this method.")

#     def validate_step(self, data):
#         self.model.eval()
#         with torch.no_grad():
#             loss = self.train_step(data)
#         return loss

#     def save_checkpoint(self, epoch, best=False):
#         unwrapped_model = self.accelerator.unwrap_model(self.model)
#         checkpoint_path = self.checkpoints_folder / ('best_model.pth' if best else f'model-{epoch}.pth')
#         checkpoint_data = {
#             'model_state_dict': unwrapped_model.state_dict(),
#             'optimizer_state_dict': self.optimizer.state_dict(),
#             'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
#             'epoch': epoch + 1,
#             'random_seed': self.random_seed,
#             'loss_history': self.loss_history,
#             'val_loss_history': self.val_loss_history,
#             'best_val_loss': self.best_val_loss,
#             'best_epoch': self.best_epoch,
#         }
#         self.accelerator.wait_for_everyone()
#         if self.accelerator.is_main_process:
#             self.accelerator.save(checkpoint_data, checkpoint_path)
#             logging.info(f"Checkpoint {'best model' if best else f'epoch {epoch}'} saved to {checkpoint_path}")

#     def load_checkpoint(self, checkpoint_path):
#         logging.info(f"Loading checkpoint from {checkpoint_path}")
#         checkpoint = torch.load(checkpoint_path, map_location=self.device)
#         self.model.load_state_dict(checkpoint['model_state_dict'])
#         self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         if self.scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
#             self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
#         self.start_epoch = checkpoint['epoch']
#         self.loss_history = checkpoint.get('loss_history', [])
#         self.val_loss_history = checkpoint.get('val_loss_history', [])
#         self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
#         self.best_epoch = checkpoint.get('best_epoch', -1)
#         logging.info(f"Resumed training from epoch {self.start_epoch}, Best Val Loss = {self.best_val_loss:.4e}")

#     def plot_loss_convergence(self):
#         if self.accelerator.is_main_process:
#             plt.figure(figsize=(10, 6))
#             plt.plot(self.loss_history, label="Training Loss")
#             if self.val_loss_history:
#                 plt.plot(self.val_loss_history, label="Validation Loss")
#             plt.xlabel("Epoch")
#             plt.ylabel("Loss")
#             plt.title("Loss Convergence")
#             plt.legend()
#             plt.grid(True)
#             plt.savefig(self.results_folder / "loss_convergence.png")
#             logging.info(f"Loss convergence plot saved to {self.results_folder / 'loss_convergence.png'}")
#             plt.close()

class BaseTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        device: str = 'cpu',
        **kwargs,
    ):
        # Initialize the accelerator for distributed training
        self.accelerator = Accelerator()
        self.device = self.accelerator.device

        # Identify and store the model type
        self.model_type = identify_model_type(model)
        logging.info(f"Identified model type: {self.model_type}")

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.start_epoch = 0
        self.nepochs = kwargs.get('nepochs', 100)
        self.save_checkpoint_every = kwargs.get('save_checkpoint_every', 10)
        self.results_folder = Path(kwargs.get('results_folder', './results'))
        self.results_folder.mkdir(parents=True, exist_ok=True)
        self.checkpoints_folder = self.results_folder / 'checkpoints'
        self.checkpoints_folder.mkdir(parents=True, exist_ok=True)

        self.loss_history = []
        self.val_loss_history = []
        self.best_val_loss = float('inf')
        self.best_epoch = -1

        self.verbose = kwargs.get('verbose', False)
        self.random_seed = kwargs.get('random_seed', 42)

        # Handle checkpoint loading
        checkpoint_path = kwargs.get('checkpoint')
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

        # Prepare the model, optimizer, scheduler, and dataloaders with Accelerator
        prepare_args = [self.model, self.optimizer]
        if self.scheduler:
            prepare_args.append(self.scheduler)
        prepare_args.extend([self.train_loader, self.val_loader])

        prepared = self.accelerator.prepare(*prepare_args)
        self.model, self.optimizer = prepared[:2]
        if self.scheduler:
            self.scheduler = prepared[2]
            self.train_loader, self.val_loader = prepared[3], prepared[4]
        else:
            self.train_loader, self.val_loader = prepared[2], prepared[3]

    def train(self):
        logging.info("Starting training...")
        for epoch in range(self.start_epoch, self.nepochs):
            self.model.train()
            total_loss = 0.0

            # Setup progress bar
            if self.verbose and self.accelerator.is_main_process:
                progress_bar = tqdm(
                    self.train_loader,
                    desc=f"Epoch {epoch + 1}/{self.nepochs}",
                    disable=not self.verbose,
                )
            else:
                progress_bar = self.train_loader

            for batch_idx, data in enumerate(progress_bar):
                self.optimizer.zero_grad()
                loss = self.train_step(data)
                self.accelerator.backward(loss)
                self.optimizer.step()
                total_loss += loss.item()

                if self.verbose and self.accelerator.is_main_process:
                    current_loss = total_loss / (batch_idx + 1)
                    progress_bar.set_postfix(loss=f"{current_loss:.4e}")

            # Scheduler step
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                if self.verbose:
                    logging.info(f"Epoch {epoch + 1}: Learning rate adjusted to {current_lr}")

            # Record training loss
            avg_loss = total_loss / len(self.train_loader)
            self.loss_history.append(avg_loss)

            # Validation
            val_loss = None
            if self.val_loader:
                val_loss = self.validate()
                self.val_loss_history.append(val_loss)

            # Logging
            if self.accelerator.is_main_process:
                if val_loss is not None:
                    logging.info(
                        f"Epoch {epoch + 1}/{self.nepochs}, "
                        f"Loss: {avg_loss:.4e}, Val Loss: {val_loss:.4e}"
                    )
                else:
                    logging.info(f"Epoch {epoch + 1}/{self.nepochs}, Loss: {avg_loss:.4e}")

            # Save checkpoint
            if (epoch + 1) % self.save_checkpoint_every == 0 or (epoch + 1) == self.nepochs:
                self.save_checkpoint(epoch)

            # Save the best model based on validation loss
            if val_loss is not None and val_loss < self.best_val_loss:
                logging.info(
                    f"New best model found at epoch {epoch + 1} with Val Loss: {val_loss:.4e}"
                )
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
                self.save_checkpoint(epoch, best=True)

        # Finalize training
        if self.accelerator.is_main_process:
            self.plot_loss_convergence()
            logging.info("Training complete!")
            logging.info(f"Best Val Loss: {self.best_val_loss:.4e} at epoch {self.best_epoch}")

    def validate(self) -> float:
        self.model.eval()
        val_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for data in self.val_loader:
                loss = self.validate_step(data)
                val_loss += loss.item()
                num_batches += 1

        # Aggregate losses across all processes
        total_val_loss = torch.tensor(val_loss, device=self.accelerator.device)
        total_num_batches = torch.tensor(num_batches, device=self.accelerator.device)

        total_val_loss = self.accelerator.gather(total_val_loss).sum()
        total_num_batches = self.accelerator.gather(total_num_batches).sum()

        avg_val_loss = total_val_loss / total_num_batches
        return avg_val_loss.item()

    def train_step(self, data):
        raise NotImplementedError("Subclasses should implement this method.")

    def validate_step(self, data):
        self.model.eval()
        with torch.no_grad():
            return self.train_step(data)

    def save_checkpoint(self, epoch: int, best: bool = False):
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        checkpoint_filename = 'best_model.pth' if best else f'model-{epoch + 1}.pth'
        checkpoint_path = self.checkpoints_folder / checkpoint_filename

        checkpoint_data = {
            'model_state_dict': unwrapped_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': epoch + 1,
            'random_seed': self.random_seed,
            'loss_history': self.loss_history,
            'val_loss_history': self.val_loss_history,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
        }

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.accelerator.save(checkpoint_data, checkpoint_path)
            logging.info(f"Model checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.start_epoch = checkpoint.get('epoch', 0)
        self.random_seed = checkpoint.get('random_seed', self.random_seed)
        self.loss_history = checkpoint.get('loss_history', [])
        self.val_loss_history = checkpoint.get('val_loss_history', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_epoch = checkpoint.get('best_epoch', -1)

        logging.info(f"Resumed training from epoch {self.start_epoch}")

    def plot_loss_convergence(self):
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
        criterion = kwargs.pop('criterion', None)
        discount_factor = kwargs.pop('discount_factor', 0.9)
        lambda_ratio = kwargs.pop('lambda_ratio', 1.0)
        noise_level = kwargs.pop('noise_level', 0.0)
        
        super().__init__(**kwargs)
        
        self.criterion = criterion if criterion is not None else torch.nn.MSELoss(reduction='none')
        self.discount_factor = discount_factor
        self.lambda_ratio = lambda_ratio
        self.noise_level = noise_level
        
        logging.info(f"Using loss function: {self.criterion.__class__.__name__}")
        logging.info(f"Using discount factor: {self.discount_factor}")
        logging.info(f"Using lambda ratio for loss weighting: {self.lambda_ratio}")
        logging.info(f"Using noise level: {self.noise_level}")

    def train_step(self, batch):
        """
        Perform one training step on a batch of graphs.
        Batch contains batched initial graphs, batched target graphs, seq_lengths, 
        and possibly settings_tensor.
        """
        self.model.train()
        epsilon = 1e-8

        # Unpack batch
        if len(batch) == 4:
            batch_initial, batch_target, seq_lengths, settings_tensor = batch
        else:
            batch_initial, batch_target, seq_lengths = batch
            settings_tensor = None

        # Optionally add noise to the initial node features
        if self.noise_level > 0:
            batch_initial.x = batch_initial.x + torch.randn_like(batch_initial.x) * self.noise_level

        # Forward pass
        predicted_node_features, predicted_log_ratios = self.model_forward(
            initial_graph=batch_initial,
            settings_tensor=settings_tensor,
            batch=batch_initial.batch,
            model_type=self.model_type
        )

        # Compute actual log ratios
        # Assumes scale is graph-level [B, scale_dim]
        actual_log_ratios = torch.log(torch.abs((batch_target.scale + epsilon) / (batch_initial.scale + epsilon)))  # [B, log_ratio_dim]

        # Ensure actual_log_ratios has the same shape as predicted_log_ratios
        if actual_log_ratios.shape != predicted_log_ratios.shape:
            raise ValueError(f"Shape mismatch between actual_log_ratios {actual_log_ratios.shape} and predicted_log_ratios {predicted_log_ratios.shape}")

        # Compute node-level reconstruction loss
        # [N, node_out_dim]
        node_recon_loss_per_node = self.criterion(predicted_node_features, batch_target.x)  # [8000, 6]
        # Mean over node feature dimension to get [8000]
        if node_recon_loss_per_node.dim() > 1:
            node_recon_loss_per_node = node_recon_loss_per_node.mean(dim=1)  # [8000]

        # Debugging: Verify shapes
        # print(f"node_recon_loss_per_node.shape: {node_recon_loss_per_node.shape}")  # Expected: [8000]
        # print(f"batch_initial.batch.shape: {batch_initial.batch.shape}")  # Expected: [8000]

        # Compute log ratio loss per graph
        # [B, log_ratio_dim]
        log_ratio_loss = self.criterion(predicted_log_ratios, actual_log_ratios)  # [4, log_ratio_dim]
        # Mean over log_ratio_dim to get [4]
        if log_ratio_loss.dim() > 1:
            log_ratio_loss = log_ratio_loss.mean(dim=1)  # [4]

        # Debugging: Verify shapes
        # print(f"log_ratio_loss.shape: {log_ratio_loss.shape}")  # Expected: [4]
        # print(f"actual_log_ratios: {actual_log_ratios}")  # Expected: [4, log_ratio_dim]   

        # **Important Correction: Apply scatter_mean to node_recon_loss_per_node, not log_ratio_loss**
        # Aggregate node-level losses to per-graph losses
        node_recon_loss_per_graph = scatter_mean(
            node_recon_loss_per_node, 
            batch_initial.batch, 
            dim=0, 
            dim_size=batch_initial.num_graphs  # Ensure correct output size
        )  # [4]

        # **Do not scatter log_ratio_loss**; it's already [4]
        # Combine node and log ratio losses
        loss_per_graph = node_recon_loss_per_graph + self.lambda_ratio * log_ratio_loss  # [4]

        # Apply discount factor per graph
        discount_factors = (self.discount_factor ** (seq_lengths - 1))  # [4]
        discounted_loss_per_graph = loss_per_graph * discount_factors  # [4]

        # Average over the batch
        total_loss = discounted_loss_per_graph.mean()  # Scalar

        return total_loss


    def validate_step(self, batch):
        """
        Perform one validation step on a batch of graphs.
        """
        self.model.eval()
        epsilon = 1e-8

        # Unpack batch
        if len(batch) == 4:
            batch_initial, batch_target, seq_lengths, settings_tensor = batch
        else:
            batch_initial, batch_target, seq_lengths = batch
            settings_tensor = None

        with torch.no_grad():
            # Forward pass
            predicted_node_features, predicted_log_ratios = self.model_forward(
                initial_graph=batch_initial,
                settings_tensor=settings_tensor,
                batch=batch_initial.batch,
                model_type=self.model_type
            )

            # Compute actual log ratios
            actual_log_ratios = torch.log(torch.abs((batch_target.scale + epsilon) / (batch_initial.scale + epsilon)))  # [B, log_ratio_dim]

            # Ensure actual_log_ratios has the same shape as predicted_log_ratios
            if actual_log_ratios.shape != predicted_log_ratios.shape:
                raise ValueError(f"Shape mismatch between actual_log_ratios {actual_log_ratios.shape} and predicted_log_ratios {predicted_log_ratios.shape}")

            # Compute node-level reconstruction loss
            node_recon_loss_per_node = self.criterion(predicted_node_features, batch_target.x)  # [8000, 6]
            # Mean over node feature dimension to get [8000]
            if node_recon_loss_per_node.dim() > 1:
                node_recon_loss_per_node = node_recon_loss_per_node.mean(dim=1)  # [8000]

            # Debugging: Verify shapes
            # print(f"Validate - node_recon_loss_per_node.shape: {node_recon_loss_per_node.shape}")  # Expected: [8000]
            # print(f"Validate - batch_initial.batch.shape: {batch_initial.batch.shape}")  # Expected: [8000]

            # Compute log ratio loss per graph
            log_ratio_loss = self.criterion(predicted_log_ratios, actual_log_ratios)  # [4, log_ratio_dim]
            # Mean over log_ratio_dim to get [4]
            if log_ratio_loss.dim() > 1:
                log_ratio_loss = log_ratio_loss.mean(dim=1)  # [4]

            # Debugging: Verify shapes
            # print(f"Validate - log_ratio_loss.shape: {log_ratio_loss.shape}")  # Expected: [4]

            # **Important Correction: Apply scatter_mean to node_recon_loss_per_node, not log_ratio_loss**
            # Aggregate node-level losses to per-graph losses
            node_recon_loss_per_graph = scatter_mean(
                node_recon_loss_per_node, 
                batch_initial.batch, 
                dim=0, 
                dim_size=batch_initial.num_graphs  # Ensure correct output size
            )  # [4]

            # **Do not scatter log_ratio_loss**; it's already [4]
            # Combine node and log ratio losses
            loss_per_graph = node_recon_loss_per_graph + self.lambda_ratio * log_ratio_loss  # [4]

            # Apply discount factor per graph
            discount_factors = (self.discount_factor ** (seq_lengths - 1))  # [4]
            discounted_loss_per_graph = loss_per_graph * discount_factors  # [4]

            # Average over the batch
            total_loss = discounted_loss_per_graph.mean()  # Scalar

        return total_loss

    def model_forward(self, initial_graph, settings_tensor, batch, model_type):
        if model_type == 'ScaleAwareLogRatioConditionalGraphNetwork':
            x = initial_graph.x
            edge_index = initial_graph.edge_index
            edge_attr = initial_graph.edge_attr
            scale = initial_graph.scale
            conditions = settings_tensor

            predicted_node_features, predicted_log_ratios = self.model(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                conditions=conditions,
                scale=scale,
                batch=batch
            )
        else:
            raise NotImplementedError(f"Model type '{model_type}' is not supported.")
        
        return predicted_node_features, predicted_log_ratios
