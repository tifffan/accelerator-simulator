# trainer_accelerate.py

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import logging
import os
from accelerate import Accelerator

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
            for batch_idx, data in enumerate(progress_bar):
                # No need to move data to device; accelerator handles it
                self.optimizer.zero_grad()
                loss = self.train_step(data)
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
            for data in self.val_loader:
                loss = self.validate_step(data)
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
        # Pop out values from kwargs to prevent duplication
        criterion = kwargs.pop('criterion', None)
        discount_factor = kwargs.pop('discount_factor', 0.9)
        
        # Initialize the parent class
        super().__init__(**kwargs)
        
        self.criterion = criterion if criterion is not None else torch.nn.MSELoss()
        self.discount_factor = discount_factor
        
        logging.info(f"Using loss function: {self.criterion.__class__.__name__}")
        logging.info(f"Using discount factor: {self.discount_factor}")
        
    def train_step(self, batch):
        """
        Perform one training step with discounted loss.
        """
        # if self.include_settings:
        #     initial_graphs, target_graphs, seq_lengths, settings = batch
        # else: TODO: add settings?
        initial_graphs, target_graphs, seq_lengths = batch

        total_loss = 0
        batch_size = len(initial_graphs)

        for initial_graph, target_graph, seq_length in zip(initial_graphs, target_graphs, seq_lengths):
            # Perform forward pass
            prediction = self.model_forward(initial_graph)

            # Compute loss for the current step
            step_loss = self.criterion(prediction, target_graph.x)

            # Apply discount factor
            # Assuming k = 0 since each target_graph corresponds to a specific step
            # Modify if multiple steps per target_graph are needed
            discounted_loss = (self.discount_factor ** (seq_length)) * step_loss

            # Accumulate loss
            total_loss += discounted_loss / batch_size  # Normalize by batch size

        return total_loss

    def model_forward(self, data):
        """
        Calls the model's forward method based on the identified model type.
        """
        model_type = self.model_type
        if model_type == 'GNN_TopK':
            x_pred, _ = self.model(
                x=data.x,
                edge_index=data.edge_index,
                edge_attr=data.edge_attr,
                pos=data.pos,
                batch=data.batch
            )
        elif model_type == 'TopkMultiscaleGNN':
            x_pred, _ = self.model(
                x=data.x,
                edge_index=data.edge_index,
                pos=data.pos,
                edge_attr=data.edge_attr,
                batch=data.batch
            )
        elif model_type in ['SinglescaleGNN', 'MultiscaleGNN']:
            x_pred = self.model(
                x=data.x,
                edge_index=data.edge_index,
                pos=data.pos,
                edge_attr=data.edge_attr,
                batch=data.batch
            )
        elif model_type in ['MeshGraphNet', 'MeshGraphAutoEncoder']:
            x_pred = self.model(
                x=data.x,
                edge_index=data.edge_index,
                edge_attr=data.edge_attr,
                batch=data.batch
            )
        elif model_type in ['GraphTransformer', 'GraphTransformerAutoEncoder']:
            x_pred = self.model(
                x=data.x,
                edge_index=data.edge_index,
                edge_attr=data.edge_attr if hasattr(data, 'edge_attr') else None,
                batch=data.batch
            )
        else:  # GraphConvolutionNetwork, GraphAttentionNetwork, etc.
            x_pred = self.model(
                x=data.x,
                edge_index=data.edge_index,
                batch=data.batch
            )
        return x_pred

    # def save_checkpoint(self, epoch):
    #     # Save checkpoint as in BaseTrainer
    #     super().save_checkpoint(epoch)
