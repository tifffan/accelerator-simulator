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

# Import model classes and context models as needed.
from src.graph_models.context_models.scale_graph_networks import ScaleAwareLogRatioConditionalGraphNetwork

def identify_model_type(model):
    """
    Identifies the type of the model and returns a string identifier.
    """
    if isinstance(model, ScaleAwareLogRatioConditionalGraphNetwork):
        return 'ScaleAwareLogRatioConditionalGraphNetwork'
    # Add further cases as needed.
    raise ValueError(f"Unrecognized model type: {type(model).__name__}")

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
        """
        Base Trainer for model training and validation.
        """
        self.accelerator = Accelerator()
        self.device = self.accelerator.device

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

        checkpoint_path = kwargs.get('checkpoint')
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

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
                    logging.debug(f"Batch {batch_idx}: Loss = {current_loss:.4e}")

            if self.scheduler:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                if self.verbose:
                    logging.info(f"Epoch {epoch + 1}: Learning rate adjusted to {current_lr}")

            avg_loss = total_loss / len(self.train_loader)
            self.loss_history.append(avg_loss)

            val_loss = self.validate() if self.val_loader is not None else None
            if val_loss is not None:
                self.val_loss_history.append(val_loss)
                logging.info(f"Epoch {epoch + 1}/{self.nepochs}, Loss: {avg_loss:.4e}, Val Loss: {val_loss:.4e}")
            else:
                logging.info(f"Epoch {epoch + 1}/{self.nepochs}, Loss: {avg_loss:.4e}")

            if (epoch + 1) % self.save_checkpoint_every == 0 or (epoch + 1) == self.nepochs:
                self.save_checkpoint(epoch)
            if val_loss is not None and val_loss < self.best_val_loss:
                logging.info(f"New best model found at epoch {epoch + 1} with Val Loss: {val_loss:.4e}")
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
                self.save_checkpoint(epoch, best=True)

        if self.accelerator.is_main_process:
            self.plot_loss_convergence()
            logging.info("Training complete!")
            logging.info(f"Best Val Loss: {self.best_val_loss:.4e} at epoch {self.best_epoch}")

    def validate(self) -> float:
        logging.info("Starting validation...")
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for data in self.val_loader:
                loss = self.validate_step(data)
                total_loss += loss.item()
                num_batches += 1
        avg_val_loss = total_loss / num_batches
        return avg_val_loss

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
        logging.info(f"Loss convergence plot saved to {self.results_folder / 'loss_convergence.png'}")

class SequenceTrainerAccelerate(BaseTrainer):
    def __init__(self, **kwargs):
        # Extract specific training parameters.
        criterion = kwargs.pop('criterion', None)
        discount_factor = kwargs.pop('discount_factor', 1.0)
        lambda_ratio = kwargs.pop('lambda_ratio', 1.0)
        noise_level = kwargs.pop('noise_level', 0.0)
        # 'horizon' here should match max_prediction_horizon from the dataset.
        self.max_horizon = kwargs.pop('horizon', 1)
        super().__init__(**kwargs)
        self.criterion = criterion if criterion is not None else torch.nn.MSELoss(reduction='none')
        self.discount_factor = discount_factor
        self.lambda_ratio = lambda_ratio
        self.noise_level = noise_level
        logging.info(f"Using loss function: {self.criterion.__class__.__name__}")
        logging.info(f"Using discount factor: {self.discount_factor}")
        logging.info(f"Using lambda ratio for loss weighting: {self.lambda_ratio}")
        logging.info(f"Using noise level: {self.noise_level}")
        logging.info(f"Using prediction horizon: {self.max_horizon}")

    def train_step(self, batch):
        self.model.train()
        epsilon = 1e-8
        # Expect batch to be: (batched_input, batched_target_list, seq_lengths, [batched_settings_list])
        if len(batch) == 4:
            batch_initial, batch_targets, seq_lengths, batch_settings = batch
        else:
            batch_initial, batch_targets, seq_lengths = batch
            batch_settings = None

        if self.noise_level > 0:
            batch_initial.x = batch_initial.x + torch.randn_like(batch_initial.x) * self.noise_level

        total_loss = 0.0
        current_graph = batch_initial

        # Iterate over each horizon step.
        for h, target_graph in enumerate(batch_targets):
            if batch_settings is not None:
                settings_tensor = batch_settings[h]
            else:
                settings_tensor = None

            predicted_node_features, predicted_log_ratios = self.model_forward(
                initial_graph=current_graph,
                settings_tensor=settings_tensor,
                batch=current_graph.batch,
                model_type=self.model_type
            )

            # Compute actual log ratios from the scaling factors.
            actual_log_ratios = torch.log(torch.abs((target_graph.scale + epsilon) / (current_graph.scale + epsilon)))
            node_recon_loss_per_node = self.criterion(predicted_node_features, target_graph.x)
            if node_recon_loss_per_node.dim() > 1:
                node_recon_loss_per_node = node_recon_loss_per_node.mean(dim=1)
            log_ratio_loss = self.criterion(predicted_log_ratios, actual_log_ratios)
            if log_ratio_loss.dim() > 1:
                log_ratio_loss = log_ratio_loss.mean(dim=1)

            # Aggregate node loss to per-graph loss.
            node_recon_loss_per_graph = scatter_mean(
                node_recon_loss_per_node,
                current_graph.batch,
                dim=0,
                dim_size=current_graph.num_graphs
            )
            loss_per_graph = node_recon_loss_per_graph + self.lambda_ratio * log_ratio_loss
            discount = self.discount_factor ** h
            loss_h = discount * loss_per_graph.mean()
            total_loss += loss_h

            # Update current graph for next step.
            current_graph = self.update_graph_for_next_step(current_graph, predicted_node_features, predicted_log_ratios)

        return total_loss

    def validate_step(self, batch):
        self.model.eval()
        epsilon = 1e-8
        if len(batch) == 4:
            batch_initial, batch_targets, seq_lengths, batch_settings = batch
        else:
            batch_initial, batch_targets, seq_lengths = batch
            batch_settings = None

        total_loss = 0.0
        current_graph = batch_initial
        for h, target_graph in enumerate(batch_targets):
            if batch_settings is not None:
                settings_tensor = batch_settings[h]
            else:
                settings_tensor = None

            predicted_node_features, predicted_log_ratios = self.model_forward(
                initial_graph=current_graph,
                settings_tensor=settings_tensor,
                batch=current_graph.batch,
                model_type=self.model_type
            )
            actual_log_ratios = torch.log(torch.abs((target_graph.scale + epsilon) / (current_graph.scale + epsilon)))
            node_recon_loss_per_node = self.criterion(predicted_node_features, target_graph.x)
            if node_recon_loss_per_node.dim() > 1:
                node_recon_loss_per_node = node_recon_loss_per_node.mean(dim=1)
            log_ratio_loss = self.criterion(predicted_log_ratios, actual_log_ratios)
            if log_ratio_loss.dim() > 1:
                log_ratio_loss = log_ratio_loss.mean(dim=1)
            node_recon_loss_per_graph = scatter_mean(
                node_recon_loss_per_node,
                current_graph.batch,
                dim=0,
                dim_size=current_graph.num_graphs
            )
            loss_per_graph = node_recon_loss_per_graph + self.lambda_ratio * log_ratio_loss
            discount = self.discount_factor ** h
            loss_h = discount * loss_per_graph.mean()
            total_loss += loss_h
            current_graph = self.update_graph_for_next_step(current_graph, predicted_node_features, predicted_log_ratios)
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

    def update_graph_for_next_step(self, current_graph, predicted_node_features, predicted_log_ratios):
        # Clone the graph and update the node features and scale with the predictions.
        updated_graph = current_graph.clone()
        updated_graph.x = predicted_node_features
        updated_graph.scale = current_graph.scale * torch.exp(predicted_log_ratios)
        return updated_graph
