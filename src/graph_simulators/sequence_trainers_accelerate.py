# sequence_trainer_accelerate.py

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import logging
import os

from src.graph_models.trainers_accelerate import BaseTrainer

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
