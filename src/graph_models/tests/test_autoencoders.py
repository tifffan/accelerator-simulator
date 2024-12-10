# test_autoencoders.py

import unittest
import torch
import torch.nn as nn
from torch_geometric.data import Data
from src.graph_models.models.graph_autoencoders import (
    GraphConvolutionalAutoEncoder,
    GraphAttentionAutoEncoder,
    GraphTransformerAutoEncoder,
    MeshGraphAutoEncoder,
)

class TestAutoEncoders(unittest.TestCase):
    def setUp(self):
        # Setup common data for testing
        self.num_nodes = 100
        self.num_node_features = 16
        self.num_edge_features = 4
        self.x = torch.randn(self.num_nodes, self.num_node_features)
        self.edge_index = torch.randint(0, self.num_nodes, (2, 200))
        self.batch = torch.zeros(self.num_nodes, dtype=torch.long)
        self.edge_attr = torch.randn(self.edge_index.size(1), self.num_edge_features)
        self.data = Data(x=self.x, edge_index=self.edge_index, edge_attr=self.edge_attr, batch=self.batch)

    def test_graph_convolutional_autoencoder(self):
        model = GraphConvolutionalAutoEncoder(
            in_channels=self.num_node_features,
            hidden_dim=32,
            out_channels=self.num_node_features,
            depth=3,
            pool_ratios=[0.5, 0.5]
        )
        
        # Forward pass
        model.train()
        x_reconstructed = model(self.data.x, self.data.edge_index, self.data.batch)

        # Check output shape
        self.assertEqual(x_reconstructed.shape, self.data.x.shape, "Output shape mismatch")

        # Compute loss
        criterion = nn.MSELoss()
        loss = criterion(x_reconstructed, self.data.x)

        # Backward pass
        loss.backward()

        # print("GraphConvolutionalAutoEncoder unit test passed.")

    def test_graph_attention_autoencoder(self):
        model = GraphAttentionAutoEncoder(
            in_channels=self.num_node_features,
            hidden_dim=32,
            out_channels=self.num_node_features,
            depth=3,
            pool_ratios=[0.5, 0.5],
            heads=2
        )
        
        # Forward pass
        model.train()
        x_reconstructed = model(self.data.x, self.data.edge_index, self.data.batch)

        # Check output shape
        self.assertEqual(x_reconstructed.shape, self.data.x.shape, "Output shape mismatch")

        # Compute loss
        criterion = nn.MSELoss()
        loss = criterion(x_reconstructed, self.data.x)

        # Backward pass
        loss.backward()

        # print("GraphAttentionAutoEncoder unit test passed.")

    def test_graph_transformer_autoencoder(self):
        model = GraphTransformerAutoEncoder(
            in_channels=self.num_node_features,
            hidden_dim=32,
            out_channels=self.num_node_features,
            depth=3,
            pool_ratios=[0.5, 0.5],
            num_heads=2,
            concat=True,
            edge_dim=self.num_edge_features
        )
        
        # Forward pass
        model.train()
        x_reconstructed = model(self.data.x, self.data.edge_index, self.data.edge_attr, self.data.batch)

        # Check output shape
        self.assertEqual(x_reconstructed.shape, self.data.x.shape, "Output shape mismatch")

        # Compute loss
        criterion = nn.MSELoss()
        loss = criterion(x_reconstructed, self.data.x)

        # Backward pass
        loss.backward()

        # print("GraphTransformerAutoEncoder unit test passed.")

    def test_mesh_graph_autoencoder(self):
        model = MeshGraphAutoEncoder(
            node_in_dim=self.num_node_features,
            edge_in_dim=self.num_edge_features,
            node_out_dim=self.num_node_features,
            hidden_dim=32,
            depth=3,
            pool_ratios=[0.5, 0.5, 0.5]
        )
        
        # Forward pass
        model.train()
        x_reconstructed = model(self.data.x, self.data.edge_index, self.data.edge_attr, self.data.batch)

        # Check output shape
        self.assertEqual(x_reconstructed.shape, self.data.x.shape, "Output shape mismatch")

        # Compute loss
        criterion = nn.MSELoss()
        loss = criterion(x_reconstructed, self.data.x)

        # Backward pass
        loss.backward()

        # print("MeshGraphAutoEncoder unit test passed.")

if __name__ == '__main__':
    unittest.main()
