# test_networks.py

import unittest
import torch
import torch_geometric
from src.graph_models.models.graph_networks import GraphConvolutionNetwork, GraphAttentionNetwork, GraphTransformer, MeshGraphNet

class TestModels(unittest.TestCase):
    def setUp(self):
        self.num_nodes = 100
        self.in_channels = 10
        self.hidden_dim = 16
        self.out_channels = 5
        self.num_layers = 3
        self.pool_ratios = [1.0] * (self.num_layers - 2)  # No pooling during tests
        self.edge_index = torch.randint(0, self.num_nodes, (2, 200))
        self.batch = torch.zeros(self.num_nodes, dtype=torch.long)
        self.x = torch.randn(self.num_nodes, self.in_channels)
        self.edge_attr = torch.randn(self.edge_index.size(1), 4)

    def test_graph_convolution_network(self):
        model = GraphConvolutionNetwork(
            in_channels=self.in_channels,
            hidden_dim=self.hidden_dim,
            out_channels=self.out_channels,
            num_layers=self.num_layers,
            pool_ratios=self.pool_ratios
        )
        output = model(self.x, self.edge_index, self.batch)
        self.assertEqual(output.shape, (self.num_nodes, self.out_channels))
        
        # print("GraphConvolutionNetwork unit test passed.")

    def test_graph_attention_network(self):
        model = GraphAttentionNetwork(
            in_channels=self.in_channels,
            hidden_dim=self.hidden_dim,
            out_channels=self.out_channels,
            num_layers=self.num_layers,
            pool_ratios=self.pool_ratios,
            heads=2
        )
        output = model(self.x, self.edge_index, self.batch)
        self.assertEqual(output.shape, (self.num_nodes, self.out_channels))
        
        # print("GraphAttentionNetwork unit test passed.")

    def test_graph_transformer(self):
        model = GraphTransformer(
            in_channels=self.in_channels,
            hidden_dim=self.hidden_dim,
            out_channels=self.out_channels,
            num_layers=self.num_layers,
            pool_ratios=self.pool_ratios,
            num_heads=2,
            concat=True,
            dropout=0.1,
            edge_dim=4
        )
        output = model(self.x, self.edge_index, self.edge_attr, self.batch)
        self.assertEqual(output.shape, (self.num_nodes, self.out_channels))
        
        # print("GraphTransformer unit test passed.")

    def test_mesh_graph_net(self):
        model = MeshGraphNet(
            node_in_dim=self.in_channels,
            edge_in_dim=4,
            node_out_dim=self.out_channels,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers
        )
        output = model(self.x, self.edge_index, self.edge_attr, self.batch)
        self.assertEqual(output.shape, (self.num_nodes, self.out_channels))
        
        # print("MeshGraphNet unit test passed.")

if __name__ == '__main__':
    unittest.main()
