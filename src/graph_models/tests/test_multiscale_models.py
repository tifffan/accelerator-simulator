# test_multiscale_models.py

import unittest
import torch
from src.graph_models.models.multiscale.gnn import MultiscaleGNN, TopkMultiscaleGNN, SinglescaleGNN

class TestMultiscaleModels(unittest.TestCase):
    def setUp(self):
        self.num_nodes = 100
        self.in_channels = 10
        self.hidden_dim = 16
        self.out_channels = 5
        self.n_mlp_hidden_layers = 2  # Number of hidden layers in MLPs
        self.n_mmp_layers = 2  # Number of MMP layers
        self.n_messagePassing_layers = 2  # Number of message passing layers within MMP
        self.max_level = 1  # Max coarsening level

        self.max_level_mmp = 1
        self.max_level_topk = 1
        self.rf_topk = 2  # Reduction factor for TopK pooling

        self.edge_index = torch.randint(0, self.num_nodes, (2, 300))
        self.edge_attr_dim = 4  # Arbitrary edge attribute dimension
        self.edge_attr = torch.randn(self.edge_index.size(1), self.edge_attr_dim)
        self.x = torch.randn(self.num_nodes, self.in_channels)
        self.pos = torch.randn(self.num_nodes, 2)  # 2D positions
        self.batch = torch.zeros(self.num_nodes, dtype=torch.long)

        # Compute l_char (characteristic length scale)
        edge_lengths = torch.norm(
            self.pos[self.edge_index[0]] - self.pos[self.edge_index[1]], dim=1
        )
        self.l_char = edge_lengths.mean().item()

    def test_multiscale_gnn(self):
        model = MultiscaleGNN(
            input_node_channels=self.in_channels,
            input_edge_channels=self.edge_attr_dim,
            hidden_channels=self.hidden_dim,
            output_node_channels=self.out_channels,
            n_mlp_hidden_layers=self.n_mlp_hidden_layers,
            n_mmp_layers=self.n_mmp_layers,
            n_messagePassing_layers=self.n_messagePassing_layers,
            max_level=self.max_level,
            l_char=self.l_char,
            name='multiscale_gnn_test'
        )
        output = model(
            x=self.x,
            edge_index=self.edge_index,
            pos=self.pos,
            edge_attr=self.edge_attr,
            batch=self.batch
        )
        self.assertEqual(output.shape, (self.num_nodes, self.out_channels))
        # print("MultiscaleGNN unit test passed.")

    def test_topk_multiscale_gnn(self):
        model = TopkMultiscaleGNN(
            input_node_channels=self.in_channels,
            input_edge_channels=self.edge_attr_dim,
            hidden_channels=self.hidden_dim,
            output_node_channels=self.out_channels,
            n_mlp_hidden_layers=self.n_mlp_hidden_layers,
            n_mmp_layers=self.n_mmp_layers,
            n_messagePassing_layers=self.n_messagePassing_layers,
            max_level_mmp=self.max_level_mmp,
            l_char=self.l_char,
            max_level_topk=self.max_level_topk,
            rf_topk=self.rf_topk,
            name='topk_multiscale_gnn_test'
        )
        output, mask = model(
            x=self.x,
            edge_index=self.edge_index,
            pos=self.pos,
            edge_attr=self.edge_attr,
            batch=self.batch
        )
        self.assertEqual(output.shape, (self.num_nodes, self.out_channels))
        self.assertEqual(mask.shape, (self.num_nodes,))
        # print("TopkMultiscaleGNN unit test passed.")
        
    def test_singlescale_gnn(self):
        model = SinglescaleGNN(
            input_node_channels=self.in_channels,
            input_edge_channels=self.edge_attr_dim,
            hidden_channels=self.hidden_dim,
            output_node_channels=self.out_channels,
            n_mlp_hidden_layers=self.n_mlp_hidden_layers,
            n_messagePassing_layers=self.n_messagePassing_layers,
            name='singlescale_gnn_test'
        )
        output = model(
            x=self.x,
            edge_index=self.edge_index,
            pos=None,  # pos is not used in SinglescaleGNN
            edge_attr=self.edge_attr,
            batch=self.batch
        )
        self.assertEqual(output.shape, (self.num_nodes, self.out_channels))
        # print("SinglescaleGNN unit test passed.")

if __name__ == '__main__':
    unittest.main()
