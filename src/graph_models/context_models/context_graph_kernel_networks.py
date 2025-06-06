#!/usr/bin/env python
"""
conditional_graph_kernel_network.py

This script defines a conditional graph kernel network for learning mappings where the message-passing
kernel is conditioned on global features. The model encodes the node features and global conditions,
then concatenates the edge attributes with the appropriate global condition (broadcast via the node batch mapping)
so that the kernel network (implemented via a DenseNet) computes an edge-specific weight matrix.
This matrix is applied (via NNConv_old) during message passing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.kernel_utils import NNConv_old, DenseNet

class ConditionalGraphKernelNetwork(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, cond_in_dim, scale_dim,
                 hidden_dim, num_layers, ker_width, out_dim=1):
        """
        Args:
            node_in_dim (int): Dimension of raw input node features.
            edge_in_dim (int): Dimension of raw input edge features.
            cond_in_dim (int): Dimension of global condition features.
            scale_dim (int): Dimension of global scale features.
            hidden_dim (int): Hidden dimension for all representations (node and condition embeddings).
            num_layers (int): Number of message-passing layers (kernel convolutions).
            ker_width (int): Hidden dimension for the kernel network.
            out_dim (int): Output dimension (typically 1 for scalar regression).
        """
        super(ConditionalGraphKernelNetwork, self).__init__()
        self.num_layers = num_layers
        
        # Encode node features.
        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Encode the global condition by combining condition and scale features.
        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_in_dim + scale_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # The kernel network maps the concatenated edge features (raw edge features plus condition)
        # into a flattened weight matrix of size [hidden_dim * hidden_dim].
        # Input dimension to the kernel network: edge_in_dim + hidden_dim.
        self.kernel_network = DenseNet([edge_in_dim + hidden_dim, ker_width // 2, ker_width, hidden_dim ** 2],
                                       nn.ReLU)
        # Use NNConv_old which applies an edge-conditioned convolution.
        self.conv = NNConv_old(hidden_dim, hidden_dim, self.kernel_network, aggr='mean')
        
        # Final projection from the hidden space to the output dimension.
        self.fc_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        """
        Args:
            data (torch_geometric.data.Data): A data object that must contain:
                - x: Node features with shape [N, node_in_dim]
                - edge_index: Graph connectivity as a tensor of shape [2, E]
                - edge_attr: Edge features with shape [E, edge_in_dim]
                - conditions: Global condition features with shape [B, cond_in_dim]
                - scale: Global scale features with shape [B, scale_dim]
                - batch: Node-to-graph mapping of shape [N]
        
        Returns:
            Tuple[Tensor, Tensor]: 
                - out: The output for each node [N, out_dim].
                - u: The global condition embedding for each graph [B, hidden_dim].
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Encode node features.
        x = self.node_encoder(x)  # [N, hidden_dim]
        
        # Combine conditions and scale, then encode into a global embedding.
        # data.conditions should have shape [B, cond_in_dim] and data.scale shape [B, scale_dim]
        cond_scale = torch.cat([data.conditions, data.scale], dim=1)  # [B, cond_in_dim + scale_dim]
        u = self.cond_encoder(cond_scale)  # [B, hidden_dim]
        
        # For each edge, retrieve the corresponding global condition from the source node's graph.
        u_edge = u[batch[edge_index[0]]]  # [E, hidden_dim]
        # Augment the raw edge attributes with the condition.
        edge_attr_cond = torch.cat([edge_attr, u_edge], dim=1)  # [E, edge_in_dim + hidden_dim]
        
        # Apply multiple rounds of edge-conditioned (kernel) convolution.
        for _ in range(self.num_layers):
            x = F.relu(self.conv(x, edge_index, edge_attr_cond))
        
        # Project the node features to the desired output.
        out = self.fc_out(x)
        return out, u

# Example testing code (this block can be removed or modified in your integration)
if __name__ == '__main__':
    from torch_geometric.data import Data

    # Suppose we have a graph with 10 nodes (node_in_dim=3), 4-dimensional edge features, and a single condition and scale per graph.
    num_nodes = 10
    node_in_dim = 3
    edge_in_dim = 4
    cond_in_dim = 2
    scale_dim = 2
    hidden_dim = 64
    num_layers = 3
    ker_width = 128
    out_dim = 1

    # Create dummy node features.
    x = torch.randn(num_nodes, node_in_dim)
    # A dummy edge_index tensor (for example, 5 edges).
    edge_index = torch.tensor([[0, 1, 2, 3, 4],
                               [1, 2, 3, 4, 5]], dtype=torch.long)
    edge_attr = torch.randn(edge_index.size(1), edge_in_dim)
    # Create a dummy batch vector (assume all nodes belong to graph 0).
    batch = torch.zeros(num_nodes, dtype=torch.long)
    # Dummy global conditions and scale for one graph.
    conditions = torch.randn(1, cond_in_dim)
    scale = torch.randn(1, scale_dim)

    # Build a dummy data object.
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch,
                conditions=conditions, scale=scale)

    # Instantiate and test the model.
    model = ConditionalGraphKernelNetwork(node_in_dim, edge_in_dim, cond_in_dim, scale_dim,
                                            hidden_dim, num_layers, ker_width, out_dim)
    out, u = model(data)
    print("Output node features:", out)
    print("Global condition embedding:", u)
