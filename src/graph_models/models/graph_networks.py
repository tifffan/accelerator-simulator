# models/graph_networks.py

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, TransformerConv, TopKPooling, MessagePassing, MetaLayer, global_mean_pool
from torch_geometric.utils import add_self_loops
from torch_geometric.typing import Adj, OptTensor

class GraphConvolutionNetwork(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, num_layers, pool_ratios):
        super(GraphConvolutionNetwork, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.activations = nn.ModuleList()
        
        # First convolution layer
        self.convs.append(GCNConv(in_channels, hidden_dim))
        self.activations.append(nn.ReLU())
        
        # Intermediate layers
        for i in range(num_layers - 2):
            # Pooling layer
            pool_ratio = pool_ratios[i] if i < len(pool_ratios) else 1.0
            if pool_ratio < 1.0:
                self.pools.append(TopKPooling(hidden_dim, ratio=pool_ratio))
            else:
                self.pools.append(None)
            # Convolution layer
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.activations.append(nn.ReLU())
        
        # Final convolution layer
        self.convs.append(GCNConv(hidden_dim, out_channels))
        self.activations.append(None)  # No activation after the final layer

    def forward(self, x, edge_index, batch):
        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)
            if self.activations[i] is not None:
                x = self.activations[i](x)
            if i < len(self.pools) and self.pools[i] is not None:
                x, edge_index, _, batch, _, _ = self.pools[i](x, edge_index, batch=batch)
        return x

class GraphAttentionNetwork(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, num_layers, pool_ratios, heads=1):
        super(GraphAttentionNetwork, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.heads = heads

        # First GAT layer
        self.convs.append(GATConv(in_channels, hidden_dim, heads=self.heads, concat=True))
        self.activations.append(nn.ELU())
        hidden_dim = hidden_dim * self.heads  # Adjust hidden_dim after concatenation

        # Intermediate GAT and pooling layers
        for i in range(num_layers - 2):
            pool_ratio = pool_ratios[i] if i < len(pool_ratios) else 1.0
            if pool_ratio < 1.0:
                self.pools.append(TopKPooling(hidden_dim, ratio=pool_ratio))
            else:
                self.pools.append(None)

            self.convs.append(GATConv(hidden_dim, hidden_dim // self.heads, heads=self.heads, concat=True))
            self.activations.append(nn.ELU())
            hidden_dim = hidden_dim  # No change in hidden_dim due to concat=True

        # Final GAT layer
        self.convs.append(GATConv(hidden_dim, out_channels, heads=1, concat=False))
        self.activations.append(None)  # No activation after the final layer

    def forward(self, x, edge_index, batch):
        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)
            if self.activations[i] is not None:
                x = self.activations[i](x)
            if i < len(self.pools) and self.pools[i] is not None:
                x, edge_index, _, batch, _, _ = self.pools[i](x, edge_index, batch=batch)
        return x

class GraphTransformer(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_dim,
        out_channels,
        num_layers,
        pool_ratios,
        num_heads=4,
        concat=True,
        dropout=0.0,
        edge_dim=None,
    ):
        super(GraphTransformer, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.num_heads = num_heads
        self.concat = concat

        # Initial TransformerConv layer
        self.convs.append(
            TransformerConv(
                in_channels,
                hidden_dim,
                heads=self.num_heads,
                concat=self.concat,
                dropout=dropout,
                edge_dim=edge_dim,
                beta=True,
                bias=True,
            )
        )
        conv_out_dim = hidden_dim * num_heads if concat else hidden_dim
        self.bns.append(nn.BatchNorm1d(conv_out_dim))
        self.activations.append(nn.ReLU())

        # Hidden layers with pooling
        for i in range(num_layers - 2):
            pool_ratio = pool_ratios[i] if i < len(pool_ratios) else 1.0
            if pool_ratio < 1.0:
                self.pools.append(TopKPooling(conv_out_dim, ratio=pool_ratio))
            else:
                self.pools.append(None)

            self.convs.append(
                TransformerConv(
                    conv_out_dim,
                    hidden_dim,
                    heads=self.num_heads,
                    concat=self.concat,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    beta=True,
                    bias=True,
                )
            )
            conv_out_dim = hidden_dim * num_heads if concat else hidden_dim
            self.bns.append(nn.BatchNorm1d(conv_out_dim))
            self.activations.append(nn.ReLU())

        # Final TransformerConv layer
        self.convs.append(
            TransformerConv(
                conv_out_dim,
                out_channels,
                heads=1,
                concat=False,
                dropout=dropout,
                edge_dim=edge_dim,
                beta=True,
                bias=True,
            )
        )
        self.bns.append(None)
        self.activations.append(None)  # No activation after the final layer

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index, edge_attr)
            if self.bns[i] is not None:
                x = self.bns[i](x)
            if self.activations[i] is not None:
                x = self.activations[i](x)

            if i < len(self.pools) and self.pools[i] is not None:
                x, edge_index, edge_attr, batch, _, _ = self.pools[i](
                    x, edge_index, edge_attr=edge_attr, batch=batch
                )
        return x

# Helper modules for MeshGraphNet
class EdgeModel(nn.Module):
    def __init__(self, edge_in_dim, node_in_dim, edge_out_dim, hidden_dim):
        super(EdgeModel, self).__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in_dim + 2 * node_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_out_dim)
        )

    def forward(self, src, dest, edge_attr, u, batch):
        # src, dest: Node features [E, node_in_dim]
        # edge_attr: Edge features [E, edge_in_dim]
        out = torch.cat([src, dest, edge_attr], dim=1)
        return self.edge_mlp(out)

class NodeModel(nn.Module):
    def __init__(self, node_in_dim, edge_out_dim, node_out_dim, hidden_dim):
        super(NodeModel, self).__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(node_in_dim + edge_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_out_dim)
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: Node features [N, node_in_dim]
        # edge_index: [2, E]
        # edge_attr: Edge features [E, edge_out_dim]
        row, col = edge_index
        agg = torch.zeros_like(x)
        agg = agg.index_add(0, row, edge_attr)
        out = torch.cat([x, agg], dim=1)
        return self.node_mlp(out)

class MeshGraphNet(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, node_out_dim, hidden_dim, num_layers):
        super(MeshGraphNet, self).__init__()
        self.num_layers = num_layers

        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.processor = nn.ModuleList()
        for _ in range(num_layers):
            edge_model = EdgeModel(
                edge_in_dim=hidden_dim,
                node_in_dim=hidden_dim,
                edge_out_dim=hidden_dim,
                hidden_dim=hidden_dim
            )
            node_model = NodeModel(
                node_in_dim=hidden_dim,
                edge_out_dim=hidden_dim,
                node_out_dim=hidden_dim,
                hidden_dim=hidden_dim
            )
            self.processor.append(MetaLayer(edge_model, node_model, None))

        self.node_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_out_dim)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        # Encode node and edge features
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        # Message passing
        for layer in self.processor:
            x_res = x
            x, edge_attr, _ = layer(x, edge_index, edge_attr, u=None, batch=batch)
            x = x + x_res  # Residual connection

        # Decode node features
        x = self.node_decoder(x)
        return x
