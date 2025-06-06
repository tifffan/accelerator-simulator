# context_graph_networks.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MetaLayer, global_mean_pool
from torch_scatter import scatter_mean

class EdgeModel(nn.Module):
    def __init__(self, edge_in_dim, node_in_dim, edge_out_dim, hidden_dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in_dim + 2 * node_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_out_dim)
        )

    def forward(self, src, dest, edge_attr, u, batch):
        # Ignore u for the basic version
        combined = torch.cat([src, dest, edge_attr], dim=1)
        return self.edge_mlp(combined)

class NodeModel(nn.Module):
    def __init__(self, node_in_dim, edge_out_dim, node_out_dim, hidden_dim):
        super().__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(node_in_dim + edge_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_out_dim)
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        # Ignore u for the basic version
        edge_attr_agg = scatter_mean(edge_attr, edge_index[0], dim=0, dim_size=x.size(0))
        combined = torch.cat([x, edge_attr_agg], dim=1)
        return self.node_mlp(combined)

class EdgeModelWithGlobal(nn.Module):
    def __init__(self, edge_in_dim, node_in_dim, global_in_dim, edge_out_dim, hidden_dim):
        super(EdgeModelWithGlobal, self).__init__()
        self.global_in_dim = global_in_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in_dim + 2 * node_in_dim + global_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_out_dim)
        )

    def forward(self, src, dest, edge_attr, u, edge_batch):
        """
        src: [E, node_in_dim]
        dest: [E, node_in_dim]
        edge_attr: [E, edge_in_dim]
        u: [B, global_in_dim]
        edge_batch: [E] mapping each edge to a graph index (should be long/int)
        """
        if u is not None and self.global_in_dim > 0:
            u_expanded = u[edge_batch]  # [E, global_in_dim]
            combined = torch.cat([src, dest, edge_attr, u_expanded], dim=1)
        else:
            combined = torch.cat([src, dest, edge_attr], dim=1)
        out = self.edge_mlp(combined)
        return out


class NodeModelWithGlobal(nn.Module):
    def __init__(self, node_in_dim, edge_out_dim, global_in_dim, node_out_dim, hidden_dim):
        super(NodeModelWithGlobal, self).__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(node_in_dim + edge_out_dim + global_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_out_dim)
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, node_in_dim]
        # edge_index: [2, E]
        # edge_attr: [E, edge_out_dim]
        # u: [B, global_in_dim]
        # batch: [N] mapping each node to a graph index

        # Expand u to node-level
        u_expanded = u[batch]  # [N, global_in_dim]

        # Aggregate edge attributes to nodes
        # Use edge_index[0] (source nodes) or edge_index[1] (target nodes) depending on the model definition.
        # Here we aggregate edge features based on the source node (edge_index[0]).
        edge_attr_agg = scatter_mean(edge_attr, edge_index[0], dim=0, dim_size=x.size(0))
        # edge_attr_agg: [N, edge_out_dim]

        combined = torch.cat([x, edge_attr_agg, u_expanded], dim=1)
        out = self.node_mlp(combined)
        return out


class ScaleAwareLogRatioConditionalGraphNetwork(nn.Module):
    """Graph Network that incorporates global conditions and scale information to predict log ratios of scaling factors."""
    def __init__(self, node_in_dim, edge_in_dim, cond_in_dim, scale_dim, node_out_dim, hidden_dim, num_layers, log_ratio_dim):
        """
        Initializes the ScaleAwareLogRatioConditionalGraphNetwork.
        
        Args:
            node_in_dim (int): Dimension of input node features.
            edge_in_dim (int): Dimension of input edge features.
            cond_in_dim (int): Dimension of global condition features.
            scale_dim (int): Dimension of scale features.
            node_out_dim (int): Dimension of output node features.
            hidden_dim (int): Dimension of hidden layers.
            num_layers (int): Number of GNN layers.
            log_ratio_dim (int): Dimension of the log ratio output.
        """
        super(ScaleAwareLogRatioConditionalGraphNetwork, self).__init__()
        self.num_layers = num_layers

        # Encoders for input features
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
        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.scale_encoder = nn.Sequential(
            nn.Linear(scale_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.u_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Processor: A sequence of MetaLayers with EdgeModel and NodeModel
        self.processor = nn.ModuleList()
        for _ in range(num_layers):
            edge_model = EdgeModelWithGlobal(
                edge_in_dim=hidden_dim,
                node_in_dim=hidden_dim,
                global_in_dim=hidden_dim,
                edge_out_dim=hidden_dim,
                hidden_dim=hidden_dim
            )
            node_model = NodeModelWithGlobal(
                node_in_dim=hidden_dim,
                edge_out_dim=hidden_dim,
                global_in_dim=hidden_dim,
                node_out_dim=hidden_dim,
                hidden_dim=hidden_dim
            )
            self.processor.append(MetaLayer(edge_model=edge_model, node_model=node_model, global_model=None))

        # Decoder for node features
        self.node_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_out_dim)
        )

        # Head for predicting log ratios of scaling factors
        self.log_ratio_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, log_ratio_dim)
        )

    def forward(self, x, edge_index, edge_attr, conditions, scale, batch):
        """
        Forward pass for the ScaleAwareLogRatioConditionalGraphNetwork.
        Args:
            x (Tensor): Node features [N, node_in_dim]
            edge_index (Tensor): Edge indices [2, E]
            edge_attr (Tensor): Edge features [E, edge_in_dim]
            conditions (Tensor): Global condition features [B, cond_in_dim]
            scale (Tensor): Scale features [B, scale_dim]
            batch (Tensor): Batch indices for nodes [N]
        Returns:
            Tuple[Tensor, Tensor]: (Updated node features [N, node_out_dim], 
                                     Predicted log ratios [B, log_ratio_dim])
        """
        # Encode node, edge, global condition, and scale features
        x = self.node_encoder(x)                 # [N, hidden_dim]
        edge_attr = self.edge_encoder(edge_attr) # [E, hidden_dim]
        u_conditions = self.cond_encoder(conditions) # [B, hidden_dim]
        u_scale = self.scale_encoder(scale)          # [B, hidden_dim]

        # Combine condition and scale encodings
        u_combined = torch.cat([u_conditions, u_scale], dim=1)  # [B, 2*hidden_dim]
        u = self.u_encoder(u_combined)         # [B, hidden_dim]

        # Apply message passing layers
        for layer in self.processor:
            x_res = x  # Residual connection
            row, col = edge_index
            edge_batch = batch[row]
            edge_attr = layer.edge_model(x[row], x[col], edge_attr, u, edge_batch)
            x = layer.node_model(x, edge_index, edge_attr, u, batch)
            x = x + x_res  # Residual connection
            
        # Aggregate node-level features for graph representation
        node_global = global_mean_pool(x, batch)  # [B, hidden_dim]

        # Decode node features
        x = self.node_decoder(x)  # [N, node_out_dim]

        # Aggregate edge-level features for graph representation
        src_nodes = edge_index[0]  # [E]
        edge_batch = batch[src_nodes]  # [E]
        edge_global = global_mean_pool(edge_attr, edge_batch)  # [B, hidden_dim]

        # Combine node and edge global features
        combined_global = torch.cat([node_global, edge_global, u], dim=1)  # [B, hidden_dim + hidden_dim + hidden_dim]
    
        # Predict log ratios
        log_ratios = self.log_ratio_head(combined_global)  # [B, log_ratio_dim]

        return x, log_ratios


class ConditionalGraphNetwork(nn.Module):
    """MeshGraphNet variant that concatenates global condition to each node input."""
    def __init__(self, node_in_dim, edge_in_dim, cond_in_dim, node_out_dim, hidden_dim, num_layers):
        super(ConditionalGraphNetwork, self).__init__()
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
        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.processor = nn.ModuleList()
        for _ in range(num_layers):
            edge_model = EdgeModelWithGlobal(
                edge_in_dim=hidden_dim,
                node_in_dim=hidden_dim,
                global_in_dim=hidden_dim,
                edge_out_dim=hidden_dim,
                hidden_dim=hidden_dim
            )
            node_model = NodeModelWithGlobal(
                node_in_dim=hidden_dim,
                edge_out_dim=hidden_dim,
                global_in_dim=hidden_dim,
                node_out_dim=hidden_dim,
                hidden_dim=hidden_dim
            )
            self.processor.append(MetaLayer(edge_model, node_model, None))
        self.node_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_out_dim)
        )

    def forward(self, x, edge_index, edge_attr, conditions, batch):
        """
        Args:
            x: [N, node_in_dim] node features
            edge_index: [2, E] edge indices
            edge_attr: [E, edge_in_dim] edge features
            conditions: [B, cond_in_dim] global condition features
            batch: [N] batch indices for nodes
        Returns:
            x: [N, node_out_dim] updated node features
        """
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        u = self.cond_encoder(conditions)  # [B, hidden_dim]
        for layer in self.processor:
            x_res = x
            row, col = edge_index
            edge_batch = batch[row]
            edge_attr = layer.edge_model(x[row], x[col], edge_attr, u, edge_batch)
            x = layer.node_model(x, edge_index, edge_attr, u, batch)
            x = x + x_res
        x = self.node_decoder(x)
        return x


class ContextAwareGraphNetworkV0(nn.Module):
    """MeshGraphNet variant that concatenates global condition to each node input."""
    def __init__(self, node_in_dim, edge_in_dim, cond_in_dim, node_out_dim, hidden_dim, num_layers):
        super(ContextAwareGraphNetworkV0, self).__init__()
        self.num_layers = num_layers
        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_dim + cond_in_dim, hidden_dim),
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

    def forward(self, x, edge_index, edge_attr, conditions, batch):
        """
        Args:
            x: [N, node_in_dim] node features
            edge_index: [2, E] edge indices
            edge_attr: [E, edge_in_dim] edge features
            conditions: [B, cond_in_dim] global condition features
            batch: [N] batch indices for nodes
        Returns:
            x: [N, node_out_dim] updated node features
        """
        # Broadcast conditions to nodes
        cond_per_node = conditions[batch]  # [N, cond_in_dim]
        x = torch.cat([x, cond_per_node], dim=1)  # [N, node_in_dim + cond_in_dim]
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        for layer in self.processor:
            x_res = x
            x, edge_attr, _ = layer(x, edge_index, edge_attr, u=None, batch=batch)
            x = x + x_res
        x = self.node_decoder(x)
        return x
