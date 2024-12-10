import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.utils import scatter
from torch_geometric.nn import MetaLayer

class EdgeModel(nn.Module):
    """Defines the edge update function in the graph network."""
    def __init__(self, edge_in_dim, node_in_dim, global_in_dim, edge_out_dim, hidden_dim):
        super(EdgeModel, self).__init__()
        self.edge_mlp = Seq(
            Lin(edge_in_dim + 2 * node_in_dim + global_in_dim, hidden_dim),
            ReLU(),
            Lin(hidden_dim, edge_out_dim)
        )

    def forward(self, src, dest, edge_attr, u, batch):
        """
        Forward pass for the edge model.
        Args:
            src: Source node features [E, node_in_dim]
            dest: Destination node features [E, node_in_dim]
            edge_attr: Edge features [E, edge_in_dim]
            u: Global features [B, global_in_dim]
            batch: Batch indices for edges [E]
        Returns:
            Updated edge features [E, edge_out_dim].
        """
        out = torch.cat([src, dest, edge_attr, u[batch]], dim=1)
        return self.edge_mlp(out)

class NodeModel(nn.Module):
    """Defines the node update function in the graph network."""
    def __init__(self, node_in_dim, edge_out_dim, global_in_dim, node_out_dim, hidden_dim):
        super(NodeModel, self).__init__()
        self.node_mlp_1 = Seq(
            Lin(node_in_dim + edge_out_dim, hidden_dim),
            ReLU(),
            Lin(hidden_dim, hidden_dim)
        )
        self.node_mlp_2 = Seq(
            Lin(node_in_dim + hidden_dim + global_in_dim, hidden_dim),
            ReLU(),
            Lin(hidden_dim, node_out_dim)
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        """
        Forward pass for the node model.
        Args:
            x: Node features [N, node_in_dim]
            edge_index: Edge index tensor [2, E]
            edge_attr: Edge features [E, edge_out_dim]
            u: Global features [B, global_in_dim]
            batch: Batch indices for nodes [N]
        Returns:
            Updated node features [N, node_out_dim].
        """
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        agg = scatter(out, col, dim=0, dim_size=x.size(0), reduce='mean')
        out = torch.cat([x, agg, u[batch]], dim=1)
        return self.node_mlp_2(out)

class GlobalModel(nn.Module):
    """Defines the global update function in the graph network."""
    def __init__(self, global_in_dim, node_in_dim, global_out_dim, hidden_dim):
        super(GlobalModel, self).__init__()
        self.global_mlp = Seq(
            Lin(global_in_dim + node_in_dim, hidden_dim),
            ReLU(),
            Lin(hidden_dim, global_out_dim)
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        """
        Forward pass for the global model.
        Args:
            x: Node features [N, node_in_dim]
            edge_index: Edge index tensor [2, E]
            edge_attr: Edge features [E, edge_in_dim]
            u: Global features [B, global_in_dim]
            batch: Batch indices for nodes [N]
        Returns:
            Updated global features [B, global_out_dim].
        """
        # Aggregate node features for each graph in the batch
        agg = scatter(x, batch, dim=0, reduce='mean')
        out = torch.cat([u, agg], dim=1)
        return self.global_mlp(out)

class GeneralGraphNetwork(nn.Module):
    """Graph Network that updates node, edge, and global features simultaneously."""
    def __init__(self, node_in_dim, edge_in_dim, global_in_dim, node_out_dim, edge_out_dim, global_out_dim, hidden_dim, num_layers):
        super(GeneralGraphNetwork, self).__init__()
        self.num_layers = num_layers

        # Initial encoders
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
        self.global_encoder = nn.Sequential(
            nn.Linear(global_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Processor: Sequence of MetaLayers
        self.processors = nn.ModuleList()
        for _ in range(num_layers):
            edge_model = EdgeModel(
                edge_in_dim=hidden_dim,
                node_in_dim=hidden_dim,
                global_in_dim=hidden_dim,
                edge_out_dim=hidden_dim,
                hidden_dim=hidden_dim
            )
            node_model = NodeModel(
                node_in_dim=hidden_dim,
                edge_out_dim=hidden_dim,
                global_in_dim=hidden_dim,
                node_out_dim=hidden_dim,
                hidden_dim=hidden_dim
            )
            # global_model = GlobalModel(
            #     global_in_dim=hidden_dim,
            #     node_in_dim=hidden_dim,
            #     global_out_dim=hidden_dim,
            #     hidden_dim=hidden_dim
            # )
            self.processors.append(MetaLayer(edge_model, node_model, global_model=None))
            # self.processors.append(MetaLayer(edge_model, node_model, global_model))

        # Decoders
        self.node_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_out_dim)
        )
        # self.edge_decoder = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, edge_out_dim)
        # )
        # self.global_decoder = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, global_out_dim)
        # )

    def forward(self, x, edge_index, edge_attr, u, batch):
        """
        Forward pass for the graph network.
        Args:
            x: Node features [N, node_in_dim]
            edge_index: Edge index tensor [2, E]
            edge_attr: Edge features [E, edge_in_dim]
            u: Global features [B, global_in_dim]
            batch: Batch indices for nodes [N]
        Returns:
            Updated node features [N, node_out_dim], edge features [E, edge_out_dim], and global features [B, global_out_dim].
        """
        # Encode inputs
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        u = self.global_encoder(u)

        # Apply processors
        for processor in self.processors:
            
            x_res, edge_attr_res, = x, edge_attr  # Residual connections
            x, edge_attr, u = processor(x, edge_index, edge_attr, u=u, batch=batch)
            x = x + x_res
            edge_attr = edge_attr + edge_attr_res
            
            # x_res, edge_attr_res, u_res = x, edge_attr, u  # Residual connections
            # x, edge_attr, u = processor(x, edge_index, edge_attr, u=u, batch=batch)
            # x = x + x_res
            # edge_attr = edge_attr + edge_attr_res
            # u = u + u_res

        # Decode outputs
        x = self.node_decoder(x)
        # edge_attr = self.edge_decoder(edge_attr)
        # u = self.global_decoder(u)

        return x #, edge_attr, u

class ConditionalGraphNetwork(nn.Module):
    """Graph Network that incorporates global conditions via concatenation."""
    def __init__(self, node_in_dim, edge_in_dim, cond_in_dim, node_out_dim, hidden_dim, num_layers):
        super(ConditionalGraphNetwork, self).__init__()
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

        # Processor: A sequence of MetaLayers with EdgeModel and NodeModel
        self.processor = nn.ModuleList()
        for _ in range(num_layers):
            edge_model = EdgeModel(
                edge_in_dim=hidden_dim,
                node_in_dim=hidden_dim,
                global_in_dim=hidden_dim,
                edge_out_dim=hidden_dim,
                hidden_dim=hidden_dim
            )
            node_model = NodeModel(
                node_in_dim=hidden_dim,
                edge_out_dim=hidden_dim,
                global_in_dim=hidden_dim,
                node_out_dim=hidden_dim,
                hidden_dim=hidden_dim
            )
            self.processor.append(MetaLayer(edge_model, node_model, None))

        # Decoder for node features
        self.node_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_out_dim)
        )

    def forward(self, x, edge_index, edge_attr, conditions, batch):
        """
        Forward pass for the conditional graph network.
        Args:
            x: Node features [N, node_in_dim]
            edge_index: Edge index tensor [2, E]
            edge_attr: Edge features [E, edge_in_dim]
            conditions: Global conditions [B, cond_in_dim]
            batch: Batch indices for nodes [N]
        Returns:
            Updated node features [N, node_out_dim].
        """
        # Encode node, edge, and global features
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        u = self.cond_encoder(conditions)
        
        # Apply message passing layers
        for layer in self.processor:
            x_res = x  # Residual connection
            x, edge_attr, _ = layer(x, edge_index, edge_attr, u=u, batch=batch)
            x = x + x_res  # Residual connection

        # Decode node features
        x = self.node_decoder(x)
        return x

class AttentionNodeModel(nn.Module):
    """Node update function with attention between node features and global conditions."""
    def __init__(self, node_in_dim, edge_out_dim, global_in_dim, node_out_dim, hidden_dim, attention_layer):
        super(AttentionNodeModel, self).__init__()
        self.attention_layer = attention_layer
        self.node_mlp_1 = Seq(
            Lin(node_in_dim + edge_out_dim, hidden_dim),
            ReLU(),
            Lin(hidden_dim, hidden_dim)
        )
        self.node_mlp_2 = nn.Sequential(
            nn.Linear(node_in_dim + hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_out_dim)
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        """
        Forward pass for the node model with attention.
        Args:
            x: Node features [N, node_in_dim]
            edge_index: Edge index tensor [2, E]
            edge_attr: Edge features [E, edge_out_dim]
            u: Global condition vector [B, hidden_dim]
            batch: Batch indices for nodes [N]
        Returns:
            Updated node features [N, node_out_dim].
        """
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        agg = scatter(out, col, dim=0, dim_size=x.size(0), reduce='mean')

        # Attention mechanism
        u_expanded = u[batch]  # Expand global conditions to match batch size
        x_with_cond, _ = self.attention_layer(
            x.unsqueeze(1), u_expanded.unsqueeze(1), u_expanded.unsqueeze(1)
        )
        x_with_cond = x_with_cond.squeeze(1)  # Remove extra dimension

        # Combine aggregated edge features, node features, and attention-modulated conditions
        out = torch.cat([x, agg, x_with_cond], dim=1)
        return self.node_mlp_2(out)

class AttentionConditionalGraphNetwork(nn.Module):
    """Graph Network with attention between node features and global conditions."""
    def __init__(self, node_in_dim, edge_in_dim, cond_in_dim, node_out_dim, hidden_dim, num_layers):
        super(AttentionConditionalGraphNetwork, self).__init__()
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

        # Attention mechanism
        self.attention_layer = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, batch_first=True
        )

        # Processor: A sequence of MetaLayers with EdgeModel and AttentionNodeModel
        self.processor = nn.ModuleList()
        for _ in range(num_layers):
            edge_model = EdgeModel(
                edge_in_dim=hidden_dim,
                node_in_dim=hidden_dim,
                global_in_dim=hidden_dim,
                edge_out_dim=hidden_dim,
                hidden_dim=hidden_dim
            )
            node_model = AttentionNodeModel(
                node_in_dim=hidden_dim,
                edge_out_dim=hidden_dim,
                global_in_dim=hidden_dim,
                node_out_dim=hidden_dim,
                hidden_dim=hidden_dim,
                attention_layer=self.attention_layer
            )
            self.processor.append(MetaLayer(edge_model, node_model, None))

        # Decoder for node features
        self.node_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_out_dim)
        )

    def forward(self, x, edge_index, edge_attr, conditions, batch):
        """
        Forward pass for the attention-based conditional graph network.
        Args:
            x: Node features [N, node_in_dim]
            edge_index: Edge index tensor [2, E]
            edge_attr: Edge features [E, edge_in_dim]
            conditions: Global conditions [B, cond_in_dim]
            batch: Batch indices for nodes [N]
        Returns:
            Updated node features [N, node_out_dim].
        """
        # Encode node, edge, and global features
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        u = self.cond_encoder(conditions)

        # Apply message passing layers
        for layer in self.processor:
            x_res = x  # Residual connection
            x, edge_attr, _ = layer(x, edge_index, edge_attr, u=u, batch=batch)
            x = x + x_res  # Residual connection

        # Decode node features
        x = self.node_decoder(x)
        return x
