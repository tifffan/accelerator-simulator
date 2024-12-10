# models/graph_autoencoders_skip_concat.py
# In this version of graph_autoencoders.py, skip connections are implemented as concatenation.

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, TransformerConv, MetaLayer, TopKPooling
from torch_geometric.data import Data


class GraphConvolutionalAutoEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, depth, pool_ratios):
        """
        Initializes the GraphConvolutionalAutoEncoder.

        Args:
            in_channels (int): Number of input node features.
            hidden_dim (int): Number of hidden dimensions.
            out_channels (int): Number of output node features.
            depth (int): Depth of the encoder and decoder (number of layers).
            pool_ratios (list of float): Pooling ratios for each encoder layer.
        """
        super(GraphConvolutionalAutoEncoder, self).__init__()
        self.encoder_convs = nn.ModuleList()
        self.decoder_convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.pool_ratios = pool_ratios

        # Encoder
        self.encoder_convs.append(GCNConv(in_channels, hidden_dim))
        self.activations.append(nn.ReLU())
        for i in range(1, depth):
            self.encoder_convs.append(GCNConv(hidden_dim, hidden_dim))
            self.activations.append(nn.ReLU())

            # Pooling layer
            pool_ratio = pool_ratios[i - 1] if (i - 1) < len(pool_ratios) else 1.0
            if pool_ratio < 1.0:
                self.pools.append(TopKPooling(hidden_dim, ratio=pool_ratio))
            else:
                self.pools.append(None)

        # Decoder
        for i in range(depth - 1):
            input_dim = hidden_dim * 2  # Due to concatenation
            self.decoder_convs.append(GCNConv(input_dim, hidden_dim))
            self.activations.append(nn.ReLU())
        # Final layer
        input_dim = hidden_dim * 2  # Due to concatenation
        self.decoder_convs.append(GCNConv(input_dim, out_channels))
        self.activations.append(None)

    def forward(self, x, edge_index, batch):
        """
        Forward pass of the GraphConvolutionalAutoEncoder.

        Args:
            x (Tensor): Node feature matrix.
            edge_index (LongTensor): Graph connectivity.
            batch (Tensor): Batch vector.

        Returns:
            Tensor: Reconstructed node features.
        """
        x_enc = []
        edge_indices = []
        batches = []
        perms = []

        # Encoder
        for i in range(len(self.encoder_convs)):
            x = self.encoder_convs[i](x, edge_index)
            if self.activations[i] is not None:
                x = self.activations[i](x)
            x_enc.append(x)
            edge_indices.append(edge_index)
            batches.append(batch)

            if i < len(self.pools) and self.pools[i] is not None:
                x, edge_index, _, batch, perm, _ = self.pools[i](x, edge_index, batch=batch)
                perms.append(perm)
            else:
                perms.append(None)

        # Decoder
        for i in range(len(self.decoder_convs)):
            idx = len(self.decoder_convs) - i - 1  # Reverse order

            if perms[idx] is not None:
                # Unpooling
                perm = perms[idx]
                x_unpooled = torch.zeros(x_enc[idx].size(0), x.size(1), device=x.device)
                x_unpooled[perm] = x
                x = x_unpooled
                edge_index = edge_indices[idx]
                batch = batches[idx]
            else:
                # No unpooling needed
                x = x
                edge_index = edge_indices[idx]
                batch = batches[idx]

            # Skip connection using concatenation
            x = torch.cat([x, x_enc[idx]], dim=1)

            x = self.decoder_convs[i](x, edge_index)
            act_idx = len(self.encoder_convs) + i
            if act_idx < len(self.activations) and self.activations[act_idx] is not None:
                x = self.activations[act_idx](x)
        return x


class GraphAttentionAutoEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, depth, pool_ratios, heads=1):
        """
        Initializes the GraphAttentionAutoEncoder.

        Args:
            in_channels (int): Number of input node features.
            hidden_dim (int): Number of hidden dimensions.
            out_channels (int): Number of output node features.
            depth (int): Depth of the encoder and decoder (number of layers).
            pool_ratios (list of float): Pooling ratios for each encoder layer.
            heads (int, optional): Number of attention heads. Defaults to 1.
        """
        super(GraphAttentionAutoEncoder, self).__init__()
        self.encoder_convs = nn.ModuleList()
        self.decoder_convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.heads = heads
        self.pool_ratios = pool_ratios

        # Encoder
        self.encoder_convs.append(GATConv(in_channels, hidden_dim // heads, heads=self.heads, concat=True))
        self.activations.append(nn.ELU())
        hidden_dim_total = hidden_dim  # Due to concatenation

        for i in range(1, depth):
            # Pooling layer
            pool_ratio = pool_ratios[i - 1] if (i - 1) < len(pool_ratios) else 1.0
            if pool_ratio < 1.0:
                self.pools.append(TopKPooling(hidden_dim_total, ratio=pool_ratio))
            else:
                self.pools.append(None)

            # GATConv layer
            self.encoder_convs.append(GATConv(hidden_dim_total, hidden_dim // heads, heads=self.heads, concat=True))
            self.activations.append(nn.ELU())
            hidden_dim_total = hidden_dim  # Due to concatenation

        # Decoder
        for i in range(depth - 1):
            input_dim = hidden_dim_total * 2  # Due to concatenation
            self.decoder_convs.append(GATConv(input_dim, hidden_dim // heads, heads=self.heads, concat=True))
            self.activations.append(nn.ELU())
            hidden_dim_total = hidden_dim  # Due to concatenation

        # Final layer
        input_dim = hidden_dim_total * 2  # Due to concatenation
        self.decoder_convs.append(GATConv(input_dim, out_channels, heads=1, concat=False))
        self.activations.append(None)

    def forward(self, x, edge_index, batch):
        """
        Forward pass of the GraphAttentionAutoEncoder.

        Args:
            x (Tensor): Node feature matrix.
            edge_index (LongTensor): Graph connectivity.
            batch (Tensor): Batch vector.

        Returns:
            Tensor: Reconstructed node features.
        """
        x_enc = []
        edge_indices = []
        batches = []
        perms = []

        # Encoder
        for i in range(len(self.encoder_convs)):
            x = self.encoder_convs[i](x, edge_index)
            if self.activations[i] is not None:
                x = self.activations[i](x)
            x_enc.append(x)
            edge_indices.append(edge_index)
            batches.append(batch)

            if i < len(self.pools) and self.pools[i] is not None:
                x, edge_index, _, batch, perm, _ = self.pools[i](x, edge_index, batch=batch)
                perms.append(perm)
            else:
                perms.append(None)

        # Decoder
        for i in range(len(self.decoder_convs)):
            idx = len(self.decoder_convs) - i - 1  # Reverse order

            if perms[idx] is not None:
                # Unpooling
                perm = perms[idx]
                x_unpooled = torch.zeros(x_enc[idx].size(0), x.size(1), device=x.device)
                x_unpooled[perm] = x
                x = x_unpooled
                edge_index = edge_indices[idx]
                batch = batches[idx]
            else:
                # No unpooling needed
                x = x
                edge_index = edge_indices[idx]
                batch = batches[idx]

            # Skip connection using concatenation
            x = torch.cat([x, x_enc[idx]], dim=1)

            x = self.decoder_convs[i](x, edge_index)
            act_idx = len(self.encoder_convs) + i
            if act_idx < len(self.activations) and self.activations[act_idx] is not None:
                x = self.activations[act_idx](x)
        return x


class GraphTransformerAutoEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_dim,
        out_channels,
        depth,
        pool_ratios,
        num_heads=4,
        concat=True,
        dropout=0.0,
        edge_dim=None,
    ):
        """
        Initializes the GraphTransformerAutoEncoder.

        Args:
            in_channels (int): Number of input node features.
            hidden_dim (int): Number of hidden dimensions.
            out_channels (int): Number of output node features.
            depth (int): Depth of the encoder and decoder (number of layers).
            pool_ratios (list of float): Pooling ratios for each encoder layer.
            num_heads (int, optional): Number of attention heads. Defaults to 4.
            concat (bool, optional): Whether to concatenate heads. Defaults to True.
            dropout (float, optional): Dropout rate. Defaults to 0.0.
            edge_dim (int, optional): Dimension of edge features. Defaults to None.
        """
        super(GraphTransformerAutoEncoder, self).__init__()
        self.encoder_convs = nn.ModuleList()
        self.decoder_convs = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.pool_ratios = pool_ratios
        self.num_heads = num_heads
        self.concat = concat

        # Encoder
        self.encoder_convs.append(
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

        for i in range(1, depth):
            # Pooling layer
            pool_ratio = pool_ratios[i - 1] if (i - 1) < len(pool_ratios) else 1.0
            if pool_ratio < 1.0:
                self.pools.append(TopKPooling(conv_out_dim, ratio=pool_ratio))
            else:
                self.pools.append(None)

            self.encoder_convs.append(
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

        # Decoder
        for i in range(depth - 1):
            input_dim = conv_out_dim * 2  # Due to concatenation
            self.decoder_convs.append(
                TransformerConv(
                    input_dim,
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

        # Final layer
        input_dim = conv_out_dim * 2  # Due to concatenation
        self.decoder_convs.append(
            TransformerConv(
                input_dim,
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
        self.activations.append(None)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        Forward pass of the GraphTransformerAutoEncoder.

        Args:
            x (Tensor): Node feature matrix.
            edge_index (LongTensor): Graph connectivity.
            edge_attr (Tensor, optional): Edge feature matrix. Defaults to None.
            batch (Tensor, optional): Batch vector. Defaults to None.

        Returns:
            Tensor: Reconstructed node features.
        """
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        x_enc = []
        edge_indices = []
        edge_attrs = []
        batches = []
        perms = []

        # Encoder
        for i in range(len(self.encoder_convs)):
            x = self.encoder_convs[i](x, edge_index, edge_attr)
            if self.bns[i] is not None:
                x = self.bns[i](x)
            if self.activations[i] is not None:
                x = self.activations[i](x)
            x_enc.append(x)
            edge_indices.append(edge_index)
            edge_attrs.append(edge_attr)
            batches.append(batch)

            if i < len(self.pools) and self.pools[i] is not None:
                x, edge_index, edge_attr, batch, perm, _ = self.pools[i](
                    x, edge_index, edge_attr=edge_attr, batch=batch
                )
                perms.append(perm)
            else:
                perms.append(None)

        # Decoder
        total_layers = len(self.decoder_convs)
        for i in range(total_layers):
            idx = total_layers - i - 1  # Reverse order
            act_idx = len(self.encoder_convs) + i

            if perms[idx] is not None:
                # Unpooling
                perm = perms[idx]
                x_unpooled = torch.zeros(x_enc[idx].size(0), x.size(1), device=x.device)
                x_unpooled[perm] = x
                x = x_unpooled
                edge_index = edge_indices[idx]
                edge_attr = edge_attrs[idx]
                batch = batches[idx]
            else:
                # No unpooling needed
                x = x
                edge_index = edge_indices[idx]
                edge_attr = edge_attrs[idx]
                batch = batches[idx]

            # Skip connection using concatenation
            x = torch.cat([x, x_enc[idx]], dim=1)

            x = self.decoder_convs[i](x, edge_index, edge_attr)
            if self.bns[act_idx] is not None:
                x = self.bns[act_idx](x)
            if self.activations[act_idx] is not None:
                x = self.activations[act_idx](x)
        return x


# Helper modules
class EdgeModel(nn.Module):
    def __init__(self, edge_in_dim, node_in_dim, edge_out_dim, hidden_dim):
        """
        Initializes the EdgeModel.

        Args:
            edge_in_dim (int): Dimension of input edge features.
            node_in_dim (int): Dimension of input node features.
            edge_out_dim (int): Dimension of output edge features.
            hidden_dim (int): Dimension of hidden layers.
        """
        super(EdgeModel, self).__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in_dim + 2 * node_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_out_dim)
        )

    def forward(self, src, dest, edge_attr, u, batch):
        """
        Forward pass of the EdgeModel.

        Args:
            src (Tensor): Source node features.
            dest (Tensor): Destination node features.
            edge_attr (Tensor): Edge features.
            u (Tensor): Global features (unused).
            batch (Tensor): Batch vector.

        Returns:
            Tensor: Updated edge features.
        """
        # Concatenate source, destination, and edge features
        out = torch.cat([src, dest, edge_attr], dim=1)
        return self.edge_mlp(out)


class NodeModel(nn.Module):
    def __init__(self, node_in_dim, edge_out_dim, node_out_dim, hidden_dim):
        """
        Initializes the NodeModel.

        Args:
            node_in_dim (int): Dimension of input node features.
            edge_out_dim (int): Dimension of input edge features.
            node_out_dim (int): Dimension of output node features.
            hidden_dim (int): Dimension of hidden layers.
        """
        super(NodeModel, self).__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(node_in_dim + edge_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_out_dim)
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        """
        Forward pass of the NodeModel.

        Args:
            x (Tensor): Node features.
            edge_index (LongTensor): Graph connectivity.
            edge_attr (Tensor): Edge features.
            u (Tensor): Global features (unused).
            batch (Tensor): Batch vector.

        Returns:
            Tensor: Updated node features.
        """
        # x: Node features [N, node_in_dim]
        # edge_index: [2, E]
        # edge_attr: Edge features [E, edge_out_dim]
        row, col = edge_index

        # Aggregate edge attributes by summing for each node
        agg = torch.zeros(x.size(0), edge_attr.size(1), device=x.device)
        agg = agg.index_add(0, row, edge_attr)

        # Concatenate node features with aggregated edge attributes
        out = torch.cat([x, agg], dim=1)
        return self.node_mlp(out)


class MeshGraphAutoEncoder(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, node_out_dim, hidden_dim, depth, pool_ratios):
        """
        Initializes the MeshGraphAutoEncoder.

        Args:
            node_in_dim (int): Dimension of input node features.
            edge_in_dim (int): Dimension of input edge features.
            node_out_dim (int): Dimension of output node features.
            hidden_dim (int): Dimension of hidden layers.
            depth (int): Depth of the encoder and decoder (number of layers).
            pool_ratios (list of float): Pooling ratios for each encoder layer.
        """
        super(MeshGraphAutoEncoder, self).__init__()
        self.depth = depth
        self.pool_ratios = pool_ratios

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

        # Encoder
        self.encoder_processor = nn.ModuleList()
        self.pools = nn.ModuleList()
        for i in range(depth):
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
            self.encoder_processor.append(MetaLayer(edge_model, node_model, None))

            # Pooling layer
            pool_ratio = pool_ratios[i] if i < len(pool_ratios) else 1.0
            if pool_ratio < 1.0:
                self.pools.append(TopKPooling(hidden_dim, ratio=pool_ratio))
            else:
                self.pools.append(None)

        # Decoder
        self.decoder_processor = nn.ModuleList()
        for i in range(depth):
            input_dim = hidden_dim * 2  # Due to concatenation
            edge_model = EdgeModel(
                edge_in_dim=hidden_dim,
                node_in_dim=input_dim,
                edge_out_dim=hidden_dim,
                hidden_dim=hidden_dim
            )
            node_model = NodeModel(
                node_in_dim=input_dim,
                edge_out_dim=hidden_dim,
                node_out_dim=hidden_dim,
                hidden_dim=hidden_dim
            )
            self.decoder_processor.append(MetaLayer(edge_model, node_model, None))

        self.node_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_out_dim)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        """
        Forward pass of the MeshGraphAutoEncoder.

        Args:
            x (Tensor): Node feature matrix.
            edge_index (LongTensor): Graph connectivity.
            edge_attr (Tensor): Edge feature matrix.
            batch (Tensor): Batch vector.

        Returns:
            Tensor: Reconstructed node features.
        """
        x_enc = []
        edge_indices = []
        edge_attrs = []
        batches = []
        perms = []

        # Encode node and edge features
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        # Encoder
        for i in range(len(self.encoder_processor)):
            x_res = x
            x, edge_attr, _ = self.encoder_processor[i](x, edge_index, edge_attr, u=None, batch=batch)
            x_enc.append(x)
            edge_indices.append(edge_index)
            edge_attrs.append(edge_attr)
            batches.append(batch)

            if self.pools[i] is not None:
                x, edge_index, edge_attr, batch, perm, _ = self.pools[i](
                    x, edge_index, edge_attr=edge_attr, batch=batch
                )
                perms.append(perm)
            else:
                perms.append(None)

        # Decoder
        total_layers = len(self.decoder_processor)
        for i in range(total_layers):
            idx = total_layers - i - 1  # Reverse order

            if perms[idx] is not None:
                # Unpooling
                perm = perms[idx]
                x_unpooled = torch.zeros(x_enc[idx].size(0), x.size(1), device=x.device)
                x_unpooled[perm] = x
                x = x_unpooled
                edge_index = edge_indices[idx]
                edge_attr = edge_attrs[idx]
                batch = batches[idx]
            else:
                # No unpooling needed
                x = x
                edge_index = edge_indices[idx]
                edge_attr = edge_attrs[idx]
                batch = batches[idx]

            # Skip connection using concatenation
            x = torch.cat([x, x_enc[idx]], dim=1)

            x, edge_attr, _ = self.decoder_processor[i](x, edge_index, edge_attr, u=None, batch=batch)

        # Decode node features
        x = self.node_decoder(x)
        return x
