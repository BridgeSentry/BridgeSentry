import torch
import torch.nn as nn
from torch_geometric.nn import MLP

from torch_geometric.data import HeteroData
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

from dataset_generator.model.sehgnn_conv import SeHGNN
from dataset_generator.types import EdgeMetapath

class BridgeDefender(nn.Module):
    def __init__(
            self,
            metapath_feature_sizes: dict[EdgeMetapath, int],
            node_types: list[str],
            first_layer_channels: int,
            hidden_channels: int,
            out_channels: int,
            dropout: float = 0.5,
            input_drop: float = 0.0,
            att_drop: float = 0.0,
            n_fp_layers: int = 2,
            n_mlp_layers: int = 2,
            act: str = 'relu',
            residual: bool = False,
            pooling: str = 'mean',
            rm_semantic_fusion: bool = False,
    ):
        super(BridgeDefender, self).__init__()
        self.convs = nn.ModuleDict({})
        self.pooling = pooling
        self.hidden_channels = hidden_channels
        # mean_max pooling concatenates mean- and max-pooled embeddings, doubling the per-type width.
        self.type_emb_channels = hidden_channels * 2 if pooling == "mean_max" else hidden_channels

        # Use SeHGNNConv to generate node embeddings for each target node type, based on the pre-computed meta-path features.
        non_null_node_types = len(node_types)
        for node_type in node_types:
            node_metapath_sizes = {mp: size for mp, size in metapath_feature_sizes.items() if mp[0][0] == node_type}
            if len(node_metapath_sizes) == 0:
                non_null_node_types -= 1
                continue  # Skip node types that don't have any associated meta-paths

            self.convs[node_type] = SeHGNN(
                metapath_data_size=node_metapath_sizes,
                num_features=first_layer_channels,
                hidden_dim=hidden_channels,
                target_type=node_type,
                dropout=dropout,
                input_drop=input_drop,
                att_drop=att_drop,
                n_fp_layers=n_fp_layers,
                act=act,
                residual=residual,
                rm_semantic_fusion=rm_semantic_fusion
            )
            # For each target node type, we will also perform pooling on the node embeddings to get a
            # type-level embedding.

        # After obtaining type-level embeddings, we will concatenate them and pass through an MLP for final classification.
        self.classifier = MLP(
            in_channels=non_null_node_types * self.type_emb_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=n_mlp_layers,
            act=act,
            dropout=dropout,
            norm="layer_norm"
        )


    def forward(
            self,
            data: HeteroData,
            metapath_aggregations: dict[EdgeMetapath, torch.Tensor]
    ) -> torch.Tensor:
        # Determine number of graphs in the current batch.
        # For PyG Batch objects, `num_graphs` should be present
        batch_size = getattr(data, 'num_graphs')

        type_embeddings = []
        for node_type, conv in self.convs.items():
            # pick metapath features relevant to this node type (keys expected to be strings)
            mp_feats = metapath_aggregations

            # run the SeHGNN conv for this target node type; handle dict or tensor returns
            x = conv(data, mp_feats)
            if isinstance(x, dict):
                x = x.get(node_type, None)
            if x is None or (isinstance(x, torch.Tensor) and x.numel() == 0):
                # If this node type has no realized nodes in the current batch,
                # still append a zero embedding so concatenation stays consistent.
                device = next(self.parameters()).device
                dtype = next(self.parameters()).dtype
                type_embeddings.append(torch.zeros((batch_size, self.type_emb_channels), device=device, dtype=dtype))
                continue

            # pooling per node type
            pooling = self.pooling
            if pooling == "mean":
                pooled = global_mean_pool(x, data[node_type].batch)
            elif pooling == "max":
                pooled = global_max_pool(x, data[node_type].batch)
            elif pooling == "mean_max":
                mean_pooled = global_mean_pool(x, data[node_type].batch)
                max_pooled = global_max_pool(x, data[node_type].batch)
                pooled = torch.cat([mean_pooled, max_pooled], dim=-1)
            elif pooling == "sum":
                pooled = global_add_pool(x, data[node_type].batch)
            elif pooling == "none":
                pooled = x
            else:
                raise ValueError(f"Unsupported pooling method: {pooling}")

            # Ensure pooled output is graph-level: [batch_size, hidden_channels].
            # If a node type is missing in some graphs, global pooling already returns zeros for those graphs.
            if pooled.dim() == 1:
                pooled = pooled.unsqueeze(0)
            if pooled.dim() == 2 and pooled.size(0) != batch_size and pooling != "none":
                # Align to batch_size if a pooling op inferred a smaller batch_size.
                aligned = torch.zeros((batch_size, pooled.size(-1)), device=pooled.device, dtype=pooled.dtype)
                aligned[: pooled.size(0)] = pooled
                pooled = aligned

            type_embeddings.append(pooled)

        if len(type_embeddings) == 0:
            return torch.empty(0)

        # concatenate type-level embeddings and optionally pass through an MLP
        type_level = torch.cat(type_embeddings, dim=-1)

        x = self.classifier(type_level)

        # Output shape should be [batch_size, num_classes]
        # This can then be used for graph classification by using
        # softmax during inference and cross-entropy loss during training.
        return x