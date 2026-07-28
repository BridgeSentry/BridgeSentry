import torch
import torch.nn as nn

from torch_geometric.data import HeteroData
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

from dataset_generator.model.deep_sad import anomaly_scores
from dataset_generator.model.sehgnn_conv import SeHGNN
from dataset_generator.types import EdgeMetapath

# Fixed by the objective, not a hyperparameter: the head is bias-free, so a plain ReLU
# that gets driven negative outputs 0 forever and its gradient dies with it.
HEAD_NEGATIVE_SLOPE = 0.1


class BridgeDefender(nn.Module):
    def __init__(
            self,
            metapath_feature_sizes: dict[EdgeMetapath, int],
            node_types: list[str],
            first_layer_channels: int,
            hidden_channels: int,
            rep_dim: int = 64,
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
        self.rep_dim = rep_dim

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

        # After obtaining type-level embeddings, we concatenate them and project into the
        # rep_dim latent space where the Deep SAD hypersphere is coded.
        # According to the paper, the activation function for the final layer should be unbounded.
        head_layers: list[nn.Module] = []
        in_dim = non_null_node_types * hidden_channels
        for _ in range(max(n_mlp_layers - 1, 0)):
            head_layers.append(nn.Linear(in_dim, hidden_channels, bias=False))
            head_layers.append(nn.LeakyReLU(HEAD_NEGATIVE_SLOPE))
            head_layers.append(nn.Dropout(dropout))
            in_dim = hidden_channels
        head_layers.append(nn.Linear(in_dim, rep_dim, bias=False))
        self.head = nn.Sequential(*head_layers)

        # The hypersphere centre is fixed after initialisation, never optimised.
        # At a later time, the inclusion of an autoencoder head could be considered
        # to stabilise the centre and improve performance.
        self.register_buffer("center", torch.zeros(rep_dim))


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
                type_embeddings.append(torch.zeros((batch_size, self.hidden_channels), device=device, dtype=dtype))
                continue

            # pooling per node type
            pooling = self.pooling
            if pooling == "mean":
                pooled = global_mean_pool(x, data[node_type].batch)
            elif pooling == "max":
                pooled = global_max_pool(x, data[node_type].batch)
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

        # concatenate type-level embeddings and project into the latent space
        type_level = torch.cat(type_embeddings, dim=-1)

        z = self.head(type_level)

        # Output shape is [batch_size, rep_dim]: the latent representation phi(x).
        # Train it with deep_sad_loss, and score graphs with `self.score(z)`.
        return z

    def score(self, z: torch.Tensor) -> torch.Tensor:
        """Anomaly score for latent representations: squared distance to the centre."""
        return anomaly_scores(z, self.center)

    @torch.no_grad()
    def init_center(self, loader, aggregated_features_attr: str = "aggregated_features", eps: float = 0.1) -> torch.Tensor:
        """Fix the hypersphere centre as the mean of phi(x) over `loader`.

        `loader` must yield only *normal* graphs: the centre defines what normality
        looks like, so letting known anomalies drag it would defeat the objective.

        Latent dimensions that land too close to zero are nudged out to +/-eps. A
        near-zero coordinate is a direction the network can trivially zero out to
        shrink every distance at once, which is the first step towards a collapse.

        Returns the centre, and also stores it on the module.
        """
        was_training = self.training
        self.eval()

        device = self.center.device
        total = torch.zeros(self.rep_dim, device=device)
        count = 0
        for data in loader:
            data = data.to(device)
            z = self(data, getattr(data, aggregated_features_attr))
            total += z.sum(dim=0)
            count += z.size(0)

        if count == 0:
            raise ValueError(
                "Cannot initialise the Deep SAD centre: the loader yielded no normal graphs."
            )

        center = total / count
        center[(center.abs() < eps) & (center < 0)] = -eps
        center[(center.abs() < eps) & (center >= 0)] = eps

        self.center.copy_(center)
        if was_training:
            self.train()
        return self.center