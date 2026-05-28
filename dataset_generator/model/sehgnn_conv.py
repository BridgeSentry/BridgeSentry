import math
import hashlib
import re

import torch
from torch import nn
from torch.nn import functional as F

from torch_geometric.data import HeteroData

from dataset_generator.types import EdgeMetapath

class LinearProjectionPerMetapath(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_metapaths: int):
        super(LinearProjectionPerMetapath, self).__init__()
        self.W = nn.Parameter(torch.randn(num_metapaths, in_dim, out_dim))
        self.b = nn.Parameter(torch.zeros(num_metapaths, out_dim))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, num_metapaths, in_dim]
        out = torch.einsum("bcm,cmn->bcn", x, self.W) + self.b.unsqueeze(0)
        return out

class SemanticFusionTransformer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 1, att_drop: float = 0.0, act: str = "none"):
        super(SemanticFusionTransformer, self).__init__()
        assert hidden_dim % (num_heads * 4) == 0

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.query = nn.Linear(self.hidden_dim, self.hidden_dim // 4)
        self.key = nn.Linear(self.hidden_dim, self.hidden_dim // 4)
        self.value = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.gamma = nn.Parameter(torch.tensor([0.]))
        self.att_drop = nn.Dropout(att_drop)

        if act == "sigmoid":
            self.act = nn.Sigmoid()
        elif act == "relu":
            self.act = nn.ReLU()
        elif act == "leaky_relu":
            self.act = nn.LeakyReLU(0.2)
        elif act == "none":
            self.act = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation function: {act}")
    
    def forward(self, x, mask=None):
        # x shape: [batch_size, num_metapaths, hidden_dim (channels)]
        B, M, C = x.size()
        H = self.num_heads
        
        if mask is not None:
            assert mask.size() == torch.Size([B, M])

        D = C // (H * 4)
        q = self.query(x).view(B, M, H, D).permute(0, 2, 1, 3)   # [B, H, M, D//4]
        k = self.key(x).view(B, M, H, D).permute(0, 2, 3, 1)     # [B, H, D//4, M]
        v = self.value(x).view(B, M, H, D * 4).permute(0, 2, 1, 3)  # [B, H, M, C//H]

        # Calculate mutual attention scores and apply dropout
        beta = F.softmax(self.act(q @ k / math.sqrt(q.size(-1))), dim=-1)  # [B, H, M, M(normalized)]
        beta = self.att_drop(beta)
        if mask is not None:
            beta = beta * mask.view(B, 1, 1, M)  # Mask out invalid metapaths
            beta = beta / (beta.sum(dim=-1, keepdim=True) + 1e-12)  # Re-normalize after masking
        
        out = self.gamma * (beta @ v)
        return out.permute(0, 2, 1, 3).reshape((B, M, C)) + x

class SeHGNN(nn.Module):
    def __init__(
            self, 
            metapath_data_size: dict[EdgeMetapath, int],
            num_features: int, 
            hidden_dim: int,
            target_type: str,
            dropout: float = 0.5,
            input_drop: float = 0.0,
            att_drop: float = 0.0, 
            n_fp_layers: int = 2,
            act: str = 'relu',
            residual: bool = False
        ):
        super().__init__()
        self.metapaths = list(metapath_data_size.keys())
        self.num_mp_channels = len(self.metapaths)
        self.target_type = target_type
        self.residual = residual
        self.input_drop = nn.Dropout(input_drop)

        def mp_to_str(mp: EdgeMetapath) -> str:
            # ParameterDict keys must be strings (and should avoid '.' since it's a module path separator).
            # Prefer human-readable metapath names, but append a short hash suffix for stability/uniqueness.
            parts: list[str] = []
            for src, rel, dst in mp:
                parts.append(f"{src}_{rel}_{dst}")

            readable = "__".join(parts)
            readable = re.sub(r"[^0-9A-Za-z_]+", "_", readable)
            readable = re.sub(r"_+", "_", readable).strip("_")

            digest = hashlib.sha1(repr(mp).encode("utf-8")).hexdigest()[:8]
            max_readable_len = 60
            if len(readable) > max_readable_len:
                readable = readable[:max_readable_len].rstrip("_")
            if readable:
                return f"mp_{readable}_{digest}"
            return f"mp_{digest}"

        self.mp_to_str: dict[EdgeMetapath, str] = {mp: mp_to_str(mp) for mp in self.metapaths}

        # Each meta-path's features are projected into the same hidden space using a type-specific linear layer.
        self.embeddings = nn.ParameterDict({})
        for mp, in_dim in metapath_data_size.items():
            name = self.mp_to_str[mp]
            param = nn.Parameter(torch.empty(in_dim, num_features))
            nn.init.uniform_(param, -0.5, 0.5)
            self.embeddings[name] = param

        # PyG's lazy initialization allows us to skip explicitly defining input dimensions for embeddings
        self.feature_projection = nn.Sequential(
            LinearProjectionPerMetapath(num_features, hidden_dim, self.num_mp_channels),
            nn.LayerNorm([self.num_mp_channels, hidden_dim]),
            nn.PReLU(),
            nn.Dropout(dropout)
        )
        
        # Extend feature projection if needed
        for _ in range(n_fp_layers - 1):
            self.feature_projection.append(LinearProjectionPerMetapath(hidden_dim, hidden_dim, self.num_mp_channels))
            self.feature_projection.append(nn.LayerNorm([self.num_mp_channels, hidden_dim]))
            self.feature_projection.append(nn.PReLU())
            self.feature_projection.append(nn.Dropout(dropout))

        self.semantic_fusion = SemanticFusionTransformer(hidden_dim, num_heads=1, att_drop=att_drop, act=act)
        self.fc_after_concat = nn.Linear(self.num_mp_channels * hidden_dim, hidden_dim)

        if self.residual:
            self.res_fc = nn.Linear(num_features, hidden_dim)

    def forward(
        self,
        data: HeteroData,
        metapath_features: dict[EdgeMetapath, torch.Tensor] | None = None,
    ):
        """
        Expects a PyG HeteroData object that has passed through the PrecomputeSeHGNN transform.
        """
        target_raw_feat = data[self.target_type].x

        # Use type-specific feature projection to ensure each meta-path's features are projected into the same hidden space
        features = {}
        for mp in self.metapaths:
            emb = self.embeddings[self.mp_to_str[mp]]
            v = metapath_features.get(mp)
            if v is None:
                # Some graphs/batches may not realize every meta-path; treat as all-zeros.
                num_nodes = int(getattr(data[self.target_type], "num_nodes", 0) or 0)
                v = torch.zeros((num_nodes, emb.size(0)), device=emb.device, dtype=emb.dtype)
            features[mp] = v @ emb

        # Stack into [B, num_channels, D]
        x = torch.stack([features[mp] for mp in self.metapaths], dim=1)
        batch_size = x.shape[0]

        # Multi-layer feature Projection
        x = self.feature_projection(x)

        # Transformer-based semantic Fusion
        x = self.semantic_fusion(x).transpose(1, 2)
        x = self.fc_after_concat(x.reshape(batch_size, self.fc_after_concat.in_features))

        # Residual connection over target node features
        if self.residual:
            x = x + self.res_fc(target_raw_feat)

        # Return the node embeddings for the target node type for downstream tasks
        return x