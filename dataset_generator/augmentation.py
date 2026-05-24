import copy
import random
from typing import Optional

import torch
from torch_geometric.data import HeteroData

# Node types that are structural periphery — safe to drop without destroying anomaly semantics.
DROPPABLE_NODE_TYPES = {"user", "other_account"}

# Indices of continuous scalar features per node type (post-normalization these are all in [0, 1]).
# All other indices are binary/categorical or hash-derived floats that should not receive Gaussian noise.
CONTINUOUS_FEATURE_INDICES: dict[str, list[int]] = {
    "user":          [0, 1],
    "router":        [0, 1],
    "other_account": [0, 1],
    "token":         [0, 1],
    "log_event":     [0, 1, 14, 15, 16],  # in_degree, out_degree, args_num, input_size, amounts
    "validator":     [0, 1],
}


def drop_nodes(graph: HeteroData, drop_prob: float = 0.2) -> HeteroData:
    """
    Randomly remove non-critical nodes (user, other_account) and their respective edges.
    """
    graph = copy.deepcopy(graph)
    keep_masks = {}
    reindex_maps = {}

    for ntype in DROPPABLE_NODE_TYPES:
        if ntype not in graph.node_types:
            continue
        x = graph[ntype].x
        if x is None or x.shape[0] == 0:
            continue

        n = x.shape[0]
        mask = torch.rand(n) > drop_prob
        if not mask.any():
            mask[0] = True  # Keep at least one node

        graph[ntype].x = x[mask]
        keep_masks[ntype] = mask
        reindex_maps[ntype] = torch.cumsum(mask.long(), dim=0) - 1

    if not keep_masks:
        return graph

    # Update edge indices for all edge types, removing edges connected to dropped nodes and reindexing survivors.
    for (src_type, edge_rel, dst_type) in list(graph.edge_types):
        key = (src_type, edge_rel, dst_type)
        edge_index = graph[key].edge_index
        if edge_index is None or edge_index.numel() == 0:
            continue

        valid = torch.ones(edge_index.size(1), dtype=torch.bool)

        src_mask = keep_masks.get(src_type)
        if src_mask is not None:
            valid = valid & src_mask[edge_index[0]]

        dst_mask = keep_masks.get(dst_type)
        if dst_mask is not None:
            valid = valid & dst_mask[edge_index[1]]

        filtered = edge_index[:, valid].clone()

        if src_type in reindex_maps:
            filtered[0] = reindex_maps[src_type][filtered[0]]
        if dst_type in reindex_maps:
            filtered[1] = reindex_maps[dst_type][filtered[1]]

        graph[key].edge_index = filtered if filtered.numel() > 0 else torch.zeros((2, 0), dtype=torch.long)

    return graph


def inject_feature_noise(graph: HeteroData, noise_std: float = 0.05) -> HeteroData:
    """
    Add Gaussian noise to continuous scalar features of nodes.
    """
    graph = copy.deepcopy(graph)

    for ntype in graph.node_types:
        x = graph[ntype].x
        if x is None or x.shape[0] == 0:
            continue

        indices = CONTINUOUS_FEATURE_INDICES.get(ntype, [])
        for idx in indices:
            if idx >= x.shape[1]:
                continue

            # Add noise and clamp to ensure noise is not negative (since features are non-negative).
            noise = torch.randn(x.shape[0]) * noise_std
            x[:, idx] = (x[:, idx] + noise).clamp(min=0.0)

    return graph


def mask_node_features(graph: HeteroData, mask_prob: float = 0.1) -> HeteroData:
    """
    Randomly zero out node features with probability mask_prob.
    """
    graph = copy.deepcopy(graph)

    for ntype in graph.node_types:
        x = graph[ntype].x
        if x is None or x.shape[0] == 0:
            continue

        mask = torch.bernoulli(torch.full(x.shape, mask_prob))
        graph[ntype].x = x * (1.0 - mask)

    return graph


_STRATEGIES = ["drop_nodes", "inject_noise", "mask_features"]


def augment_anomaly_graphs(
    graphs: list[HeteroData],
    labels: list[int],
    augment_factor: int = 3,
    drop_prob: float = 0.2,
    noise_std: float = 0.05,
    mask_prob: float = 0.1,
    seed: int = 42,
) -> tuple[list[HeteroData], list[int]]:
    """Create augmented copies of anomaly graphs using random augmentation strategies.

    Each anomaly graph is duplicated `augment_factor` times. Each copy receives a
    randomly selected augmentation strategy (node dropping, feature noise, or feature masking).
    Augmented graphs are marked with `synthetic=True` and their tx_hash is suffixed with
    `_augmented_<copy_idx>` for unique identification.

    Args:
        graphs: List of HeteroData graphs (already normalized if noise is to be meaningful).
        labels: Corresponding integer labels (0=normal, non-zero=anomaly).
        augment_factor: Number of augmented copies per anomaly graph.
        drop_prob: Node drop probability for the node-dropping strategy.
        noise_std: Std deviation for Gaussian noise in the noise-injection strategy.
        mask_prob: Feature masking probability for the masking strategy.
        seed: Random seed for reproducibility.

    Returns:
        (aug_graphs, aug_labels): Lists of augmented graphs and their labels.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    aug_graphs: list[HeteroData] = []
    aug_labels: list[int] = []

    for i, (graph, label) in enumerate(zip(graphs, labels)):
        if label == 0:
            continue

        for copy_idx in range(augment_factor):
            strategy = random.choice(_STRATEGIES)

            if strategy == "drop_nodes":
                aug = drop_nodes(graph, drop_prob=drop_prob)
            elif strategy == "inject_noise":
                aug = inject_feature_noise(graph, noise_std=noise_std)
            else:
                aug = mask_node_features(graph, mask_prob=mask_prob)

            aug.synthetic = True
            aug.y = graph.y.clone()
            aug.y_str = graph.y_str
            aug.bridge = graph.bridge
            aug.tx_hash = f"{graph.tx_hash}_augmented_{copy_idx}"

            aug_graphs.append(aug)
            aug_labels.append(label)

    return aug_graphs, aug_labels
