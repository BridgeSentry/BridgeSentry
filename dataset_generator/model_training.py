from venv import logger

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
import torch
from torch_geometric.data import Batch, HeteroData
from torch_geometric.loader import DataLoader

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from collections import defaultdict
from typing import DefaultDict
from dataset_generator.cctx_dataset import CrossChainTransactionsDataset
from dataset_generator.early_stopping import EarlyStopping
from dataset_generator.feature_extraction import AMOUNTS_INDEX, ARGS_NUM_INDEX, IN_DEGREE_INDEX, INPUT_SIZE_INDEX, OUT_DEGREE_INDEX
from dataset_generator.model.bridge_defender import BridgeDefender
from dataset_generator.types import DATASET_TYPE, CanonicalEdgeType, EdgeMetapath
from repository.db.graph_label import GraphNodeType
from dataset_generator.types import DATASET_CLASS
from dataset_generator.augmentation import augment_anomaly_graphs
from dataset_generator.training_reporter import TrainingReporter

def get_adjacency_matrices(
    graph: HeteroData,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> dict[CanonicalEdgeType, torch.Tensor]:
    """Build 1-hop adjacency matrices for each canonical edge type.

    Keys are (src_node_type, edge_type, dst_node_type). Values are coalesced
    sparse COO tensors of shape [num_src_nodes, num_dst_nodes] with unit weights.
    """
    adjacency_matrices: dict[CanonicalEdgeType, torch.Tensor] = {}
    for src_type, edge_type, dst_type in graph.edge_types:
        key: CanonicalEdgeType = (src_type, edge_type, dst_type)
        edge_index = graph[key].edge_index
        num_src = graph[src_type].num_nodes
        num_dst = graph[dst_type].num_nodes

        if edge_index.numel() == 0:
            continue

        target_device = device if device is not None else edge_index.device
        edge_index = edge_index.to(target_device)
        values = torch.ones(edge_index.size(1), device=target_device, dtype=dtype)
        adj = torch.sparse_coo_tensor(
            edge_index,
            values,
            size=(num_src, num_dst),
            device=target_device,
            dtype=dtype,
        ).coalesce()
        adjacency_matrices[key] = adj

    return adjacency_matrices

def count_metapaths_occurrences(
    graph: HeteroData,
    max_hops: int = 5,
    device: torch.device | str | None = None,
) -> dict[EdgeMetapath, float]:
    """Count meta-path occurrences while respecting edge types in a BFS-traversal.

    We model a meta-path as a sequence of canonical edge types:
        ((src0, rel0, dst0), (src1, rel1, dst1), ...)
    where dst(i) == src(i+1).

    Counting is done without materializing the full commuting matrix.
    For a meta-path with adjacency matrices A1..Ak, the total number of walks is:
        1^T (A1 A2 ... Ak) 1
    which we compute via dynamic programming on vectors:
        v0 = 1 (size num_src)
        v_{i+1} = A_{i+1}^T v_i
        count = sum(v_k)
    """
    if max_hops < 1:
        return {}

    adjacency_matrices = get_adjacency_matrices(graph, device=device)

    # Get a list of outgoing edge types for each source node type
    # to efficiently find compatible edge types when extending meta-paths.
    outgoing_by_src: DefaultDict[str, list[CanonicalEdgeType]] = defaultdict(list)
    for edge_key in adjacency_matrices.keys():
        outgoing_by_src[edge_key[0]].append(edge_key)

    # Level-1 vectors for each 1-hop canonical edge type.
    prefix_vectors: dict[EdgeMetapath, torch.Tensor] = {}
    path_instance_counts: dict[EdgeMetapath, float] = {}

    for edge_key, adj in adjacency_matrices.items():
        # v0 -> start a walk from every possible source node
        # (hence the all-ones vector of size num_src).
        ones_src = torch.ones((adj.size(0), 1), device=adj.device, dtype=adj.dtype)
        v1 = torch.sparse.mm(adj.transpose(0, 1), ones_src) # shape: [num_dst, 1] -> number of 1-hop walks to each dst node
        mp: EdgeMetapath = (edge_key,)
        prefix_vectors[mp] = v1
        path_instance_counts[mp] = float(v1.sum().item()) # Total number of 1-hop walks for this edge type.

    # Extend meta-paths by chaining compatible canonical edge types.
    for _ in range(2, max_hops + 1):
        next_prefix_vectors: dict[EdgeMetapath, torch.Tensor] = {}
        for mp, v in prefix_vectors.items():
            # Get the last destination node type from the current meta-path to find compatible next edges.
            last_edge = mp[-1]
            current_dst_type = last_edge[2]

            for next_edge_key in outgoing_by_src.get(current_dst_type, []):
                # Get the adjacency matrix for the next hop and multiply to extend the walks.
                next_adj = adjacency_matrices[next_edge_key]
                if next_adj.size(0) != v.size(0):
                    # Should not happen, but protect against inconsistent graphs.
                    continue
                v_next = torch.sparse.mm(next_adj.transpose(0, 1), v)
                mp_next = mp + (next_edge_key,)
                next_prefix_vectors[mp_next] = v_next
                path_instance_counts[mp_next] = float(v_next.sum().item()) # Total number of walks for this meta-path.

        if not next_prefix_vectors:
            break
        prefix_vectors = next_prefix_vectors

    return path_instance_counts

def differential_metapath_extraction(
    dataset: list[HeteroData],
    max_hops: int = 5,
    threshold: float = 0.5,
) -> list[EdgeMetapath]:
    metapath_counts = {}
    label_counts = {
        'normal': 0,
        'anomaly': 0,
    }

    # Count the number of occurrences of each meta-path in the dataset, 
    # as well as the distribution of graph labels for each meta-path.
    for data in dataset:
        path_instances = count_metapaths_occurrences(data, max_hops=max_hops)
        label = 'normal' if int(data.y.item()) == 0 else 'anomaly'

        label_counts[label] += 1
        for metapath in path_instances:
            if metapath in metapath_counts:
                metapath_counts[metapath][label] = metapath_counts[metapath].get(label, 0) + path_instances[metapath]
            else:
                metapath_counts[metapath] = {label: path_instances[metapath]}
    
    # Divide the count of each meta-path by the total number of graphs with that label 
    # to obtain the average frequency of each meta-path for each label.
    differential_metapaths = []
    for metapath, mp_labels in metapath_counts.items():
        if 'normal' not in mp_labels:
            mp_labels['normal'] = 0.0
        if 'anomaly' not in mp_labels:
            mp_labels['anomaly'] = 0.0

        for label, count in label_counts.items():
            mp_labels[label] = mp_labels[label] / (count + 1e-12)

        if abs(mp_labels['normal'] - mp_labels['anomaly']) > threshold:
            differential_metapaths.append(metapath)
    
    return differential_metapaths

def aggregate_metapath_features(
    graph: HeteroData,
    metapaths: list[EdgeMetapath],
    device: torch.device | str | None = None,
    expected_node_types: list[str] | None = None,
    base_feature_dims: dict[str, int] | None = None,
):
    """Aggregate node features along canonical-edge meta-paths.

    Contract:
    - Returns **only** meta-path features (keys are the provided `metapaths`).
      It does *not* include per-node-type 0-hop keys like `(node_type,)`.
    - For batching stability (PyG collation), every meta-path key in `metapaths`
      is ensured to exist in the returned dict; missing ones are filled with
      all-zero tensors of the correct shape.

    Args:
        graph: Input `HeteroData` with per-node-type features in `data[node_type].x`.
        metapaths: List of meta-paths (each is a tuple of canonical edge types).
        device: Optional device to move features/adjacencies to.
        expected_node_types: Node types to consider when building base features.
        base_feature_dims: If provided, used to validate/construct feature dims
            for missing node types and for zero-filling meta-path outputs.

    Returns:
        (metapath_features, feature_sizes)
        - metapath_features: dict mapping each meta-path to a tensor of shape
          [num_start_nodes, dim_end].
        - feature_sizes: dict mapping each meta-path to `dim_end`.
    """
    def normalize_row(adj: torch.Tensor) -> torch.Tensor:
        # Row-normalize sparse COO adjacency: D^{-1} A
        row_sum = torch.sparse.sum(adj, dim=1).to_dense()
        inv_row_sum = torch.where(row_sum > 0, 1.0 / row_sum, torch.zeros_like(row_sum))
        norm_values = adj.values() * inv_row_sum[adj.indices()[0]]
        return torch.sparse_coo_tensor(
            adj.indices(),
            norm_values,
            size=adj.size(),
            device=adj.device,
            dtype=adj.dtype,
        ).coalesce()
    
    # Internal base features per node type (used only to seed meta-path aggregation).
    expected = expected_node_types if expected_node_types is not None else list(graph.node_types)
    target_device = torch.device(device) if device is not None else None
    base_features: dict[str, torch.Tensor] = {}

    for node_type in expected:
        x = None
        if node_type in graph.node_types and hasattr(graph[node_type], 'x'):
            x = graph[node_type].x

        if x is None:
            if base_feature_dims is None or node_type not in base_feature_dims:
                raise ValueError(
                    f"Missing features for node type '{node_type}' and no base_feature_dims provided to create empty features."
                )
            dim = int(base_feature_dims[node_type])
            num_nodes = int(getattr(graph[node_type], 'num_nodes', 0)) if node_type in graph.node_types else 0
            dev = target_device if target_device is not None else torch.device('cpu')
            base_features[node_type] = torch.zeros((num_nodes, dim), dtype=torch.float32, device=dev)
        else:
            if base_feature_dims is not None:
                dim = int(x.size(1))
                expected_dim = int(base_feature_dims.get(node_type, dim))
                if dim != expected_dim:
                    raise ValueError(
                        f"Inconsistent feature dim for node type '{node_type}': got {dim}, expected {expected_dim}."
                    )
            if target_device is not None:
                x = x.to(target_device)
            base_features[node_type] = x

    if len(metapaths) == 0:
        # By contract we don't return 0-hop node-type features anymore.
        return {}, {}
    
    # Step 1: Obtain normalized adjacency matrices for each canonical edge type.
    adjacency_matrices = get_adjacency_matrices(graph, device=device)
    adjacency_matrices = {k: normalize_row(v) for k, v in adjacency_matrices.items()}

    # Step 2: Aggregate features along each meta-path.
    # For a meta-path (e1, e2, ... ek) where ek ends at node type T,
    # we compute A_e1 A_e2 ... A_ek X_T. This yields features on the *start* node type.
    metapath_features: dict[EdgeMetapath, torch.Tensor] = {}
    sorted_metapaths = sorted(metapaths, key=lambda x: (x, len(x)))
    for mp in sorted_metapaths:
        if any(edge_key not in adjacency_matrices for edge_key in mp):
            continue    # Skip metapaths that contain edge types not present in the graph.

        end_type = mp[-1][2]
        if end_type not in base_features:
            continue

        feat = base_features[end_type]
        # Apply adjacencies from the end backwards: A_k @ X_end, then A_{k-1} @ ..., ...
        for edge_key in reversed(mp):
            adj = adjacency_matrices[edge_key]
            feat = torch.sparse.mm(adj, feat)
        metapath_features[mp] = feat

    # Ensure every metapath key exists (for PyG batching). Fill missing ones with zeros.
    for mp in metapaths:
        if mp in metapath_features:
            continue
        start_type = mp[0][0]
        end_type = mp[-1][2]

        num_start = int(getattr(graph[start_type], 'num_nodes', 0)) if start_type in graph.node_types else 0
        dim_end: int | None
        if base_feature_dims is not None and end_type in base_feature_dims:
            dim_end = int(base_feature_dims[end_type])
        elif end_type in base_features:
            dim_end = int(base_features[end_type].size(1))
        else:
            dim_end = None

        if dim_end is None:
            continue

        dev = target_device if target_device is not None else torch.device('cpu')
        metapath_features[mp] = torch.zeros((num_start, dim_end), dtype=torch.float32, device=dev)

    # Also obtain the dimensionality of the aggregated features for each meta-path,
    # which will be needed for the input projection layers in the model.
    # Feature dimensionality is determined by the end node type features.
    if base_feature_dims is not None:
        feature_sizes = {mp: int(base_feature_dims.get(mp[-1][2], 0)) for mp in metapaths}
    else:
        feature_sizes = {mp: metapath_features[mp].size(1) if mp in metapath_features else 0 for mp in metapaths}

    return metapath_features, feature_sizes


def train(dataset_type: DATASET_TYPE, dataset_path: str, force_reload: bool, model_args: dict, **kwargs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = device if kwargs.get("gpu", "cuda") == "cuda" else "cpu"
    print("Early stopping:", kwargs.get("early_stopping", False))

    dataset = DATASET_CLASS[dataset_type](root=dataset_path, force_reload=force_reload)
    torch.manual_seed(42)

    # Separate pre-generated synthetic (augmented) graphs from real ones.
    # Synthetic graphs must ONLY appear in the training split, never in the test set.
    real_graphs = [g for g in dataset if not getattr(g, 'synthetic', False)]
    synthetic_graphs = [g for g in dataset if getattr(g, 'synthetic', False)]
    if synthetic_graphs:
        print(f"Loaded {len(synthetic_graphs)} pre-generated synthetic graphs (will be added to train/val only).")

    # First, split the dataset into a training/validation set for each fold using K-Fold cross-validation,
    # and a test set that is held out for final evaluation after training is complete.
    print(f"Splitting dataset into train/val/test sets with test size 15% and {kwargs.get('k_folds', 5)} folds for cross-validation.")
    X = real_graphs
    y = [int(data.y.item()) for data in real_graphs]
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

    # Record how many REAL graphs are in the pool before appending synthetics.
    # StratifiedKFold will split only the first n_real_train_val entries; synthetic graphs
    # are appended after that boundary and routed to folds by source-graph membership.
    n_real_train_val = len(X_train_val)

    # Append all synthetic graphs to the combined pool (they will never enter the val fold —
    # see the fold loop below for how this is enforced).
    if synthetic_graphs:
        X_train_val = X_train_val + synthetic_graphs
        y_train_val = y_train_val + [int(g.y.item()) for g in synthetic_graphs]
        print(f"Loaded {len(synthetic_graphs)} synthetic graphs into combined pool "
              f"({sum(y_train_val[n_real_train_val:])} anomalies).")

    # Build a mapping from each real graph's position in X_train_val[:n_real_train_val]
    # to the positions of its synthetic copies (indices >= n_real_train_val).
    # Synthetic graphs follow the naming convention "{original_tx_hash}_augmented_{n}",
    # so we recover the source by stripping the last "_augmented_N" suffix.
    from collections import defaultdict as _defaultdict
    source_to_synthetic_idx: dict[int, list[int]] = _defaultdict(list)
    if synthetic_graphs:
        real_tx_hash_to_idx = {g.tx_hash: i for i, g in enumerate(X_train_val[:n_real_train_val])}
        for syn_offset, syn_g in enumerate(synthetic_graphs):
            original_tx_hash = syn_g.tx_hash.rsplit('_augmented_', 1)[0]
            real_idx = real_tx_hash_to_idx.get(original_tx_hash)
            if real_idx is not None:
                source_to_synthetic_idx[real_idx].append(n_real_train_val + syn_offset)

    # Now, we need to normalize some numerical features in the dataset. 
    # Specifically, we will apply min-max normalization
    # to the node features of each node type across the entire training/validation set, and 
    # then apply the same scale to the test set to avoid data leakage.
    print("Normalizing node features using min-max normalization.")
    max_degrees = torch.zeros(1, dtype=torch.float32)
    max_args_num = torch.zeros(1, dtype=torch.float32)
    max_inputs_size = torch.zeros(1, dtype=torch.float32)
    max_amounts = torch.zeros(1, dtype=torch.float32)
    for graph in X_train_val:
        for node_type in graph.node_types:
            if hasattr(graph[node_type], 'x') and graph[node_type].x is not None:
                in_degrees = graph[node_type].x[:, IN_DEGREE_INDEX]
                out_degrees = graph[node_type].x[:, OUT_DEGREE_INDEX]
                max_degrees = torch.max(max_degrees, in_degrees.max(dim=0).values if in_degrees.shape[0] > 0 else torch.zeros_like(max_degrees))
                max_degrees = torch.max(max_degrees, out_degrees.max(dim=0).values if out_degrees.shape[0] > 0 else torch.zeros_like(max_degrees))

            if node_type == GraphNodeType.LOG_EVENT.value:
                # For log event nodes, we also want to normalize the args_num feature (which is at index 14),
                # the inputs_size feature (which is at index 15),
                # and the amounts feature (which is at index 16)
                if hasattr(graph[node_type], 'x') and graph[node_type].x is not None:
                    args_num = graph[node_type].x[:, ARGS_NUM_INDEX]
                    inputs_size = graph[node_type].x[:, INPUT_SIZE_INDEX]
                    amounts = graph[node_type].x[:, AMOUNTS_INDEX]
                    max_args_num = torch.max(max_args_num, args_num.max(dim=0).values if args_num.shape[0] > 0 else torch.zeros_like(max_args_num))
                    max_inputs_size = torch.max(max_inputs_size, inputs_size.max(dim=0).values if inputs_size.shape[0] > 0 else torch.zeros_like(max_inputs_size))
                    max_amounts = torch.max(max_amounts, amounts.max(dim=0).values if amounts.shape[0] > 0 else torch.zeros_like(max_amounts))

    for graph in X_train_val + X_test:
        for node_type in graph.node_types:
            if hasattr(graph[node_type], 'x') and graph[node_type].x is not None:
                x = graph[node_type].x
                x[:, IN_DEGREE_INDEX] = x[:, IN_DEGREE_INDEX] / (max_degrees[0] + 1e-12)
                x[:, OUT_DEGREE_INDEX] = x[:, OUT_DEGREE_INDEX] / (max_degrees[0] + 1e-12)

            if node_type == GraphNodeType.LOG_EVENT.value:
                if hasattr(graph[node_type], 'x') and graph[node_type].x is not None:
                    x = graph[node_type].x
                    x[:, ARGS_NUM_INDEX] = x[:, ARGS_NUM_INDEX] / (max_args_num[0] + 1e-12)
                    x[:, INPUT_SIZE_INDEX] = x[:, INPUT_SIZE_INDEX] / (max_inputs_size[0] + 1e-12)
                    x[:, AMOUNTS_INDEX] = x[:, AMOUNTS_INDEX] / (max_amounts[0] + 1e-12)

    # Optional runtime augmentation (applied AFTER normalization so noise operates in [0, 1] space).
    # Only active when --augment-factor > 0 is passed to the train command.
    runtime_augment_factor = kwargs.get("augment_factor", 0)
    if runtime_augment_factor > 0:
        print(f"Applying runtime augmentation with factor {runtime_augment_factor}.")
        aug_graphs, aug_labels = augment_anomaly_graphs(
            X_train_val,
            y_train_val,
            augment_factor=runtime_augment_factor,
        )
        X_train_val = X_train_val + aug_graphs
        y_train_val = y_train_val + aug_labels
        print(f"Training set after runtime augmentation: {len(X_train_val)} graphs "
              f"({sum(y_train_val)} anomalies).")

    # Determine the graph label weights based on the ratio of each label in the dataset
    # Less frequent labels (i.e. anomalies) should have higher weights to penalize
    # misclassification more than the normal label
    label_counts = torch.bincount(torch.tensor(y_train_val), minlength=4)
    class_weights = 1.0 / (label_counts + 1e-12)  # Add small value to avoid division by zero
    class_weights = class_weights / class_weights.sum()  # Normalize to sum to 1
    class_weights = class_weights.to(device)

    # Perform Differential Meta-path Extraction on the training/validation set
    # to identify the most discriminative meta-paths for the classification task
    metapaths = differential_metapath_extraction(X_train_val, threshold=0.5, max_hops=5)
    print(f"Identified {len(metapaths)} differential meta-paths for training.")

    # As a pre-processing step, we will calculate the aggregated meta-path features for each graph in the dataset,
    # and then use those features as input to the model during training.
    # Also infer per-node-type feature dimensionalities so we can create empty features
    # for missing node types and keep a consistent aggregated_features schema across graphs.
    base_feature_dims: dict[str, int] = {}
    for data in X_train_val + X_test:
        for node_type in data.node_types:
            if not hasattr(data[node_type], 'x') or data[node_type].x is None:
                continue
            dim = int(data[node_type].x.size(1))
            prev = base_feature_dims.get(node_type)
            if prev is not None and prev != dim:
                raise ValueError(f"Inconsistent feature dim for node type '{node_type}': saw {prev} and {dim}.")
            base_feature_dims[node_type] = dim

    expected_node_types = sorted(base_feature_dims.keys())

    all_feature_sizes = dict()
    for data in X_train_val:
        data.aggregated_features, feature_sizes = aggregate_metapath_features(
            data,
            metapaths,
            expected_node_types=expected_node_types,
            base_feature_dims=base_feature_dims,
        )
        all_feature_sizes.update(feature_sizes)
    for data in X_test:
        data.aggregated_features, feature_sizes = aggregate_metapath_features(
            data,
            metapaths,
            expected_node_types=expected_node_types,
            base_feature_dims=base_feature_dims,
        )
        all_feature_sizes.update(feature_sizes)

    k_folds = kwargs.get("k_folds", 5)
    num_epochs = kwargs.get("num_epochs", 100)
    kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    reporter = TrainingReporter(
        reports_root="reports",
        name_prefix=kwargs.get("run_name_prefix", "train"),
        params={**model_args, **{k: v for k, v in kwargs.items() if k != "run_name_prefix"}},
    )

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    def collate_fn(batch: list[HeteroData]) -> Batch:
        pyg_batch = Batch.from_data_list(batch)
        pyg_batch.aggregated_features = {
            mp: torch.cat([g.aggregated_features[mp] for g in batch], dim=0)
            for mp in metapaths
        }
        return pyg_batch

    fold_model_states: list[tuple[int, dict]] = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_val[:n_real_train_val], y_train_val[:n_real_train_val])):
        # Each fold's train index covers real training graphs only; add synthetic graphs
        # whose *source* real graph also falls in this fold's training partition.
        # This prevents augmented copies of validation graphs from leaking into training.
        extra_syn = []
        for real_idx in train_idx:
            extra_syn.extend(source_to_synthetic_idx.get(int(real_idx), []))
        train_idx = list(train_idx) + extra_syn
        model = BridgeDefender(
            metapath_feature_sizes=all_feature_sizes,
            node_types=dataset[0].node_types,
            first_layer_channels=model_args.get("first_layer_channels", 128),
            hidden_channels=model_args.get("hidden_channels", 64),
            out_channels=model_args.get("out_channels", 4),
            dropout=model_args.get("dropout", 0.5),
            input_drop=model_args.get("input_drop", 0.0),
            att_drop=model_args.get("att_drop", 0.0),
            n_fp_layers=model_args.get("n_fp_layers", 2),
            n_mlp_layers=model_args.get("n_mlp_layers", 2),
            act=model_args.get("act", 'relu'),
            residual=model_args.get("residual", False),
            pooling=model_args.get("pooling", 'mean'),
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=model_args.get("learning_rate", 0.01),
            weight_decay=model_args.get("weight_decay", 5e-4),
        )
        early_stopping = EarlyStopping(patience=10, delta=0, device=device)

        train_subset = torch.utils.data.Subset(X_train_val, train_idx)
        val_subset = torch.utils.data.Subset(X_train_val, val_idx)

        train_loader = DataLoader(train_subset, batch_size=model_args.get("batch_size", 32), shuffle=True, num_workers=kwargs.get("num_workers", 0), collate_fn=collate_fn)
        val_loader = DataLoader(val_subset, batch_size=model_args.get("batch_size", 32), shuffle=False, num_workers=kwargs.get("num_workers", 0), collate_fn=collate_fn)

        fold_best_val_loss = float("inf")
        fold_best_model_state: dict | None = None

        for epoch in range(num_epochs):
            model.train()
            print(f"Fold {fold + 1}/{k_folds}, epoch {epoch + 1}/{num_epochs} — training...")
            train_loss = 0.0
            for data in train_loader:
                data = data.to(device)
                output = model(data, data.aggregated_features)
                loss = criterion(output, data.y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_loss += loss.item()
            avg_train_loss = train_loss / len(train_loader)

            model.eval()
            val_loss = 0
            print(f"Fold {fold + 1}/{k_folds}, epoch {epoch + 1}/{num_epochs} — validation...")
            with torch.no_grad():
                all_labels = []
                all_preds = []
                all_val_probs = []
                for data in val_loader:
                    data = data.to(device)
                    output = model(data, data.aggregated_features)
                    loss = criterion(output, data.y)
                    val_loss += loss.item()
                    probs = torch.softmax(output, dim=1)
                    all_labels.extend(data.y.detach().cpu().tolist())
                    all_preds.extend(output.argmax(dim=1).detach().cpu().tolist())
                    all_val_probs.extend(probs.detach().cpu().tolist())

                avg_val_loss = val_loss / len(val_loader)
                print(f"  Fold {fold + 1}, epoch {epoch + 1}, val loss: {avg_val_loss:.4f}")
                print(classification_report(all_labels, all_preds, zero_division=0))

                all_val_probs_np = np.array(all_val_probs)
                val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
                    all_labels, all_preds, labels=[0, 1], zero_division=0
                )
                reporter.record_val_epoch(fold, epoch, {
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "accuracy": accuracy_score(all_labels, all_preds),
                    "precision_normal": val_precision[0],
                    "recall_normal": val_recall[0],
                    "f1_normal": val_f1[0],
                    "precision_anomaly": val_precision[1],
                    "recall_anomaly": val_recall[1],
                    "f1_anomaly": val_f1[1],
                    "mcc": matthews_corrcoef(all_labels, all_preds),
                    "roc_auc": roc_auc_score(all_labels, all_val_probs_np[:, 1]),
                    "pr_auc": average_precision_score(all_labels, all_val_probs_np[:, 1]),
                })

                early_stopping(avg_val_loss, model)
                if kwargs.get("early_stopping", False) and early_stopping.early_stop:
                    print(f"Early stopping triggered at epoch {epoch + 1} for fold {fold + 1}.")
                    break

                if avg_val_loss < fold_best_val_loss:
                    fold_best_val_loss = avg_val_loss
                    fold_best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if fold_best_model_state is not None:
            fold_model_states.append((fold, fold_best_model_state))

    if not fold_model_states:
        print("No model was trained; skipping test evaluation.")
        return

    print("=== Test set evaluation (averaged across K-fold models) ===")

    test_loader = DataLoader(
        X_test,
        batch_size=model_args.get("batch_size", 32),
        shuffle=False,
        num_workers=kwargs.get("num_workers", 0),
        collate_fn=collate_fn,
    )
    label_names = ["normal", "anomaly"]
    fold_metrics = []

    for fold_idx, model_state in fold_model_states:
        fold_model = BridgeDefender(
            metapath_feature_sizes=all_feature_sizes,
            node_types=dataset[0].node_types,
            first_layer_channels=model_args.get("first_layer_channels", 128),
            hidden_channels=model_args.get("hidden_channels", 64),
            out_channels=model_args.get("out_channels", 4),
            dropout=model_args.get("dropout", 0.5),
            input_drop=model_args.get("input_drop", 0.0),
            att_drop=model_args.get("att_drop", 0.0),
            n_fp_layers=model_args.get("n_fp_layers", 2),
            n_mlp_layers=model_args.get("n_mlp_layers", 2),
            act=model_args.get("act", 'relu'),
            residual=model_args.get("residual", False),
            pooling=model_args.get("pooling", 'mean'),
        ).to(device)
        fold_model.load_state_dict({k: v.to(device) for k, v in model_state.items()})
        fold_model.eval()

        test_loss = 0
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                output = fold_model(data, data.aggregated_features)
                loss = criterion(output, data.y)
                test_loss += loss.item()
                probs = torch.softmax(output, dim=1)
                all_labels.extend(data.y.detach().cpu().tolist())
                all_probs.extend(probs.detach().cpu().tolist())

        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        all_preds = all_probs.argmax(axis=1)

        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, labels=[0, 1], zero_division=0
        )
        acc = accuracy_score(all_labels, all_preds)
        mcc = matthews_corrcoef(all_labels, all_preds)
        roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
        pr_auc = average_precision_score(all_labels, all_probs[:, 1])
        avg_test_loss = test_loss / len(test_loader)

        fold_metrics.append({
            "test_loss": avg_test_loss,
            "accuracy": acc,
            "precision_normal": precision[0],
            "recall_normal": recall[0],
            "f1_normal": f1[0],
            "precision_anomaly": precision[1],
            "recall_anomaly": recall[1],
            "f1_anomaly": f1[1],
            "mcc": mcc,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
        })
        reporter.record_test_fold(fold_idx, fold_metrics[-1])

        print(f"\n--- Fold {fold_idx + 1} test results ---")
        print(f"Test loss: {avg_test_loss:.4f}")
        print(classification_report(all_labels, all_preds, target_names=label_names, zero_division=0))
        print(f"ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | MCC: {mcc:.4f}")

    print("\n=== K-Fold test summary (mean ± std across folds) ===")
    metric_display = [
        ("test_loss",         "Test loss"),
        ("accuracy",          "Accuracy"),
        ("precision_normal",  "Precision  (normal)"),
        ("recall_normal",     "Recall     (normal)"),
        ("f1_normal",         "F1-score   (normal)"),
        ("precision_anomaly", "Precision  (anomaly)"),
        ("recall_anomaly",    "Recall     (anomaly)"),
        ("f1_anomaly",        "F1-score   (anomaly)"),
        ("roc_auc",           "ROC-AUC"),
        ("pr_auc",            "PR-AUC"),
        ("mcc",               "MCC"),
    ]
    for key, label in metric_display:
        values = [m[key] for m in fold_metrics]
        print(f"  {label:<26} {np.mean(values):.4f} ± {np.std(values):.4f}")

    reporter.save(metric_display, k_folds)