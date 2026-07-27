import torch
from torch_geometric.data import Batch, HeteroData
from torch_geometric.loader import DataLoader

from dataset_generator.model.bridge_defender import BridgeDefender
from dataset_generator.model_training import aggregate_metapath_features, apply_normalization


def load_checkpoint(path: str, device: str | torch.device = "cpu") -> dict:
    """Load a fold checkpoint saved by `TrainingReporter.save_model` during training."""
    return torch.load(path, map_location=device)


def build_model_from_checkpoint(checkpoint: dict, device: str | torch.device = "cpu") -> BridgeDefender:
    """Reconstruct a fold's BridgeDefender model from a checkpoint and load its weights."""
    model = BridgeDefender(
        metapath_feature_sizes=checkpoint["metapath_feature_sizes"],
        node_types=checkpoint["node_types"],
        **checkpoint["model_kwargs"],
    )
    model.load_state_dict({k: v.to(device) for k, v in checkpoint["model_state_dict"].items()})
    model.to(device)
    model.eval()
    return model


def prepare_graphs(graphs: list[HeteroData], checkpoint: dict) -> list[HeteroData]:
    """Apply the checkpoint's normalization scale and meta-path feature aggregation to
    new graphs in place, so they match what the model was trained on."""
    apply_normalization(graphs, checkpoint["normalization"])
    for graph in graphs:
        graph.aggregated_features, _ = aggregate_metapath_features(
            graph,
            checkpoint["metapaths"],
            expected_node_types=checkpoint["expected_node_types"],
            base_feature_dims=checkpoint["base_feature_dims"],
        )
    return graphs


@torch.no_grad()
def predict(
    model: BridgeDefender,
    graphs: list[HeteroData],
    checkpoint: dict,
    device: str | torch.device = "cpu",
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run inference on graphs already prepared via `prepare_graphs`.

    Returns (scores, preds):
      - scores: [N] Deep SAD anomaly score, the squared distance from phi(x) to the
        hypersphere centre. Higher means a larger deviation from learned normality.
        Unbounded above — it is a distance, not a probability.
      - preds: [N] binary labels, `scores > checkpoint["threshold"]`, using the
        operating point fitted on the validation fold during training.
    """
    metapaths = checkpoint["metapaths"]
    threshold = checkpoint.get("threshold")
    if threshold is None:
        raise KeyError(
            "Checkpoint has no 'threshold'; it predates the Deep SAD head and cannot be "
            "used for binary prediction."
        )

    def collate_fn(batch: list[HeteroData]) -> Batch:
        pyg_batch = Batch.from_data_list(batch)
        pyg_batch.aggregated_features = {
            mp: torch.cat([g.aggregated_features[mp] for g in batch], dim=0)
            for mp in metapaths
        }
        return pyg_batch

    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    all_scores = []
    for data in loader:
        data = data.to(device)
        z = model(data, data.aggregated_features)
        all_scores.append(model.score(z).cpu())

    scores = torch.cat(all_scores, dim=0)
    preds = (scores > threshold).long()
    return scores, preds
