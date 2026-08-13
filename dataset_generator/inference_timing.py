import time

import numpy as np
import torch
from torch_geometric.data import Batch, HeteroData
from torch_geometric.loader import DataLoader

from dataset_generator.model_evaluation import _discover_checkpoint_paths
from dataset_generator.model_inference import build_model_from_checkpoint, load_checkpoint
from dataset_generator.model_training import aggregate_metapath_features, apply_normalization
from dataset_generator.timing_reporter import TimingReporter
from dataset_generator.types import DATASET_CLASS


def _stage_row(stage: str, batch_size: int | str, per_tx_seconds: list[float], total_seconds: float) -> dict:
    per_tx_ms = np.array(per_tx_seconds) * 1000.0
    return {
        "stage": stage,
        "batch_size": batch_size,
        "n_samples": len(per_tx_ms),
        "mean_ms_per_tx": float(per_tx_ms.mean()),
        "std_ms_per_tx": float(per_tx_ms.std()),
        "min_ms_per_tx": float(per_tx_ms.min()),
        "max_ms_per_tx": float(per_tx_ms.max()),
        "total_seconds": total_seconds,
    }


def _time_preprocessing(graphs: list[HeteroData], checkpoint: dict) -> list[dict]:
    """Times `prepare_graphs`' two steps separately, then reports their sum:
    - `normalization`: one-shot `apply_normalization` call over all graphs (cheap,
      only one wall-clock measurement exists so it's amortized evenly across N).
    - `metapath_aggregation`: `aggregate_metapath_features`, timed individually per
      graph (the heavier, sparse-adjacency part) so it gets a real per-graph distribution.
    - `preprocessing`: the two stages combined, per graph.
    """
    t0 = time.perf_counter()
    apply_normalization(graphs, checkpoint["normalization"])
    norm_total = time.perf_counter() - t0
    norm_per_tx = norm_total / len(graphs)
    normalization_row = _stage_row("normalization", "", [norm_per_tx], norm_total)

    metapath_seconds = []
    for graph in graphs:
        t0 = time.perf_counter()
        graph.aggregated_features, _ = aggregate_metapath_features(
            graph,
            checkpoint["metapaths"],
            expected_node_types=checkpoint["expected_node_types"],
            base_feature_dims=checkpoint["base_feature_dims"],
        )
        metapath_seconds.append(time.perf_counter() - t0)
    metapath_row = _stage_row("metapath_aggregation", "", metapath_seconds, sum(metapath_seconds))

    preprocessing_seconds = [d + norm_per_tx for d in metapath_seconds]
    preprocessing_row = _stage_row(
        "preprocessing", "", preprocessing_seconds, norm_total + sum(metapath_seconds)
    )

    return [normalization_row, metapath_row, preprocessing_row]


def _time_forward_pass(
    model,
    graphs: list[HeteroData],
    checkpoint: dict,
    device: str,
    batch_size: int,
    warmup_batches: int,
    repeats: int,
    num_workers: int,
) -> dict:
    """Times the pure `model(...)` forward call (plus `model.score(z)` on the
    DeepSAD branch, detected via duck-typing since that method only exists there).

    Small test sets mean a handful of warmup batches can exceed the whole dataset
    at larger batch sizes, so the graph list is looped `repeats` times before the
    first `warmup_batches` batches of that stream are discarded.
    """
    metapaths = checkpoint["metapaths"]
    is_deepsad = hasattr(model, "score")

    def collate_fn(batch: list[HeteroData]) -> Batch:
        pyg_batch = Batch.from_data_list(batch)
        pyg_batch.aggregated_features = {
            mp: torch.cat([g.aggregated_features[mp] for g in batch], dim=0)
            for mp in metapaths
        }
        return pyg_batch

    timed_graphs = graphs * repeats
    loader = DataLoader(
        timed_graphs, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn
    )

    batch_durations = []
    batch_sizes_actual = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            data = data.to(device)
            if i < warmup_batches:
                if is_deepsad:
                    model.score(model(data, data.aggregated_features))
                else:
                    model(data, data.aggregated_features)
                continue

            t0 = time.perf_counter()
            if is_deepsad:
                model.score(model(data, data.aggregated_features))
            else:
                model(data, data.aggregated_features)
            batch_durations.append(time.perf_counter() - t0)
            batch_sizes_actual.append(data.num_graphs)

    if not batch_durations:
        raise ValueError(
            f"No batches left to time at batch_size={batch_size} after {warmup_batches} warmup batches "
            f"over {len(timed_graphs)} graphs (repeats={repeats}). Increase --repeats or reduce --warmup-batches."
        )

    per_tx_seconds = [d / n for d, n in zip(batch_durations, batch_sizes_actual)]
    total_seconds = sum(batch_durations)
    return _stage_row(f"forward_batch{batch_size}", batch_size, per_tx_seconds, total_seconds)


def _end_to_end_row(preprocessing_row: dict, forward_row: dict) -> dict:
    return {
        "stage": f"end_to_end_batch{forward_row['batch_size']}",
        "batch_size": forward_row["batch_size"],
        "n_samples": forward_row["n_samples"],
        "mean_ms_per_tx": preprocessing_row["mean_ms_per_tx"] + forward_row["mean_ms_per_tx"],
        "std_ms_per_tx": "",
        "min_ms_per_tx": "",
        "max_ms_per_tx": "",
        "total_seconds": preprocessing_row["total_seconds"] + forward_row["total_seconds"],
    }


def measure_inference_time(
    load_data: str,
    checkpoint_dir: str,
    fold: int,
    tags: str | None,
    device: str,
    warmup_batches: int = 3,
    repeats: int = 3,
    batch_sizes: list[int] = (1, 32),
    num_workers: int = 0,
    force_reload: bool = False,
) -> None:
    checkpoint_paths = _discover_checkpoint_paths(checkpoint_dir)
    if fold < 1 or fold > len(checkpoint_paths):
        raise ValueError(f"--fold {fold} out of range: found {len(checkpoint_paths)} checkpoints in '{checkpoint_dir}'.")
    checkpoint = load_checkpoint(checkpoint_paths[fold - 1], device)
    print(f"Loaded fold {fold} checkpoint from '{checkpoint_paths[fold - 1]}'.")

    dataset_type = checkpoint["dataset_type"]
    dataset = DATASET_CLASS[dataset_type](root=load_data, force_reload=force_reload)
    real_graphs = [g for g in dataset if not getattr(g, "synthetic", False)]
    print(f"Loaded {len(real_graphs)} real graphs from '{load_data}'.")

    model = build_model_from_checkpoint(checkpoint, device)

    reporter = TimingReporter(
        checkpoint_dir=checkpoint_dir,
        tags=tags,
        dataset_type=dataset_type,
        dataset_path=load_data,
        device=device,
        fold=fold,
        warmup_batches=warmup_batches,
        repeats=repeats,
    )

    rows = _time_preprocessing(real_graphs, checkpoint)
    preprocessing_row = rows[-1]
    for row in rows:
        print(f"{row['stage']}: {row['mean_ms_per_tx']:.4f} ms/tx (n={row['n_samples']})")

    for batch_size in batch_sizes:
        forward_row = _time_forward_pass(
            model, real_graphs, checkpoint, device, batch_size, warmup_batches, repeats, num_workers
        )
        rows.append(forward_row)
        rows.append(_end_to_end_row(preprocessing_row, forward_row))
        print(
            f"forward_batch{batch_size}: {forward_row['mean_ms_per_tx']:.4f} ms/tx "
            f"(n={forward_row['n_samples']} batches)"
        )

    reporter.save_timing_report(rows)
    print(f"\nTiming report saved to: {reporter.run_dir}")