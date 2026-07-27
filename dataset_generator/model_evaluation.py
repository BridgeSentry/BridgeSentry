import glob
import os
import re

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)

from dataset_generator.evaluation_reporter import EvaluationReporter
from dataset_generator.model.deep_sad import DEFAULT_EPS, deep_sad_loss_from_scores
from dataset_generator.model_inference import build_model_from_checkpoint, load_checkpoint, predict, prepare_graphs
from dataset_generator.model_training import fbeta_score
from dataset_generator.types import DATASET_CLASS

_FOLD_FILE_RE = re.compile(r"fold_(\d+)\.pt$")


def _discover_checkpoint_paths(checkpoint_dir: str) -> list[str]:
    paths = glob.glob(os.path.join(checkpoint_dir, "fold_*.pt"))
    matches = [(int(_FOLD_FILE_RE.search(p).group(1)), p) for p in paths if _FOLD_FILE_RE.search(p)]
    if not matches:
        raise ValueError(f"No fold_*.pt checkpoints found in '{checkpoint_dir}'.")
    return [p for _, p in sorted(matches)]


def evaluate(
    load_data: str,
    checkpoint_dir: str,
    tags: str | None,
    device: str,
    batch_size: int = 32,
    num_workers: int = 0,
    force_reload: bool = False,
) -> None:
    checkpoint_paths = _discover_checkpoint_paths(checkpoint_dir)
    checkpoints = [load_checkpoint(p, device) for p in checkpoint_paths]
    print(f"Loaded {len(checkpoints)} fold checkpoints from '{checkpoint_dir}'.")

    # All fold checkpoints from the same training run share identical dataset_type,
    # metapaths, feature dims and normalization stats — only weights differ per fold.
    reference_checkpoint = checkpoints[0]
    dataset_type = reference_checkpoint["dataset_type"]

    dataset = DATASET_CLASS[dataset_type](root=load_data, force_reload=force_reload)
    real_graphs = [g for g in dataset if not getattr(g, "synthetic", False)]
    print(f"Loaded {len(real_graphs)} real graphs from '{load_data}'.")

    prepared_graphs = prepare_graphs(real_graphs, reference_checkpoint)
    labels = np.array([int(g.y.item()) for g in prepared_graphs])
    labels_t = torch.tensor(labels)
    bridges = [getattr(g, "bridge", "unknown") for g in prepared_graphs]
    tx_hashes = [g.tx_hash for g in prepared_graphs]

    reporter = EvaluationReporter(
        checkpoint_dir=checkpoint_dir,
        tags=tags,
        dataset_type=dataset_type,
        dataset_path=load_data,
        dataset_graphs=real_graphs,
        model_kwargs=reference_checkpoint["model_kwargs"],
    )

    fold_metrics = []
    fold_preds: dict[int, np.ndarray] = {}
    fold_scores: dict[int, np.ndarray] = {}

    for checkpoint in checkpoints:
        fold_num = checkpoint["fold"] + 1
        model = build_model_from_checkpoint(checkpoint, device)
        scores, preds = predict(model, prepared_graphs, checkpoint, device, batch_size, num_workers)
        preds_np = preds.numpy()
        scores_np = scores.numpy()
        fold_preds[fold_num] = preds_np
        fold_scores[fold_num] = scores_np

        # Deep SAD objective evaluated on this dataset. Computed from the scores
        # (which are already the squared distances the loss is built from), so it
        # needs no second forward pass.
        loss = deep_sad_loss_from_scores(
            scores, labels_t, eta=checkpoint.get("eta", 1.0), eps=checkpoint.get("deep_sad_eps", DEFAULT_EPS)
        ).item()

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds_np, labels=[0, 1], zero_division=0
        )
        acc = accuracy_score(labels, preds_np)
        mcc = matthews_corrcoef(labels, preds_np)
        # The score is a distance: higher already means more anomalous, so it feeds
        # the ranking metrics directly.
        roc_auc = roc_auc_score(labels, scores_np)
        pr_auc = average_precision_score(labels, scores_np)
        f2_anomaly = fbeta_score(precision[1], recall[1])

        fold_metrics.append({
            "fold": fold_num,
            "loss": loss,
            "threshold": checkpoint["threshold"],
            "accuracy": acc,
            "precision_normal": precision[0],
            "recall_normal": recall[0],
            "f1_normal": f1[0],
            "precision_anomaly": precision[1],
            "recall_anomaly": recall[1],
            "f1_anomaly": f1[1],
            "f2_anomaly": f2_anomaly,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "mcc": mcc,
        })

        reporter.save_confusion_matrix(fold_num, labels, preds_np)
        reporter.save_bridge_confusion_matrices(fold_num, bridges, labels, preds_np)

        print(f"Fold {fold_num}: accuracy={acc:.4f} roc_auc={roc_auc:.4f} pr_auc={pr_auc:.4f} mcc={mcc:.4f}")

    reporter.save_eval_stats(fold_metrics)
    reporter.save_infer_report(bridges, tx_hashes, labels, fold_preds, fold_scores)

    print(f"\nEvaluation report saved to: {reporter.run_dir}")
