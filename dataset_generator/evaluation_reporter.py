import csv
import math
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from dataset_generator.training_reporter import compute_dataset_stats

LABEL_NAMES = ["normal", "anomaly"]

# Mirrors TrainingReporter.TEST_METRIC_COLUMNS, with "test_loss" renamed to "loss"
# since this isn't necessarily the training run's own held-out test split.
EVAL_METRIC_COLUMNS = [
    "fold",
    "loss",
    "accuracy",
    "precision_normal",
    "recall_normal",
    "f1_normal",
    "precision_anomaly",
    "recall_anomaly",
    "f1_anomaly",
    "f2_anomaly",
    "roc_auc",
    "pr_auc",
    "mcc",
]


def _build_tag_suffix(tags: str | None) -> str:
    tag_list = [tag.strip() for tag in (tags or "").split(",") if tag.strip()]
    return "-".join(tag_list)


class EvaluationReporter:
    """Records per-fold evaluation results for an `eval` CLI run and persists them
    to a timestamped folder placed beside the checkpoint folder it evaluated."""

    def __init__(
        self,
        checkpoint_dir: str,
        tags: str | None,
        dataset_type: str,
        dataset_path: str,
        dataset_graphs: list,
        model_kwargs: dict,
    ):
        run_datetime = datetime.now()
        timestamp = run_datetime.strftime("%Y%m%d%H%M")
        tag_suffix = _build_tag_suffix(tags)
        run_name = f"inference_{tag_suffix}_{timestamp}" if tag_suffix else f"inference_{timestamp}"

        run_dir_parent = os.path.dirname(os.path.normpath(checkpoint_dir))
        self.run_dir = os.path.join(run_dir_parent, run_name)
        os.makedirs(self.run_dir, exist_ok=True)

        run_info = {
            "out_dir": self.run_dir,
            "run_datetime": run_datetime.isoformat(timespec="seconds"),
            "checkpoint_dir": checkpoint_dir,
        }
        dataset_info = {
            "type": dataset_type,
            "path": dataset_path,
            "stats": compute_dataset_stats(dataset_graphs),
        }
        document = {
            "run_info": run_info,
            "dataset": dataset_info,
            "params": dict(sorted(model_kwargs.items())),
        }
        with open(os.path.join(self.run_dir, "params.yaml"), "w") as f:
            yaml.dump(document, f, default_flow_style=False, sort_keys=False)

    def save_eval_stats(self, fold_metrics: list[dict]) -> None:
        path = os.path.join(self.run_dir, "eval_stats.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=EVAL_METRIC_COLUMNS)
            writer.writeheader()
            for entry in fold_metrics:
                row = {col: entry.get(col, "") for col in EVAL_METRIC_COLUMNS}
                row["fold"] = f"fold_{entry['fold']}"
                writer.writerow(row)

    def save_infer_report(
        self,
        bridges: list[str],
        tx_hashes: list[str],
        labels: np.ndarray,
        fold_preds: dict[int, np.ndarray],
    ) -> None:
        fold_numbers = sorted(fold_preds.keys())
        path = os.path.join(self.run_dir, "infer_report.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["bridge", "tx_hash", "true_label"] + [f"fold_{n}" for n in fold_numbers])
            for i, (bridge, tx_hash) in enumerate(zip(bridges, tx_hashes)):
                row = [bridge, tx_hash, int(labels[i])] + [int(fold_preds[n][i]) for n in fold_numbers]
                writer.writerow(row)

    def save_confusion_matrix(self, fold: int, labels: np.ndarray, preds: np.ndarray) -> None:
        cm = confusion_matrix(labels, preds, labels=[0, 1])
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(cm, display_labels=LABEL_NAMES).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"Fold {fold} — confusion matrix")
        fig.tight_layout()

        path = os.path.join(self.run_dir, f"confusion_fold_{fold}.svg")
        fig.savefig(path, format="svg")
        plt.close(fig)

    def save_bridge_confusion_matrices(
        self,
        fold: int,
        bridges: list[str],
        labels: np.ndarray,
        preds: np.ndarray,
    ) -> None:
        bridges_arr = np.array(bridges)
        unique_bridges = sorted(set(bridges))
        n = len(unique_bridges)
        ncols = math.ceil(math.sqrt(n))
        nrows = math.ceil(n / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
        for idx, bridge in enumerate(unique_bridges):
            ax = axes[idx // ncols][idx % ncols]
            mask = bridges_arr == bridge
            cm = confusion_matrix(labels[mask], preds[mask], labels=[0, 1])
            ConfusionMatrixDisplay(cm, display_labels=LABEL_NAMES).plot(ax=ax, cmap="Blues", colorbar=False)
            ax.set_title(f"{bridge} (n={int(mask.sum())})")

        # Hide unused subplots in the grid.
        for idx in range(n, nrows * ncols):
            axes[idx // ncols][idx % ncols].axis("off")

        fig.suptitle(f"Fold {fold} — confusion matrix per bridge")
        fig.tight_layout()

        path = os.path.join(self.run_dir, f"confusion_bridges_{fold}.svg")
        fig.savefig(path, format="svg")
        plt.close(fig)
