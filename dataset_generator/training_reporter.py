import csv
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import yaml


class TrainingReporter:
    """Records per-epoch validation metrics and per-fold test metrics, then
    persists them to a timestamped run folder under `reports/`."""

    # Validation metrics recorded each epoch (matches keys in record_val_epoch)
    VAL_METRICS = [
        "train_loss",
        "val_loss",
        "accuracy",
        "precision_normal",
        "recall_normal",
        "f1_normal",
        "precision_anomaly",
        "recall_anomaly",
        "f1_anomaly",
        "mcc",
        "roc_auc",
        "pr_auc",
    ]

    # Test metrics recorded per fold (matches keys in record_test_fold)
    TEST_METRIC_COLUMNS = [
        "fold",
        "test_loss",
        "accuracy",
        "precision_normal",
        "recall_normal",
        "f1_normal",
        "precision_anomaly",
        "recall_anomaly",
        "f1_anomaly",
        "roc_auc",
        "pr_auc",
        "mcc",
    ]

    def __init__(self, reports_root: str, name_prefix: str, params: dict):
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        self.run_name = f"{name_prefix}_{timestamp}"
        self.run_dir = os.path.join(reports_root, self.run_name)
        self.charts_dir = os.path.join(self.run_dir, "charts")

        os.makedirs(self.charts_dir, exist_ok=True)

        with open(os.path.join(self.run_dir, "params.yaml"), "w") as f:
            yaml.dump(params, f, default_flow_style=False, sort_keys=True)

        # val_history[metric_name][fold_idx][epoch_idx] = float value
        # e.g. val_history["accuracy"][2][10] = 0.93  (fold 2, epoch 10)
        self.val_history: dict[str, dict[int, dict[int, float]]] = {
            m: {} for m in self.VAL_METRICS
        }

        # One dict per fold, same keys as TEST_METRIC_COLUMNS (minus "fold")
        self.test_metrics: list[dict] = []

    def record_val_epoch(self, fold: int, epoch: int, metrics: dict) -> None:
        for metric_name in self.VAL_METRICS:
            if metric_name not in self.val_history:
                self.val_history[metric_name] = {}
            if fold not in self.val_history[metric_name]:
                self.val_history[metric_name][fold] = {}
            self.val_history[metric_name][fold][epoch] = metrics[metric_name]

    def record_test_fold(self, fold: int, metrics: dict) -> None:
        self.test_metrics.append({"fold": fold, **metrics})

    def save(self, metric_display: list[tuple[str, str]], k_folds: int) -> None:
        self._save_val_csvs(k_folds)
        self._save_val_charts(k_folds)
        self._save_test_csv()
        self._save_summary_txt(metric_display)
        print(f"\nReport saved to: {self.run_dir}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    # Save per-epoch validation metrics to CSVs, one file per metric, with the
    # first column as epoch number and subsequent columns as fold values.
    def _save_val_csvs(self, k_folds: int) -> None:
        for metric in self.VAL_METRICS:
            fold_data = self.val_history[metric]
            if not fold_data:
                continue

            max_epoch = max(
                max(epochs.keys()) for epochs in fold_data.values()
            )
            header = ["epoch"] + [f"fold_{f + 1}" for f in range(k_folds)]
            rows = []
            for epoch in range(max_epoch + 1):
                row = [epoch + 1]
                for fold in range(k_folds):
                    row.append(fold_data.get(fold, {}).get(epoch, ""))
                rows.append(row)

            path = os.path.join(self.run_dir, f"val_{metric}.csv")
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(rows)

    # Generate and save line charts for each validation metric, with epochs on
    # the x-axis and metric values on the y-axis, one line per fold.
    def _save_val_charts(self, k_folds: int) -> None:
        for metric in self.VAL_METRICS:
            fold_data = self.val_history[metric]
            if not fold_data:
                continue

            fig, ax = plt.subplots()
            for fold in range(k_folds):
                if fold not in fold_data:
                    continue
                epochs_dict = fold_data[fold]
                sorted_epochs = sorted(epochs_dict.keys())
                x = [e + 1 for e in sorted_epochs]
                y = [epochs_dict[e] for e in sorted_epochs]
                ax.plot(x, y, label=f"Fold {fold + 1}")

            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_title(f"{metric.replace('_', ' ').title()} per Epoch")
            ax.legend()
            fig.tight_layout()

            path = os.path.join(self.charts_dir, f"val_{metric}.svg")
            fig.savefig(path, format="svg")
            plt.close(fig)

    # Save per-fold test metrics to a single CSV, with one row per fold and columns for each metric.
    def _save_test_csv(self) -> None:
        path = os.path.join(self.run_dir, "test_metrics.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.TEST_METRIC_COLUMNS)
            writer.writeheader()
            for entry in self.test_metrics:
                row = {col: entry.get(col, "") for col in self.TEST_METRIC_COLUMNS}
                # Use 1-based fold label for readability
                row["fold"] = f"fold_{int(entry['fold']) + 1}"
                writer.writerow(row)

    # Register the output print lines for the test summary into a text file.
    def _save_summary_txt(self, metric_display: list[tuple[str, str]]) -> None:
        lines = ["=== K-Fold test summary (mean ± std across folds) ===\n"]
        for key, label in metric_display:
            values = [m[key] for m in self.test_metrics if key in m]
            if values:
                lines.append(
                    f"  {label:<26} {np.mean(values):.4f} ± {np.std(values):.4f}\n"
                )

        path = os.path.join(self.run_dir, "summary.txt")
        with open(path, "w") as f:
            f.writelines(lines)
