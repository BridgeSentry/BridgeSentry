import csv
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import yaml

from repository.db.graph_label import GraphNodeType, GraphEdgeType

_NODE_TYPE_SHORT_NAMES = {
    GraphNodeType.USER.value: "U",
    GraphNodeType.ROUTER.value: "R",
    GraphNodeType.TOKEN.value: "T",
    GraphNodeType.OTHER_ACCOUNT.value: "O",
    GraphNodeType.LOG_EVENT.value: "L",
    GraphNodeType.VALIDATOR.value: "V",
}
_EDGE_TYPE_SHORT_NAMES = {
    GraphEdgeType.TRANSACTION.value: "x",
    GraphEdgeType.TOKEN_TRANSFER.value: "t",
    GraphEdgeType.TOKEN_AUTH.value: "a",
    GraphEdgeType.FUNCTION_CALL.value: "f",
    GraphEdgeType.LOG_RELATION.value: "l",
    GraphEdgeType.CROSS_CHAIN_RELATION.value: "v",
}


def build_metapath_short_name(metapath) -> str:
    parts = [_NODE_TYPE_SHORT_NAMES[metapath[0][0]]]
    for _src, edge, dst in metapath:
        parts.append(_EDGE_TYPE_SHORT_NAMES[edge])
        parts.append(_NODE_TYPE_SHORT_NAMES[dst])
    return "".join(parts)


def compute_dataset_stats(graphs: list) -> dict:
    """Compute normal/anomaly sample counts and ratios, overall and per bridge.

    Each graph is expected to expose a `.y` tensor (0 = normal, 1 = anomaly)
    and a `.bridge` attribute naming the bridge it was generated from.
    """
    total_samples = len(graphs)
    normal_samples = sum(1 for g in graphs if int(g.y.item()) == 0)
    anomaly_samples = total_samples - normal_samples

    by_bridge: dict[str, dict[str, int]] = {}
    for g in graphs:
        bridge = getattr(g, "bridge", "unknown")
        counts = by_bridge.setdefault(
            bridge, {"num_samples": 0, "normal_samples": 0, "anomaly_samples": 0}
        )
        counts["num_samples"] += 1
        if int(g.y.item()) == 0:
            counts["normal_samples"] += 1
        else:
            counts["anomaly_samples"] += 1

    bridges = []
    for name, c in sorted(
        by_bridge.items(), key=lambda item: item[1]["num_samples"], reverse=True
    ):
        bridges.append(
            {
                "name": name,
                "num_samples": c["num_samples"],
                "normal_samples": c["normal_samples"],
                "anomaly_samples": c["anomaly_samples"],
                "anomaly_ratio": round(c["anomaly_samples"] / c["num_samples"], 4),
                "bridge_ratio": round(c["num_samples"] / total_samples, 4),
            }
        )

    return {
        "total_samples": total_samples,
        "normal_samples": normal_samples,
        "anomaly_samples": anomaly_samples,
        "anomaly_ratio": round(anomaly_samples / total_samples, 4) if total_samples else 0.0,
        "bridges": bridges,
    }

class TrainingReporter:
    """Records per-epoch validation metrics and per-fold test metrics, then
    persists them to a timestamped run folder."""

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
        "f2_anomaly",
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
        "f2_anomaly",
        "roc_auc",
        "pr_auc",
        "mcc",
    ]

    def __init__(
        self,
        reports_root: str,
        name_prefix: str,
        params: dict,
        k_folds: int,
        dataset_type: str,
        dataset_path: str,
        dataset_graphs: list,
        dme_threshold: float,
        reported_metapaths: int,
    ):
        run_datetime = datetime.now()
        timestamp = run_datetime.strftime("%Y%m%d%H%M")
        self.run_name = f"{name_prefix}_{timestamp}"
        self.run_dir = os.path.join(reports_root, self.run_name)
        self.charts_dir = os.path.join(self.run_dir, "charts")
        self.k_folds = k_folds

        os.makedirs(self.charts_dir, exist_ok=True)

        run_info = {
            "out_dir": self.run_dir,
            "run_datetime": run_datetime.isoformat(timespec="seconds"),
        }
        dataset_info = {
            "type": dataset_type,
            "path": dataset_path,
            "stats": compute_dataset_stats(dataset_graphs),
        }
        dme_info = {
            "dme_threshold": dme_threshold,
            "reported_metapaths": reported_metapaths,
        }
        document = {
            "run_info": run_info,
            "params": dict(sorted(params.items())),
            "dataset": dataset_info,
            "dme": dme_info,
        }
        with open(os.path.join(self.run_dir, "params.yaml"), "w") as f:
            yaml.dump(document, f, default_flow_style=False, sort_keys=False)

        # val_history[metric_name][fold_idx][epoch_idx] = float value
        # e.g. val_history["accuracy"][2][10] = 0.93  (fold 2, epoch 10)
        self.val_history: dict[str, dict[int, dict[int, float]]] = {
            m: {} for m in self.VAL_METRICS
        }

        # One dict per fold, same keys as TEST_METRIC_COLUMNS (minus "fold")
        self.test_metrics: list[dict] = []

        # fold_durations[fold_idx] = wall-clock seconds spent training that fold
        self.fold_durations: dict[int, float] = {}

    def record_val_epoch(self, fold: int, epoch: int, metrics: dict) -> None:
        for metric_name in self.VAL_METRICS:
            if metric_name not in self.val_history:
                self.val_history[metric_name] = {}
            if fold not in self.val_history[metric_name]:
                self.val_history[metric_name][fold] = {}
            self.val_history[metric_name][fold][epoch] = metrics[metric_name]
        self._save_val_csvs()

    def record_test_fold(self, fold: int, metrics: dict) -> None:
        self.test_metrics.append({"fold": fold, **metrics})
        self._save_test_csv()

    def record_fold_duration(self, fold: int, duration_seconds: float) -> None:
        self.fold_durations[fold] = duration_seconds
        self._save_fold_duration_csv()

    def save_metapath_report(self, metapaths: list, differential_values: dict) -> None:
        entries = [
            {
                "metapath": build_metapath_short_name(mp),
                "difference": round(differential_values[mp], 4),
            }
            for mp in metapaths
        ]
        entries.sort(key=lambda e: abs(e["difference"]), reverse=True)

        path = os.path.join(self.run_dir, "dme_report.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["metapath", "difference"])
            writer.writeheader()
            writer.writerows(entries)

    def flush_charts(self) -> None:
        """Regenerate validation charts from the current in-memory history.

        Call after each fold completes so charts on disk reflect progress
        without waiting for the whole run to finish.
        """
        self._save_val_charts()

    def save(self, metric_display: list[tuple[str, str]]) -> None:
        self.flush_charts()
        self._save_summary_txt(metric_display)
        print(f"\nReport saved to: {self.run_dir}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    # Save per-epoch validation metrics to CSVs, one file per metric, with the
    # first column as epoch number and subsequent columns as fold values.
    def _save_val_csvs(self) -> None:
        for metric in self.VAL_METRICS:
            fold_data = self.val_history[metric]
            if not fold_data:
                continue

            max_epoch = max(
                max(epochs.keys()) for epochs in fold_data.values()
            )
            header = ["epoch"] + [f"fold_{f + 1}" for f in range(self.k_folds)]
            rows = []
            for epoch in range(max_epoch + 1):
                row = [epoch + 1]
                for fold in range(self.k_folds):
                    row.append(fold_data.get(fold, {}).get(epoch, ""))
                rows.append(row)

            path = os.path.join(self.run_dir, f"val_{metric}.csv")
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(rows)

    # Generate and save line charts for each validation metric, with epochs on
    # the x-axis and metric values on the y-axis, one line per fold.
    def _save_val_charts(self) -> None:
        for metric in self.VAL_METRICS:
            fold_data = self.val_history[metric]
            if not fold_data:
                continue

            fig, ax = plt.subplots()
            for fold in range(self.k_folds):
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

    # Save per-fold training durations to a single CSV, with one row per fold.
    def _save_fold_duration_csv(self) -> None:
        path = os.path.join(self.run_dir, "fold_durations.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["fold", "time"])
            for fold in sorted(self.fold_durations):
                writer.writerow([f"fold_{fold + 1}", f"{self.fold_durations[fold]:.2f}"])

    # Register the output print lines for the test summary into a text file.
    def _save_summary_txt(self, metric_display: list[tuple[str, str]]) -> None:
        lines = ["=== K-Fold test summary (mean ± std across folds) ===\n"]
        for key, label in metric_display:
            values = [m[key] for m in self.test_metrics if key in m]
            if values:
                lines.append(
                    f"  {label:<26} {np.mean(values):.4f} ± {np.std(values):.4f}\n"
                )

        if self.fold_durations:
            durations = list(self.fold_durations.values())
            lines.append(
                f"  {'Avg fold training time':<26} "
                f"{np.mean(durations):.2f}s ± {np.std(durations):.2f}s\n"
            )

        path = os.path.join(self.run_dir, "summary.txt")
        with open(path, "w") as f:
            f.writelines(lines)
