"""Backfill val_best_metrics.csv for a training report that predates it.

Usage:
    python scripts/generate_val_best.py <report_folder>

Replays the same sequential F2 + PR-AUC tiebreak comparison that
EarlyStopping (dataset_generator/early_stopping.py) uses during training,
epoch by epoch over the already-recorded val_f2_anomaly.csv / val_pr_auc.csv,
to find which epoch's weights were actually selected as the best checkpoint
for each fold. That epoch's values are then read out of the other
val_<metric>.csv files to build one row per fold, in the same column layout
as test_metrics.csv (with "test_loss" renamed "val_loss") plus "epoch".

The column set is read from the report's own test_metrics.csv header, so
this works unmodified on both DeepSAD-branch reports (which have a
"threshold" column) and MLP-branch reports (which don't).
"""
import argparse
import csv
import os
import sys

import pandas as pd
import yaml

# Must match the `delta` literal passed to EarlyStopping(...) in
# dataset_generator/model_training.py.
EARLY_STOPPING_DELTA = 0.00001


def select_best_epoch(epochs, f2_values, pr_auc_values) -> int:
    """Replay EarlyStopping.__call__'s comparison logic (dataset_generator/
    early_stopping.py) over one fold's recorded epochs, in order, to find
    which epoch its best checkpoint came from.
    """
    best_f2 = best_tiebreak = best_epoch = None
    for epoch, f2, pr_auc in zip(epochs, f2_values, pr_auc_values):
        if best_f2 is None:
            best_f2, best_tiebreak, best_epoch = f2, pr_auc, epoch
        elif f2 > best_f2 + EARLY_STOPPING_DELTA:
            best_f2, best_tiebreak, best_epoch = f2, pr_auc, epoch
        elif f2 >= best_f2 - EARLY_STOPPING_DELTA and pr_auc > best_tiebreak + EARLY_STOPPING_DELTA:
            best_tiebreak, best_epoch = pr_auc, epoch
        # else: non-improving epoch — best_epoch unchanged. (During real
        # training this would increment the patience counter; we don't need
        # to track that here, since the CSVs already only contain the
        # epochs that actually ran.)
    return best_epoch


def main(report_folder: str) -> None:
    if not os.path.isdir(report_folder):
        sys.exit(f"Error: {report_folder} is not a directory.")

    required = ["test_metrics.csv", "val_f2_anomaly.csv", "val_pr_auc.csv"]
    missing = [f for f in required if not os.path.isfile(os.path.join(report_folder, f))]
    if missing:
        sys.exit(f"Error: {report_folder} is missing required file(s): {', '.join(missing)}")

    test_metrics = pd.read_csv(os.path.join(report_folder, "test_metrics.csv"))
    metric_columns = [c for c in test_metrics.columns if c != "fold"]
    val_metric_names = ["val_loss" if c == "test_loss" else c for c in metric_columns]

    f2_df = pd.read_csv(os.path.join(report_folder, "val_f2_anomaly.csv")).set_index("epoch")
    pr_auc_df = pd.read_csv(os.path.join(report_folder, "val_pr_auc.csv")).set_index("epoch")

    patience = None
    params_path = os.path.join(report_folder, "params.yaml")
    if os.path.isfile(params_path):
        with open(params_path) as f:
            params_doc = yaml.safe_load(f) or {}
        patience = params_doc.get("params", {}).get("early_stopping")

    # Read every metric CSV once, indexed by epoch, for row lookups below.
    metric_dfs = {}
    for metric_name in val_metric_names:
        path = os.path.join(report_folder, f"val_{metric_name}.csv")
        if not os.path.isfile(path):
            sys.exit(f"Error: {report_folder} is missing {os.path.basename(path)} "
                      f"(needed for column '{metric_name}').")
        metric_dfs[metric_name] = pd.read_csv(path).set_index("epoch")

    fold_columns = sorted(
        (c for c in f2_df.columns if c.startswith("fold_")),
        key=lambda c: int(c.split("_")[1]),
    )

    out_columns = ["fold", "epoch"] + val_metric_names
    rows = []
    for fold_col in fold_columns:
        f2_fold = f2_df[fold_col].dropna()
        pr_auc_fold = pr_auc_df[fold_col].dropna()
        if f2_fold.empty:
            continue

        best_epoch = select_best_epoch(f2_fold.index, f2_fold.values, pr_auc_fold.values)
        trained_epochs = int(f2_fold.index.max())
        patience_str = patience if patience is not None else "unknown"
        print(f"{fold_col}: selected epoch {best_epoch} of {trained_epochs} trained "
              f"(patience={patience_str})")

        row = {"fold": fold_col, "epoch": int(best_epoch)}
        for metric_name in val_metric_names:
            row[metric_name] = metric_dfs[metric_name].loc[best_epoch, fold_col]
        rows.append(row)

    out_path = os.path.join(report_folder, "val_best_metrics.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_folder", help="Path to a training run's report directory")
    args = parser.parse_args()
    main(args.report_folder)
