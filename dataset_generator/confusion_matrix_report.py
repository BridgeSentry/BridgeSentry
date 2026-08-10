import math
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

LABEL_NAMES = ["normal", "anomaly"]


# Shared confusion-matrix plotting, used by both EvaluationReporter (standalone
# `eval` CLI runs) and TrainingReporter (final test-set evaluation right after
# training), so the two commands render identical figures into their own run_dir.
def plot_confusion_matrix(run_dir: str, fold: int, labels: np.ndarray, preds: np.ndarray) -> None:
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(cm, display_labels=LABEL_NAMES).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Fold {fold} — confusion matrix")
    fig.tight_layout()

    path = os.path.join(run_dir, f"confusion_fold_{fold}.svg")
    fig.savefig(path, format="svg")
    plt.close(fig)


def plot_bridge_confusion_matrices(
    run_dir: str,
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

    path = os.path.join(run_dir, f"confusion_bridges_{fold}.svg")
    fig.savefig(path, format="svg")
    plt.close(fig)
