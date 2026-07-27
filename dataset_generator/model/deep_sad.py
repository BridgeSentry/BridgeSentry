import numpy as np
import torch
from sklearn.metrics import precision_recall_curve

# Machine epsilon added to the denominator of the inverse term for numerical
# stability (see footnote 1 of the paper).
DEFAULT_EPS = 1e-6


def deep_sad_loss(
    z: torch.Tensor,
    center: torch.Tensor,
    y: torch.Tensor,
    eta: float = 1.0,
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    """
    Deep SAD objective, without weight-decay.
    Args:
        z: Latent representations, [B, rep_dim].
        center: The fixed hypersphere centre c, [rep_dim].
        y: Graph labels, [B]. 0 = normal/unlabeled, 1 = known anomaly.
        eta: Balances the labeled-anomaly term against the unlabeled term. eta=0
            drops the anomaly term entirely, degenerating to unsupervised Deep SVDD.
        eps: Denominator guard for the inverse term.

    Returns:
        Scalar loss, averaged over the batch (matching the paper's 1/(n+m)).
    """
    return deep_sad_loss_from_scores((z - center).pow(2).sum(dim=1), y, eta=eta, eps=eps)


def deep_sad_loss_from_scores(
    dist: torch.Tensor,
    y: torch.Tensor,
    eta: float = 1.0,
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    # Deep SAD objective from precomputed squared distances, [B].
    # Normal samples minimise the squared distance; anomalies minimise the inverse of the squared distance.
    anomaly_term = eta * (dist + eps).reciprocal()
    return torch.where(y == 0, dist, anomaly_term).mean()


def anomaly_scores(z: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    """
    Squared distance to the centre, [B]. Higher means more anomalous.

    We use the squared distance rather than the Euclidean norm as comparing
    distances is the same as comparing squared distances, and it's cheaper to compute.
    """
    return (z - center).pow(2).sum(dim=1)


def f2_optimal_threshold(labels, scores, beta: float = 2.0) -> float:
    """
    Return the score threshold maximising F-beta for the anomaly class.

    A distance score has no natural decision boundary the way a softmax argmax
    does, so the operating point is fitted on the validation fold and then carried
    in the checkpoint alongside the weights it was fitted for.

    Predictions are taken as `scores > tau`, so candidate thresholds come from
    `precision_recall_curve`, which reports the precision/recall attained at
    `scores >= thresholds[i]`.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)

    # Degenerate folds (single class present) have no meaningful operating point;
    # fall back to a threshold above every score, i.e. predict all-normal.
    if labels.size == 0 or len(np.unique(labels)) < 2:
        return float(scores.max()) + 1.0 if scores.size else 0.0

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    # precision/recall have one more entry than thresholds; the last pair is the
    # trivial (precision=1, recall=0) point which has no threshold.
    precision, recall = precision[:-1], recall[:-1]

    beta_sq = beta ** 2
    denom = beta_sq * precision + recall
    fbeta = np.divide(
        (1 + beta_sq) * precision * recall,
        denom,
        out=np.zeros_like(denom),
        where=denom > 0,
    )

    if fbeta.size == 0:
        return float(scores.max()) + 1.0

    # Nudge just below the chosen threshold so the `>` comparison used at predict
    # time reproduces the `>=` semantics of precision_recall_curve.
    return float(np.nextafter(thresholds[int(fbeta.argmax())], -np.inf))


def pretrain_autoencoder(*args, **kwargs):
    # TODO Currently unused, but we may want to implement this in the future in case performance is bad.
    raise NotImplementedError(
        "Autoencoder pre-training is not implemented."
    )
