"""Evaluation metrics for crowdsourced classification models.

Every model in this repo is compared on the same quantities.  This module
provides one canonical implementation so that experiment scripts don't
duplicate scoring logic.

Functions accept plain ``numpy`` arrays (or anything array-like) and return
Python scalars or 1-D numpy arrays.
"""

from __future__ import annotations

from typing import Any

import numpy as np


__all__ = [
    "label_accuracy",
    "balanced_accuracy",
    "confusion_error",
    "expected_calibration_error",
    "cross_entropy",
    "worker_accuracy_correlation",
]


# -----------------------------------------------------------------------
# Classification metrics
# -----------------------------------------------------------------------


def label_accuracy(pred: Any, true: Any) -> float:
    """Fraction of items whose inferred label is correct.

    Args:
        pred: Array of predicted labels, shape ``(N,)``.
        true: Array of ground-truth labels, shape ``(N,)``.

    Returns:
        float: Accuracy in ``[0, 1]``.
    """
    pred = np.asarray(pred)
    true = np.asarray(true)
    if pred.shape != true.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs true {true.shape}")
    if pred.size == 0:
        return 0.0
    return float(np.mean(pred == true))


def balanced_accuracy(pred: Any, true: Any) -> float:
    """Mean per-class recall.

    Crowd datasets are usually class-imbalanced; plain accuracy hides this.
    For each class *c*, compute recall = TP_c / (TP_c + FN_c), then average
    over classes.  Classes present in *true* but never predicted still count
    (recall 0).

    Args:
        pred: Array of predicted labels, shape ``(N,)``.
        true: Array of ground-truth labels, shape ``(N,)``.

    Returns:
        float: Balanced accuracy in ``[0, 1]``.
    """
    pred = np.asarray(pred)
    true = np.asarray(true)
    if pred.shape != true.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs true {true.shape}")
    classes = np.unique(true)
    recalls: list[float] = []
    for c in classes:
        mask = true == c
        if not np.any(mask):
            continue
        recalls.append(float(np.mean(pred[mask] == c)))
    # Classes that exist in true but never get predicted contribute 0
    n_classes = max(len(classes), len(np.unique(pred)))
    while len(recalls) < n_classes:
        recalls.append(0.0)
    return float(np.mean(recalls)) if recalls else 0.0


def confusion_error(estimated: Any, true: Any) -> float:
    """Mean absolute error over per-worker confusion-matrix entries.

    Each confusion matrix has shape ``(3, 3)`` with axes
    ``[A, C_obs, C_true]`` where the three columns represent the latent
    class assigned to an item by the annotator's response pattern.

    Args:
        estimated: Estimated confusion array, shape ``(W, 3, 3)`` or
            flattened ``(W, 9)``.
        true: Ground-truth confusion array, same shape as *estimated*.

    Returns:
        float: Mean absolute error across all entries.
    """
    est = np.asarray(estimated, dtype=np.float64).ravel()
    tru = np.asarray(true, dtype=np.float64).ravel()
    if est.shape != tru.shape:
        raise ValueError(f"Shape mismatch: estimated {est.shape} vs true {tru.shape}")
    if est.size == 0:
        return 0.0
    return float(np.mean(np.abs(est - tru)))


# -----------------------------------------------------------------------
# Probabilistic / calibration metrics
# -----------------------------------------------------------------------


def expected_calibration_error(
    probs: Any,
    true: Any,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE).

    A model that is 90 % confident should be right 90 % of the time.  For
    crowdsourcing this often matters more than accuracy because the posterior
    is used downstream (e.g. active learning acquisition).

    Bins predictions by their max confidence, then computes the weighted
    absolute difference between average confidence and average accuracy in each
    bin.

    Args:
        probs: Predicted probabilities, shape ``(N, C)``.
        true: Ground-truth labels, shape ``(N,)``.
        n_bins: Number of equal-width bins (default 10).

    Returns:
        float: ECE in ``[0, 1]``.
    """
    probs = np.asarray(probs, dtype=np.float64)
    true = np.asarray(true)

    if probs.ndim == 1:
        # Binary classification: treat positive-class probability
        confidences = probs
    else:
        confidences = np.max(probs, axis=1)

    predictions = np.argmax(probs, axis=1) if probs.ndim > 1 else (probs >= 0.5).astype(int)
    accuracies = predictions == true

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        count = int(np.sum(mask))
        if count == 0:
            continue
        avg_conf = float(np.mean(confidences[mask]))
        avg_acc = float(np.mean(accuracies[mask]))
        ece += abs(avg_conf - avg_acc) * count

    total = probs.shape[0]
    return ece / total if total > 0 else 0.0


def cross_entropy(probs: Any, true: Any) -> float:
    """Cross-entropy loss between predicted probabilities and true labels.

    Args:
        probs: Predicted probabilities, shape ``(N, C)``.
        true: Ground-truth labels, shape ``(N,)`` as integer class indices.

    Returns:
        float: Mean cross-entropy (nats).
    """
    probs = np.asarray(probs, dtype=np.float64)
    true = np.asarray(true, dtype=int)

    N = probs.shape[0]
    if N == 0:
        return 0.0

    # Clip to avoid log(0)
    eps = 1e-12
    probs_clipped = np.clip(probs, eps, 1.0)

    # Extract probability of the true class
    if probs.ndim > 1:
        true_probs = probs_clipped[np.arange(N), true]
    else:
        # Binary case
        true_probs = np.where(true == 1, probs_clipped, 1.0 - probs_clipped)

    return float(-np.mean(np.log(true_probs)))


# -----------------------------------------------------------------------
# Worker-level metrics
# -----------------------------------------------------------------------


def worker_accuracy_correlation(
    estimated_confusion: Any,
    true_confusion: Any,
) -> float:
    """Pearson correlation between estimated and true worker confusion matrices.

    Flattens each worker's ``(3, 3)`` confusion matrix to a vector and
    correlates the estimated vector with the ground-truth vector.

    Args:
        estimated_confusion: Shape ``(W, 3, 3)`` or ``(W, 9)``.
        true_confusion: Same shape as *estimated_confusion*.

    Returns:
        float: Pearson correlation coefficient in ``[-1, 1]``.
    """
    est = np.asarray(estimated_confusion, dtype=np.float64)
    tru = np.asarray(true_confusion, dtype=np.float64)

    if est.size == 0:
        return 0.0

    est = est.reshape(len(est), -1)
    tru = tru.reshape(len(tru), -1)

    if est.shape != tru.shape:
        raise ValueError(f"Shape mismatch: estimated {est.shape} vs true {tru.shape}")
    if est.shape[0] == 0:
        return 0.0

    # Per-worker correlation, then mean
    correlations: list[float] = []
    for w in range(est.shape[0]):
        e_w = est[w].ravel()
        t_w = tru[w].ravel()
        # Skip workers with zero-variance confusion matrices
        if np.std(e_w) < 1e-10 or np.std(t_w) < 1e-10:
            continue
        corr_mat = np.corrcoef(e_w, t_w)
        correlations.append(float(corr_mat[0, 1]))

    return float(np.mean(correlations)) if correlations else 0.0
