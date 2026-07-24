"""Synthetic crowdsourcing data with known ground truth.

Not a convenience. Without a dataset whose true labels *and* true confusion
matrices are known, there is no way to distinguish a working crowd model from a
broken one: a broken model still produces a rising ELBO, still converges, and
still emits confident labels. Every claim about correctness in this library is
ultimately checked here.

The generator produces a deliberately *mixed* worker population -- some
reliable, some close to random -- because a model that cannot separate the two
has not done anything a vote count could not. It also produces features that
genuinely carry signal (clustered by true class), so the latent GP has
something to learn and the ``latent`` ELBO term can be expected to rise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import CrowdLabels

__all__ = ["SyntheticCrowd", "make_synthetic"]


@dataclass
class SyntheticCrowd:
    """A generated dataset together with the quantities used to generate it.

    Attributes:
        X (np.ndarray): Features. Shape ``[N, D]``.
        z (np.ndarray): True class per item. Shape ``[N]``. Never visible to
            the model; used only for evaluation.
        confusion (np.ndarray): True confusion matrices in this library's
            ``[A, C_obs, C_true]`` convention, columns summing to 1.
        labels (CrowdLabels): The observed sparse annotations -- all the model
            is allowed to see.
    """

    X: np.ndarray
    z: np.ndarray
    confusion: np.ndarray
    labels: CrowdLabels


def make_synthetic(
    num_items: int = 400,
    num_features: int = 2,
    num_classes: int = 3,
    num_workers: int = 20,
    labels_per_item: int = 4,
    good_worker_fraction: float = 0.6,
    good_accuracy: float = 0.85,
    cluster_separation: float = 3.0,
    seed: int = 0,
) -> SyntheticCrowd:
    """Generates clustered features, true labels, and noisy crowd annotations.

    Args:
        num_items: Number of items ``N``.
        num_features: Feature dimension ``D``.
        num_classes: Number of classes ``C``.
        num_workers: Number of annotators ``A``.
        labels_per_item: Annotations drawn per item, sampled without
            replacement so no worker labels the same item twice.
        good_worker_fraction: Share of workers that are reliable.
        good_accuracy: Diagonal probability for the reliable workers.
        cluster_separation: Distance between class centres relative to the unit
            within-class noise. Around 3.0 gives overlapping but learnable
            clusters; push it up to make the task trivial for the GP, down to
            make the annotations carry all the information.
        seed: RNG seed.

    Returns:
        SyntheticCrowd: Features, true labels, true confusions and annotations.

    Note:
        Unreliable workers are given Dirichlet-sampled confusion matrices
        rather than uniform ones. A uniform spammer is easy to detect -- their
        votes carry no signal at all -- whereas a worker with a random but
        *consistent* bias looks informative and is the case that actually
        separates a working annotator model from a vote count.

        Note ``confusion[a, :, z[n]]`` when sampling: the column indexes the
        true class, matching the library's convention. Sampling from the row
        instead generates data under the transposed model, which then appears
        as an unfixable accuracy ceiling rather than as an error.
    """
    rng = np.random.default_rng(seed)

    centres = rng.normal(scale=cluster_separation, size=(num_classes, num_features))
    z = rng.integers(0, num_classes, size=num_items)
    X = centres[z] + rng.normal(scale=1.0, size=(num_items, num_features))

    num_good = int(round(good_worker_fraction * num_workers))
    confusion = np.empty((num_workers, num_classes, num_classes))
    for a in range(num_workers):
        if a < num_good:
            mat = np.full((num_classes, num_classes), (1 - good_accuracy) / (num_classes - 1))
            np.fill_diagonal(mat, good_accuracy)
        else:
            mat = rng.dirichlet(np.ones(num_classes) * 2.0, size=num_classes).T
        confusion[a] = mat

    k = min(labels_per_item, num_workers)
    item_idx, worker_idx, label = [], [], []
    for n in range(num_items):
        for a in rng.choice(num_workers, size=k, replace=False):
            item_idx.append(n)
            worker_idx.append(a)
            label.append(rng.choice(num_classes, p=confusion[a, :, z[n]]))

    labels = CrowdLabels(
        item_idx=np.array(item_idx),
        worker_idx=np.array(worker_idx),
        label=np.array(label),
        num_items=num_items,
        worker_keys=np.arange(num_workers),
        class_keys=np.arange(num_classes),
    )
    return SyntheticCrowd(X=X, z=z, confusion=confusion, labels=labels)