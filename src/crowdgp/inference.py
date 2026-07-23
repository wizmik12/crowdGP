"""Training utilities.

Minibatching
------------
Only *item indices* are shuffled. Features, annotations and ``q(Z)`` rows are
all gathered from the same index vector downstream, so they cannot fall out of
alignment. The reference implementation instead constructs three separate
``Minibatch`` objects -- over ``X``, over the label array, and over the index
array -- kept in step only by passing each the same ``seed=0``. Nothing
enforces that agreement, and if it ever fails the model simply pairs features
with the wrong annotations and trains to a worse optimum, silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator

import numpy as np
import tensorflow as tf

from .data import CrowdLabels
from .models import GPCrowdModel

__all__ = ["batch_iterator", "train", "TrainingHistory"]


@dataclass
class TrainingHistory:
    """Per-iteration trace of the ELBO and each of its components.

    Storing the decomposition rather than the total alone is what makes a bad
    run diagnosable. The characteristic crowdsourcing failure is a rising ELBO
    in which ``crowd`` climbs steadily while ``latent`` stays flat: the model is
    fitting the annotators and ignoring the features, so it will predict
    nothing useful on unannotated data.

    Attributes:
        elbo (list[float]): Total scaled ELBO per iteration.
        latent (list[float]): Latent GP evidence term.
        crowd (list[float]): Crowdsourcing evidence term.
        entropy (list[float]): Entropy of ``q(Z)``.
        kl_latent (list[float]): ``KL(q(u) || p(u))``.
        kl_annotator (list[float]): ``KL(q(R) || p(R))``.
    """

    elbo: list[float] = field(default_factory=list)
    latent: list[float] = field(default_factory=list)
    crowd: list[float] = field(default_factory=list)
    entropy: list[float] = field(default_factory=list)
    kl_latent: list[float] = field(default_factory=list)
    kl_annotator: list[float] = field(default_factory=list)


def batch_iterator(num_items: int, batch_size: int | None, seed: int = 0) -> Iterator[tf.Tensor]:
    """Yields shuffled item-index batches indefinitely.

    Args:
        num_items: Dataset size ``N``.
        batch_size: Items per batch, or None for full batch.
        seed: RNG seed for the shuffle.

    Yields:
        tf.Tensor: Int32 index vectors, length ``batch_size``.

    Note:
        Epochs are reshuffled and the trailing partial batch is dropped, so
        every yielded batch has the same length. That keeps ``tf.function``
        from retracing on the last batch of each epoch -- retracing is correct
        but expensive, and doing it once per epoch is a common, invisible
        source of slow training.
    """
    rng = np.random.default_rng(seed)

    if batch_size is None or batch_size >= num_items:
        full = tf.constant(np.arange(num_items), dtype=tf.int32)
        while True:
            yield full

    while True:
        perm = rng.permutation(num_items)
        for start in range(0, num_items - batch_size + 1, batch_size):
            yield tf.constant(perm[start : start + batch_size], dtype=tf.int32)


def train(
    model: GPCrowdModel,
    X: np.ndarray,
    labels: CrowdLabels,
    iterations: int = 500,
    batch_size: int | None = None,
    learning_rate: float = 0.01,
    seed: int = 0,
    compile_graph: bool = True,
    callback: Callable[[int, float], None] | None = None,
) -> TrainingHistory:
    """Fits the model by maximising the ELBO with Adam.

    Args:
        model: The composed model.
        X: Feature matrix. Shape ``[N, D]``.
        labels: The annotations.
        iterations: Number of optimiser steps.
        batch_size: Items per step, or None for full batch.
        learning_rate: Adam learning rate.
        seed: Shuffling seed.
        compile_graph: Wrap the step in ``tf.function``. Leave True normally;
            set False when debugging, since eager execution gives readable
            tracebacks and lets you print intermediate tensors.
        callback: Optional ``(iteration, elbo)`` hook, for progress reporting
            or early stopping.

    Returns:
        TrainingHistory: The ELBO and component traces.

    Note:
        ``model.trainable_variables`` is collected fresh inside the step rather
        than captured once, because ``gpflow.Module`` discovers parameters by
        traversing attributes. Capturing the list outside would silently miss
        any component attached after construction.

        Converting each term to ``float`` forces a device sync every iteration.
        Fine at this scale, and worth the cost for the diagnostics; if you
        later train on GPU with thousands of fast steps, accumulate on device
        and transfer periodically instead.
    """
    X_tf = tf.constant(np.asarray(X, dtype=np.float64))
    opt = tf.optimizers.Adam(learning_rate)
    history = TrainingHistory()

    def step(idx: tf.Tensor):
        batch = labels.gather_batch(X_tf, idx)
        with tf.GradientTape() as tape:
            terms = model.elbo_terms(batch)
            loss = -terms.total
        variables = model.trainable_variables
        opt.apply_gradients(zip(tape.gradient(loss, variables), variables))
        return terms

    step_fn = tf.function(step) if compile_graph else step

    for i, idx in zip(range(iterations), batch_iterator(labels.num_items, batch_size, seed)):
        terms = step_fn(idx)
        history.elbo.append(float(terms.total))
        history.latent.append(float(terms.latent))
        history.crowd.append(float(terms.crowd))
        history.entropy.append(float(terms.entropy))
        history.kl_latent.append(float(terms.kl_latent))
        history.kl_annotator.append(float(terms.kl_annotator))
        if callback is not None:
            callback(i, float(terms.total))

    return history