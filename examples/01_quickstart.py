"""Quickstart: fit a crowdsourcing GP and check it against majority vote.

This is the sanity check every model in this library must clear before it is
trusted on real data: on a dataset where votes alone are not enough (only half
the workers reliable, three annotations per item), does combining the worker
model with the latent classifier actually beat a plain vote count?

Run with::

    ./run.sh examples/01_quickstart.py
"""

from __future__ import annotations

import gpflow
import numpy as np
import tensorflow as tf

from gpcrowdkit import (
    FreeCategoricalZ,
    GPCrowdModel,
    SVGPLatent,
    VariationalDirichletAnnotator,
    init_alpha_tilde,
    make_synthetic,
    train,
)


def accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.asarray(pred) == np.asarray(true)))


def main() -> None:
    gpflow.config.set_default_float(np.float64)

    # A deliberately hard population: only half the workers are reliable, and
    # each item only gets three votes, so majority vote has real room to fail.
    data = make_synthetic(
        num_items=300,
        num_features=2,
        num_classes=3,
        num_workers=10,
        labels_per_item=3,
        good_worker_fraction=0.5,
        seed=1,
    )
    labels = data.labels
    print(labels)

    # Initialise q(Z) and the annotator concentrations from the vote
    # histogram rather than uniformly -- see synthetic.py and posteriors.py
    # for why a symmetric start risks the non-convex objective settling on a
    # self-consistent permutation of the true labelling.
    class_probs = labels.empirical_class_probs()

    model = GPCrowdModel(
        latent=SVGPLatent(
            kernel=gpflow.kernels.SquaredExponential(lengthscales=2.0),
            num_classes=labels.num_classes,
            inducing_points=data.X[:25].copy(),
        ),
        annotator=VariationalDirichletAnnotator(
            labels.num_workers,
            labels.num_classes,
            alpha_tilde_init=init_alpha_tilde(labels, class_probs),
        ),
        num_data=labels.num_items,
        q_z=FreeCategoricalZ(labels.num_items, labels.num_classes, init_probs=class_probs),
    )

    def report(iteration: int, elbo: float) -> None:
        if iteration % 50 == 0:
            print(f"  iter {iteration:4d}   elbo {elbo:12.2f}")

    print("\nTraining ...")
    history = train(model, data.X, labels, iterations=300, learning_rate=0.05, callback=report)

    mv_acc = accuracy(labels.majority_vote(), data.z)
    model_acc = accuracy(model.infer_true_labels(tf.constant(data.X), labels), data.z)

    print("\nELBO decomposition, first vs last 10 iterations:")
    for name, series in [
        ("latent", history.latent),
        ("crowd", history.crowd),
        ("entropy", history.entropy),
        ("kl_latent", history.kl_latent),
        ("kl_annotator", history.kl_annotator),
    ]:
        print(f"  {name:12s} {np.mean(series[:10]):12.2f} -> {np.mean(series[-10:]):12.2f}")

    print("\nAccuracy against the (normally hidden) true labels:")
    print(f"  majority vote : {mv_acc:.3f}")
    print(f"  gpcrowdkit model : {model_acc:.3f}")

    est_confusion = model.annotator.confusion_matrices().numpy()
    confusion_mae = np.abs(est_confusion - data.confusion).mean()
    print(f"\nMean absolute error of recovered worker confusion matrices: {confusion_mae:.3f}")

    assert model_acc > mv_acc, "sanity check failed: the model did not beat majority vote"
    print("\nSanity check passed: the model beat majority vote.")


if __name__ == "__main__":
    main()
