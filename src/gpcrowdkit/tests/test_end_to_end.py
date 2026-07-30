"""End-to-end tests: does the assembled model actually work?

The unit tests elsewhere check that each component computes what it claims.
These check the only thing that ultimately matters -- that putting them
together recovers labels a vote count could not, and recovers the worker
confusion matrices that generated the data.

Both are properties no amount of correct-looking code guarantees. A model with
a transposed confusion convention, an unscaled KL term, or misaligned batches
trains happily, produces a rising ELBO, and fails here.

These are slow by unit-test standards -- a few hundred optimiser steps each.
Mark them and skip them on quick runs::

    pytest -m "not slow"
"""

from __future__ import annotations

import gpflow
import numpy as np
import pytest
import tensorflow as tf

from gpcrowdkit import (
    FreeCategoricalZ,
    GPCrowdModel,
    OneCoinAnnotator,
    SoftmaxPointAnnotator,
    SVGPLatent,
    VariationalDirichletAnnotator,
    init_alpha_tilde,
    make_synthetic,
    train,
)

gpflow.config.set_default_float(np.float64)

ANNOTATORS = [VariationalDirichletAnnotator, SoftmaxPointAnnotator, OneCoinAnnotator]


def build(data, annotator_cls=VariationalDirichletAnnotator, num_inducing=25, **kw):
    """Assembles a model over a synthetic dataset, initialised from the votes.

    Both the annotator concentrations and ``q(Z)`` start from the smoothed vote
    histogram, as in the reference implementation. From a uniform start the
    non-convex objective can settle on a permutation of the true labelling --
    self-consistent, high-ELBO, and useless -- which would make these tests
    flaky for reasons unrelated to the code under test.
    """
    labels = data.labels
    probs = labels.empirical_class_probs()

    if annotator_cls is VariationalDirichletAnnotator:
        annot = annotator_cls(
            labels.num_workers,
            labels.num_classes,
            alpha_tilde_init=init_alpha_tilde(labels, probs),
        )
    else:
        annot = annotator_cls(labels.num_workers, labels.num_classes)

    return GPCrowdModel(
        latent=SVGPLatent(
            kernel=gpflow.kernels.SquaredExponential(lengthscales=2.0),
            num_classes=labels.num_classes,
            inducing_points=data.X[:num_inducing].copy(),
            **kw,
        ),
        annotator=annot,
        num_data=labels.num_items,
        q_z=FreeCategoricalZ(labels.num_items, labels.num_classes, init_probs=probs),
    )


def accuracy(pred, true) -> float:
    return float(np.mean(np.asarray(pred) == np.asarray(true)))


# ------------------------------------------------------------------- smoke


@pytest.mark.parametrize("cls", ANNOTATORS)
def test_elbo_is_a_finite_scalar(cls):
    """Every strategy must assemble into a usable objective."""
    data = make_synthetic(num_items=80, num_workers=6, seed=0)
    model = build(data, cls, num_inducing=10)
    elbo = model.elbo(data.labels.full_batch(tf.constant(data.X)))
    assert elbo.shape == ()
    assert np.isfinite(float(elbo))


def test_minibatch_elbo_is_unbiased():
    """The N/B scale must make a batch estimate the full-data ELBO.

    A KL term scaled along with the data terms would show up here as a large
    systematic gap, which is otherwise nearly impossible to spot: the loss
    curve looks normal and the model merely underfits.
    """
    data = make_synthetic(num_items=200, num_workers=10, seed=3)
    model = build(data, num_inducing=15)
    X = tf.constant(data.X)

    full = float(model.elbo(data.labels.full_batch(X)))
    rng = np.random.default_rng(0)
    ests = [
        float(
            model.elbo(
                data.labels.gather_batch(
                    X, tf.constant(rng.choice(200, 50, replace=False), tf.int32)
                )
            )
        )
        for _ in range(40)
    ]
    assert abs(np.mean(ests) - full) < 0.15 * abs(full)


@pytest.mark.parametrize("cls", ANNOTATORS)
def test_compiles_under_tf_function(cls):
    """Two batch sizes force a retrace, where static-shape assumptions surface."""
    data = make_synthetic(num_items=80, num_workers=6, seed=0)
    model = build(data, cls, num_inducing=10)
    X = tf.constant(data.X)

    @tf.function
    def compiled(idx):
        return model.elbo(data.labels.gather_batch(X, idx))

    assert np.isfinite(float(compiled(tf.constant([0, 1, 2, 3], tf.int32))))
    assert np.isfinite(float(compiled(tf.constant([4, 5, 6, 7, 8], tf.int32))))


# ----------------------------------------------------------------- learning


@pytest.mark.slow
def test_elbo_increases():
    data = make_synthetic(num_items=200, num_workers=10, seed=1)
    model = build(data, num_inducing=15)
    hist = train(model, data.X, data.labels, iterations=100, learning_rate=0.05)
    assert np.mean(hist.elbo[-10:]) > np.mean(hist.elbo[:10])


@pytest.mark.slow
def test_latent_term_actually_improves():
    """The GP must learn something, not merely let the annotator term carry it.

    This is the failure the total ELBO hides: ``crowd`` climbs, ``latent`` stays
    flat, the model reproduces the annotations and generalises to nothing.
    """
    data = make_synthetic(num_items=300, num_workers=10, seed=4)
    model = build(data)
    hist = train(model, data.X, data.labels, iterations=200, learning_rate=0.05)
    assert np.mean(hist.latent[-20:]) > np.mean(hist.latent[:20])


@pytest.mark.slow
def test_beats_majority_vote():
    """The headline claim, on data where votes alone are not enough.

    Three annotations per item and only half the workers reliable, so majority
    vote leaves real signal on the table -- signal the features can supply.
    """
    data = make_synthetic(
        num_items=300, num_workers=10, num_classes=3,
        labels_per_item=3, good_worker_fraction=0.5, seed=1,
    )
    mv = accuracy(data.labels.majority_vote(), data.z)

    model = build(data)
    train(model, data.X, data.labels, iterations=300, learning_rate=0.05)
    acc = accuracy(model.infer_true_labels(tf.constant(data.X), data.labels), data.z)

    assert acc > mv, f"model {acc:.3f} did not beat majority vote {mv:.3f}"


@pytest.mark.slow
def test_recovers_confusion_matrices():
    """Worker parameters must converge on the ones that generated the data.

    A transposed convention passes every shape check and most accuracy checks --
    the model can still label items well by learning the transpose consistently.
    This is the test that catches it.
    """
    data = make_synthetic(num_items=400, num_workers=8, labels_per_item=5, seed=2)
    model = build(data)
    train(model, data.X, data.labels, iterations=400, learning_rate=0.05)

    est = model.annotator.confusion_matrices().numpy()
    np.testing.assert_allclose(est.sum(axis=1), 1.0, atol=1e-6)

    err = np.abs(est - data.confusion).mean()
    err_transposed = np.abs(np.swapaxes(est, 1, 2) - data.confusion).mean()
    assert err < 0.15, f"confusion MAE {err:.3f}"
    assert err < err_transposed, "estimate matches the transpose better than the truth"


@pytest.mark.slow
def test_predicts_on_unannotated_items():
    """The GP must generalise to items no worker has labelled.

    This is the whole reason for fitting a latent function rather than merely
    denoising the annotation matrix, and it is untestable without held-out data.
    """
    data = make_synthetic(num_items=400, num_workers=12, seed=5)
    train_n = 300

    held_out_X = data.X[train_n:]
    held_out_z = data.z[train_n:]

    keep = data.labels.item_idx < train_n
    from gpcrowdkit import CrowdLabels

    sub = CrowdLabels(
        item_idx=data.labels.item_idx[keep],
        worker_idx=data.labels.worker_idx[keep],
        label=data.labels.label[keep],
        num_items=train_n,
        worker_keys=data.labels.worker_keys,
        class_keys=data.labels.class_keys,
    )

    probs = sub.empirical_class_probs()
    model = GPCrowdModel(
        latent=SVGPLatent(
            gpflow.kernels.SquaredExponential(lengthscales=2.0),
            sub.num_classes,
            inducing_points=data.X[:25].copy(),
        ),
        annotator=VariationalDirichletAnnotator(
            sub.num_workers, sub.num_classes, alpha_tilde_init=init_alpha_tilde(sub, probs)
        ),
        num_data=sub.num_items,
        q_z=FreeCategoricalZ(sub.num_items, sub.num_classes, init_probs=probs),
    )
    train(model, data.X[:train_n], sub, iterations=300, learning_rate=0.05)

    pred = np.argmax(model.predict_class_probs(tf.constant(held_out_X)).numpy(), axis=1)
    assert accuracy(pred, held_out_z) > 1.0 / sub.num_classes + 0.15