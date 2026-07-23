"""Tests for the annotator Strategy base classes.

Both classes under test are abstract, so the suite defines minimal stub
strategies and exercises the concrete machinery through them. The stubs return
arithmetically transparent values -- worker ids, label ids, digit-encoded
tensor entries -- so that every expected result can be computed by hand and
written into the assertion as a literal. A test whose expected value is
produced by re-running the implementation proves nothing.

That the stubs are five lines each is itself the point: if writing a new
strategy required more than its own mathematics, the contract would be wrong.
"""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

import gpflow

from crowdgp.annotators.base import AnnotatorModel, ConfusionAnnotator
from crowdgp.data import CrowdLabels

FLOAT = tf.float64


# --------------------------------------------------------------------- fixtures


@pytest.fixture
def labels() -> CrowdLabels:
    """Five annotations over four items; item 1 deliberately has none.

    ==========  ========  =======
    item        worker    label
    ==========  ========  =======
    0           0         0
    0           3         0
    2           1         1
    2           4         1
    3           2         1
    ==========  ========  =======

    Gives ``N=4, A=5, C=2, L=5``. Item 1 exercises the empty-segment path,
    which is where an off-by-one in the ragged construction would surface.
    """
    return CrowdLabels(
        item_idx=np.array([0, 0, 2, 2, 3]),
        worker_idx=np.array([0, 3, 1, 4, 2]),
        label=np.array([0, 0, 1, 1, 1]),
        num_items=4,
    )


@pytest.fixture
def X() -> tf.Tensor:
    """Feature matrix, values irrelevant -- only its row count is used."""
    return tf.constant(np.arange(8, dtype=np.float64).reshape(4, 2))


class StubAnnotator(AnnotatorModel):
    """Returns ``[worker_idx, label]`` as the per-annotation term row.

    Not a model. The point is that the two numbers are distinct and known, so
    the aggregation performed by the base class can be verified exactly.
    """

    def label_log_terms(self, batch):
        return tf.stack(
            [tf.cast(batch.worker_idx, FLOAT), tf.cast(batch.label, FLOAT)], axis=1
        )

    def kl_divergence(self):
        return tf.constant(0.0, dtype=FLOAT)


class StubConfusionAnnotator(ConfusionAnnotator):
    """Digit-encoded confusion tensor: entry ``(a, i, j)`` equals ``100a + 10i + j``.

    Every entry is unique and readable, so a gather that picks the wrong
    worker, the wrong observed class, or the wrong axis produces a visibly
    wrong number rather than a plausible one.
    """

    def expected_log_confusion(self):
        a = np.arange(self.A)[:, None, None] * 100
        i = np.arange(self.C)[None, :, None] * 10
        j = np.arange(self.C)[None, None, :]
        return tf.constant((a + i + j).astype(np.float64))

    def kl_divergence(self):
        return tf.constant(0.0, dtype=FLOAT)


# ------------------------------------------------------------ contract enforcement


def test_abstract_classes_cannot_be_instantiated():
    """abc.ABC must reject incomplete strategies at construction, not at first use."""
    with pytest.raises(TypeError):
        AnnotatorModel(5, 2)
    with pytest.raises(TypeError):
        ConfusionAnnotator(5, 2)


def test_missing_abstract_method_is_rejected():
    class Incomplete(AnnotatorModel):
        def label_log_terms(self, batch):  # kl_divergence deliberately absent
            return tf.zeros((0, self.C), dtype=FLOAT)

    with pytest.raises(TypeError, match="kl_divergence"):
        Incomplete(5, 2)


def test_gpflow_module_tracks_parameters():
    """Parameters assigned as plain attributes must reach the optimiser."""

    class WithParam(StubAnnotator):
        def __init__(self, A, C):
            super().__init__(A, C)
            self.theta = gpflow.Parameter(np.zeros((A, C), dtype=np.float64))

    assert len(WithParam(5, 2).trainable_variables) == 1
    assert len(StubAnnotator(5, 2).trainable_variables) == 0


# ------------------------------------------------------------------- reductions


def test_crowd_log_per_item_aggregates_by_item(labels, X):
    """[L, C] -> [B, C] must sum each item's own annotations and no others."""
    annot = StubAnnotator(labels.num_workers, labels.num_classes)
    batch = labels.full_batch(X)

    terms = annot.label_log_terms(batch)
    assert terms.shape == (5, 2)

    # item 0: workers 0 and 3, both label 0  -> [0+3, 0+0]
    # item 1: no annotations                 -> [0, 0]
    # item 2: workers 1 and 4, both label 1  -> [1+4, 1+1]
    # item 3: worker 2, label 1              -> [2, 1]
    expected = np.array([[3.0, 0.0], [0.0, 0.0], [5.0, 2.0], [2.0, 1.0]])
    np.testing.assert_allclose(annot.crowd_log_per_item(batch).numpy(), expected)


def test_item_without_annotations_contributes_zero(labels, X):
    """An empty segment is no evidence, which must be an exact zero row."""
    annot = StubAnnotator(labels.num_workers, labels.num_classes)
    per_item = annot.crowd_log_per_item(labels.full_batch(X)).numpy()
    np.testing.assert_array_equal(per_item[1], np.zeros(2))


def test_expected_log_likelihood_weights_by_gamma(labels, X):
    """The ELBO term is the gamma-weighted sum of the per-item evidence."""
    annot = StubAnnotator(labels.num_workers, labels.num_classes)
    batch = labels.full_batch(X)
    gamma = tf.constant([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [0.25, 0.75]], dtype=FLOAT)

    # 1*3 + 0*0  |  0.5*0 + 0.5*0  |  0*5 + 1*2  |  0.25*2 + 0.75*1
    assert float(annot.expected_log_likelihood(batch, gamma)) == pytest.approx(6.25)


def test_subbatch_aggregation_is_local(labels, X):
    """item_local indexes the batch, not the dataset: a 2-item batch gives 2 rows."""
    annot = StubAnnotator(labels.num_workers, labels.num_classes)
    batch = labels.gather_batch(X, tf.constant([2, 0], tf.int32))

    per_item = annot.crowd_log_per_item(batch).numpy()
    assert per_item.shape == (2, 2)
    np.testing.assert_allclose(per_item, np.array([[5.0, 2.0], [3.0, 0.0]]))


# ---------------------------------------------------------- confusion gather path


def test_gather_nd_selects_worker_and_observed_row(labels, X):
    """gather_nd must consume axes (A, C_obs) and keep C_true."""
    annot = StubConfusionAnnotator(labels.num_workers, labels.num_classes)
    batch = labels.full_batch(X)

    # (worker, label) pairs in ragged order: (0,0) (3,0) (1,1) (4,1) (2,1)
    # entry = 100*worker + 10*label + j
    expected = np.array(
        [[0.0, 1.0], [300.0, 301.0], [110.0, 111.0], [410.0, 411.0], [210.0, 211.0]]
    )
    np.testing.assert_allclose(annot.label_log_terms(batch).numpy(), expected)


def test_confusion_annotator_provides_no_default_estimator(labels):
    """Regression guard.

    ``exp(E[log R])`` normalised is the geometric mean, not the posterior mean
    ``E[R]``; it is more peaked and overstates accuracy, worst for the workers
    with the fewest annotations. It coincides with the truth only for
    point-estimate strategies, so a default here would test clean against the
    simplest subclass and silently misreport the Dirichlet one.
    """
    annot = StubConfusionAnnotator(labels.num_workers, labels.num_classes)
    with pytest.raises(NotImplementedError, match="StubConfusionAnnotator"):
        annot.confusion_matrices()


# ------------------------------------------------------------------- graph mode


def test_compiles_and_retraces_under_tf_function(labels, X):
    """Guards the failure mode of Python loops over annotators.

    A per-annotator loop is correct eagerly but unrolls into A copies of the
    subgraph here. Two different batch sizes force a retrace, which is where
    static-shape assumptions surface.
    """
    annot = StubAnnotator(labels.num_workers, labels.num_classes)

    @tf.function
    def compiled(idx):
        return annot.crowd_log_per_item(labels.gather_batch(X, idx))

    a = compiled(tf.constant([0, 2], tf.int32))
    b = compiled(tf.constant([1, 2, 3], tf.int32))
    assert a.shape == (2, 2) and b.shape == (3, 2)
    np.testing.assert_allclose(a.numpy(), np.array([[3.0, 0.0], [5.0, 2.0]]))