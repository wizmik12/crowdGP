"""Concrete annotator strategies.

Three interchangeable worker-noise models, all satisfying the
[AnnotatorModel][crowdgp.annotators.base.AnnotatorModel] object. The core engine
never learns which one it holds, so adding a fourth requires no change
anywhere else in the library.

They span the useful range of complexity:

============================  ==================  ==============================
Strategy                      Parameters          Worker uncertainty
============================  ==================  ==============================
VariationalDirichletAnnotator ``A * C * C``       full posterior over ``R``
SoftmaxPointAnnotator         ``A * C * C``       none (point estimate)
OneCoinAnnotator              ``A``               none (point estimate)
============================  ==================  ==============================

[OneCoinAnnotator][crowdgp.annotators.strategies.OneCoinAnnotator] is the one that justifies the shape of the base
class. It carries a single scalar per worker, so an interface promising an
``[A, C, C]`` parameter tensor would have been the wrong abstraction -- it
happens to *build* such a tensor, but it does not store one.

Index convention throughout: tensors are ``[A, C_obs, C_true]``, normalised
down axis 1, so each column is one distribution over observed labels given a
fixed true class.
"""

from __future__ import annotations

import gpflow
import numpy as np
import tensorflow as tf
from gpflow.utilities import positive

from ..data import CrowdLabels
from .base import ConfusionAnnotator

__all__ = [
    "VariationalDirichletAnnotator",
    "SoftmaxPointAnnotator",
    "OneCoinAnnotator",
    "init_alpha_tilde",
]

FLOAT = tf.float64


class VariationalDirichletAnnotator(ConfusionAnnotator):
    """Full variational Dirichlet posterior over confusion matrices (SVGPCR).

    Places an independent Dirichlet on each *column* of each worker's confusion
    matrix -- one distribution over observed labels per true class::

        q(R^a_{.,j}) = Dir(alpha_tilde^a_{.,j})
        p(R^a_{.,j}) = Dir(alpha^a_{.,j})

    This is the strategy from SVGPCR:
    
    Morales-Alvarez, P., Ruiz, P., Coughlin, S., Molina, R., & Katsaggelos, A. K. (2022). 
    Scalable Variational Gaussian Processes for Crowdsourcing: Glitch Detection in LIGO. 
    IEEE transactions on pattern analysis and machine intelligence, 44(3), 1534-1551.
    
    The only one here
    that represents *uncertainty* about a worker rather than a best guess. That
    matters when workers are sparse: a worker with three annotations and a
    worker with three thousand can have identical point estimates but wildly
    different posteriors, and only this strategy can tell them apart.

    Attributes:
        alpha (gpflow.Parameter): Prior concentrations, non-trainable.
            Shape ``[A, C_obs, C_true]``.
        alpha_tilde (gpflow.Parameter): Variational concentrations, trainable,
            constrained positive. Shape ``[A, C_obs, C_true]``.
    """

    def __init__(
        self,
        num_workers: int,
        num_classes: int,
        alpha_prior: np.ndarray | float = 1.0,
        alpha_tilde_init: np.ndarray | None = None,
    ) -> None:
        """Initialises the prior and variational Dirichlet concentrations.

        Args:
            num_workers: Number of annotators ``A``.
            num_classes: Number of classes ``C``.
            alpha_prior: Prior concentrations: a scalar broadcast over
                ``[A, C, C]``, or an explicit array of that shape. A flat 1.0
                (the reference implementation's choice) is the uniform
                Dirichlet -- no prior opinion about any worker.
            alpha_tilde_init: Optional ``[A, C, C]`` starting point. Defaults to
                a mild diagonal bias, encoding the assumption that workers are
                better than chance. See [init_alpha_tilde][crowdgp.annotators.strategies.init_alpha_tilde] for the
                data-driven alternative, which converges considerably faster.

        Note:
            ``alpha`` is stored as a non-trainable ``Parameter`` rather than a
            ``tf.constant`` so that it appears in ``gpflow.utilities.print_summary``
            alongside everything else, and so that a subclass can make the
            prior learnable (empirical Bayes) by flipping one flag.
        """
        super().__init__(num_workers, num_classes)
        shape = (self.A, self.C, self.C)

        prior = np.broadcast_to(np.asarray(alpha_prior, dtype=np.float64), shape).copy()
        self.alpha = gpflow.Parameter(prior, transform=positive(), trainable=False)

        if alpha_tilde_init is None:
            alpha_tilde_init = np.full(shape, 1.0 / self.C) + np.stack(
                [np.eye(self.C) for _ in range(self.A)]
            )
        self.alpha_tilde = gpflow.Parameter(
            np.asarray(alpha_tilde_init, dtype=np.float64), transform=positive()
        )

    def expected_log_confusion(self) -> tf.Tensor:
        """``E_q[log R] = psi(alpha_tilde) - psi(sum_i alpha_tilde_{i,j})``.

        The standard Dirichlet identity. Note this is an expectation of a
        logarithm, computed exactly -- not the logarithm of the mean, which
        would be a different and biased quantity.

        Returns:
            tf.Tensor: Shape ``[A, C_obs, C_true]``, float64.
        """
        a = self.alpha_tilde
        return tf.math.digamma(a) - tf.math.digamma(tf.reduce_sum(a, axis=1, keepdims=True))

    def kl_divergence(self) -> tf.Tensor:
        """Analytic Dirichlet-Dirichlet KL, summed over workers and columns.

        For a single column, ``KL(Dir(q) || Dir(p))`` is::

            log B(p) - log B(q) + sum_i (q_i - p_i) (psi(q_i) - psi(sum q))

        The implementation evaluates all ``A * C`` columns at once by
        distributing that expression across the tensor.

        Returns:
            tf.Tensor: Scalar, non-negative, and exactly zero when
            ``alpha_tilde == alpha``.

        Note:
            ``tf.math.lbeta`` reduces the *last* axis, but the Dirichlet lives
            along axis 1 (observed classes). ``matrix_transpose`` swaps the last
            two axes so that the reduction lands on the right one. Omitting it
            computes a log-beta over true classes instead: still finite, still
            differentiable, silently wrong.
        """
        q, p = self.alpha_tilde, self.alpha
        diff = q - p
        term1 = tf.reduce_sum(diff * tf.math.digamma(q))
        term2 = -tf.reduce_sum(
            tf.math.digamma(tf.reduce_sum(q, axis=1)) * tf.reduce_sum(diff, axis=1)
        )
        term3 = tf.reduce_sum(
            tf.math.lbeta(tf.linalg.matrix_transpose(p))
            - tf.math.lbeta(tf.linalg.matrix_transpose(q))
        )
        return term1 + term2 + term3

    def confusion_matrices(self) -> tf.Tensor:
        """Posterior mean ``E_q[R] = alpha_tilde / sum_i alpha_tilde_{i,j}``.

        The Dirichlet is the normalised vector of independent Gammas, so its
        mean is each concentration over their sum.

        Returns:
            tf.Tensor: Shape ``[A, C_obs, C_true]``, columns summing to 1.

        Note:
            Deliberately *not* ``softmax(expected_log_confusion())``. That is
            the normalised geometric mean, which is more sharply peaked than
            the posterior mean and so overstates worker accuracy -- by around
            0.06 on the diagonal at ``alpha = [10, 1, 1]``, and more as the
            concentrations shrink. The bias is therefore largest exactly where
            data is scarce and the estimate matters most.
        """
        return self.alpha_tilde / tf.reduce_sum(self.alpha_tilde, axis=1, keepdims=True)


class SoftmaxPointAnnotator(ConfusionAnnotator):
    """Deterministic point-estimate confusion matrices, ``R^a = softmax(W^a)``.

    Same parameter count as the Dirichlet strategy but no distribution over
    them, so no KL term and no notion of how confident the estimate is. Cheaper
    and often adequate when every worker has plenty of annotations.

    Attributes:
        logits (gpflow.Parameter): Unconstrained logits. Shape ``[A, C, C]``.
    """

    def __init__(self, num_workers: int, num_classes: int, diagonal_init: float = 2.0) -> None:
        """Initialises the logits with a diagonal bias.

        Args:
            num_workers: Number of annotators ``A``.
            num_classes: Number of classes ``C``.
            diagonal_init: Logit mass on the diagonal at initialisation.
                Softmax of ``2.0`` on the diagonal gives a competent-but-not-
                certain worker, a reasonable neutral start.

        Note:
            No constraint transform is needed: the softmax in
            [expected_log_confusion][crowdgp.annotators.base.ConfusionAnnotator.expected_log_confusion] handles normalisation, so the raw
            logits are free parameters over all of R.
        """
        super().__init__(num_workers, num_classes)
        init = np.tile(np.eye(num_classes) * diagonal_init, (num_workers, 1, 1))
        self.logits = gpflow.Parameter(init.astype(np.float64))

    def expected_log_confusion(self) -> tf.Tensor:
        """``log R = log_softmax(logits)`` down the observed-class axis.

        Exact rather than an expectation, since ``R`` is deterministic here.

        Returns:
            tf.Tensor: Shape ``[A, C_obs, C_true]``, float64.

        Note:
            ``log_softmax`` rather than ``log(softmax(...))``: it subtracts the
            max internally, so it stays finite for logits where the naive
            composition would produce ``log(0) = -inf``.
        """
        return tf.nn.log_softmax(self.logits, axis=1)

    def kl_divergence(self) -> tf.Tensor:
        """Zero: a point estimate carries no distribution to penalise."""
        return tf.constant(0.0, dtype=FLOAT)

    def confusion_matrices(self) -> tf.Tensor:
        """Exact confusion matrices, ``softmax(logits)``.

        No expectation is involved, so unlike the Dirichlet case this really is
        the exponential of [expected_log_confusion][crowdgp.annotators.base.ConfusionAnnotator.expected_log_confusion].

        Returns:
            tf.Tensor: Shape ``[A, C_obs, C_true]``, columns summing to 1.
        """
        return tf.nn.softmax(self.logits, axis=1)


class OneCoinAnnotator(ConfusionAnnotator):
    """Single scalar accuracy per worker -- the classic one-coin model.

    Worker ``a`` is correct with probability ``beta_a`` and otherwise spreads
    the remaining mass uniformly::

        R^a_{ij} = beta_a               if i == j
                   (1 - beta_a)/(C-1)   otherwise

    Strong assumption -- it cannot express a worker who systematically confuses
    two particular classes -- but with ``A`` parameters instead of ``A*C*C`` it
    is far better behaved when annotations per worker are scarce, which is the
    usual situation.

    It is also the design test for the base contract: it stores ``A`` scalars,
    not a confusion tensor. An interface that had promised ``[A, C, C]``
    parameters would have been the wrong abstraction.

    Attributes:
        beta_logit (gpflow.Parameter): Unconstrained per-worker accuracy
            logits. Shape ``[A]``.
    """

    def __init__(self, num_workers: int, num_classes: int, init_accuracy: float = 0.7) -> None:
        """Initialises per-worker accuracies.

        Args:
            num_workers: Number of annotators ``A``.
            num_classes: Number of classes ``C``, at least 2.
            init_accuracy: Initial accuracy shared by all workers, in ``(0, 1)``.

        Raises:
            ValueError: If ``num_classes < 2`` (the off-diagonal mass would be
                divided by zero), or if ``init_accuracy`` is not in ``(0, 1)``.

        Note:
            Stored as a logit with the sigmoid applied in [beta][crowdgp.annotators.strategies.OneCoinAnnotator.beta], rather
            than as a constrained ``Parameter``. Both work; the logit keeps the
            dependency surface small and makes the unconstrained optimisation
            explicit at the point of use.
        """
        if num_classes < 2:
            raise ValueError("OneCoinAnnotator requires at least two classes.")
        if not 0.0 < init_accuracy < 1.0:
            raise ValueError("init_accuracy must lie strictly in (0, 1).")
        super().__init__(num_workers, num_classes)
        logit = float(np.log(init_accuracy / (1.0 - init_accuracy)))
        self.beta_logit = gpflow.Parameter(np.full(num_workers, logit, dtype=np.float64))

    @property
    def beta(self) -> tf.Tensor:
        """Per-worker accuracy in ``(0, 1)``. Shape ``[A]``."""
        return tf.sigmoid(self.beta_logit)

    def _confusion(self) -> tf.Tensor:
        """Builds the ``[A, C, C]`` confusion tensor from the ``A`` scalars.

        Returns:
            tf.Tensor: Shape ``[A, C_obs, C_true]``, columns summing to 1.
        """
        beta = self.beta[:, None, None]  # [A, 1, 1]
        eye = tf.eye(self.C, dtype=FLOAT)[None, :, :]  # [1, C, C]
        off = (1.0 - beta) / tf.constant(self.C - 1, dtype=FLOAT)
        return eye * beta + (1.0 - eye) * off

    def expected_log_confusion(self) -> tf.Tensor:
        """Log of the constructed confusion tensor.

        Returns:
            tf.Tensor: Shape ``[A, C_obs, C_true]``, float64.

        Note:
            The ``1e-12`` floor guards ``log(0)`` if the sigmoid saturates at
            0 or 1 during optimisation. Saturation is itself a symptom -- a
            worker driven to perfect accuracy usually means too few
            annotations and no prior to regularise them.
        """
        return tf.math.log(self._confusion() + 1e-12)

    def kl_divergence(self) -> tf.Tensor:
        """Zero: a point estimate carries no distribution to penalise."""
        return tf.constant(0.0, dtype=FLOAT)

    def confusion_matrices(self) -> tf.Tensor:
        """Exact confusion matrices built from the per-worker accuracies.

        Returns:
            tf.Tensor: Shape ``[A, C_obs, C_true]``, columns summing to 1.
        """
        return self._confusion()


def init_alpha_tilde(
    labels: CrowdLabels, class_probs: np.ndarray, prior_strength: float = 1.0
) -> np.ndarray:
    """Data-driven initialisation of the variational Dirichlet concentrations.

    Reproduces ``_init_behaviors`` from the reference implementation: each
    annotation ``(n, a, y)`` adds the soft class assignment ``class_probs[n]``
    to ``alpha_tilde[a, y, :]``, so a worker who labels an item "cat" when the
    votes point to "cat" accumulates diagonal mass.

    Starting here rather than from a flat prior matters. The ELBO is not convex,
    and from a uniform start the model can settle into a labelling that is a
    permutation of the truth -- self-consistent, high-likelihood, and useless.
    Vote-based initialisation breaks that symmetry before optimisation begins.

    Args:
        labels: The annotations.
        class_probs: Soft per-item class assignments ``[N, C]``, typically
            [empirical_class_probs][crowdgp.data.CrowdLabels.empirical_class_probs].
        prior_strength: Constant added to every entry, keeping concentrations
            comfortably positive and the digamma well conditioned.

    Returns:
        np.ndarray: Shape ``[A, C_obs, C_true]``, float64.

    Note:
        The two nested Python loops of the original become two ``np.add.at``
        scatter-adds, turning an ``O(L)`` interpreter loop into two vectorised
        passes. On a dataset with millions of annotations this is the
        difference between minutes and seconds of setup.
    """
    A, C = labels.num_workers, labels.num_classes
    acc = np.full((A, C, C), 1.0 / C)
    counts = np.ones((A, C))

    np.add.at(acc, (labels.worker_idx, labels.label), class_probs[labels.item_idx])
    np.add.at(counts, (labels.worker_idx, labels.label), 1.0)

    acc /= counts[:, :, None]
    acc *= (counts / counts.sum(axis=1, keepdims=True))[:, :, None]
    acc /= acc.sum(axis=1, keepdims=True)
    return acc + prior_strength