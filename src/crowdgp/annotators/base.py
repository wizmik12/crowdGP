"""Abstract Strategy contract for annotator (worker noise) models.

Design of the contract
----------------------
The single method every annotator must implement is

    label_log_terms(batch) -> [L, C]

whose entry ``(l, c)`` is the expected log-likelihood of annotation ``l`` under
the hypothesis that its item's true class is ``c``::

    t_{l,c} = E_{q(R)}[ log p(y_l | z_{i(l)} = c, R) ]

Everything else the model needs is a fixed reduction of that tensor, and is
implemented once in :class:`AnnotatorModel` rather than in every subclass:

* summing over the annotations of each item gives the ``[B, C]`` crowdsourcing
  evidence (:meth:`AnnotatorModel.crowd_log_per_item`);
* weighting that by ``q(Z)`` and summing gives the scalar ELBO term
  (:meth:`AnnotatorModel.expected_log_likelihood`);
* adding it to the latent term and taking a softmax gives the closed-form
  update for ``q(Z)`` (see :mod:`gpcrowd.posteriors`).

Choosing the scalar ``expected_log_likelihood`` as the primitive instead would
force every subclass to repeat the aggregation, and would discard the per-item
breakdown that the closed-form ``q(Z)`` update needs -- recovering it later
would mean changing this interface, which is precisely the modification the
Open/Closed principle is meant to avoid.

What the contract deliberately omits
------------------------------------
It does not promise a confusion matrix. A one-coin annotator carries ``A``
scalars rather than ``A * C * C`` entries, and an instance-dependent annotator
carries none at all, computing its confusions from features on the fly. Models
that *do* have confusion matrices inherit from :class:`ConfusionAnnotator`,
which supplies the gather that maps them onto observed annotations.

Two expectations, two jobs
--------------------------
For any strategy holding a *distribution* over ``R`` rather than a point
estimate, these are different quantities and neither substitutes for the other:

* ``E_q(R)[log R]``, for a Dirichlet ``psi(a) - psi(sum a)``, is what the ELBO
  contains -- a genuine expectation of a logarithm, computed exactly by the
  digamma function. Replacing it with ``log(E[R])`` would break the bound.
* ``E_q(R)[R]``, for a Dirichlet ``a / sum(a)``, is the posterior mean, used
  for reporting, prediction and initialising other models.

Going from the first to the second by normalising ``exp(E[log R])`` gives the
*geometric* mean, not the posterior mean. It is more sharply peaked, so it
overstates worker accuracy -- worst for the workers with the fewest
annotations, where the estimate matters most. This module therefore offers no
generic confusion-matrix implementation; each concrete strategy supplies the
estimator correct for its own posterior.

Index convention
----------------
Confusion tensors are ``[A, C_obs, C_true]`` and are normalised down **axis 1**,
so each *column* is one distribution over observed labels given a fixed true
class. This matches the reference SVGPCR implementation. Assuming rows instead
produces code that runs, returns finite numbers, and learns the transpose of
the intended model.
"""

from __future__ import annotations

import abc

import gpflow
import tensorflow as tf

from ..data import CrowdBatch

__all__ = ["AnnotatorModel", "ConfusionAnnotator"]


class AnnotatorModel(gpflow.Module, abc.ABC):
    """Abstract Strategy interface for crowdsourced annotator noise models.

    Subclasses supply :meth:`label_log_terms` and :meth:`kl_divergence`; the
    reductions built on top of them are concrete methods here.

    Attributes:
        A (int): Number of annotators.
        C (int): Number of categorical classes.

    Note:
        Inheriting from ``gpflow.Module`` (itself a ``tf.Module``) is what makes
        ``gpflow.Parameter`` attributes discoverable via ``trainable_variables``
        without any registration boilerplate. Assign parameters as plain
        attributes and the optimiser will find them.
    """

    def __init__(self, num_workers: int, num_classes: int, name: str | None = None) -> None:
        """Initialises the shared annotator attributes.

        Args:
            num_workers: Number of unique annotators ``A``.
            num_classes: Number of categorical classes ``C``.
            name: Optional module name, forwarded to ``gpflow.Module``.
        """
        super().__init__(name=name)
        self.A: int = int(num_workers)
        self.C: int = int(num_classes)

    # ------------------------------------------------------------- contract

    @abc.abstractmethod
    def label_log_terms(self, batch: CrowdBatch) -> tf.Tensor:
        """Per-annotation expected log-likelihood under each candidate class.

        Args:
            batch (CrowdBatch): Batch supplying ``worker_idx`` and ``label``
                for every annotation, and ``X`` for models whose worker
                behaviour depends on the item.

        Returns:
            tf.Tensor: Shape ``[L, C]``, float64. Row ``l`` is annotation
            ``l``'s log-likelihood under each of the ``C`` hypotheses for its
            item's true class.
        """

    @abc.abstractmethod
    def kl_divergence(self) -> tf.Tensor:
        """Parameter KL penalty ``KL(q(params) || p(params))``.

        Returns:
            tf.Tensor: Scalar. Exactly zero for point-estimate strategies,
            which hold no distribution over their parameters.
        """

    # ---------------------------------------------------------- derived terms

    def crowd_log_per_item(self, batch: CrowdBatch) -> tf.Tensor:
        """Aggregates the annotation terms onto the items they belong to.

        This is the ``[L, C] -> [B, C]`` reduction: for each item, sum the
        log-likelihood contributions of all its annotations, separately under
        each candidate true class.

        Args:
            batch (CrowdBatch): The batch.

        Returns:
            tf.Tensor: Shape ``[B, C]``, float64.

        Note:
            ``unsorted_segment_sum`` does this in one vectorised op using
            ``item_local`` as the segment id, replacing the reference
            implementation's Python ``for a in range(self.A)`` loop over
            annotators. That loop works eagerly but unrolls into ``A`` copies of
            the subgraph under ``tf.function``, so compile time grows with the
            number of annotators -- fatal on real crowdsourcing datasets.
            Items with no annotations correctly receive a row of zeros.
        """
        return tf.math.unsorted_segment_sum(
            self.label_log_terms(batch), batch.item_local, batch.size
        )

    def expected_log_likelihood(self, batch: CrowdBatch, gamma: tf.Tensor) -> tf.Tensor:
        """Crowdsourcing term of the ELBO, ``E_{q(Z)q(R)}[log p(Y | Z, R)]``.

        Args:
            batch (CrowdBatch): The batch.
            gamma (tf.Tensor): Variational posterior ``q(z_n = c)``. Shape
                ``[B, C]``, rows summing to 1.

        Returns:
            tf.Tensor: Scalar.
        """
        return tf.reduce_sum(gamma * self.crowd_log_per_item(batch))

    # -------------------------------------------------------------- optional

    def confusion_matrices(self) -> tf.Tensor:
        """Posterior mean of the per-worker confusion matrices, for reporting.

        Not part of the required contract, and deliberately not implemented
        generically: the correct estimator depends on the posterior the
        strategy holds. A concrete subclass should return ``E_q(R)[R]``, which
        for a Dirichlet is ``alpha / sum(alpha)`` down axis 1, and for a point
        estimate is simply the parameter itself.

        Returns:
            tf.Tensor: Shape ``[A, C_obs, C_true]``, columns summing to 1.

        Raises:
            NotImplementedError: If this strategy has no confusion-matrix view,
                or has not supplied its posterior mean.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not expose confusion matrices. "
            "Concrete strategies should return the posterior mean E[R]."
        )


class ConfusionAnnotator(AnnotatorModel):
    """Base class for strategies defined by per-worker confusion matrices.

    Subclasses implement :meth:`expected_log_confusion` for the ELBO, and
    :meth:`~AnnotatorModel.confusion_matrices` for reporting. The gather that
    maps the log confusion tensor onto observed annotations is written once,
    here.

    Note:
        This class intentionally provides no ``confusion_matrices`` default.
        The tempting one -- normalising ``exp(expected_log_confusion())`` -- is
        exact only for point-estimate strategies, where no expectation is
        involved. For a distributional strategy it returns the normalised
        geometric mean rather than the posterior mean: more peaked, and so
        overstating accuracy, worst for the workers with the least data. A
        silently wrong default that happens to be correct for the simplest
        subclass is worse than no default at all.
    """

    @abc.abstractmethod
    def expected_log_confusion(self) -> tf.Tensor:
        """Expected log confusion tensor ``E_{q(R)}[log R^a_{i,j}]``.

        Entry ``(a, i, j)`` is worker ``a``'s expected log-probability of
        reporting observed class ``i`` when the true class is ``j``.

        Returns:
            tf.Tensor: Shape ``[A, C_obs, C_true]``, float64. This is the
            quantity the ELBO needs. It is *not* the log of the posterior mean,
            and the two must not be interchanged.
        """

    def label_log_terms(self, batch: CrowdBatch) -> tf.Tensor:
        """See :meth:`AnnotatorModel.label_log_terms`.

        Selects, for each annotation, the row of the expected log confusion
        tensor corresponding to ``(the worker who gave it, the label they
        gave)``, leaving a vector over candidate true classes.

        Note:
            ``tf.gather_nd`` with ``[L, 2]`` indices into an
            ``[A, C_obs, C_true]`` tensor consumes the first two axes and keeps
            the third, yielding ``[L, C_true]`` in a single op -- no loop over
            annotators, and no dependence on ``A``.
        """
        idx = tf.stack([batch.worker_idx, batch.label], axis=-1)  # [L, 2]
        return tf.gather_nd(self.expected_log_confusion(), idx)  # [L, C_true]