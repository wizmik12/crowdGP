"""Variational posterior over the latent ground-truth labels ``q(Z)``.

``q(z_n = c) = gamma_nc`` is the model's belief about what item ``n`` really
is. It is where the two evidence streams meet -- the latent function says what
the features imply, the annotator model says what the workers imply -- and it
is usually the output you care about: inferred labels for an unlabelled
dataset.

This module follows the reference SVGPCR implementation (Morales-Alvarez et
al.), where ``gamma`` is a free variational parameter optimised by gradient
alongside everything else. In the reference code it is ``self.q_unn``, an
``[N, K]`` positive ``Parameter`` normalised row-wise at use::

    self.q_unn = Parameter(q_unn, transform=transforms.positive)
    q_mb = q_unn_mb / tf.reduce_sum(q_unn_mb, axis=1, keepdims=True)

[FreeCategoricalZ][crowdgp.posteriors.FreeCategoricalZ] reproduces exactly that.

An alternative worth knowing about
----------------------------------
Because ``z_n`` appears only in terms local to item ``n``, the ELBO-optimal
``q(z_n)`` given the other parameters has a closed form: writing
``t_nc = gp_log_nc + crowd_log_nc`` for the total log evidence, item ``n``
contributes ``sum_c gamma_nc t_nc - sum_c gamma_nc log gamma_nc``, which is
maximised at ``gamma_n = softmax(t_n)``. That update is exact, stays exact
under minibatching, and stores no parameters at all -- it is the E-step of EM.

The reference implementation does not use it, so neither does this module. It
is a natural second strategy to add once the free version is working and you
have a baseline to compare against; the abstract base class below exists so
that adding it later requires no change to the core engine.
"""

from __future__ import annotations

import abc

import gpflow
import numpy as np
import tensorflow as tf
from gpflow.utilities import positive

from .data import CrowdBatch

__all__ = ["PosteriorZ", "FreeCategoricalZ"]

FLOAT = tf.float64


class PosteriorZ(gpflow.Module, abc.ABC):
    """Abstract variational posterior over categorical ground-truth labels.

    Note:
        [gamma][crowdgp.posteriors.PosteriorZ.gamma] receives both evidence tensors even though the free
        strategy ignores them. The uniform signature is what would let a
        closed-form strategy drop in later without altering the call site in
        ``models.py``; a signature tailored to the free version would leak that
        choice into the engine.
    """

    @abc.abstractmethod
    def gamma(self, batch: CrowdBatch, gp_log: tf.Tensor, crowd_log: tf.Tensor) -> tf.Tensor:
        """Responsibilities ``q(z_n = c)`` for the items in the batch.

        Args:
            batch (CrowdBatch): The batch, supplying ``item_global`` for
                strategies that index dataset-sized parameters.
            gp_log (tf.Tensor): Latent evidence ``E_q(f)[log p(z=c|f)]``.
                Shape ``[B, C]``.
            crowd_log (tf.Tensor): Aggregated annotation evidence.
                Shape ``[B, C]``.

        Returns:
            tf.Tensor: Shape ``[B, C]``, rows summing to 1.
        """

    @staticmethod
    def entropy(gamma: tf.Tensor) -> tf.Tensor:
        """Entropy ``H[q(Z)] = -sum_n sum_c gamma_nc log gamma_nc``.

        Args:
            gamma (tf.Tensor): Responsibilities. Shape ``[B, C]``.

        Returns:
            tf.Tensor: Scalar, non-negative.

        Note:
            Returned as a *positive* entropy, to be added to the ELBO. The
            reference implementation computes ``qentComp = sum(q log q)`` --
            the negative entropy -- and subtracts it. Same bound, opposite
            convention; mixing the two produces an objective that decreases
            monotonically while otherwise looking correct.

            The ``1e-12`` floor keeps ``log(0)`` out of the graph if a
            responsibility saturates. Mathematically ``0 * log 0 = 0``, but in
            floating point ``0 * -inf = nan``, and a single NaN poisons every
            gradient in the step.
        """
        g = gamma + 1e-12
        return -tf.reduce_sum(g * tf.math.log(g))


class FreeCategoricalZ(PosteriorZ):
    """Free ``[N, C]`` parameters optimised jointly with the rest of the model.

    Reproduces the reference implementation: a positive-constrained tensor,
    normalised row-wise on use. ``gamma`` is learned indirectly, through the
    gradient of the ELBO, rather than being read off the evidence.

    Attributes:
        q_unn (gpflow.Parameter): Unnormalised positive responsibilities.
            Shape ``[N, C]``.

    Note:
        This parameterisation is scale-degenerate: multiplying any row by a
        constant leaves ``gamma`` unchanged, so every item carries one flat
        direction along which the optimiser can drift without altering the
        objective. Harmless, but it does mean the raw parameter values are not
        interpretable on their own and only their ratios matter. Storing
        logits instead removes the redundancy; this class keeps the reference
        form so that results can be compared against it directly.
    """

    def __init__(
        self, num_items: int, num_classes: int, init_probs: np.ndarray | None = None
    ) -> None:
        """Initialises the responsibilities.

        Args:
            num_items: Dataset size ``N``.
            num_classes: Number of classes ``C``.
            init_probs: Optional ``[N, C]`` starting probabilities, normally
                [empirical_class_probs][crowdgp.data.CrowdLabels.empirical_class_probs], which
                is the smoothed vote histogram the reference implementation
                uses. Defaults to uniform.

        Note:
            Uniform initialisation is a symmetric point: every class is equally
            plausible for every item, so nothing distinguishes the true
            labelling from a permutation of it, and the non-convex objective
            can settle into one. The reference implementation initialises from
            vote counts for exactly this reason, and so should you on any real
            dataset.
        """
        super().__init__()
        init = (
            np.full((num_items, num_classes), 1.0 / num_classes)
            if init_probs is None
            else np.asarray(init_probs, dtype=np.float64)
        )
        self.q_unn = gpflow.Parameter(init, transform=positive())

    def gamma(self, batch, gp_log, crowd_log):
        """See [PosteriorZ.gamma][crowdgp.posteriors.PosteriorZ.gamma]. Uses only ``batch.item_global``.

        Gathers this batch's rows and normalises them. The evidence arguments
        are ignored: for this strategy the evidence reaches ``gamma`` only
        through the gradient of the ELBO, over many steps.
        """
        rows = tf.gather(self.q_unn, batch.item_global)  # [B, C]
        return rows / tf.reduce_sum(rows, axis=1, keepdims=True)