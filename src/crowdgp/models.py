"""The composed core engine.

===============  ================================================
ELBO term        Supplied by
===============  ================================================
``latent``       :class:`~crowdgp.latent.base.LatentFunction`
``crowd``        :class:`~crowdgp.annotators.base.AnnotatorModel`
``entropy``      :class:`~crowdgp.posteriors.PosteriorZ`
``kl_latent``    :class:`~crowdgp.latent.base.LatentFunction`
``kl_annotator`` :class:`~crowdgp.annotators.base.AnnotatorModel`
===============  ================================================

"""

from __future__ import annotations

from typing import NamedTuple

import gpflow
import numpy as np
import tensorflow as tf

from .annotators.base import AnnotatorModel
from .data import CrowdBatch, CrowdLabels
from .latent.base import LatentFunction
from .posteriors import PosteriorZ

__all__ = ["GPCrowdModel", "ELBOTerms"]

FLOAT = tf.float64


class ELBOTerms(NamedTuple):
    """Decomposition of the ELBO into its five terms plus the batch scale.

    Kept as a structured return rather than folded into a scalar because the
    individual terms are the primary diagnostic for a crowdsourcing model. The
    characteristic failure is not divergence but a plausible-looking ELBO in
    which ``crowd`` dominates and ``latent`` flatlines: the model has learned to
    reproduce the annotations and is ignoring the features entirely, so it
    generalises to nothing. That is invisible in the total and obvious in the
    decomposition. The reference implementation keeps the same list, as
    ``self.decomp``.

    Attributes:
        latent (tf.Tensor): ``sum_n sum_c gamma_nc E_q(f)[log p(z=c|f)]``.
        crowd (tf.Tensor): ``sum_n sum_c gamma_nc`` times the aggregated
            annotation evidence.
        entropy (tf.Tensor): ``H[q(Z)]`` over the batch, positive.
        kl_latent (tf.Tensor): ``KL(q(u) || p(u))``.
        kl_annotator (tf.Tensor): ``KL(q(R) || p(R))``, zero for
            point-estimate strategies.
        scale (tf.Tensor): ``N / B``, the minibatch correction.
    """

    latent: tf.Tensor
    crowd: tf.Tensor
    entropy: tf.Tensor
    kl_latent: tf.Tensor
    kl_annotator: tf.Tensor
    scale: tf.Tensor

    @property
    def total(self) -> tf.Tensor:
        """The scaled ELBO.

        Returns:
            tf.Tensor: Scalar.

        Note:
            The scale multiplies the three data-dependent terms and *not* the
            two KLs. The data terms are sums over items, so a batch estimates
            ``B/N`` of the total and must be rescaled; the KLs are properties
            of global parameters, complete in every batch. Scaling them too
            inflates the complexity penalty by a factor of ``N/B`` and produces
            a model that underfits for reasons that are very hard to diagnose
            from the loss curve alone.
        """
        return (
            (self.latent + self.crowd + self.entropy) * self.scale
            - self.kl_latent
            - self.kl_annotator
        )


class GPCrowdModel(gpflow.Module):
    """Gaussian-process model for learning from crowds.

    Composed of three pluggable parts, none of which this class inspects:

    * a :class:`~crowdgp.latent.base.LatentFunction` mapping features to
      per-class log evidence,
    * an :class:`~crowdgp.annotators.base.AnnotatorModel` describing worker
      noise,
    * a :class:`~crowdgp.posteriors.PosteriorZ` over the unknown true labels.

With :class:`~crowdgp.latent.svgp.SVGPLatent`,
    :class:`~crowdgp.annotators.strategies.VariationalDirichletAnnotator` and
    :class:`~crowdgp.posteriors.FreeCategoricalZ` this reproduces SVGPCR
    (Morales-Alvarez et al.).

    Attributes:
        latent (LatentFunction): Latent classifier.
        annotator (AnnotatorModel): Worker noise strategy.
        q_z (PosteriorZ): Ground-truth posterior strategy.
        num_data (int): Dataset size ``N``, used to rescale minibatch terms.
    """

    def __init__(
        self,
        latent: LatentFunction,
        annotator: AnnotatorModel,
        num_data: int,
        q_z: PosteriorZ,
    ) -> None:
        """Assembles the model from its components.

        Args:
            latent: Latent classifier strategy.
            annotator: Annotator noise strategy.
            num_data: Total number of items ``N``. Given explicitly rather than
                inferred from a batch, since a minibatch cannot know it.
            q_z: Ground-truth posterior strategy, e.g.
                :class:`~crowdgp.posteriors.FreeCategoricalZ`. Required rather
                than defaulted: it needs ``N`` and ``C``, which this
                constructor does not have, and the choice is a modelling
                decision rather than an implementation detail.

        Note:
            Nothing here validates that the components agree on ``C``. A
            mismatch surfaces immediately as a shape error on the first ELBO
            call, which is a clearer message than anything this constructor
            would produce, and avoids duplicating the classes' own knowledge of
            their shapes.
        """
        super().__init__()
        self.latent = latent
        self.annotator = annotator
        self.q_z = q_z
        self.num_data = int(num_data)

    def elbo_terms(self, batch: CrowdBatch) -> ELBOTerms:
        """Computes the ELBO decomposition for one minibatch.

        The whole method: ask the latent function what the features imply, ask
        the annotator what the workers imply, ask ``q(Z)`` to combine them,
        then weight and add.

        Args:
            batch (CrowdBatch): Aligned features and annotations.

        Returns:
            ELBOTerms: The five terms and the scale factor.
        """
        gp_log = self.latent.expected_log_p_z(batch.X)  # [B, C]
        crowd_log = self.annotator.crowd_log_per_item(batch)  # [B, C]
        gamma = self.q_z.gamma(batch, gp_log, crowd_log)  # [B, C]

        return ELBOTerms(
            latent=tf.reduce_sum(gamma * gp_log),
            crowd=tf.reduce_sum(gamma * crowd_log),
            entropy=self.q_z.entropy(gamma),
            kl_latent=self.latent.prior_kl(),
            kl_annotator=self.annotator.kl_divergence(),
            scale=tf.cast(self.num_data, FLOAT) / tf.cast(batch.size, FLOAT),
        )

    def elbo(self, batch: CrowdBatch) -> tf.Tensor:
        """Evidence lower bound for one minibatch.

        Args:
            batch (CrowdBatch): Aligned features and annotations.

        Returns:
            tf.Tensor: Scalar. Under minibatching this is an unbiased estimate
            of the full-data ELBO, not the value itself.
        """
        return self.elbo_terms(batch).total

    def maximum_log_likelihood_objective(self, batch: CrowdBatch) -> tf.Tensor:
        """GPflow-conventional alias for :meth:`elbo`."""
        return self.elbo(batch)

    def training_loss(self, batch: CrowdBatch) -> tf.Tensor:
        """Negative ELBO, for minimisation.

        Args:
            batch (CrowdBatch): Aligned features and annotations.

        Returns:
            tf.Tensor: Scalar.
        """
        return -self.elbo(batch)

    def predict_class_probs(self, X: tf.Tensor) -> tf.Tensor:
        """Predictive class probabilities for inputs with no annotations.

        Delegates to the latent function alone: the annotator model is
        meaningless here, since no worker has labelled these items.

        Args:
            X (tf.Tensor): Feature matrix. Shape ``[B, D]``.

        Returns:
            tf.Tensor: Shape ``[B, C]``, rows summing to 1.
        """
        return self.latent.predict_class_probs(X)

    def infer_true_labels(self, X: tf.Tensor, labels: CrowdLabels) -> np.ndarray:
        """Posterior estimate of the latent ground-truth label of each item.

        Combines both evidence streams, which is what distinguishes this from
        majority vote: an item whose three annotators disagree can still be
        confidently labelled if its features place it among items the model has
        learned to classify.

        Args:
            X (tf.Tensor): Full feature matrix. Shape ``[N, D]``.
            labels (CrowdLabels): The annotations.

        Returns:
            np.ndarray: Shape ``[N]``, the argmax class per item.

        Note:
            Runs on the full dataset in one pass. For large ``N`` this is the
            method to batch, not :meth:`elbo`.
        """
        batch = labels.full_batch(X)
        gp_log = self.latent.expected_log_p_z(batch.X)
        crowd_log = self.annotator.crowd_log_per_item(batch)
        gamma = self.q_z.gamma(batch, gp_log, crowd_log)
        return np.argmax(gamma.numpy(), axis=1)