"""Abstract contract for the latent classifier ``p(z | x)``.

The latent function is the half of the model that looks at *features*. While
the annotator model asks "given this worker said cat, what was the truth?",
this asks "given this image, what is the truth?". The two sources of evidence
are combined by ``q(Z)``, and the whole point of a GP crowdsourcing model is
that they regularise each other: features smooth over noisy annotations, and
annotations supply the labels the features are fitted against.

Why this is an interface rather than a hardcoded SVGP
----------------------------------------------------
The obvious reason is symmetry with the annotator strategy -- a deep-kernel
backbone, a plain GP, or a neural classifier should be swappable without
touching the ELBO.

The substantive reason is that this interface isolates an *approximation*. The
quantity the bound requires is

    E_{q(f)}[ log p(z = c | f) ]

which is not ``log softmax(E[f])``. Substituting the latter is a real
approximation with real bias: it discards the posterior variance of ``f``
entirely, so a confidently-wrong region of feature space looks identical to an
uncertain one. Hidden inside a model constructor, that substitution is an
invisible assumption. Behind this interface it is a constructor flag you can
switch and measure.

Shape contract
--------------
:meth:`LatentFunction.expected_log_p_z` returns ``[B, C]``, not a scalar. This
is deliberate. The scalar latent term of the ELBO is recoverable from it, but
the reverse is not, and the per-class breakdown is exactly what the closed-form
``q(Z)`` update consumes. Returning a scalar here would make
:class:`~gpcrowd.posteriors.ClosedFormZ` impossible to write without changing
this interface later.
"""

from __future__ import annotations

import abc

import gpflow
import tensorflow as tf

__all__ = ["LatentFunction"]


class LatentFunction(gpflow.Module, abc.ABC):
    """Abstract latent classifier supplying per-class log-likelihood terms.

    Implementations own whatever variational machinery they need -- inducing
    points, kernel hyperparameters, network weights. The core model knows only
    the three methods below.
    """

    @abc.abstractmethod
    def expected_log_p_z(self, X: tf.Tensor) -> tf.Tensor:
        """Expected log-likelihood of each class under the variational posterior.

        Args:
            X (tf.Tensor): Feature matrix. Shape ``[B, D]``, float64.

        Returns:
            tf.Tensor: Shape ``[B, C]``, float64. Entry ``(n, c)`` is
            ``E_{q(f_n)}[log p(z_n = c | f_n)]``.

        Note:
            These are unnormalised log-likelihood terms, not log-probabilities:
            the rows need not sum to anything in particular, and generally will
            not. Normalisation happens later, in ``q(Z)``, after the annotation
            evidence has been added.
        """

    @abc.abstractmethod
    def prior_kl(self) -> tf.Tensor:
        """KL divergence between the variational and prior latent processes.

        For a sparse variational GP this is ``KL(q(u) || p(u))`` over the
        inducing values -- the complexity penalty that keeps the fitted
        function from chasing noise.

        Returns:
            tf.Tensor: Scalar, non-negative.
        """

    @abc.abstractmethod
    def predict_class_probs(self, X: tf.Tensor) -> tf.Tensor:
        """Posterior predictive class probabilities for new inputs.

        This is the method that makes the trained model useful on data with no
        annotations at all -- the payoff of having fitted a classifier rather
        than merely denoising a label matrix.

        Args:
            X (tf.Tensor): Feature matrix. Shape ``[B, D]``, float64.

        Returns:
            tf.Tensor: Shape ``[B, C]``, rows summing to 1.
        """