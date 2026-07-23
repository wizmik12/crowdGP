"""Abstract contract for the latent classifier ``p(z | x)``.

The latent function is the part of the model that looks at *features* and 
predicts the *ground-truth* label. It can be thought as a latent classifier, 
and is the part of the model that generalizes to new items. 


Shape
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