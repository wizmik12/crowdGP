"""Sparse variational Gaussian process backbone.

Wraps ``gpflow.models.SVGP`` by composition rather than inheritance. The GPflow
model is a component this class *has*, not a base class it *is*, which matters
because ``SVGP`` carries an inference API (``elbo``, ``training_loss``,
``predict_y``) built around observed labels ``Y``. In a crowdsourcing model
there are no observed labels -- only annotations and a variational posterior
over the truth -- so inheriting that API would expose a large surface of
methods that are meaningless or, worse, silently answer the wrong question.
"""

from __future__ import annotations

import inspect

import gpflow
import numpy as np
import tensorflow as tf

from .base import LatentFunction

__all__ = ["SVGPLatent"]

FLOAT = tf.float64


def _variational_expectations(
    likelihood: gpflow.likelihoods.Likelihood,
    X: tf.Tensor,
    Fmu: tf.Tensor,
    Fvar: tf.Tensor,
    Y: tf.Tensor,
) -> tf.Tensor:
    """Calls ``variational_expectations`` across GPflow 2.x signature changes.

    GPflow added a leading ``X`` argument in 2.7 to support input-dependent
    likelihoods. Dispatching on the parameter count keeps the library working
    either side of that change.

    Args:
        likelihood: The GPflow likelihood.
        X (tf.Tensor): Inputs. Shape ``[B, D]``.
        Fmu (tf.Tensor): Latent means. Shape ``[B, C]``.
        Fvar (tf.Tensor): Latent variances. Shape ``[B, C]``.
        Y (tf.Tensor): Candidate labels. Shape ``[B, 1]``, int32.

    Returns:
        tf.Tensor: Per-point expected log density, shape ``[B]`` or ``[B, 1]``.
    """
    n_params = len(inspect.signature(likelihood.variational_expectations).parameters)
    if n_params >= 4:
        return likelihood.variational_expectations(X, Fmu, Fvar, Y)
    return likelihood.variational_expectations(Fmu, Fvar, Y)


class SVGPLatent(LatentFunction):
    """Sparse variational multi-class GP over the latent ground truth.

    ``C`` independent latent GPs share a kernel; a multi-class likelihood maps
    them to class probabilities. Sparsity comes from ``M`` inducing points, so
    cost is ``O(B M^2)`` per step rather than ``O(N^3)``, which is what makes
    minibatch training over large annotation sets feasible at all.

    Attributes:
        svgp (gpflow.models.SVGP): The underlying GPflow model, held by
            composition. Its kernel hyperparameters, inducing locations and
            variational parameters are tracked automatically through
            ``tf.Module``.
        num_classes (int): Number of classes ``C``.
        quadrature (bool): Selects the approximation to
            ``E_{q(f)}[log p(z|f)]``. True uses the likelihood's own quadrature
            over ``q(f)``, matching the reference SVGPCR implementation. False
            uses the cheaper ``log softmax(E[f])`` plug-in, which ignores the
            posterior variance.
    """

    def __init__(
        self,
        kernel: gpflow.kernels.Kernel,
        num_classes: int,
        inducing_points: np.ndarray,
        likelihood: gpflow.likelihoods.Likelihood | None = None,
        mean_function: gpflow.mean_functions.MeanFunction | None = None,
        whiten: bool = True,
        quadrature: bool = True,
    ) -> None:
        """Builds the SVGP backbone.

        Args:
            kernel: Kernel shared by the ``C`` latent GPs.
            num_classes: Number of classes ``C``.
            inducing_points: Initial inducing locations. Shape ``[M, D]``.
                A k-means summary of the data, or a random subset, both work;
                they are optimised along with everything else.
            likelihood: Multi-class likelihood. Defaults to
                ``MultiClass(C, invlink=RobustMax(C))``, the reference choice,
                whose expectations are available in closed form.
            mean_function: Optional GP mean function; zero if omitted.
            whiten: Use the whitened inducing representation ``u = L v``. Kept
                True by default: it decorrelates the variational parameters
                from the kernel hyperparameters and markedly improves
                conditioning during joint optimisation.
            quadrature: See the class attributes.

        Note:
            ``num_latent_gps=num_classes`` is what makes this multi-class. Omit
            it and GPflow defaults to one latent function, which trains without
            complaint and can only ever express a binary decision.
        """
        super().__init__()
        self.num_classes = int(num_classes)
        self.quadrature = bool(quadrature)

        if likelihood is None:
            likelihood = gpflow.likelihoods.MultiClass(
                num_classes, invlink=gpflow.likelihoods.RobustMax(num_classes)
            )
        self.svgp = gpflow.models.SVGP(
            kernel=kernel,
            likelihood=likelihood,
            inducing_variable=np.asarray(inducing_points, dtype=np.float64),
            num_latent_gps=num_classes,
            mean_function=mean_function,
            whiten=whiten,
        )

    def expected_log_p_z(self, X: tf.Tensor) -> tf.Tensor:
        """See [expected_log_p_z][crowdgp.latent.base.LatentFunction.expected_log_p_z].

        Because the true class is unknown, the term is evaluated once per
        candidate class and the results assembled column-wise. The loop runs
        over ``C``, a small fixed number known at trace time, so it unrolls
        harmlessly under ``tf.function`` -- unlike a loop over annotators.

        Args:
            X (tf.Tensor): Feature matrix. Shape ``[B, D]``.

        Returns:
            tf.Tensor: Shape ``[B, C]``, float64.
        """
        f_mean, f_var = self.svgp.predict_f(X, full_cov=False, full_output_cov=False)

        if not self.quadrature:
            # Plug-in approximation: discards f_var entirely.
            return tf.nn.log_softmax(f_mean, axis=-1)

        n = tf.shape(X)[0]
        columns = []
        for c in range(self.num_classes):
            Y_c = tf.fill([n, 1], tf.constant(c, dtype=tf.int32))
            ve = _variational_expectations(self.svgp.likelihood, X, f_mean, f_var, Y_c)
            columns.append(tf.reshape(ve, [n, 1]))
        return tf.concat(columns, axis=-1)

    def prior_kl(self) -> tf.Tensor:
        """See [prior_kl][crowdgp.latent.base.LatentFunction.prior_kl].

        Delegates to GPflow, which handles the whitened and unwhitened cases.

        Returns:
            tf.Tensor: Scalar.
        """
        return self.svgp.prior_kl()

    def predict_class_probs(self, X: tf.Tensor) -> tf.Tensor:
        """See [predict_class_probs][crowdgp.latent.base.LatentFunction.predict_class_probs].

        Args:
            X (tf.Tensor): Feature matrix. Shape ``[B, D]``.

        Returns:
            tf.Tensor: Shape ``[B, C]``, rows summing to 1.

        Note:
            Uses ``predict_y``, which propagates the posterior variance of
            ``f`` through the likelihood, rather than ``invlink(f_mean)``,
            which does not. The difference shows up exactly where it matters:
            far from the inducing points, where the GP is uncertain and should
            say so.
        """
        return self.svgp.predict_y(X)[0]