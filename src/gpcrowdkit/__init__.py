"""gpcrowdkit: Gaussian-process models for learning from crowds.

A minimal, composable implementation of scalable variational GP
crowdsourcing models in the style of SVGPCR (Morales-Alvarez et al., 2022).
A [GPCrowdModel][gpcrowdkit.models.GPCrowdModel] is assembled from three independently
swappable strategy objects, and knows nothing about which concrete strategy
it holds:

* a [LatentFunction][gpcrowdkit.latent.base.LatentFunction] -- maps item features to a
  distribution over the true class (the part that generalises to items no
  one has annotated);
* an [AnnotatorModel][gpcrowdkit.annotators.base.AnnotatorModel] -- describes each
  worker's labelling noise;
* a [PosteriorZ][gpcrowdkit.posteriors.PosteriorZ] -- the variational posterior over
  the unknown ground-truth labels, where the two evidence streams combine.

Quickstart::

    from gpcrowdkit import (
        FreeCategoricalZ, GPCrowdModel, SVGPLatent,
        VariationalDirichletAnnotator, make_synthetic, train,
    )
    import gpflow

    data = make_synthetic(num_items=300, num_workers=10)
    labels = data.labels
    probs = labels.empirical_class_probs()

    model = GPCrowdModel(
        latent=SVGPLatent(
            gpflow.kernels.SquaredExponential(),
            num_classes=labels.num_classes,
            inducing_points=data.X[:25],
        ),
        annotator=VariationalDirichletAnnotator(labels.num_workers, labels.num_classes),
        num_data=labels.num_items,
        q_z=FreeCategoricalZ(labels.num_items, labels.num_classes, init_probs=probs),
    )
    train(model, data.X, labels, iterations=300)
    inferred = model.infer_true_labels(data.X, labels)

See the ``examples/`` directory for complete, runnable scripts, and each
submodule's docstring for the design rationale behind that component.
"""

from __future__ import annotations

from .annotators.base import AnnotatorModel, ConfusionAnnotator
from .annotators.strategies import (
    OneCoinAnnotator,
    SoftmaxPointAnnotator,
    VariationalDirichletAnnotator,
    init_alpha_tilde,
)
from .data import CrowdBatch, CrowdLabels
from .inference import TrainingHistory, batch_iterator, train
from .latent.base import LatentFunction
from .latent.svgp import SVGPLatent
from .models import ELBOTerms, GPCrowdModel
from .posteriors import FreeCategoricalZ, PosteriorZ
from .synthetic import SyntheticCrowd, make_synthetic

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # data
    "CrowdBatch",
    "CrowdLabels",
    # latent
    "LatentFunction",
    "SVGPLatent",
    # annotators
    "AnnotatorModel",
    "ConfusionAnnotator",
    "VariationalDirichletAnnotator",
    "SoftmaxPointAnnotator",
    "OneCoinAnnotator",
    "init_alpha_tilde",
    # posteriors
    "PosteriorZ",
    "FreeCategoricalZ",
    # model + training
    "GPCrowdModel",
    "ELBOTerms",
    "train",
    "batch_iterator",
    "TrainingHistory",
    # synthetic data
    "SyntheticCrowd",
    "make_synthetic",
]
