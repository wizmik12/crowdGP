"""Annotator (worker-noise) strategies.

See [gpcrowdkit.annotators.base][gpcrowdkit.annotators.base] for the abstract contract every strategy
implements, and [gpcrowdkit.annotators.strategies][gpcrowdkit.annotators.strategies] for the concrete
strategies shipped with the library.
"""

from __future__ import annotations

from .base import AnnotatorModel, ConfusionAnnotator
from .strategies import (
    OneCoinAnnotator,
    SoftmaxPointAnnotator,
    VariationalDirichletAnnotator,
    init_alpha_tilde,
)

__all__ = [
    "AnnotatorModel",
    "ConfusionAnnotator",
    "VariationalDirichletAnnotator",
    "SoftmaxPointAnnotator",
    "OneCoinAnnotator",
    "init_alpha_tilde",
]
