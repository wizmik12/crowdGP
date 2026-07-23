"""Latent classifier strategies, ``p(z | x)``.

See [crowdgp.latent.base][crowdgp.latent.base] for the abstract contract, and
[crowdgp.latent.svgp][crowdgp.latent.svgp] for the sparse variational GP implementation
shipped with the library.
"""

from __future__ import annotations

from .base import LatentFunction
from .svgp import SVGPLatent

__all__ = ["LatentFunction", "SVGPLatent"]
