"""Latent classifier strategies, ``p(z | x)``.

See [gpcrowdkit.latent.base][gpcrowdkit.latent.base] for the abstract contract, and
[gpcrowdkit.latent.svgp][gpcrowdkit.latent.svgp] for the sparse variational GP implementation
shipped with the library.
"""

from __future__ import annotations

from .base import LatentFunction
from .svgp import SVGPLatent

__all__ = ["LatentFunction", "SVGPLatent"]
