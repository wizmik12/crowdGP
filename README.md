# gpcrowdkit

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Status: research, active development](https://img.shields.io/badge/status-research%20%2F%20active%20development-orange.svg)](#status)

Gaussian-process models for learning from crowds: fitting a classifier from
noisy, incomplete crowdsourced annotations instead of clean labels, jointly
with a model of each annotator's reliability.

The default configuration reproduces **SVGPCR**:

> Morales-Álvarez, P., Ruiz, P., Coughlin, S., Molina, R., & Katsaggelos, A. K.
> (2022). *Scalable Variational Gaussian Processes for Crowdsourcing: Glitch
> Detection in LIGO.* IEEE TPAMI, 44(3), 1534-1551.

## Status

Research code under active development, released alongside ongoing work from
the authors' group. The public API (`gpcrowdkit.*`, see [Architecture](#architecture))
is reasonably stable, but interfaces can still change between versions ahead
of a first tagged release -- pin a commit if you depend on it for a
publication. Issues and pull requests are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md)
for which areas are open and how contributions are credited. A citation entry
(BibTeX / paper reference) will be added here once the accompanying
publication is available.

## Why a GP, and not just a confusion-matrix model

A pure crowd-aggregation model (Dawid-Skene and its descendants) only
denoises the label matrix: it has nothing to say about an item no one has
annotated. Here the ground-truth posterior is coupled to a sparse variational
GP over the item *features*, so the fitted classifier generalises to
unannotated data -- the payoff for the extra modelling effort, and the thing
[`examples/`](examples/) is built to check for.

## Install

```bash
pip install -e ".[dev,examples]"
```

`dev` pulls in `pytest`; `examples` pulls in `matplotlib` for the plotting
demo. Both are optional -- the core library only needs `numpy`, `tensorflow`
and `gpflow`.

## Quickstart

```python
import gpflow
from gpcrowdkit import (
    FreeCategoricalZ, GPCrowdModel, SVGPLatent,
    VariationalDirichletAnnotator, init_alpha_tilde, make_synthetic, train,
)

data = make_synthetic(num_items=300, num_workers=10)     # or CrowdLabels.from_pairs(...) on real data
labels = data.labels
class_probs = labels.empirical_class_probs()              # smoothed vote histogram, used to warm-start

model = GPCrowdModel(
    latent=SVGPLatent(
        gpflow.kernels.SquaredExponential(),
        num_classes=labels.num_classes,
        inducing_points=data.X[:25],
    ),
    annotator=VariationalDirichletAnnotator(
        labels.num_workers, labels.num_classes,
        alpha_tilde_init=init_alpha_tilde(labels, class_probs),
    ),
    num_data=labels.num_items,
    q_z=FreeCategoricalZ(labels.num_items, labels.num_classes, init_probs=class_probs),
)

train(model, data.X, labels, iterations=300)

model.infer_true_labels(data.X, labels)     # denoised labels for the annotated items
model.predict_class_probs(new_X)            # predictions for items no one has annotated
```

See [`examples/`](examples/) for complete, runnable scripts, including a
sanity check that the fitted model beats plain majority vote, and a plotting
script for the ELBO decomposition and confusion-matrix recovery.

## Architecture

`GPCrowdModel` is assembled from three independently swappable strategy
objects and never inspects which concrete implementation it holds -- it only
calls each one's abstract contract. Composition, not inheritance: adding a
fourth annotator model or a closed-form `q(Z)` update is a new class, not a
change to the core engine.

| Component | Abstract base | Job | Shipped implementations |
|---|---|---|---|
| Latent classifier | [`LatentFunction`](src/gpcrowdkit/latent/base.py) | maps item features to per-class evidence | [`SVGPLatent`](src/gpcrowdkit/latent/svgp.py) (sparse variational multi-class GP) |
| Annotator model | [`AnnotatorModel`](src/gpcrowdkit/annotators/base.py) | describes each worker's labelling noise | [`VariationalDirichletAnnotator`](src/gpcrowdkit/annotators/strategies.py), `SoftmaxPointAnnotator`, `OneCoinAnnotator` |
| Ground-truth posterior | [`PosteriorZ`](src/gpcrowdkit/posteriors.py) | combines both evidence streams into `q(z_n = c)` | [`FreeCategoricalZ`](src/gpcrowdkit/posteriors.py) |

`GPCrowdModel.elbo_terms` (in [`models.py`](src/gpcrowdkit/models.py)) is the
entire core engine: ask the latent function what the features imply, ask the
annotator what the workers imply, ask `q(Z)` to combine them, weight and add.
Everything else is in the strategies.

**To add a new strategy**, subclass the relevant abstract base and implement
its contract -- e.g. a new `AnnotatorModel` need only implement
`label_log_terms` (per-annotation expected log-likelihood under each
candidate true class) and `kl_divergence`; the aggregation onto items, the
ELBO term, and the interaction with `q(Z)` are all handled generically by the
base class. See the docstrings in `annotators/base.py` and `latent/base.py`
for the full contract and the reasoning behind its shape.

Data flows through the library as sparse `(item, worker, label)` triples via
[`CrowdLabels`](src/gpcrowdkit/data.py) -- see its module docstring for why COO
storage and structural batch alignment were chosen over a dense label matrix.

## Theory

**[`docs/theory.md`](docs/theory.md)** derives the full generative model and
the evidence lower bound this library optimises -- the sparse variational
multi-class GP, the robust-max link and its Gaussian-quadrature expectation,
the three annotator confusion-matrix models (including the Dirichlet KL and
posterior-mean formulas), and the variational posterior over ground truth --
with every term tied to the line of code that computes it. Read it before
changing modelling code or adding a new strategy.

## Documentation site

The same theory, an auto-generated API reference, and the examples are also
published as a browsable site (MkDocs + Material, [`mkdocs.yml`](mkdocs.yml)),
built by [Read the Docs](https://readthedocs.org) from
[`.readthedocs.yaml`](.readthedocs.yaml) on every push to `main`, once the
repository is imported there (**Sign in → Add project → select
`wizmik12/gpcrowdkit`** on readthedocs.org; a one-time step only a project owner
can do). It will then live at `https://gpcrowdkit.readthedocs.io/`.

The docs build only needs the MkDocs/mkdocstrings toolchain
([`docs/requirements.txt`](docs/requirements.txt)), not the library's own
dependencies -- `mkdocstrings` reads docstrings via static analysis, never by
importing `gpcrowdkit`, so TensorFlow and GPflow are never installed just to
build docs. To build it locally:

```bash
pip install -e ".[docs]"
mkdocs serve       # live preview at http://127.0.0.1:8000
mkdocs build        # static site in site/
```

## Testing

```bash
pytest -m "not slow"   # unit tests + smoke tests, seconds
pytest                  # + end-to-end training tests that verify the model
                        # actually beats majority vote and recovers the true
                        # confusion matrices, ~2 minutes
```

`src/gpcrowdkit/tests/test_end_to_end.py` is the test module to read first if
you are changing modelling code: unlike the unit tests, it checks properties
that no amount of correct-looking code guarantees on its own.

## License

Licensed under the [Apache License, Version 2.0](LICENSE) -- permissive,
patent-granting, and the same license used by TensorFlow and GPflow, which
this library depends on. You may use, modify, and redistribute this code,
including commercially, provided you retain the copyright and license
notices; see the [LICENSE](LICENSE) file for the full text.
