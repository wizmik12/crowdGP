# crowdgp

Gaussian-process models for learning from crowds: fitting a classifier from
noisy, incomplete crowdsourced annotations instead of clean labels, jointly
with a model of each annotator's reliability.

The default configuration reproduces **SVGPCR**:

> Morales-Álvarez, P., Ruiz, P., Coughlin, S., Molina, R., & Katsaggelos, A. K.
> (2022). *Scalable Variational Gaussian Processes for Crowdsourcing: Glitch
> Detection in LIGO.* IEEE TPAMI, 44(3), 1534-1551.

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
from crowdgp import (
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
| Latent classifier | [`LatentFunction`](src/crowdgp/latent/base.py) | maps item features to per-class evidence | [`SVGPLatent`](src/crowdgp/latent/svgp.py) (sparse variational multi-class GP) |
| Annotator model | [`AnnotatorModel`](src/crowdgp/annotators/base.py) | describes each worker's labelling noise | [`VariationalDirichletAnnotator`](src/crowdgp/annotators/strategies.py), `SoftmaxPointAnnotator`, `OneCoinAnnotator` |
| Ground-truth posterior | [`PosteriorZ`](src/crowdgp/posteriors.py) | combines both evidence streams into `q(z_n = c)` | [`FreeCategoricalZ`](src/crowdgp/posteriors.py) |

`GPCrowdModel.elbo_terms` (in [`models.py`](src/crowdgp/models.py)) is the
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
[`CrowdLabels`](src/crowdgp/data.py) -- see its module docstring for why COO
storage and structural batch alignment were chosen over a dense label matrix.

## Testing

```bash
pytest -m "not slow"   # unit tests + smoke tests, seconds
pytest                  # + end-to-end training tests that verify the model
                        # actually beats majority vote and recovers the true
                        # confusion matrices, ~2 minutes
```

`src/crowdgp/tests/test_end_to_end.py` is the test module to read first if
you are changing modelling code: unlike the unit tests, it checks properties
that no amount of correct-looking code guarantees on its own.
