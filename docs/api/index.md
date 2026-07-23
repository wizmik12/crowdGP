# API reference

Generated from the library's docstrings. Start with [`GPCrowdModel`](models.md)
-- it is the whole core engine -- then the three strategy families it
composes:

| Page | Contents |
|---|---|
| [Data (`CrowdLabels`)](data.md) | Sparse COO storage for annotations, minibatch gathering, majority vote / vote-histogram baselines. |
| [Model (`GPCrowdModel`)](models.md) | The composed model and the ELBO decomposition. |
| [Latent classifier](latent.md) | The abstract `LatentFunction` contract and the `SVGPLatent` implementation. |
| [Annotator strategies](annotators.md) | The abstract `AnnotatorModel`/`ConfusionAnnotator` contract and the three shipped worker-noise strategies. |
| [Ground-truth posterior](posteriors.md) | The abstract `PosteriorZ` contract and `FreeCategoricalZ`. |
| [Training](inference.md) | The minibatching and optimisation loop. |
| [Synthetic data](synthetic.md) | Synthetic crowds with known ground truth, for testing and the examples. |

See [Theory](../theory.md) for the mathematics behind these objects, and
[Examples](../examples.md) for runnable end-to-end scripts.
