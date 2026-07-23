# Examples

Runnable sanity checks for the library, in increasing order of detail. Each
is a plain script, not a notebook, so it can be run in CI or from the command
line as a regression check on the modelling code. The full source for each
is in [`examples/`](https://github.com/wizmik12/crowdGP/tree/main/examples)
in the repository.

## [`01_quickstart.py`](https://github.com/wizmik12/crowdGP/blob/main/examples/01_quickstart.py)

End-to-end fit on a deliberately hard synthetic crowd (only half the workers
reliable, three annotations per item). Asserts that the fitted model beats
plain majority vote, and prints the ELBO decomposition and the mean absolute
error of the recovered worker confusion matrices.

## [`02_compare_annotator_strategies.py`](https://github.com/wizmik12/crowdGP/blob/main/examples/02_compare_annotator_strategies.py)

Trains the same latent GP against the same data with each of the three
shipped [`AnnotatorModel`](api/annotators.md) strategies in turn -- the
demonstration that they really are interchangeable behind `GPCrowdModel`,
with only the one line that constructs `annotator=` changing.

## [`03_diagnostics_plots.py`](https://github.com/wizmik12/crowdGP/blob/main/examples/03_diagnostics_plots.py)

Plots the ELBO decomposition (checking that the `latent` term does not
flatline while `crowd` climbs -- the characteristic crowdsourcing failure
mode) and true-vs-recovered confusion matrices, saved to `examples/output/`.
Requires `pip install -e ".[examples]"` for `matplotlib`.

## Running them

```bash
./run.sh examples/01_quickstart.py
```

`run.sh` configures the project's virtualenv and GPU library paths; run the
script directly with your own interpreter if you manage the environment
differently.
