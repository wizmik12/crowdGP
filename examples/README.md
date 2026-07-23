# Examples

Runnable sanity checks for the library, in increasing order of detail. Each is
a plain script, not a notebook, so it can be run in CI or from the command
line as a regression check on the modelling code.

| Script | What it checks |
|---|---|
| [`01_quickstart.py`](01_quickstart.py) | End-to-end fit; asserts the model beats majority vote. |
| [`02_compare_annotator_strategies.py`](02_compare_annotator_strategies.py) | The three shipped `AnnotatorModel` strategies are interchangeable behind the same `GPCrowdModel`. |
| [`03_diagnostics_plots.py`](03_diagnostics_plots.py) | Plots the ELBO decomposition and true-vs-recovered confusion matrices to `output/`. Requires `pip install -e ".[examples]"`. |

Run any of them with the project's virtualenv and GPU library paths already
configured:

```bash
./run.sh examples/01_quickstart.py
```
