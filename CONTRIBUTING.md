# Contributing

Thanks for your interest in contributing! Bug reports, bug fixes, documentation, tests, and new features are all welcome.

This project accompanies ongoing academic research, so this guide also explains — plainly, and up front — how contributions are credited and how that relates to authorship on any paper describing the software. The short version: contributing is welcome and always credited, and code credit and paper authorship are handled separately.

## Getting started

1. **Open an issue first** for anything beyond a small fix, so we can agree on the approach before you invest time. Some parts of the library are under active research and may not be open to outside contributions yet — opening an issue first means we can let you know before you start, rather than after.
2. **Fork** the repository and create a branch from `main`.
3. **Make your changes.** Add or update tests where it makes sense, and keep the change focused — one logical change per pull request is much easier to review than many.
4. **Open a pull request** describing what you changed and why. Link the issue it addresses.
5. A maintainer will review it, may suggest changes, and will merge it once it's ready. Requests for changes are a normal part of review, not a rejection.

## Development setup

```bash
git clone https://github.com/wizmik12/crowdGP
cd crowdGP
pip install -e ".[dev]"
pytest -m "not slow"      # fast unit + smoke tests, a few seconds
pytest                     # everything, including end-to-end training (~2 min)
```

## Code guidelines

A few conventions keep this codebase consistent and, more importantly, correct. The first two are not style preferences — getting them wrong produces code that runs and returns plausible numbers while being silently wrong.

- **The confusion-matrix convention is load-bearing.** Confusion tensors are `[A, C_obs, C_true]`, normalised down **axis 1**, so each *column* is a distribution over observed labels given a fixed true class. Code that touches confusion matrices should include a test comparing against the transpose, not only against the expected values — a transposed convention passes most other checks.
- **No Python loops over annotators or annotations, and no Python branching on tensor shapes**, in anything that runs inside `tf.function`. Loops unroll into one subgraph copy per iteration; shape branches work eagerly and raise in graph mode. Use `gather_nd`, `unsorted_segment_sum`, or `np.add.at`.
- **float64 throughout.** GPflow defaults to it and the variational maths needs it; call `gpflow.config.set_default_float(np.float64)`.
- **Docstrings** in Google style, and every tensor argument documents its shape. Scientific code is unreadable without this.
- **Tests** should use hand-computed expected values, not values produced by running the implementation — the latter pin current behaviour rather than catching regressions. New models should be checked against `make_synthetic`, where the true labels and confusion matrices are known. Mark anything slow with `@pytest.mark.slow`.
- **Style**: line length 100, checked with `ruff`; types checked with `mypy`. Both are configured in `pyproject.toml`.

## Code of conduct

Be respectful and constructive. Disagreements about code are welcome; personal attacks are not.

## Licensing of contributions

This project is licensed under the [Apache License 2.0](LICENSE). By opening a pull request, you agree that your contribution is provided under that same license (this is what Apache-2.0 Section 5 already states for contributions to a project carrying its notice). You retain copyright in the code you write — contributing does not transfer ownership.

---

## How contributions are credited

We want this clear before you contribute, so nothing comes as a surprise later. There are two separate systems, and they are not the same thing.

### Software credit — for everyone

Every merged contribution is credited, regardless of size:

- your name in [`CONTRIBUTORS.md`](CONTRIBUTORS.md),
- permanent attribution in the git history,
- a mention in release notes where relevant.

When the software has a tagged, archived release (e.g. a Zenodo DOI), it can be cited in its own right, independently of any paper.

### Paper authorship — a separate decision

**Contributing code, docs, tests, or reviews does not by itself make you an author on a paper describing this software.** This is standard practice, not a judgement about the value of the work.

Authorship follows the [ICMJE criteria](https://www.icmje.org/recommendations/browse/roles-and-responsibilities/defining-the-role-of-authors-and-contributors.html), which require *all* of: a substantial intellectual contribution to the work; involvement in drafting or revising the manuscript; approval of the final version; and accountability for it. Code alone therefore never suffices — authorship also means engaging with the paper itself.

In practice:

- Bug fixes, refactoring, tests, documentation, and small features are credited as software contributions (above), and meaningful contributions are also named in the paper's **Acknowledgements**.
- Larger contributions — designing a method the paper builds on, leading a component it evaluates, or contributing to the writing and analysis — are assessed individually.

If you think your contribution is heading toward that level, **please raise it early**, while you're working on it rather than after a paper is submitted. Journal author lists are fixed at submission and awkward to change afterward, so an early conversation genuinely serves both sides. Co-authorship also carries obligations: co-authors provide an ORCID iD, review and approve the manuscript, and stay reachable through submission and revision.

Questions about any of this are welcome — open an issue, or contact the maintainer listed in [`CITATION.cff`](CITATION.cff).
