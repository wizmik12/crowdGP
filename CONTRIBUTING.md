# Contributing

Thank you for considering contributing to this project! We welcome bug reports, bug fixes, documentation improvements, and new features.

Before contributing, please read this document in full — it explains our process **and** clarifies how code contributions relate (and do not relate) to authorship on any research papers describing this software.

## How to contribute

1. Open an issue first for anything beyond a trivial fix, so we can discuss the approach before you spend time on it.
2. Fork the repo and create a branch from `main`.
3. Make your changes, with tests where applicable.
4. Ensure your commits are signed off (see [Developer Certificate of Origin](#developer-certificate-of-origin-dco) below).
5. Open a pull request describing what you changed and why.
6. A maintainer will review your PR, may request changes, and will merge it once it's ready.

## Developer Certificate of Origin (DCO)

We use the [Developer Certificate of Origin](https://developercertificate.org/) instead of a Contributor License Agreement. This simply confirms you have the right to submit your contribution, and that it's licensed under this project's license (Apache License 2.0).

To sign off, add `-s` when committing:

```
git commit -s -m "Your commit message"
```

This adds a line to your commit message:

```
Signed-off-by: Your Name <your.email@example.com>
```

Pull requests with unsigned commits will not be merged until this is fixed.

## Code of conduct

Be respectful and constructive. Disagreements about code are fine; personal attacks are not.

---

## Authorship policy (please read before contributing)

This repository accompanies ongoing academic research. We want to be upfront and fair about how contributions are recognized, so there are no surprises later.

**Contributing code, documentation, bug reports, or reviews to this repository does not by itself confer authorship on any paper describing this software.**

We operate two separate recognition systems:

### 1. Software / code credit (this repo)

All contributors are credited here, regardless of the size of their contribution:

- Listed in [`CONTRIBUTORS.md`](CONTRIBUTORS.md)
- Listed in the `contributors` section of [`CITATION.cff`](CITATION.cff), which allows your contribution to be cited independently of any paper
- Mentioned in release notes when relevant
- Visible permanently in the git history

This is real, citable, indexable credit — many contributors prefer to cite the software directly via its own DOI (e.g., via Zenodo) rather than being listed on an unrelated paper.

### 2. Paper authorship (separate process)

Authorship on any manuscript describing this software is decided by the corresponding author(s)/PI, following standard [ICMJE authorship criteria](https://www.icmje.org/recommendations/browse/roles-and-responsibilities/defining-the-role-of-authors-and-contributors.html): substantial contribution to conception/design, or analysis/interpretation of data; drafting or substantively revising the manuscript; final approval; and accountability for the work.

In practice, this generally means:

- **Usually qualifies for authorship consideration:** designing a new core algorithm/method that becomes central to the paper's contribution, leading a major feature that the paper specifically evaluates, substantial writing/analysis contributions to the manuscript itself.
- **Usually does not, on its own:** bug fixes, refactoring, adding tests, documentation, minor features, code review, dependency updates.

This list is a guideline, not an exhaustive rule — if you believe your contribution meets the bar for authorship, **please raise it with the maintainers before or while opening your PR**, not after the paper is submitted. We're happy to have that conversation early.

Contributors who don't meet authorship criteria but made a meaningful contribution will typically be named in the paper's **Acknowledgments** section, which is the standard venue for this kind of recognition.

Questions about this policy can be raised as a GitHub issue or directed to the maintainers listed in `CITATION.cff`.
