# Theory: probabilistic formulation

This document derives the generative model and the variational inference
scheme that `gpcrowdkit` implements, and connects every term to the code that
computes it. It assumes familiarity with sparse variational Gaussian
processes and mean-field variational inference; see the
[References](#references) section for the papers this reproduces.

If you only want to *use* the library, the [README](https://github.com/wizmik12/gpcrowdkit/blob/main/README.md) is enough.
Read this document when you are changing modelling code, adding a new
strategy, or need to know exactly which quantity a given line of code is
computing.

## Contents

- [Notation](#notation)
- [The generative model](#the-generative-model)
- [Variational family](#variational-family)
- [The evidence lower bound](#the-evidence-lower-bound)
- [The latent classifier: sparse variational multi-class GP](#the-latent-classifier-sparse-variational-multi-class-gp)
- [The annotator noise model](#the-annotator-noise-model)
- [The posterior over ground truth, *q(Z)*](#the-posterior-over-ground-truth-qz)
- [Stochastic optimization over minibatches](#stochastic-optimization-over-minibatches)
- [Prediction](#prediction)
- [Relationship to the literature](#relationship-to-the-literature)
- [References](#references)

## Notation

| Symbol | Meaning | Code |
|---|---|---|
| $N$ | number of items | `CrowdLabels.num_items` |
| $A$ | number of annotators (workers) | `CrowdLabels.num_workers` |
| $C$ | number of classes | `CrowdLabels.num_classes` |
| $L$ | number of annotations (one worker labelling one item, once) | `CrowdLabels.num_labels` |
| $x_n \in \mathbb{R}^D$ | features of item $n$ | rows of `X` |
| $z_n \in \{1,\dots,C\}$ | true (unobserved) class of item $n$ | never observed |
| $y_l \in \{1,\dots,C\}$ | the label given by annotation $l$ | `label` |
| $i(l), a(l)$ | the item and worker of annotation $l$ | `item_idx`, `worker_idx` |
| $f_c(\cdot)$, $c=1,\dots,C$ | latent GPs, one per class, sharing one kernel | `SVGPLatent.svgp` |
| $R^a \in [0,1]^{C\times C}$ | worker $a$'s confusion matrix, columns sum to 1 | `AnnotatorModel.confusion_matrices()` |
| $\gamma_{nc} = q(z_n = c)$ | variational responsibility | `PosteriorZ.gamma` |
| $B$ | minibatch size (items) | `CrowdBatch.size` |

Throughout, $R^a_{ij} = P(\text{observed} = i \mid \text{true} = j)$ for worker
$a$ — **columns**, not rows, are the probability vectors. This is the
convention fixed in [`annotators/base.py`](https://github.com/wizmik12/gpcrowdkit/blob/main/src/gpcrowdkit/annotators/base.py);
sampling or normalising along the wrong axis produces a model that trains
happily and learns the transpose of the intended one (see
`test_recovers_confusion_matrices` in
[`test_end_to_end.py`](https://github.com/wizmik12/gpcrowdkit/blob/main/src/gpcrowdkit/tests/test_end_to_end.py), which exists
specifically to catch that).

## The generative model

For $c = 1, \dots, C$, a zero-mean GP prior over a latent function, all $C$
functions sharing one kernel $k(\cdot,\cdot)$ and one set of $M$ inducing
locations $Z = \{z_m\}_{m=1}^M$:

$$
f_c \sim \mathcal{GP}(0, k), \qquad u_c = f_c(Z) \sim \mathcal{N}(0, K_{ZZ}).
$$

For each item $n$, the true class is drawn through a **robust-max** link
(Hernández-Lobato et al., 2011) rather than a softmax, because it admits a
closed-form-ish expectation under a Gaussian $q(f)$ (below):

$$
p(z_n = c \mid f(x_n)) =
\begin{cases}
1-\varepsilon & c = \arg\max_{c'} f_{c'}(x_n) \\
\varepsilon / (C-1) & \text{otherwise,}
\end{cases}
$$

with $\varepsilon$ a small, fixed label-noise floor (GPflow's default
$\varepsilon = 10^{-3}$, held non-trainable). This is the prior half of the
model — it is what lets the fitted classifier answer `predict_class_probs`
for an item **no worker has ever seen**, which a pure vote-aggregation model
cannot do.

For each worker $a$ and each true class $j$, a confusion column with a
Dirichlet prior:

$$
R^a_{\cdot, j} \sim \mathrm{Dir}(\alpha^a_{\cdot,j}), \qquad
p(y_l = i \mid z_{i(l)} = j,\, R) = R^{a(l)}_{ij}.
$$

Annotations are conditionally independent given the true label and the
annotator's own confusion matrix — the standard Dawid–Skene assumption, and
the reason `crowd_log_per_item` (below) is a **sum** over one item's
annotations rather than something more elaborate.

The full joint distribution is

$$
p(X, f, Z, Y, R) \;=\; \underbrace{\prod_{c=1}^{C} p(u_c)\,p(f_c \mid u_c)}_{\text{GP prior}}
\; \prod_{n=1}^{N} p(z_n \mid f(x_n))
\; \underbrace{\prod_{a=1}^A \prod_{j=1}^C p(R^a_{\cdot,j})}_{\text{annotator prior}}
\; \prod_{l=1}^{L} p(y_l \mid z_{i(l)}, R).
$$

$X$, $Y$ (the annotations) are observed; $f$, $Z$, $R$ are not.

## Variational family

Mean-field across the three latent groups, matching the three objects
composed by [`GPCrowdModel`](https://github.com/wizmik12/gpcrowdkit/blob/main/src/gpcrowdkit/models.py):

$$
q(f, Z, R) \;=\; \Big[\prod_{c=1}^C p(f_c \mid u_c)\, q(u_c)\Big]
\;\times\; \Big[\prod_{n=1}^N q(z_n)\Big]
\;\times\; \Big[\prod_{a,j} q(R^a_{\cdot,j})\Big].
$$

* $q(u_c) = \mathcal{N}(m_c, S_c)$ — handled by
  [`SVGPLatent`](https://github.com/wizmik12/gpcrowdkit/blob/main/src/gpcrowdkit/latent/svgp.py) (a `gpflow.models.SVGP`);
  $p(f_c\mid u_c)$ is kept unintegrated on both sides, in the usual sparse-GP
  trick, so it cancels in the ELBO below and never needs to be evaluated.
* $q(z_n) = \mathrm{Categorical}(\gamma_n)$, $\gamma_n$ a free point in the
  $(C-1)$-simplex — [`FreeCategoricalZ`](https://github.com/wizmik12/gpcrowdkit/blob/main/src/gpcrowdkit/posteriors.py).
* $q(R^a_{\cdot,j})$ is either a Dirichlet
  ($\mathrm{Dir}(\tilde\alpha^a_{\cdot,j})$,
  [`VariationalDirichletAnnotator`](https://github.com/wizmik12/gpcrowdkit/blob/main/src/gpcrowdkit/annotators/strategies.py))
  or a Dirac point mass at a deterministic estimate
  ([`SoftmaxPointAnnotator`](https://github.com/wizmik12/gpcrowdkit/blob/main/src/gpcrowdkit/annotators/strategies.py),
  [`OneCoinAnnotator`](https://github.com/wizmik12/gpcrowdkit/blob/main/src/gpcrowdkit/annotators/strategies.py)) — see
  [below](#the-annotator-noise-model).

## The evidence lower bound

Substituting the variational family into the standard bound and cancelling
$p(f_c\mid u_c)$ against its counterpart in $q$:

$$
\log p(Y) \;\ge\;
\underbrace{\sum_{c=1}^C \mathbb{E}_{q(u_c)}\big[\log p(u_c) - \log q(u_c)\big]}_{-\,\mathrm{KL}(q(u)\,\|\,p(u))}
\;+\;
\underbrace{\sum_n \mathbb{E}_{q(z_n)}\mathbb{E}_{q(f(x_n))}\big[\log p(z_n \mid f(x_n))\big]}_{\texttt{latent}}
$$
$$
+\;
\underbrace{\sum_{a,j}\mathbb{E}_{q(R^a_{\cdot,j})}\big[\log p(R^a_{\cdot,j}) - \log q(R^a_{\cdot,j})\big]}_{-\,\mathrm{KL}(q(R)\,\|\,p(R))}
\;+\;
\underbrace{\sum_n \mathbb{E}_{q(z_n)}\mathbb{E}_{q(R)}\big[\log p(y_{\cdot} \mid z_n, R)\big]}_{\texttt{crowd}}
\;+\;
\underbrace{\sum_n \mathrm{H}[q(z_n)]}_{\texttt{entropy}}.
$$

This is exactly the five-term decomposition returned by
[`GPCrowdModel.elbo_terms`](https://github.com/wizmik12/gpcrowdkit/blob/main/src/gpcrowdkit/models.py) as
`ELBOTerms(latent, crowd, entropy, kl_latent, kl_annotator, scale)`, and

```
total = (latent + crowd + entropy) * scale - kl_latent - kl_annotator
```

is exactly the bound above (the `scale` factor is the minibatch correction,
[discussed later](#stochastic-optimization-over-minibatches); it multiplies
only the three sums over items, never the two KL terms, because the KLs are
already exact for the *whole* dataset in every minibatch — they are
properties of global parameters, not of the batch).

Reporting these five numbers separately, rather than only their sum, is not
cosmetic. The characteristic crowdsourcing failure is a model where `crowd`
climbs steadily while `latent` stays flat: the total ELBO still rises, the
model still converges, and it has learned to reproduce the annotations while
ignoring the features entirely — useless on anything unannotated, and
invisible unless you look at the decomposition. This is exactly what
[`examples/03_diagnostics_plots.py`](https://github.com/wizmik12/gpcrowdkit/blob/main/examples/03_diagnostics_plots.py)
plots, and what `test_latent_term_actually_improves` in
[`test_end_to_end.py`](https://github.com/wizmik12/gpcrowdkit/blob/main/src/gpcrowdkit/tests/test_end_to_end.py) checks for.

## The latent classifier: sparse variational multi-class GP

**`latent` term.** For item $n$ and each *candidate* true class $c$,

$$
\texttt{gp\_log}_{n,c} \;=\; \mathbb{E}_{q(f(x_n))}\big[\log p(z_n = c \mid f(x_n))\big],
$$

returned as an $[N,C]$ (or $[B,C]$ for a batch) tensor by
`SVGPLatent.expected_log_p_z` — **not yet weighted by $\gamma$**. The
weighting happens once, centrally, in `GPCrowdModel.elbo_terms`:
`latent = sum(gamma * gp_log)`. Returning the unweighted per-class tensor
rather than folding in $\gamma$ here is what lets $q(Z)$ be swapped out later
without touching this class (see [below](#the-posterior-over-ground-truth-qz)).

Two ways to compute the expectation, selected by `SVGPLatent(quadrature=...)`:

* **`quadrature=True`** (default, matches the reference model). $q(f(x_n))$
  is Gaussian with mean/variance $(\mu_n, \sigma_n^2) \in \mathbb{R}^C\times\mathbb{R}^C$
  from the sparse GP posterior. Writing $p_c = \Pr_{q(f)}\big(\arg\max f = c\big)$
  (a multivariate orthant probability, approximated by 20-point Gauss–Hermite
  quadrature against the marginal of $f_c$; this is GPflow's
  `RobustMax.prob_is_largest`, following Hernández-Lobato et al., 2011):

  $$
  \texttt{gp\_log}_{n,c} \;=\; p_c \log(1-\varepsilon) + (1-p_c)\log\frac{\varepsilon}{C-1}.
  $$

  This is an *expectation of a log-likelihood*, exact given the quadrature
  approximation to $p_c$ — it uses the full posterior, mean **and**
  variance.

* **`quadrature=False`** (cheaper plug-in). $\texttt{gp\_log}_{n,c} =
  \log\mathrm{softmax}_c(\mu_n)$ — evaluated at the posterior mean only,
  discarding $\sigma_n^2$ entirely, and silently substituting a softmax link
  for the robust-max prior. It is a fast, common approximation, not a
  different exact quantity for the same model; prefer `quadrature=True`
  unless profiling says otherwise.

**`kl_latent` term.** With the whitened parameterisation ($u_c = L_c v_c$,
$L_c L_c^\top = K_{ZZ}$, prior $v_c \sim \mathcal{N}(0, I_M)$, posterior
$q(v_c) = \mathcal{N}(m_c, S_c)$):

$$
\mathrm{KL}(q(u)\,\|\,p(u)) \;=\; \sum_{c=1}^C \tfrac12\Big[\mathrm{tr}(S_c) + m_c^\top m_c - M - \log\det S_c\Big],
$$

computed by `gpflow.models.SVGP.prior_kl()` and exposed as
`SVGPLatent.prior_kl()`. Whitening is what keeps this well-conditioned when
the kernel hyperparameters are also being optimised jointly (Hensman et al.,
2015) — see the constructor docstring in
[`latent/svgp.py`](https://github.com/wizmik12/gpcrowdkit/blob/main/src/gpcrowdkit/latent/svgp.py).

## The annotator noise model

**`crowd` term.** For each annotation $l$ and each candidate true class $c$
of its item:

$$
\texttt{label\_log\_terms}_{l,c} \;=\; \mathbb{E}_{q(R)}\big[\log p(y_l \mid z_{i(l)} = c,\, R)\big]
\;=\; \mathbb{E}_{q(R)}\big[\log R^{a(l)}_{y_l, c}\big],
$$

aggregated onto items by summing the annotations that share an item (the
conditional-independence assumption stated above):

$$
\texttt{crowd\_log}_{n,c} \;=\; \sum_{l:\, i(l)=n} \texttt{label\_log\_terms}_{l,c},
$$

which is `AnnotatorModel.crowd_log_per_item` — an
`unsorted_segment_sum` over `item_local`. `crowd` in the ELBO is then
`sum(gamma * crowd_log_per_item)`, the same gamma-weighting pattern as
`latent`.

Three interchangeable strategies supply
`label_log_terms`/`expected_log_confusion` and `kl_divergence`; the engine
above never inspects which one it holds.

### `VariationalDirichletAnnotator` — full Bayesian confusion matrices

This is the strategy from SVGPCR (Morales-Álvarez et al., 2022). Each column
of each worker's confusion matrix carries an independent Dirichlet posterior,
$q(R^a_{\cdot,j}) = \mathrm{Dir}(\tilde\alpha^a_{\cdot,j})$ against a prior
$p(R^a_{\cdot,j}) = \mathrm{Dir}(\alpha^a_{\cdot,j})$ (flat, $\alpha=1$, by
default: no prior opinion about any worker).

**Expected log confusion** — the standard Dirichlet identity, an expectation
of a logarithm computed *exactly* by the digamma function (not the logarithm
of the mean, which is a different, biased quantity):

$$
\mathbb{E}_{q}[\log R^a_{ij}] \;=\; \psi(\tilde\alpha^a_{ij}) - \psi\Big(\textstyle\sum_{i'} \tilde\alpha^a_{i'j}\Big).
$$

**KL divergence**, for one Dirichlet column ($q=\tilde\alpha^a_{\cdot,j}$,
$p=\alpha^a_{\cdot,j}$), summed over every $(a,j)$:

$$
\mathrm{KL}(\mathrm{Dir}(q)\,\|\,\mathrm{Dir}(p)) \;=\; \log B(p) - \log B(q) + \sum_i (q_i - p_i)\big(\psi(q_i) - \psi(\textstyle\sum_{i'} q_{i'})\big),
$$

with $B(x) = \prod_i \Gamma(x_i) / \Gamma(\sum_i x_i)$. `tf.math.lbeta`
reduces the *last* axis, so the implementation transposes to put the
observed-class axis (axis 1, the Dirichlet's own axis) last before calling
it — see the note in
[`annotators/strategies.py`](https://github.com/wizmik12/gpcrowdkit/blob/main/src/gpcrowdkit/annotators/strategies.py).

**Posterior mean**, for reporting via `confusion_matrices()`:
$\mathbb{E}_q[R^a_{ij}] = \tilde\alpha^a_{ij} / \sum_{i'}\tilde\alpha^a_{i'j}$
— deliberately **not** $\mathrm{softmax}(\mathbb{E}_q[\log R])$, which is the
normalised *geometric* mean and is more sharply peaked than the true
posterior mean, overstating accuracy worst for the workers with the fewest
annotations (exactly where the estimate matters most).

Why a Dirichlet is worth having at all: it is the only strategy here that
represents genuine *uncertainty* about a worker. A worker with three
annotations and one with three thousand can have identical point estimates
but very different posteriors — only this strategy tells them apart, through
the concentration $\sum_i \tilde\alpha^a_{ij}$.

### `SoftmaxPointAnnotator` — deterministic, same parameter count

A point estimate, $R^a = \mathrm{softmax}(W^a)$ (softmax along the observed-class
axis), with $W^a \in \mathbb{R}^{C\times C}$ free logits. No distribution, so
$\mathrm{KL}=0$ and $\mathbb{E}[\log R^a_{ij}] = \log\mathrm{softmax}(W^a)_{ij}$
exactly (via `log_softmax`, which stays finite where
$\log(\mathrm{softmax}(\cdot))$ composed naively would not). Same $A\times C^2$
parameter count as the Dirichlet strategy, cheaper to evaluate, with no
notion of confidence in the estimate — reasonable when every worker has
plenty of annotations.

### `OneCoinAnnotator` — the classic one-coin model

A single scalar accuracy $\beta_a \in (0,1)$ per worker (Dawid & Skene's
simplest special case):

$$
R^a_{ij} = \begin{cases}\beta_a & i = j \\ (1-\beta_a)/(C-1) & i \ne j.\end{cases}
$$

$A$ parameters instead of $A\times C^2$; cannot express a worker who
systematically confuses two *particular* classes, but far better behaved
when annotations per worker are scarce — the usual situation in practice.
$\beta_a$ is stored as a logit with the sigmoid applied at use, so the
optimiser sees an unconstrained parameter; $\mathrm{KL}=0$, as for any point
estimate.

### The design test these three pass together

`OneCoinAnnotator` is why the abstract contract in
[`annotators/base.py`](https://github.com/wizmik12/gpcrowdkit/blob/main/src/gpcrowdkit/annotators/base.py) does not promise a
confusion-matrix parameter: it *builds* one on the fly from a single scalar,
it does not *store* one. An interface requiring an `[A,C,C]` parameter tensor
would have been the wrong abstraction for exactly this strategy — the
contract instead requires only `label_log_terms` and `kl_divergence`, and
everything downstream (item aggregation, the ELBO term, the interaction with
$q(Z)$) is a fixed reduction implemented once, on the base class.

## The posterior over ground truth, *q(Z)*

`FreeCategoricalZ` treats $\gamma_n = q(z_n=\cdot)$ as $N\times C$ free
parameters (`q_unn`, positive-constrained, row-normalised at use),
optimised by the same gradient step as every other parameter in the model.
This reproduces the reference implementation; it is not the only valid
choice.

**A closed form exists, and is worth knowing.** Because $z_n$ appears only in
terms local to item $n$, holding every other parameter fixed, item $n$'s
contribution to the ELBO as a function of $\gamma_n$ alone is

$$
\sum_c \gamma_{nc}\, t_{nc} \;-\; \sum_c \gamma_{nc}\log\gamma_{nc}, \qquad t_{nc} := \texttt{gp\_log}_{nc} + \texttt{crowd\_log}_{nc}.
$$

Maximising over the simplex $\sum_c \gamma_{nc}=1$ with a Lagrange multiplier
gives the exact optimum

$$
\gamma_n = \mathrm{softmax}(t_n),
$$

which is precisely the E-step of a mean-field / variational-EM coordinate
ascent scheme: no parameters to store for $q(Z)$ at all, and the update is
exact rather than gradient-approximate at every step. This library does not
implement it — the reference SVGPCR model does not either — but
`PosteriorZ.gamma`'s signature (it receives `gp_log` and `crowd_log` even
though `FreeCategoricalZ` ignores them) exists precisely so that a
closed-form `q(Z)` strategy could be added later without changing
`GPCrowdModel.elbo_terms` at all. See the module docstring in
[`posteriors.py`](https://github.com/wizmik12/gpcrowdkit/blob/main/src/gpcrowdkit/posteriors.py).

**Entropy.** $\mathrm{H}[q(z_n)] = -\sum_c \gamma_{nc}\log\gamma_{nc} \ge 0$,
added to the ELBO (some derivations of this bound subtract a *negative*
entropy term instead — same bound, opposite sign convention; the two must
not be mixed, or the objective decreases monotonically while otherwise
looking correct).

## Stochastic optimization over minibatches

A minibatch is a set of $B$ **items**, plus every annotation belonging to
them ([`CrowdBatch`](https://github.com/wizmik12/gpcrowdkit/blob/main/src/gpcrowdkit/data.py)). `latent`, `crowd`, `entropy`
are all sums over items, so a batch estimates $B/N$ of the full sum, and the
unbiased full-data estimate requires the correction

$$
\texttt{total} = \big(\texttt{latent} + \texttt{crowd} + \texttt{entropy}\big)\cdot\frac{N}{B} \;-\; \texttt{kl\_latent} \;-\; \texttt{kl\_annotator}.
$$

The two KL terms are **not** scaled: they are properties of global
parameters ($q(u)$, $q(R)$) that are complete and exact regardless of which
items are in the batch. Scaling them by $N/B$ as well would inflate the
complexity penalty by that same factor and produce a model that
systematically underfits — a failure mode that is very hard to read off the
loss curve alone, since the (wrongly scaled) total still decreases
monotonically. `test_minibatch_elbo_is_unbiased` in
[`test_end_to_end.py`](https://github.com/wizmik12/gpcrowdkit/blob/main/src/gpcrowdkit/tests/test_end_to_end.py) checks this
directly, by comparing the mean of many random-batch ELBO estimates against
the full-data value.

All trainable parameters — kernel hyperparameters, inducing locations,
$q(u)$'s mean and covariance, $q(Z)$'s free parameters, and $q(R)$'s
concentrations or point logits — are optimised **jointly** by a single Adam
optimiser on the negative ELBO ([`inference.train`](https://github.com/wizmik12/gpcrowdkit/blob/main/src/gpcrowdkit/inference.py)).
This is the simplest scheme that works; natural gradients on the variational
Gaussian parameters, or the closed-form $q(Z)$ update above, are documented
extension points rather than implemented alternatives.

## Prediction

Two distinct predictive quantities, for two distinct situations:

* **`infer_true_labels(X, labels)`** — for items that *do* have annotations.
  Combines both evidence streams exactly as training does:
  $\gamma_n = q(Z)$ evaluated with `gp_log` and `crowd_log` both present,
  then $\hat z_n = \arg\max_c \gamma_{nc}$. This is what makes the result
  different from majority vote: an item whose annotators disagree can still
  be labelled confidently if its features place it among items the GP has
  learned to classify.

* **`predict_class_probs(X)`** — for items with **no** annotations at all.
  Delegates to the latent GP alone (`SVGPLatent.predict_class_probs`, via
  GPflow's `predict_y`, which propagates the posterior variance of $f$
  through the likelihood rather than evaluating the link at the mean only).
  The annotator model is meaningless here — no worker has said anything
  about these items — and this is the method that only exists because a
  latent *classifier* was fit, rather than merely a denoised label matrix.
  `test_predicts_on_unannotated_items` in
  [`test_end_to_end.py`](https://github.com/wizmik12/gpcrowdkit/blob/main/src/gpcrowdkit/tests/test_end_to_end.py) is the
  test that would fail if this generalisation were illusory.

## Relationship to the literature

* **Dawid & Skene (1979)** — the confusion-matrix observation model
  ($p(y\mid z, R)$) and the conditional-independence-given-truth assumption
  are theirs; `OneCoinAnnotator` is their simplest special case.
* **Raykar et al. (2010)** — extends Dawid–Skene with a parametric classifier
  (logistic regression) over features, so the model generalises to
  unannotated data; the two-evidence-stream structure here (features *and*
  annotations, combined in $q(Z)$) descends from this idea.
* **Rodrigues, Pereira & Ribeiro (2014), "Gaussian Process Classification and
  Active Learning with Multiple Annotators"** — replaces the parametric
  classifier with a GP, non-sparse.
* **Hernández-Lobato, Houlsby & Ghahramani (2011)** — the robust-max
  multi-class link and its Gaussian-quadrature expectation, used unchanged
  in `SVGPLatent`.
* **Hensman, Matthews & Ghahramani (2013, 2015)** — sparse variational GP
  classification, inducing points, and the whitened parameterisation that
  `SVGPLatent` inherits from `gpflow.models.SVGP`.
* **Morales-Álvarez, Ruiz, Coughlin, Molina & Katsaggelos (2022), "Scalable
  Variational Gaussian Processes for Crowdsourcing: Glitch Detection in
  LIGO"** ("SVGPCR") — combines all of the above into the exact model this
  library's default configuration reproduces: sparse variational multi-class
  GP classifier, per-worker Dirichlet confusion matrices, and a free
  variational $q(Z)$, all optimised jointly by gradient ascent on one ELBO.

## References

1. Dawid, A. P., & Skene, A. M. (1979). Maximum likelihood estimation of
   observer error-rates using the EM algorithm. *Applied Statistics*, 28(1), 20–28.
2. Raykar, V. C., Yu, S., Zhao, L. H., Valadez, G. H., Florin, C., Bogoni, L.,
   & Moy, L. (2010). Learning from crowds. *JMLR*, 11, 1297–1322.
3. Hernández-Lobato, D., Houlsby, N., & Ghahramani, Z. (2011). Probabilistic
   modelling of skill classification. *NeurIPS Workshop*; the robust-max link
   used here is detailed in Hernández-Lobato & Hernández-Lobato,
   "Robust Multi-Class Gaussian Process Classification".
4. Hensman, J., Matthews, A. G. de G., & Ghahramani, Z. (2015). Scalable
   Variational Gaussian Process Classification. *AISTATS*.
5. Rodrigues, F., Pereira, F., & Ribeiro, B. (2014). Gaussian Process
   Classification and Active Learning with Multiple Annotators. *ICML*.
6. Morales-Álvarez, P., Ruiz, P., Coughlin, S., Molina, R., & Katsaggelos, A. K.
   (2022). Scalable Variational Gaussian Processes for Crowdsourcing: Glitch
   Detection in LIGO. *IEEE TPAMI*, 44(3), 1534–1551.
