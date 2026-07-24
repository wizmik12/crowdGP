"""Visual diagnostics: the ELBO decomposition and confusion-matrix recovery.

Two plots, saved to ``examples/output/``:

``elbo_decomposition.png``
    The trace every component of the ELBO takes during training. The
    characteristic crowdsourcing failure mode is a rising total ELBO in which
    ``crowd`` climbs steadily while ``latent`` stays flat: the model has learned
    to reproduce the annotations and is ignoring the features, so it will not
    generalise to a single unannotated item. That failure is invisible in the
    total and obvious in the decomposition -- which is the whole reason
    :class:`~gpcrowdkit.models.ELBOTerms` reports it separately (see models.py).

``confusion_recovery.png``
    True vs. recovered confusion matrix for a handful of workers -- a visual
    check that the Dirichlet posterior mean converges on the matrices the data
    was actually generated from, and not their transpose (a bug that would
    otherwise still produce a plausible-looking, high-accuracy model; see the
    note in ``synthetic.py`` and the ``test_recovers_confusion_matrices`` test).

Requires matplotlib (``pip install -e ".[examples]"``). Run with::

    ./run.sh examples/03_diagnostics_plots.py
"""

from __future__ import annotations

from pathlib import Path

import gpflow
import matplotlib

matplotlib.use("Agg")  # headless: this script only saves figures, never shows them
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from gpcrowdkit import (
    FreeCategoricalZ,
    GPCrowdModel,
    SVGPLatent,
    VariationalDirichletAnnotator,
    init_alpha_tilde,
    make_synthetic,
    train,
)

OUTPUT_DIR = Path(__file__).parent / "output"

# Fixed categorical colors, one per ELBO component -- chosen so each keeps the
# same identity across both this script's plots and any figure a user builds
# from TrainingHistory themselves.
COLORS = {
    "elbo": "#2a78d6",
    "latent": "#eb6834",
    "crowd": "#1baf7a",
    "entropy": "#4a3aa7",
    "kl_latent": "#e87ba4",
    "kl_annotator": "#008300",
}
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"


def _style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_elbo_decomposition(history, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.plot(history.elbo, color=COLORS["elbo"], linewidth=2, label="total ELBO")
    ax.set_title("Total ELBO", color=INK)
    ax.set_xlabel("iteration", color=INK)
    _style_axes(ax)

    ax = axes[1]
    for name in ["latent", "crowd", "entropy", "kl_latent", "kl_annotator"]:
        ax.plot(getattr(history, name), color=COLORS[name], linewidth=1.8, label=name)
    ax.set_title("ELBO components -- 'latent' must not flatline", color=INK)
    ax.set_xlabel("iteration", color=INK)
    ax.legend(frameon=False, labelcolor=INK, fontsize=9)
    _style_axes(ax)

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def plot_confusion_recovery(true_confusion, est_confusion, workers, class_keys, path: Path) -> None:
    n = len(workers)
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.4))
    cmap = plt.get_cmap("Blues")  # sequential: one hue, light -> dark, for magnitude

    for col, a in enumerate(workers):
        for row, (mat, label) in enumerate(
            [(true_confusion[a], "true"), (est_confusion[a], "recovered")]
        ):
            ax = axes[row, col]
            im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1)
            ax.set_xticks(range(len(class_keys)))
            ax.set_yticks(range(len(class_keys)))
            ax.set_xticklabels(class_keys, color=MUTED, fontsize=8)
            ax.set_yticklabels(class_keys, color=MUTED, fontsize=8)
            if row == 0:
                ax.set_title(f"worker {a}", color=INK)
            if col == 0:
                ax.set_ylabel(f"{label}\nobserved class", color=INK, fontsize=9)
            if row == 1:
                ax.set_xlabel("true class", color=INK, fontsize=9)
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    ax.text(
                        j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        color="white" if mat[i, j] > 0.6 else INK, fontsize=7,
                    )

    fig.colorbar(im, ax=axes, shrink=0.7, label="P(observed | true)")
    fig.suptitle("Worker confusion matrices: true vs. recovered posterior mean", color=INK)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def main() -> None:
    gpflow.config.set_default_float(np.float64)
    OUTPUT_DIR.mkdir(exist_ok=True)

    data = make_synthetic(num_items=300, num_classes=3, num_workers=8, labels_per_item=4, seed=3)
    labels = data.labels
    class_probs = labels.empirical_class_probs()

    model = GPCrowdModel(
        latent=SVGPLatent(
            kernel=gpflow.kernels.SquaredExponential(lengthscales=2.0),
            num_classes=labels.num_classes,
            inducing_points=data.X[:25].copy(),
        ),
        annotator=VariationalDirichletAnnotator(
            labels.num_workers,
            labels.num_classes,
            alpha_tilde_init=init_alpha_tilde(labels, class_probs),
        ),
        num_data=labels.num_items,
        q_z=FreeCategoricalZ(labels.num_items, labels.num_classes, init_probs=class_probs),
    )

    print("Training ...")
    history = train(model, data.X, labels, iterations=300, learning_rate=0.05)

    elbo_path = OUTPUT_DIR / "elbo_decomposition.png"
    plot_elbo_decomposition(history, elbo_path)
    print(f"Wrote {elbo_path}")

    est_confusion = model.annotator.confusion_matrices().numpy()
    confusion_path = OUTPUT_DIR / "confusion_recovery.png"
    plot_confusion_recovery(
        data.confusion, est_confusion,
        workers=list(range(min(4, labels.num_workers))),
        class_keys=labels.class_keys,
        path=confusion_path,
    )
    print(f"Wrote {confusion_path}")


if __name__ == "__main__":
    main()
