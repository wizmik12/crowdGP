"""Sparse containers for crowdsourced annotations.

This is the foundation file of the library: every other module consumes a
[CrowdBatch][gpcrowdkit.data.CrowdBatch] produced here, so the guarantees this file makes are the
guarantees the whole model rests on.

Storage format
--------------
Annotations are held in **COO (coordinate) form** -- three parallel arrays of
length ``L``, one entry per *annotation*::

    item_idx[l], worker_idx[l], label[l]

meaning "worker ``worker_idx[l]`` labelled item ``item_idx[l]`` as class
``label[l]``".

The obvious alternatives are worse. A dense ``[N, A]`` matrix with a ``-1``
sentinel wastes memory in proportion to ``N * A``, and real crowdsourcing
datasets have a handful of labels per item spread over thousands of workers,
so the matrix is over 99% sentinel. A padded ``[N, S, 2]`` block (used by the
reference SVGPCR implementation) wastes memory in proportion to the *maximum*
number of labels on any one item, which one pathological item can blow up.
COO costs exactly ``3L``.

Batching
--------
The awkward part of COO is that a minibatch is defined by a set of *items*,
but the annotations for those items are scattered through the ``L`` arrays.
[CrowdLabels.as_ragged][gpcrowdkit.data.CrowdLabels.as_ragged] solves this by re-expressing the annotations as
a ``tf.RaggedTensor`` indexed by item, so that ``tf.gather`` retrieves exactly
the annotations belonging to a batch -- with no padding, and with the
item-to-annotation mapping recovered for free from ``value_rowids()``.

This matters for correctness, not just speed. The reference implementation
keeps three separate ``Minibatch`` objects (features, labels, indices) aligned
only by passing them the same random seed. Nothing enforces that; if it ever
breaks, features get paired with the wrong labels and the model simply trains
to a worse optimum with no error raised. Here, alignment is structural: there
is one index vector, and everything is gathered from it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import tensorflow as tf

__all__ = ["CrowdLabels", "CrowdBatch"]


@dataclass(frozen=True)
class CrowdBatch:
    """One minibatch: some items, plus every annotation belonging to them.

    Frozen because a batch is a value, not a workspace -- nothing downstream
    should be mutating index arrays.

    Attributes:
        X (tf.Tensor): Features of the batch items. Shape ``[B, D]``, float64.
        worker_idx (tf.Tensor): Annotator of each annotation, in ``[0, A)``.
            Shape ``[L]``, int32.
        label (tf.Tensor): Observed class of each annotation, in ``[0, C)``.
            Shape ``[L]``, int32.
        item_local (tf.Tensor): Row of ``X`` that each annotation belongs to,
            in ``[0, B)``. Shape ``[L]``, int32. This is the segment id used to
            aggregate per-annotation quantities onto items.
        item_global (tf.Tensor): Dataset-level index of each row of ``X``, in
            ``[0, N)``. Shape ``[B]``, int32. Used to index dataset-sized
            objects such as a stored ``q(Z)``.

    Note:
        ``item_local`` and ``item_global`` are both "the item index" but live
        in different coordinate systems, and swapping them is the one serious
        bug this design still permits. The identity that connects them is
        ``tf.gather(item_global, item_local)``, which yields the dataset-level
        item of every annotation.
    """

    X: tf.Tensor
    worker_idx: tf.Tensor
    label: tf.Tensor
    item_local: tf.Tensor
    item_global: tf.Tensor

    @property
    def size(self) -> tf.Tensor:
        """Number of items ``B`` in the batch, as a scalar int32 tensor.

        Returned as a tensor rather than a Python int so that this works
        unchanged inside ``tf.function``, where the batch dimension may be
        dynamic.
        """
        return tf.shape(self.X)[0]

    @property
    def num_labels(self) -> tf.Tensor:
        """Number of annotations ``L`` in the batch, as a scalar int32 tensor."""
        return tf.shape(self.label)[0]


class CrowdLabels:
    """Crowdsourced annotations in sparse COO form.

    Attributes:
        item_idx (np.ndarray): Item index per annotation. Shape ``[L]``, int32.
        worker_idx (np.ndarray): Worker index per annotation. Shape ``[L]``, int32.
        label (np.ndarray): Observed class per annotation. Shape ``[L]``, int32.
        num_items (int): Dataset size ``N``. Given explicitly rather than
            inferred, because items with zero annotations are legal and would
            otherwise be invisible.
        worker_keys (np.ndarray): The caller's original annotator identifiers,
            where position ``a`` holds the key mapped to internal index ``a``.
            Shape ``[A]``.
        class_keys (np.ndarray): The caller's original class identifiers.
            Shape ``[C]``.

    Note:
        Everything in this class is NumPy except [as_ragged][gpcrowdkit.data.CrowdLabels.as_ragged] and
        [gather_batch][gpcrowdkit.data.CrowdLabels.gather_batch]. Preprocessing runs once, outside the training
        graph, so there is no reason to pay TensorFlow's overhead for it.
    """

    def __init__(
        self,
        item_idx: np.ndarray,
        worker_idx: np.ndarray,
        label: np.ndarray,
        num_items: int,
        worker_keys: np.ndarray | None = None,
        class_keys: np.ndarray | None = None,
    ) -> None:
        """Stores the COO arrays and the key lookup tables.

        Args:
            item_idx: Item index per annotation. Shape ``[L]``.
            worker_idx: Worker index per annotation. Shape ``[L]``.
            label: Observed class per annotation. Shape ``[L]``.
            num_items: Dataset size ``N``.
            worker_keys: Original annotator identifiers. Defaults to
                ``0..max(worker_idx)``.
            class_keys: Original class identifiers. Defaults to
                ``0..max(label)``.

        Raises:
            ValueError: If the three arrays do not all have shape ``[L]``.
                Checked eagerly because passing a single ``[L, 3]`` array is an
                easy mistake that would otherwise surface much later, inside a
                gather, with an unhelpful message.
        """
        self.item_idx = np.asarray(item_idx, dtype=np.int32)
        self.worker_idx = np.asarray(worker_idx, dtype=np.int32)
        self.label = np.asarray(label, dtype=np.int32)
        if not (self.item_idx.shape == self.worker_idx.shape == self.label.shape):
            raise ValueError(
                "item_idx, worker_idx and label must all have shape [L]; got "
                f"{self.item_idx.shape}, {self.worker_idx.shape}, {self.label.shape}"
            )
        self.num_items = int(num_items)
        self.worker_keys = (
            np.arange(self.worker_idx.max() + 1)
            if worker_keys is None
            else np.asarray(worker_keys)
        )
        self.class_keys = (
            np.arange(self.label.max() + 1) if class_keys is None else np.asarray(class_keys)
        )
        # Built lazily and cached: it is derived from arrays that never change,
        # so rebuilding it per training step would dominate the runtime.
        self._ragged: tf.RaggedTensor | None = None

    # ------------------------------------------------------------------ sizes

    @property
    def num_workers(self) -> int:
        """Number of distinct annotators ``A``."""
        return len(self.worker_keys)

    @property
    def num_classes(self) -> int:
        """Number of distinct classes ``C``."""
        return len(self.class_keys)

    @property
    def num_labels(self) -> int:
        """Total number of annotations ``L``."""
        return int(self.item_idx.size)

    def __repr__(self) -> str:
        density = self.num_labels / (self.num_items * self.num_workers)
        return (
            f"CrowdLabels(N={self.num_items}, A={self.num_workers}, "
            f"C={self.num_classes}, L={self.num_labels}, density={density:.2%})"
        )

    # ------------------------------------------------------------ constructors

    @classmethod
    def from_pairs(cls, Y: Sequence[np.ndarray], num_items: int | None = None) -> "CrowdLabels":
        """Builds from the reference SVGPCR input format.

        That format is a length-``N`` sequence of ``[S_n, 2]`` arrays of
        ``(annotator_key, class_key)`` pairs, where the keys may be strings or
        any other dtype.

        The key-to-index mapping uses ``np.unique(..., return_inverse=True)``,
        which returns both the sorted unique keys and, for each element, its
        position among them -- the lookup table and the mapped column in a
        single pass. The reference implementation instead evaluates
        ``np.flatnonzero(v == keys)[0]`` once per annotation, which is
        ``O(L * A)``; that is what its timing printouts are measuring.

        Args:
            Y: Sequence of ``[S_n, 2]`` arrays of (annotator, class) pairs.
            num_items: Dataset size ``N``. Defaults to ``len(Y)``.

        Returns:
            CrowdLabels: The COO representation, with ``worker_keys`` and
            ``class_keys`` recording the original identifiers.
        """
        rows = [np.atleast_2d(np.asarray(y)) for y in Y]
        counts = np.array([r.shape[0] if r.size else 0 for r in rows], dtype=np.int64)
        stacked = np.concatenate([r for r in rows if r.size], axis=0)
        # np.repeat expands the per-item counts into one item index per annotation:
        # counts [2, 0, 1] -> item_idx [0, 0, 2].
        item_idx = np.repeat(np.arange(len(rows), dtype=np.int32), counts)

        worker_keys, worker_idx = np.unique(stacked[:, 0], return_inverse=True)
        class_keys, label = np.unique(stacked[:, 1], return_inverse=True)
        return cls(
            item_idx=item_idx,
            worker_idx=worker_idx,
            label=label,
            num_items=num_items or len(rows),
            worker_keys=worker_keys,
            class_keys=class_keys,
        )

    @classmethod
    def from_dense(cls, Y: np.ndarray, missing: Any = -1) -> "CrowdLabels":
        """Builds from a dense ``[N, A]`` label matrix with a missing sentinel.

        Args:
            Y: Dense label matrix. Shape ``[N, A]``.
            missing: Value marking an absent annotation.

        Returns:
            CrowdLabels: The COO representation.
        """
        Y = np.asarray(Y)
        item_idx, worker_idx = np.nonzero(Y != missing)
        class_keys, label = np.unique(Y[item_idx, worker_idx], return_inverse=True)
        return cls(
            item_idx=item_idx,
            worker_idx=worker_idx,
            label=label,
            num_items=Y.shape[0],
            worker_keys=np.arange(Y.shape[1]),
            class_keys=class_keys,
        )

    # -------------------------------------------------------------- tf interop

    def as_ragged(self) -> tf.RaggedTensor:
        """Re-expresses the annotations as a ragged ``[N, None, 2]`` tensor.

        Row ``n`` holds every ``(worker, label)`` pair for item ``n``, so
        gathering a set of item indices retrieves precisely the annotations for
        those items. Rows for items with no annotations are empty, which is
        both legal and common.

        The construction is: stable-sort the annotations by item; stack
        ``(worker, label)`` into an ``[L, 2]`` value matrix in that order;
        count annotations per item with ``bincount``; hand both to
        ``from_row_lengths``, which slices the flat values into rows.

        Returns:
            tf.RaggedTensor: Shape ``[N, None, 2]``, int32. Cached after the
            first call.

        Note:
            ``minlength=self.num_items`` is load-bearing. Without it,
            ``bincount`` stops at the highest item index that actually has an
            annotation, the ragged tensor ends up with too few rows, and every
            subsequent gather silently indexes the wrong item.
        """
        if self._ragged is None:
            with tf.init_scope():  # build from NumPy constants in eager context
                order = np.argsort(self.item_idx, kind="stable")
                values = np.stack([self.worker_idx[order], self.label[order]], axis=1)
                row_lengths = np.bincount(self.item_idx, minlength=self.num_items)
                self._ragged = tf.RaggedTensor.from_row_lengths(
                    values=tf.constant(values, dtype=tf.int32),
                    row_lengths=tf.constant(row_lengths, dtype=tf.int64),
                )
        return self._ragged

    def gather_batch(self, X: tf.Tensor, item_global: tf.Tensor) -> CrowdBatch:
        """Assembles an aligned [CrowdBatch][gpcrowdkit.data.CrowdBatch] for the given items.

        Everything in the batch derives from the single ``item_global`` vector,
        which is what makes misalignment structurally impossible rather than
        merely unlikely.

        Args:
            X (tf.Tensor): Full feature matrix. Shape ``[N, D]``.
            item_global (tf.Tensor): Item indices to gather. Shape ``[B]``.

        Returns:
            CrowdBatch: Features, annotations and both index maps.

        Note:
            There is deliberately no Python-level branching on tensor shapes in
            this method. A construct such as ``if tf.shape(X)[0] != ...`` works
            eagerly and raises inside ``tf.function``, and that failure appears
            only once you enable graph mode -- typically long after the module
            seemed finished.
        """
        item_global = tf.cast(item_global, tf.int32)
        sub = tf.gather(self.as_ragged(), item_global)  # ragged [B, None, 2]
        flat = sub.flat_values  # [L, 2]
        return CrowdBatch(
            X=tf.gather(X, item_global),
            worker_idx=flat[:, 0],
            label=flat[:, 1],
            # value_rowids() is int64; cast so it matches the int32 segment ids
            # expected downstream by unsorted_segment_sum.
            item_local=tf.cast(sub.value_rowids(), tf.int32),
            item_global=item_global,
        )

    def full_batch(self, X: tf.Tensor) -> CrowdBatch:
        """Returns a [CrowdBatch][gpcrowdkit.data.CrowdBatch] covering the entire dataset.

        Args:
            X (tf.Tensor): Full feature matrix. Shape ``[N, D]``.

        Returns:
            CrowdBatch: Every item and every annotation.
        """
        return self.gather_batch(X, tf.range(self.num_items, dtype=tf.int32))

    # -------------------------------------------------------------- baselines

    def majority_vote(self) -> np.ndarray:
        """Per-item majority-vote label, ties broken by lowest class index.

        Kept here rather than in a baselines module because it is also the
        sanity check every crowd model must clear: a model that cannot beat
        majority vote is not doing anything.

        Returns:
            np.ndarray: Shape ``[N]``, int32. Items with no annotations get
            class 0 by the tie-break rule.
        """
        counts = np.zeros((self.num_items, self.num_classes), dtype=np.int64)
        # Scatter-add: counts[item_idx[l], label[l]] += 1 for every l, in one
        # vectorised pass rather than a Python loop over annotations.
        np.add.at(counts, (self.item_idx, self.label), 1)
        return counts.argmax(axis=1).astype(np.int32)

    def empirical_class_probs(self, smoothing: float = 1.0) -> np.ndarray:
        """Smoothed per-item label histogram.

        Used to initialise ``q(Z)`` and the variational annotator parameters:
        starting from the observed vote distribution converges far faster than
        starting from uniform.

        Args:
            smoothing: Additive (Laplace) count added to every class, which
                also keeps items with no annotations well defined.

        Returns:
            np.ndarray: Row-normalised probabilities. Shape ``[N, C]``, float64.
        """
        counts = np.zeros((self.num_items, self.num_classes), dtype=np.float64)
        np.add.at(counts, (self.item_idx, self.label), 1.0)
        counts += smoothing
        return counts / counts.sum(axis=1, keepdims=True)