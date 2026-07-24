"""Tests for gpcrowdkit.metrics."""

from __future__ import annotations

import numpy as np
import pytest

from gpcrowdkit.metrics import (
    balanced_accuracy,
    confusion_error,
    cross_entropy,
    expected_calibration_error,
    label_accuracy,
    worker_accuracy_correlation,
)


class TestLabelAccuracy:
    def test_perfect(self):
        assert label_accuracy([0, 1, 2], [0, 1, 2]) == pytest.approx(1.0)

    def test_all_wrong(self):
        assert label_accuracy([0, 0, 0], [1, 2, 3]) == pytest.approx(0.0)

    def test_mixed(self):
        pred = [0, 1, 2, 0, 1]
        true = [0, 1, 1, 2, 1]
        # correct at indices 0, 1, 4 → 3/5
        assert label_accuracy(pred, true) == pytest.approx(0.6)

    def test_numpy_arrays(self):
        pred = np.array([0, 1])
        true = np.array([0, 0])
        assert label_accuracy(pred, true) == pytest.approx(0.5)

    def test_empty(self):
        assert label_accuracy([], []) == pytest.approx(0.0)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            label_accuracy([0, 1], [0])


class TestBalancedAccuracy:
    def test_perfect(self):
        assert balanced_accuracy([0, 1, 0, 1], [0, 1, 0, 1]) == pytest.approx(1.0)

    def test_imbalanced(self):
        """Class 0: 1/2 recall, class 1: 1/1 recall → (0.5+1)/2=0.75"""
        pred = [0, 1, 1]
        true = [0, 1, 0]  # class 0: idx 0 correct (TP), idx 2 wrong (FN)
        result = balanced_accuracy(pred, true)
        assert result == pytest.approx(0.75, abs=0.01)

    def test_missing_class_predicted(self):
        """Class 2 exists in true but never predicted → contributes 0."""
        pred = [0, 1, 0]
        true = [0, 1, 2]
        result = balanced_accuracy(pred, true)
        # class 0: 1/1, class 1: 1/1, class 2: 0/1 → (1+1+0)/3
        assert result == pytest.approx(2 / 3, abs=0.01)

    def test_empty(self):
        assert balanced_accuracy([], []) == pytest.approx(0.0)


class TestConfusionError:
    def test_identical(self):
        mat = np.ones((3, 3))
        assert confusion_error(mat, mat) == pytest.approx(0.0)

    def test_known_error(self):
        est = np.zeros((2, 3, 3))
        tru = np.ones((2, 3, 3))
        assert confusion_error(est, tru) == pytest.approx(1.0)

    def test_flattened_input(self):
        est = np.array([0.0, 0.5, 1.0])
        tru = np.array([0.0, 0.0, 1.0])
        # |0-0| + |0.5-0| + |1-1| = 0.5, mean = 0.5/3
        assert confusion_error(est, tru) == pytest.approx(0.5 / 3)

    def test_empty(self):
        assert confusion_error(np.array([]), np.array([])) == pytest.approx(0.0)


class TestExpectedCalibrationError:
    def test_perfectly_calibrated(self):
        probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]])
        true = np.array([0, 0, 0])  # all predicted correctly with high conf
        # With only 3 samples and 10 bins, binning noise is expected
        assert expected_calibration_error(probs, true) < 0.25

    def test_poorly_calibrated(self):
        N = 100
        probs = np.random.rand(N, 2)
        probs /= probs.sum(axis=1, keepdims=True)
        true = np.random.randint(0, 2, size=N)
        # Random predictions should give moderate ECE
        ece = expected_calibration_error(probs, true, n_bins=10)
        assert 0.0 <= ece <= 1.0

    def test_binary_probs_1d(self):
        probs = np.array([0.9, 0.6, 0.3])
        true = np.array([1, 1, 0])
        ece = expected_calibration_error(probs, true, n_bins=3)
        assert 0.0 <= ece <= 1.0

    def test_empty(self):
        probs = np.empty((0, 2))
        true = np.empty(0, dtype=int)
        assert expected_calibration_error(probs, true) == pytest.approx(0.0)


class TestCrossEntropy:
    def test_perfect_prediction(self):
        probs = np.array([[1.0, 0.0], [0.0, 1.0]])
        true = np.array([0, 1])
        assert cross_entropy(probs, true) == pytest.approx(0.0, abs=1e-6)

    def test_wrong_prediction(self):
        probs = np.array([[0.001, 0.999], [0.999, 0.001]])
        true = np.array([0, 1])
        ce = cross_entropy(probs, true)
        assert ce > 0.0

    def test_uniform_probs(self):
        C = 3
        N = 10
        probs = np.full((N, C), 1.0 / C)
        true = np.zeros(N, dtype=int)
        ce = cross_entropy(probs, true)
        # -log(1/C) = log(C)
        import math
        assert ce == pytest.approx(math.log(C), rel=1e-4)

    def test_empty(self):
        assert cross_entropy(np.empty((0, 2)), np.empty(0, dtype=int)) == pytest.approx(0.0)


class TestWorkerAccuracyCorrelation:
    def test_identical_matrices(self):
        W = 4
        mats = np.random.rand(W, 3, 3)
        assert worker_accuracy_correlation(mats, mats) == pytest.approx(1.0)

    def test_uncorrelated(self):
        W = 5
        est = np.random.rand(W, 9).reshape(W, 3, 3)
        tru = np.random.rand(W, 9).reshape(W, 3, 3)
        corr = worker_accuracy_correlation(est, tru)
        assert -1.0 <= corr <= 1.0

    def test_constant_matrix_skipped(self):
        W = 3
        est = np.ones((W, 3, 3)) * 0.5
        tru = np.full((W, 3, 3), 0.5)
        # All constant → zero variance → skipped → returns 0
        assert worker_accuracy_correlation(est, tru) == pytest.approx(0.0)

    def test_empty(self):
        empty = np.empty((0, 3, 3))
        assert worker_accuracy_correlation(empty, empty) == pytest.approx(0.0)
