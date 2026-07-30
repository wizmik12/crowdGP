import pytest
import numpy as np
import math
from gpcrowdkit.metrics import (
    label_accuracy,
    balanced_accuracy,
    confusion_error,
    expected_calibration_error,
    cross_entropy,
    worker_accuracy_correlation,
)

def test_label_accuracy_perfect():
    pred = np.array([1, 0, 1, 0])
    true = np.array([1, 0, 1, 0])
    assert label_accuracy(pred, true) == 1.0
    assert type(label_accuracy(pred, true)) is float

def test_label_accuracy_half():
    pred = np.array([1, 1, 1, 1])
    true = np.array([1, 0, 1, 0])
    assert label_accuracy(pred, true) == 0.5
    assert type(label_accuracy(pred, true)) is float

def test_balanced_accuracy():
    true = np.array([0, 0, 0, 1, 1, 2])
    pred = np.array([0, 0, 1, 1, 1, 2])
    # Class 0: 2/3, Class 1: 2/2 = 1, Class 2: 1/1 = 1
    # Mean: (2/3 + 1 + 1) / 3 = 8/9
    expected = 8.0 / 9.0
    result = balanced_accuracy(pred, true)
    assert result == pytest.approx(expected)
    assert type(result) is float

def test_confusion_error():
    estimated = np.array([
        [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
        [[2.0, 3.0, 4.0], [2.0, 3.0, 4.0], [2.0, 3.0, 4.0]],
    ])  # shape (2, 3, 3)
    true = estimated + 0.5  # every element off by 0.5
    result = confusion_error(estimated, true)
    assert result == pytest.approx(0.5)
    assert type(result) is float

def test_confusion_error_shape_mismatch():
    estimated = np.zeros((2, 3, 3))
    true = np.zeros((3, 3, 2))  # wrong/transposed shape
    with pytest.raises(ValueError):
        confusion_error(estimated, true)

def test_label_accuracy_empty():
    result = label_accuracy(np.array([]), np.array([]))
    assert result == 0.0

def test_cross_entropy_empty():
    probs = np.array([]).reshape(0, 2)
    true = np.array([])
    result = cross_entropy(probs, true)
    assert result == 0.0

def test_expected_calibration_error_perfect():
    # 10 items, all prob=0.9 for the correct class (all correct predictions)
    # The confidence is 0.9, accuracy is 1.0
    # ECE = |1.0 - 0.9| = 0.1
    probs = np.array([[0.1, 0.9] for _ in range(10)])
    true = np.array([1 for _ in range(10)])
    result = expected_calibration_error(probs, true, n_bins=10)
    assert result == pytest.approx(0.1)
    assert type(result) is float

def test_cross_entropy():
    probs = np.array([[0.9, 0.1], [0.1, 0.9]])
    true = np.array([0, 1])
    # -(ln(0.9) + ln(0.9)) / 2
    expected = -(math.log(0.9) + math.log(0.9)) / 2.0
    result = cross_entropy(probs, true)
    assert result == pytest.approx(expected)
    assert type(result) is float

def test_worker_accuracy_correlation():
    # 3 workers, 2 classes.
    # estimated diagonals: [0.8, 0.9, 0.7] -> [0.8, 0.8], [0.9, 0.9], [0.7, 0.7]
    estimated = np.array([
        [[0.8, 0.2], [0.2, 0.8]],
        [[0.9, 0.1], [0.1, 0.9]],
        [[0.7, 0.3], [0.3, 0.7]]
    ])
    # true diagonals: [0.75, 0.95, 0.65]
    true_conf = np.array([
        [[0.75, 0.25], [0.25, 0.75]],
        [[0.95, 0.05], [0.05, 0.95]],
        [[0.65, 0.35], [0.35, 0.65]]
    ])
    
    # Ranks for estimated means [0.8, 0.9, 0.7] -> [2, 3, 1]
    # Ranks for true means [0.75, 0.95, 0.65] -> [2, 3, 1]
    # Correlation is 1.0
    result = worker_accuracy_correlation(estimated, true_conf)
    assert result == pytest.approx(1.0)
    assert type(result) is float

def test_worker_accuracy_correlation_with_ties():
    estimated = np.array([
        [[0.8, 0.2], [0.2, 0.8]], # 0.8
        [[0.9, 0.1], [0.1, 0.9]], # 0.9
        [[0.9, 0.1], [0.1, 0.9]], # 0.9
        [[0.7, 0.3], [0.3, 0.7]]  # 0.7
    ])
    true_conf = np.array([
        [[0.75, 0.25], [0.25, 0.75]], # 0.75
        [[0.95, 0.05], [0.05, 0.95]], # 0.95
        [[0.95, 0.05], [0.05, 0.95]], # 0.95
        [[0.65, 0.35], [0.35, 0.65]]  # 0.65
    ])
    
    # Ranks for estimated [0.8, 0.9, 0.9, 0.7] -> [2, 3.5, 3.5, 1]
    # Ranks for true [0.75, 0.95, 0.95, 0.65] -> [2, 3.5, 3.5, 1]
    # Correlation is 1.0
    result = worker_accuracy_correlation(estimated, true_conf)
    assert result == pytest.approx(1.0)
    assert type(result) is float

def test_worker_accuracy_correlation_zero_std():
    # All same diagonals
    estimated = np.array([
        [[0.8, 0.2], [0.2, 0.8]],
        [[0.8, 0.2], [0.2, 0.8]],
    ])
    true_conf = np.array([
        [[0.7, 0.3], [0.3, 0.7]],
        [[0.7, 0.3], [0.3, 0.7]],
    ])
    result = worker_accuracy_correlation(estimated, true_conf)
    assert result == 0.0
    assert type(result) is float
