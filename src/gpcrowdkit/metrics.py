import numpy as np

def label_accuracy(pred, true):
    """
    Fraction of items whose inferred label matches ground truth.

    Args:
        pred (np.ndarray): shape (n_items,), predicted labels.
        true (np.ndarray): shape (n_items,), true labels.

    Returns:
        float: Accuracy score between 0.0 and 1.0.
    """
    if len(pred) == 0:
        return 0.0
    return float(np.mean(pred == true))

def balanced_accuracy(pred, true):
    """
    Mean per-class recall.

    Args:
        pred (np.ndarray): shape (n_items,), predicted labels.
        true (np.ndarray): shape (n_items,), true labels.

    Returns:
        float: Balanced accuracy score between 0.0 and 1.0.
    """
    classes = np.unique(true)
    recalls = []
    for c in classes:
        mask = (true == c)
        if np.sum(mask) > 0:
            recalls.append(np.sum((pred == c) & mask) / np.sum(mask))
    return float(np.mean(recalls)) if recalls else 0.0

def confusion_error(estimated, true):
    """
    Mean absolute error over [A, C_obs, C_true].

    Args:
        estimated (np.ndarray): shape (n_workers, C, C), estimated confusion matrices.
        true (np.ndarray): shape (n_workers, C, C), true confusion matrices.

    Returns:
        float: Mean absolute error.

    Raises:
        ValueError: If inputs do not have the expected (n_workers, C, C) shape or shapes do not match.
    """
    if (estimated.ndim != 3
            or estimated.shape[1] != estimated.shape[2]
            or estimated.shape != true.shape):
        raise ValueError(f"Expected matching [A, C, C]; got {estimated.shape} and {true.shape}")
    return float(np.mean(np.abs(estimated - true)))

def expected_calibration_error(probs, true, n_bins=10):
    """
    Expected Calibration Error (ECE)

    Args:
        probs (np.ndarray): shape (n_items, n_classes), predicted probabilities.
        true (np.ndarray): shape (n_items,), true labels.
        n_bins (int): Number of probability bins.

    Returns:
        float: ECE score between 0.0 and 1.0.
    """
    confidence = np.max(probs, axis=1)
    predicted = np.argmax(probs, axis=1)
    
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_items = len(true)
    
    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]
        
        if i == n_bins - 1:
            in_bin = (confidence >= lower) & (confidence <= upper)
        else:
            in_bin = (confidence >= lower) & (confidence < upper)
            
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(predicted[in_bin] == true[in_bin])
            bin_conf = np.mean(confidence[in_bin])
            ece += np.abs(bin_acc - bin_conf) * (bin_size / n_items)
            
    return float(ece)

def cross_entropy(probs, true):
    """
    Average cross-entropy loss.

    Args:
        probs (np.ndarray): shape (n_items, n_classes), predicted probabilities.
        true (np.ndarray): shape (n_items,), true labels.

    Returns:
        float: Cross-entropy loss >= 0.0.
    """
    if len(probs) == 0:
        return 0.0
    probs_clipped = np.clip(probs, 1e-15, 1 - 1e-15)
    true_probs = probs_clipped[np.arange(len(true)), true]
    return float(-np.mean(np.log(true_probs)))

def worker_accuracy_correlation(estimated_confusion, true_confusion):
    """
    Spearman correlation between estimated and true per-worker diagonal means.

    Args:
        estimated_confusion (np.ndarray): shape (n_workers, n_classes, n_classes), estimated confusion matrices.
        true_confusion (np.ndarray): shape (n_workers, n_classes, n_classes), true confusion matrices.

    Returns:
        float: Spearman correlation coefficient between -1.0 and 1.0.
    """
    n_workers = estimated_confusion.shape[0]
    est_diags = np.array([np.mean(estimated_confusion[i].diagonal()) for i in range(n_workers)])
    true_diags = np.array([np.mean(true_confusion[i].diagonal()) for i in range(n_workers)])
    
    def rankdata(a):
        temp = np.argsort(a)
        ranks = np.empty_like(temp, dtype=float)
        ranks[temp] = np.arange(len(a)) + 1.0
        
        sorted_a = a[temp]
        i = 0
        while i < len(a):
            j = i + 1
            while j < len(a) and sorted_a[j] == sorted_a[i]:
                j += 1
            if j - i > 1:
                avg_rank = np.mean(ranks[temp[i:j]])
                ranks[temp[i:j]] = avg_rank
            i = j
        return ranks
        
    rank_est = rankdata(est_diags)
    rank_true = rankdata(true_diags)
    
    std_est = np.std(rank_est)
    std_true = np.std(rank_true)
    
    if std_est == 0 or std_true == 0:
        return 0.0
        
    cov = np.mean((rank_est - np.mean(rank_est)) * (rank_true - np.mean(rank_true)))
    return float(cov / (std_est * std_true))
