import numpy as np, tensorflow as tf
from crowdgp.data import CrowdLabels

def test_alignment():
    rng = np.random.default_rng(0)
    N, A, C, L = 50, 6, 3, 200
    item = rng.integers(0, N, L); worker = rng.integers(0, A, L)
    _, keep = np.unique(item * A + worker, return_index=True)   # dedupe (item, worker)
    item, worker = item[keep], worker[keep]
    label = rng.integers(0, C, len(item))

    labels = CrowdLabels(item, worker, label, num_items=N)
    X = tf.constant(rng.normal(size=(N, 2)))
    idx = tf.constant([5, 1, 30, 3], tf.int32)
    batch = labels.gather_batch(X, idx)

    lookup = {(i, a): y for i, a, y in zip(item, worker, label)}
    recovered = tf.gather(idx, batch.item_local).numpy()
    for n, a, y in zip(recovered, batch.worker_idx.numpy(), batch.label.numpy()):
        assert lookup[(n, a)] == y
    assert batch.X.shape[0] == 4