"""The science test: does the manifold pipeline rediscover the true regimes?

Synthetic bars carry ground-truth Markov-state labels. We build the train feature
matrix with the same RollingFeaturePipeline the engine uses, fit ManifoldModelImpl,
recover per-train-point cluster assignments, align them to the ground-truth labels
(offset by the pipeline warmup), and assert recovery beats chance.

Thresholds are deliberately lenient: PCA+KMeans on noisy GBM features will not
perfectly partition a 3-state Markov chain, but it should land well above random.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from mannofold.engine.engine import build_training
from mannofold.features.pipeline import RollingFeaturePipeline
from mannofold.feed.synthetic import SyntheticConfig, generate_bars
from mannofold.manifold.model import ManifoldModelImpl


def _cluster_purity(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """Fraction of points whose cluster's majority true-label matches their own."""
    total = len(true_labels)
    correct = 0
    for cluster in np.unique(pred_labels):
        mask = pred_labels == cluster
        truth = true_labels[mask]
        if truth.size:
            counts = np.bincount(truth)
            correct += int(counts.max())
    return correct / total if total else 0.0


def _recover():
    bars, labels = generate_bars(SyntheticConfig(n_bars=3000, seed=7))
    pipe = RollingFeaturePipeline()

    with threadpool_limits(limits=1):
        # build_training fits the pipeline and returns the train feature matrix +
        # aligned forward returns; rows start at bar index (warmup - 1).
        X, fwd = build_training(bars, pipe, horizon=10)
        model = ManifoldModelImpl(
            embedder="pca", n_components=3, n_regimes=4, n_neighbors=25
        )
        model.fit(X, fwd)

    # Re-derive per-point cluster ids with the SAME fitted embedder + kmeans the
    # model used internally, so we score exactly what the engine would see.
    emb = model._embedder.transform(np.nan_to_num(X))
    assert model._km is not None, "expected kmeans regime method"
    pred = np.asarray(model._km.predict(emb))

    # Ground-truth labels aligned to the feature rows (offset by warmup - 1).
    true = np.asarray(labels[pipe.warmup - 1 :], dtype=int)
    assert len(true) == len(pred) == len(emb)
    return true, pred


def test_regime_recovery_beats_chance():
    true, pred = _recover()

    ari = adjusted_rand_score(true, pred)
    purity = _cluster_purity(true, pred)

    # Lenient but meaningful: random assignment gives ARI ~ 0 and purity near the
    # majority-class frequency. We require a clear margin over either bar.
    assert ari > 0.1 or purity > 0.5, (
        f"regime recovery no better than chance: ARI={ari:.3f}, purity={purity:.3f}"
    )


def test_recovery_finds_multiple_clusters():
    """A degenerate one-blob result would trivially pass purity; guard against it."""
    _, pred = _recover()
    assert len(np.unique(pred)) >= 2
