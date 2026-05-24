"""Tests for the UMAP embedder and HDBSCAN regime backend.

PCA + KMeans remain the defaults (covered by test_correctness). These tests
exercise the optional manifold-upgrade backends behind the same interfaces:

* UMAP out-of-sample ``transform`` shape + determinism across two seeded fits;
* HDBSCAN online assignment flagging an out-of-distribution point as the
  ANOMALY_REGIME (noise) / high anomaly_score;
* a smoke test combining UMAP + HDBSCAN end-to-end through ManifoldModelImpl.
"""

from __future__ import annotations

import math

import numpy as np
from threadpoolctl import threadpool_limits

from mannofold.contracts.models import ANOMALY_REGIME
from mannofold.features import RollingFeaturePipeline
from mannofold.feed.synthetic import SyntheticConfig, generate_bars
from mannofold.manifold._umap import UMAPEmbedder, make_umap
from mannofold.manifold.model import ManifoldModelImpl


def _build_features(n_bars: int = 1200, seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """Feature matrix + aligned forward returns from synthetic bars."""
    bars, _ = generate_bars(SyntheticConfig(n_bars=n_bars, seed=seed))
    pipe = RollingFeaturePipeline()
    pipe.fit(bars)
    closes = [b.close for b in bars]
    horizon = 10
    rows: list[list[float]] = []
    fwd: list[float] = []
    for i in range(pipe.warmup - 1, len(bars)):
        fv = pipe.transform(bars[i - pipe.warmup + 1 : i + 1])
        rows.append(fv.values)
        if i + horizon < len(bars):
            fwd.append(math.log(closes[i + horizon] / max(closes[i], 1e-9)))
        else:
            fwd.append(float("nan"))
    return np.asarray(rows, dtype=float), np.asarray(fwd, dtype=float)


def test_umap_transform_shape_and_determinism():
    X, _ = _build_features(n_bars=900, seed=3)
    train, unseen = X[:600], X[600:660]

    with threadpool_limits(limits=1):
        e1 = UMAPEmbedder(n_components=3, random_state=42)
        e1.fit(train)
        out1 = e1.transform(unseen)

        e2 = UMAPEmbedder(n_components=3, random_state=42)
        e2.fit(train)
        out2 = e2.transform(unseen)

    assert out1.shape == (len(unseen), 3)
    assert np.isfinite(out1).all()
    # Same seed + single-threaded -> reproducible out-of-sample embedding.
    assert np.allclose(out1, out2, atol=1e-6)


def test_make_umap_factory():
    e = make_umap("umap", n_components=2)
    assert e.n_components == 2
    p = make_umap("parametric_umap", n_components=3)
    assert p.n_components == 3


def test_hdbscan_flags_out_of_distribution_point():
    X, fwd = _build_features(n_bars=1200, seed=11)
    model = ManifoldModelImpl(
        embedder="pca",
        n_components=3,
        regime_method="hdbscan",
    )
    with threadpool_limits(limits=1):
        model.fit(X, fwd)
        # A wildly out-of-distribution feature vector (far off the manifold).
        ood = np.full(X.shape[1], 50.0)
        state = model.transform_online(ood)

    assert state.regime_id == ANOMALY_REGIME or state.anomaly_score > 0.9
    assert math.isfinite(state.anomaly_score)
    # HDBSCAN produced at least one real regime in the legend.
    assert any(r.regime_id >= 0 for r in model.regimes)


def test_umap_hdbscan_smoke():
    X, fwd = _build_features(n_bars=900, seed=5)
    model = ManifoldModelImpl(
        embedder="umap",
        n_components=3,
        regime_method="hdbscan",
    )
    with threadpool_limits(limits=1):
        model.fit(X, fwd)
        state = model.transform_online(X[100])

    assert len(state.embedding) == 3
    assert all(math.isfinite(v) for v in state.embedding)
    assert math.isfinite(state.anomaly_score)
    assert math.isfinite(state.fwd_return_mean)
    assert math.isfinite(state.regime_prob)
    assert 0.0 <= state.anomaly_score <= 1.0
