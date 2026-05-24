"""Composed manifold model: embedding + regimes + neighbourhood forward returns.

Fit once on a TRAIN matrix, then assign new points online with no re-fit. Cluster
ids are stable only within one fitted model, so the engine versions them per
walk-forward refit.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from mannofold.contracts.models import ANOMALY_REGIME, ManifoldState, Regime
from mannofold.manifold.embedding import make_embedder

_EPS = 1e-9
_SENTINEL_TS = datetime(1970, 1, 1, tzinfo=UTC)

# Qualitative palette (regime id -> colour), reused by the dashboard legend.
_PALETTE = [
    "#4e79a7",
    "#f28e2b",
    "#e15759",
    "#76b7b2",
    "#59a14f",
    "#edc948",
    "#b07aa1",
    "#ff9da7",
]


class ManifoldModelImpl:
    def __init__(
        self,
        embedder: str = "pca",
        n_components: int = 3,
        n_regimes: int = 4,
        n_neighbors: int = 25,
    ):
        self._embedder = make_embedder(embedder, n_components)
        self._n_regimes = n_regimes
        self._k = n_neighbors
        self._km: KMeans | None = None
        self._knn: NearestNeighbors | None = None
        self._fwd = np.empty(0)
        self._dist_ref = 1.0
        self._regimes: list[Regime] = []

    def fit(self, X: np.ndarray, fwd_returns: np.ndarray) -> None:
        X = np.nan_to_num(np.asarray(X, dtype=float))
        self._embedder.fit(X)
        emb = self._embedder.transform(X)

        k_reg = min(self._n_regimes, max(2, len(emb) // 50))
        self._km = KMeans(n_clusters=k_reg, random_state=0, n_init=10).fit(emb)
        n_nb = min(self._k, max(2, len(emb) - 1))
        self._knn = NearestNeighbors(n_neighbors=n_nb).fit(emb)
        self._fwd = np.asarray(fwd_returns, dtype=float)

        dist, _ = self._knn.kneighbors(emb)
        self._dist_ref = float(np.percentile(dist.mean(axis=1), 95)) + _EPS

        labels = self._km.labels_
        regimes: list[Regime] = []
        for rid in range(k_reg):
            mask = labels == rid
            fwd = self._fwd[mask]
            mean_fwd = float(np.nanmean(fwd)) if np.isfinite(fwd).any() else 0.0
            regimes.append(
                Regime(
                    regime_id=rid,
                    label=self._auto_label(mean_fwd, emb[mask]),
                    color=_PALETTE[rid % len(_PALETTE)],
                    size=int(mask.sum()),
                    mean_fwd_return=mean_fwd,
                )
            )
        self._regimes = regimes

    @staticmethod
    def _auto_label(mean_fwd: float, pts: np.ndarray) -> str:
        spread = float(np.mean(np.std(pts, axis=0))) if len(pts) else 0.0
        tone = "bull" if mean_fwd > 1e-4 else "bear" if mean_fwd < -1e-4 else "neutral"
        vol = "high-vol" if spread > 1.0 else "low-vol"
        return f"{vol} {tone}"

    def transform_online(self, x: np.ndarray) -> ManifoldState:
        if self._km is None or self._knn is None:
            raise RuntimeError("ManifoldModelImpl.transform_online called before fit")
        x = np.nan_to_num(np.asarray(x, dtype=float)).reshape(1, -1)
        emb = self._embedder.transform(x)[0]

        dist, idx = self._knn.kneighbors(emb.reshape(1, -1))
        neigh_fwd = self._fwd[idx[0]]
        valid = neigh_fwd[np.isfinite(neigh_fwd)]
        fwd_mean = float(np.mean(valid)) if valid.size else 0.0
        fwd_std = float(np.std(valid)) if valid.size else 0.0

        mean_dist = float(dist[0].mean())
        anomaly = float(np.clip(mean_dist / self._dist_ref, 0.0, 2.0) / 2.0)
        density = float(1.0 / (1.0 + mean_dist))

        centers = self._km.cluster_centers_
        cdist = np.linalg.norm(centers - emb, axis=1)
        probs = np.exp(-cdist)
        probs = probs / (probs.sum() + _EPS)
        regime_id = int(np.argmin(cdist))

        return ManifoldState(
            ts=_SENTINEL_TS,
            symbol="",
            embedding=emb[:3].tolist(),
            regime_id=regime_id if anomaly < 0.95 else ANOMALY_REGIME,
            regime_prob=float(probs[regime_id]),
            density=density,
            anomaly_score=anomaly,
            fwd_return_mean=fwd_mean,
            fwd_return_std=fwd_std,
            velocity=[],
        )

    @property
    def regimes(self) -> list[Regime]:
        return list(self._regimes)
