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
        regime_method: str = "kmeans",
    ):
        self._embedder = make_embedder(embedder, n_components)
        self._n_regimes = n_regimes
        self._k = n_neighbors
        self._regime_method = regime_method.lower()
        if self._regime_method not in ("kmeans", "hdbscan"):
            raise ValueError(f"unknown regime_method: {regime_method!r}")
        self._km: KMeans | None = None
        self._hdb: object | None = None  # hdbscan.HDBSCAN, fitted with prediction_data
        self._centers: np.ndarray | None = None  # regime-id -> embedding centroid
        self._center_ids: list[int] = []
        self._knn: NearestNeighbors | None = None
        self._fwd = np.empty(0)
        self._dist_ref = 1.0
        self._regimes: list[Regime] = []

    def fit(self, X: np.ndarray, fwd_returns: np.ndarray) -> None:
        X = np.nan_to_num(np.asarray(X, dtype=float))
        self._embedder.fit(X)
        emb = self._embedder.transform(X)

        n_nb = min(self._k, max(2, len(emb) - 1))
        self._knn = NearestNeighbors(n_neighbors=n_nb).fit(emb)
        self._fwd = np.asarray(fwd_returns, dtype=float)

        dist, _ = self._knn.kneighbors(emb)
        self._dist_ref = float(np.percentile(dist.mean(axis=1), 95)) + _EPS

        if self._regime_method == "hdbscan":
            labels = self._fit_hdbscan(emb)
        else:
            labels = self._fit_kmeans(emb)

        self._regimes = self._build_regimes(labels, emb)

    def _fit_kmeans(self, emb: np.ndarray) -> np.ndarray:
        k_reg = min(self._n_regimes, max(2, len(emb) // 50))
        self._km = KMeans(n_clusters=k_reg, random_state=0, n_init=10).fit(emb)
        self._hdb = None
        return np.asarray(self._km.labels_)

    def _fit_hdbscan(self, emb: np.ndarray) -> np.ndarray:
        import hdbscan

        # min_cluster_size scaled to the train window so we recover a handful of
        # regimes rather than one blob or hundreds of micro-clusters. min_samples
        # is kept small (decoupled from min_cluster_size, whose default would make
        # the conservative density estimate label everything noise on the
        # smoothly-varying market embedding) so real regimes are actually found.
        min_cluster_size = max(15, len(emb) // 50)
        min_samples = min(5, min_cluster_size)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            prediction_data=True,
        )
        clusterer.fit(emb)
        self._hdb = clusterer
        self._km = None
        labels = np.asarray(clusterer.labels_)
        # Precompute centroids per cluster id for regime_prob softmax online.
        ids = sorted(rid for rid in set(labels.tolist()) if rid >= 0)
        if ids:
            self._centers = np.vstack(
                [emb[labels == rid].mean(axis=0) for rid in ids]
            )
            self._center_ids = ids
        else:
            self._centers = np.empty((0, emb.shape[1]))
            self._center_ids = []
        return labels

    def _build_regimes(self, labels: np.ndarray, emb: np.ndarray) -> list[Regime]:
        regimes: list[Regime] = []
        ids = sorted(set(labels.tolist()))
        for rid in ids:
            if rid == ANOMALY_REGIME:
                # Noise points: surface as the anomaly regime in the legend too.
                mask = labels == rid
                fwd = self._fwd[mask]
                mean_fwd = float(np.nanmean(fwd)) if np.isfinite(fwd).any() else 0.0
                regimes.append(
                    Regime(
                        regime_id=ANOMALY_REGIME,
                        label="anomaly",
                        color="#888888",
                        size=int(mask.sum()),
                        mean_fwd_return=mean_fwd,
                    )
                )
                continue
            mask = labels == rid
            fwd = self._fwd[mask]
            mean_fwd = float(np.nanmean(fwd)) if np.isfinite(fwd).any() else 0.0
            regimes.append(
                Regime(
                    regime_id=int(rid),
                    label=self._auto_label(mean_fwd, emb[mask]),
                    color=_PALETTE[int(rid) % len(_PALETTE)],
                    size=int(mask.sum()),
                    mean_fwd_return=mean_fwd,
                )
            )
        return regimes

    @staticmethod
    def _auto_label(mean_fwd: float, pts: np.ndarray) -> str:
        spread = float(np.mean(np.std(pts, axis=0))) if len(pts) else 0.0
        tone = "bull" if mean_fwd > 1e-4 else "bear" if mean_fwd < -1e-4 else "neutral"
        vol = "high-vol" if spread > 1.0 else "low-vol"
        return f"{vol} {tone}"

    def transform_online(self, x: np.ndarray) -> ManifoldState:
        if self._knn is None or (self._km is None and self._hdb is None):
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

        if self._hdb is not None:
            regime_id, regime_prob = self._assign_hdbscan(emb)
        else:
            regime_id, regime_prob = self._assign_kmeans(emb)

        # Distance-based anomaly still overrides to the noise/off-manifold regime.
        if anomaly >= 0.95:
            regime_id = ANOMALY_REGIME

        return ManifoldState(
            ts=_SENTINEL_TS,
            symbol="",
            embedding=emb[:3].tolist(),
            regime_id=regime_id,
            regime_prob=regime_prob,
            density=density,
            anomaly_score=anomaly,
            fwd_return_mean=fwd_mean,
            fwd_return_std=fwd_std,
            velocity=[],
        )

    def _assign_kmeans(self, emb: np.ndarray) -> tuple[int, float]:
        assert self._km is not None
        centers = self._km.cluster_centers_
        cdist = np.linalg.norm(centers - emb, axis=1)
        probs = np.exp(-cdist)
        probs = probs / (probs.sum() + _EPS)
        regime_id = int(np.argmin(cdist))
        return regime_id, float(probs[regime_id])

    def _assign_hdbscan(self, emb: np.ndarray) -> tuple[int, float]:
        import hdbscan

        labels, strengths = hdbscan.approximate_predict(
            self._hdb, emb.reshape(1, -1)
        )
        regime_id = int(labels[0])
        prob = float(strengths[0])
        if regime_id == ANOMALY_REGIME:
            # Noise -> off-manifold/crash signal; no soft assignment strength.
            return ANOMALY_REGIME, prob
        return regime_id, prob

    @property
    def regimes(self) -> list[Regime]:
        return list(self._regimes)
