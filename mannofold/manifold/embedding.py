"""Swappable embedding φ: R^D → R^k.

Baseline is PCA — deterministic, with a true linear ``transform`` (no online
instability). The manifold-upgrade workstream adds ``UMAPEmbedder`` /
``ParametricUMAPEmbedder`` in ``_umap.py`` implementing the same Protocol; the
factory is the only place that needs to learn about new kinds.

CRITICAL: ``transform`` on a new point must use the FROZEN fitted model. Never
``fit_transform`` online. UMAP's ``transform`` is supported but slower/less
stable than the fitted embedding — see vault/Manifold Embedding.md.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.decomposition import PCA


class Embedder(Protocol):
    def fit(self, X: np.ndarray) -> None: ...

    def transform(self, X: np.ndarray) -> np.ndarray: ...

    @property
    def n_components(self) -> int: ...


class PCAEmbedder:
    def __init__(self, n_components: int = 3):
        self._n = n_components
        self._pca = PCA(n_components=n_components, random_state=0)

    def fit(self, X: np.ndarray) -> None:
        self._pca.fit(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._pca.transform(X))

    @property
    def n_components(self) -> int:
        return self._n


def make_embedder(kind: str = "pca", n_components: int = 3) -> Embedder:
    kind = kind.lower()
    if kind == "pca":
        return PCAEmbedder(n_components)
    if kind in ("umap", "parametric_umap"):
        # Provided by the manifold-upgrade workstream; optional `manifold` extra.
        from mannofold.manifold._umap import make_umap

        return make_umap(kind, n_components)
    raise ValueError(f"unknown embedder kind: {kind!r}")
