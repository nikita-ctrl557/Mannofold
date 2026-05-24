"""UMAP-based embedders implementing the :class:`Embedder` Protocol.

These are optional, heavier alternatives to PCA. The contract is identical:

* ``fit(X)`` is called ONCE on the TRAIN matrix at a walk-forward refit boundary;
* ``transform(X)`` applies the FROZEN fitted model to new points — it MUST NOT
  re-fit. UMAP supports out-of-sample ``transform`` (it embeds new points into
  the previously learned space), which is exactly what the online step needs.

Determinism: a fixed ``random_state`` makes UMAP reproducible, at the cost of a
single-threaded layout optimisation (UMAP emits an ``n_jobs`` override warning —
expected and accepted). For bit-stable comparisons across two fits, also pin the
BLAS/OpenMP thread count (``threadpoolctl.threadpool_limits(1)``).

``ParametricUMAPEmbedder`` uses a learned neural encoder when TensorFlow +
``umap.ParametricUMAP`` are importable; otherwise it transparently falls back to
the standard :class:`UMAPEmbedder` so this module stays import-safe everywhere.
"""

from __future__ import annotations

import warnings

import numpy as np
import umap

_DEFAULT_SEED = 0


class UMAPEmbedder:
    """Out-of-sample UMAP embedding with a frozen fitted model.

    ``n_neighbors`` and ``min_dist`` are clamped at fit time to remain valid for
    small train windows.
    """

    def __init__(
        self,
        n_components: int = 3,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        random_state: int = _DEFAULT_SEED,
    ):
        self._n = int(n_components)
        self._n_neighbors = int(n_neighbors)
        self._min_dist = float(min_dist)
        self._random_state = int(random_state)
        self._model: umap.UMAP | None = None

    def fit(self, X: np.ndarray) -> None:
        X = np.nan_to_num(np.asarray(X, dtype=float))
        n = len(X)
        # UMAP requires 2 <= n_neighbors <= n_samples - 1.
        n_neighbors = max(2, min(self._n_neighbors, n - 1)) if n > 2 else 2
        # n_components must stay below the sample count for the spectral init.
        n_components = max(1, min(self._n, max(1, n - 2)))
        with warnings.catch_warnings():
            # Pinning random_state forces n_jobs=1 -> a benign UserWarning.
            warnings.simplefilter("ignore")
            model = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=self._min_dist,
                random_state=self._random_state,
                transform_seed=self._random_state,
                init="spectral",
                verbose=False,
            )
            model.fit(X)
        self._model = model

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("UMAPEmbedder.transform called before fit")
        X = np.nan_to_num(np.asarray(X, dtype=float))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            emb = np.asarray(self._model.transform(X), dtype=float)
        # Pad to the requested width when the train set forced fewer components.
        if emb.shape[1] < self._n:
            pad = np.zeros((emb.shape[0], self._n - emb.shape[1]), dtype=float)
            emb = np.hstack([emb, pad])
        return emb

    @property
    def n_components(self) -> int:
        return self._n


class ParametricUMAPEmbedder:
    """Parametric (neural-encoder) UMAP with a graceful fallback.

    If ``umap.ParametricUMAP`` (TensorFlow) is unavailable or fails to fit, this
    degrades to a standard :class:`UMAPEmbedder` so callers always get a working
    embedder with the same interface.
    """

    def __init__(
        self,
        n_components: int = 3,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        random_state: int = _DEFAULT_SEED,
    ):
        self._n = int(n_components)
        self._n_neighbors = int(n_neighbors)
        self._min_dist = float(min_dist)
        self._random_state = int(random_state)
        self._model: object | None = None
        self._fallback: UMAPEmbedder | None = None

    def _new_fallback(self) -> UMAPEmbedder:
        return UMAPEmbedder(
            n_components=self._n,
            n_neighbors=self._n_neighbors,
            min_dist=self._min_dist,
            random_state=self._random_state,
        )

    def fit(self, X: np.ndarray) -> None:
        X = np.nan_to_num(np.asarray(X, dtype=float))
        ParametricUMAP = getattr(umap, "ParametricUMAP", None)
        if ParametricUMAP is None:
            self._fallback = self._new_fallback()
            self._fallback.fit(X)
            return
        n = len(X)
        n_neighbors = max(2, min(self._n_neighbors, n - 1)) if n > 2 else 2
        n_components = max(1, min(self._n, max(1, n - 2)))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ParametricUMAP(
                    n_components=n_components,
                    n_neighbors=n_neighbors,
                    min_dist=self._min_dist,
                    random_state=self._random_state,
                    verbose=False,
                )
                model.fit(X)
            self._model = model
            self._fallback = None
        except Exception:
            # Missing TensorFlow / unsupported backend -> standard UMAP.
            self._model = None
            self._fallback = self._new_fallback()
            self._fallback.fit(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._fallback is not None:
            return self._fallback.transform(X)
        if self._model is None:
            raise RuntimeError("ParametricUMAPEmbedder.transform called before fit")
        X = np.nan_to_num(np.asarray(X, dtype=float))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            emb = np.asarray(self._model.transform(X), dtype=float)
        if emb.shape[1] < self._n:
            pad = np.zeros((emb.shape[0], self._n - emb.shape[1]), dtype=float)
            emb = np.hstack([emb, pad])
        return emb

    @property
    def n_components(self) -> int:
        return self._n


def make_umap(kind: str = "umap", n_components: int = 3):
    kind = kind.lower()
    if kind == "umap":
        return UMAPEmbedder(n_components=n_components)
    if kind == "parametric_umap":
        return ParametricUMAPEmbedder(n_components=n_components)
    raise ValueError(f"unknown umap kind: {kind!r}")
