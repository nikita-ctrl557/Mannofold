"""Manifold layer: embedding φ + regimes + neighbourhood forward-return model.

The embedder is swappable behind :func:`make_embedder` (PCA baseline → UMAP →
Parametric UMAP) without touching the composed :class:`ManifoldModelImpl`.
"""

from mannofold.manifold.embedding import PCAEmbedder, make_embedder
from mannofold.manifold.model import ManifoldModelImpl

__all__ = ["PCAEmbedder", "make_embedder", "ManifoldModelImpl"]
