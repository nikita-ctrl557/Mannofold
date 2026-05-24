---
title: ADR-0003 — PCA baseline, UMAP swappable
tags: [adr, decision, manifold]
---

# ADR-0003 — PCA baseline embedder, UMAP/HDBSCAN swappable behind a factory

Back to [[Home]] · Related: [[Manifold Embedding]] · [[Regime Detection]] ·
[[No-Lookahead]] · [[Architecture]] · [[ADR-0001-stack]] · [[Glossary]]

Status: **Accepted**

## Context

The embedding φ ([[Manifold Embedding]]) and regime clusterer ([[Regime
Detection]]) are the most experimental parts of the system. UMAP and HDBSCAN are
powerful but are **heavy compiled dependencies** (slower installs, platform
build risk) and UMAP's online `transform` is approximate and can jitter. The
[[No-Lookahead]] golden-equivalence test also demands **determinism**, which PCA
gives for free and UMAP/KMeans only with pinned seeds + single-thread BLAS.

## Decision

- **Baseline = PCA(3)** (`PCAEmbedder`, `mannofold/manifold/embedding.py`) and
  **KMeans** regimes (`ManifoldModelImpl`, `model.py`). Both are in core
  `scikit-learn`, deterministic, and have a true linear / nearest-centroid
  `transform` with no online drift.
- **UMAP / Parametric UMAP** (`mannofold/manifold/_umap.py`) and **HDBSCAN**
  implement the *same* `Embedder` Protocol / `regime_method` switch and are
  reached only through `make_embedder(kind, n_components)` and the
  `regime_method` constructor arg. They live behind the **optional `manifold`
  extra** (`umap-learn`, `hdbscan` in `pyproject.toml`).
- New embedder kinds touch **only the factory** — nothing downstream knows which
  φ it received.

## Consequences

- The baseline always installs and runs (and passes the deterministic
  equivalence test); the heavy libs are opt-in ([[ADR-0001-stack]]).
- Upgrading to UMAP/HDBSCAN is a config change (`embedder=`, `regime_method=`),
  not a refactor — the manifold-upgrade workstream is decoupled.
- PCA is linear and may under-represent a curved manifold; that is the explicit
  trade for determinism + zero install cost, with UMAP available when the
  representational power is worth it.
- `ParametricUMAPEmbedder` degrades gracefully to standard `UMAPEmbedder` when
  TensorFlow is unavailable, so the module is import-safe everywhere.
