---
title: Manifold Embedding
tags: [theory, embedding]
---

# Manifold Embedding — φ: R^D → R^k

Back to [[Home]] · Related: [[Mannofold Theory]] · [[Regime Detection]] ·
[[No-Lookahead]] · [[ADR-0003-pca-baseline-umap-swappable]] · [[Glossary]]

The embedding `φ` maps the 11-dimensional market state ([[Mannofold Theory]]) to
2 or 3 dimensions. It does two jobs at once: it is the **coordinate system the
geometry signals are computed in** *and* it is the **visualization** rendered by
deck.gl ([[Architecture]]). Code: `mannofold/manifold/embedding.py`.

## The `Embedder` Protocol

```text
Embedder.fit(X)            # learn φ on the TRAIN matrix
Embedder.transform(X)      # apply the FROZEN φ to new points
Embedder.n_components      # k (2 or 3)
```

`make_embedder(kind, n_components)` is the single factory. `kind="pca"` returns
`PCAEmbedder`; `kind in {"umap","parametric_umap"}` lazily imports
`mannofold.manifold._umap.make_umap` (the optional `manifold` extra:
`umap-learn`, `hdbscan` in `pyproject.toml`). New embedder kinds only touch the
factory — nothing downstream knows which `φ` it got.

## PCA → UMAP → Parametric UMAP

| Stage | What | Transform stability |
|-------|------|---------------------|
| **PCA** (baseline, default `n_components=3`) | linear projection onto top variance directions | **Exact, deterministic.** `transform(x)` is a matrix multiply — the same `x` always maps to the same point. No online drift. |
| **UMAP** | nonlinear; preserves local neighbourhood topology, unfolds curved manifolds PCA flattens | `transform()` on a fitted model is supported but **slower and less stable** — it optimises new points against the frozen embedding, so it is approximate and can jitter. |
| **Parametric UMAP** | UMAP whose `φ` is a trained neural net | `transform()` is a forward pass: fast, deterministic, the most "online-friendly" — the eventual target for live use. |

The progression trades determinism + speed (PCA) for representational power
(UMAP). Decision and rationale: [[ADR-0003-pca-baseline-umap-swappable]].

## Why fit-on-train-only

`Embedder.fit` is called by `ManifoldModelImpl.fit` (`mannofold/manifold/model.py`)
which the engine only ever calls on **past** bars (the train slice, expanding up
to `max_train`). New bars are scored with `transform_online`, which calls
`embedder.transform` on the **frozen** model — never `fit_transform`. Fitting `φ`
on data that includes the bar being scored would leak the future into its own
coordinate. This is one of the invariants the [[No-Lookahead]] test enforces.

## Why signals must not depend on a refit "display" embedding

PCA/UMAP coordinates are only defined **up to sign, rotation and (for UMAP)
arbitrary layout** — a refit can flip or rotate the whole cloud. If signals were
read off a *display* embedding that gets refit for the dashboard, a cosmetic
refit could silently change positions and thus trades. Mannofold avoids this by:

- Computing all geometry signals inside `transform_online` from the **frozen**
  fitted model that produced this run's embedding — the embedding the signal sees
  is the embedding it was trained against.
- Versioning regime ids **per walk-forward refit** (cluster ids are only stable
  within one fitted model — see the module docstring in `model.py` and
  [[Regime Detection]]).
- Persisting the actual embedding coordinates in each `StepResult` (`store.py`
  flattens `emb_x/y/z`), so the frontend renders *recorded* coordinates rather
  than re-embedding.

The display layer therefore consumes embeddings; it never produces the ones the
strategy trades on.
