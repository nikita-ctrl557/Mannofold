---
title: Regime Detection
tags: [theory, regimes, clustering]
---

# Regime Detection

Back to [[Home]] · Related: [[Manifold Embedding]] · [[Mannofold Theory]] ·
[[Signal Geometry]] · [[No-Lookahead]] · [[Glossary]]

A **regime** is a cluster of the embedded train cloud — a recurring market mode
(e.g. quiet uptrend, choppy range, selloff). Code: `mannofold/manifold/model.py`
(`ManifoldModelImpl`). The synthetic generator (`mannofold/feed/synthetic.py`)
emits a 3-state Markov-switching process (low-vol trend / high-vol revert /
crash), giving ground-truth labels the pipeline is meant to rediscover.

Each fitted regime is a `Regime` model (`mannofold/contracts/models.py`) with
`regime_id`, auto `label`, `color`, `size`, `mean_fwd_return`. Auto-labels combine
forward-return tone (`bull`/`bear`/`neutral`) with cloud spread (`high-vol`/
`low-vol`) — see `_auto_label`.

## KMeans (baseline) vs HDBSCAN (optional)

| | **KMeans** (default) | **HDBSCAN** (`regime_method="hdbscan"`) |
|---|---|---|
| Shape | spherical, every point assigned | density-based, arbitrary shapes |
| Count | fixed `k` (`n_regimes`, capped at `len(emb)//50`, min 2) | discovered; `min_cluster_size = max(15, len/50)` |
| Noise | none — all points get a regime | **`-1` = noise**, surfaced as the anomaly regime |
| Online assign | nearest centroid + softmax `regime_prob` (`_assign_kmeans`) | `approximate_predict` → label + membership strength (`_assign_hdbscan`) |
| Dependency | core (`scikit-learn`) | optional `manifold` extra (`hdbscan`) |

KMeans is the always-installable baseline; HDBSCAN is more honest about density
and noise but adds a compiled dependency. Rationale parallels
[[ADR-0003-pca-baseline-umap-swappable]].

## `approximate_predict` — no online re-cluster

When scoring a new point, Mannofold **never re-clusters**. KMeans assigns to the
nearest frozen centroid; HDBSCAN calls `hdbscan.approximate_predict(self._hdb, x)`
against the model fitted with `prediction_data=True`. Re-clustering online would
(a) leak the new point into the cluster definition (a [[No-Lookahead]] violation)
and (b) shuffle cluster ids every bar. Ids are stable only **within one fitted
model**, so the engine versions them per walk-forward refit ([[Architecture]]).

## `-1` noise = anomaly / crash regime

`ANOMALY_REGIME = -1` (`mannofold/contracts/models.py`) is the sentinel for
off-manifold states. Two paths land here:

1. **HDBSCAN noise.** Points `approximate_predict` labels `-1` — too sparse to
   belong to any density peak.
2. **Distance override.** Regardless of clusterer, if `anomaly_score >= 0.95`
   (kNN distance far beyond the train 95th-percentile `_dist_ref`),
   `transform_online` forces `regime_id = ANOMALY_REGIME`.

So `-1` means the same thing both ways: the market is somewhere the model has not
learned to price. Downstream this drives **de-grossing** rather than a directional
bet — see [[Signal Geometry]] and [[Risk Model]]. The dashboard paints the `-1`
regime grey (`#888888`) in the [[Architecture#frontend|legend]].
