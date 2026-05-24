---
title: No-Lookahead
tags: [architecture, correctness, testing]
---

# No-Lookahead — point-in-time correctness

Back to [[Home]] · Related: [[Architecture]] · [[Mannofold Theory]] ·
[[Manifold Embedding]] · [[Regime Detection]] · [[Signal Geometry]] ·
[[Risk Model]] · [[Glossary]]

**Lookahead bias** is the single most dangerous bug in a trading system: a
decision at bar *t* that secretly uses information from bar *t+1…*. Mannofold is
designed so that lookahead is structurally hard to introduce, and two tests in
`tests/test_correctness.py` enforce it.

## The correctness rules

### 1. Scaler lives inside the pipeline

`RollingFeaturePipeline` (`mannofold/features/pipeline.py`) owns its
`StandardScaler`. It is `fit()` on TRAIN bars only; `transform()` merely applies
the frozen scaler. No caller can accidentally fit it on the full series — the
most common source of leakage. Indicators (`indicators.py`) are all **causal**:
each takes a trailing array whose last element is the current bar and looks only
backwards.

### 2. Fit-on-train-only (walk-forward)

`Engine._fit` calls `build_training` → `pipe.fit` and `model.fit` only on the
trailing TRAIN slice (`buf[-max_train:]`), at the initial fit and at each
`refit_every` boundary. New bars are scored with `transform`/`transform_online`
on the **frozen** model — never `fit_transform` online. See [[Architecture]]
(walk-forward refit) and [[Manifold Embedding]] (why signals must not depend on a
refit display embedding).

### 3. UMAP `transform` (not `fit_transform`)

`mannofold/manifold/_umap.py` documents and enforces: `fit(X)` once on TRAIN at a
refit boundary, `transform(X)` applies the frozen embedding to new points
(out-of-sample). Fitting φ on data that includes the bar being scored would leak
that bar into its own coordinate ([[Manifold Embedding]]).

### 4. HDBSCAN `approximate_predict` (no online re-cluster)

`ManifoldModelImpl._assign_hdbscan` calls
`hdbscan.approximate_predict(self._hdb, x)` against a model fitted with
`prediction_data=True`; KMeans assigns to the nearest frozen centroid. The system
**never re-clusters** online — re-clustering would let the new point reshape the
cluster definitions ([[Regime Detection]]).

### 5. Forward-return alignment excludes the unrealized tail

`build_training` (`mannofold/engine/engine.py`) aligns each train row to its
realized forward return `log(close[i+horizon] / close[i])`, but writes **NaN**
for rows where `i + horizon >= len(bars)` (no realized future yet). The kNN
forward-return model drops NaN neighbours when averaging
(`transform_online`), so an unrealized future is never used as a label.

### 6. Velocity uses only past embeddings

`ManifoldState.velocity` is `Δembedding` over the last `velocity_lookback`
embeddings the engine has already seen — never a future embedding.

### 7. Strategy/sizer state depends only on the past

`ManifoldStrategy` keeps a per-symbol EMA of the target ([[Signal Geometry]]);
`VolTargetRiskSizer` is a pure function of the current target + book
([[Risk Model]]). Both are deterministic for a given ordered stream of inputs and
introduce no lookahead.

## The two enforcing tests

`tests/test_correctness.py` (both pin BLAS/OpenMP to one thread via
`threadpool_limits(1)` — KMeans' threaded float reductions are otherwise
non-deterministic at the ULP level and would mask the real invariant).

### Golden equivalence — `test_backtest_equals_paper`

Runs the same synthetic series through `HistoricalReplayFeed` (backtest) and
`LiveReplayFeed(speed=0)` (paper) and asserts step-for-step equality of
`embedding`, `regime_id`, `target_weight`, and `equity`. Proves the **single
online step** drives both modes — if any code read the wall clock or a future
bar, the two would diverge. See the DataFeed≡invariant in [[Architecture]].

### No-lookahead — `test_no_lookahead`

Takes a series, **mutates every bar at/after `divergence = 1200`** (×1.5 close &
high), runs both the original and mutated series, and asserts that every
`StepResult` for bars strictly *before* the mutation point is byte-for-byte
identical (`model_dump()` equality). If any decision before *t* used a bar at or
after *t*, the pre-mutation steps would change — and the test fails with
`lookahead leak at seq N`.

Together these are the two **load-bearing correctness tests** for the engine.
