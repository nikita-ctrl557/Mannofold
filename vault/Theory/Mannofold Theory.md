---
title: Mannofold Theory
tags: [theory, manifold]
---

# Mannofold Theory — the market-manifold thesis

Back to [[Home]] · Related: [[Manifold Embedding]] · [[Regime Detection]] ·
[[Signal Geometry]] · [[Risk Model]] · [[Glossary]]

## The high-dimensional market state `x_t`

At each bar the market is described by a feature vector `x_t ∈ R^D` built from a
trailing window. In Mannofold `D = 11` and the vector is produced by
`RollingFeaturePipeline` in `mannofold/features/pipeline.py`, with the indicator
maths in `mannofold/features/indicators.py`. The exact features (order matters —
it is the wire layout of `FeatureVector.values`):

| # | Name | Meaning | Indicator fn |
|---|------|---------|--------------|
| 0 | `ret_1` | last log return | `log_returns` |
| 1 | `mom_5` | 5-bar log momentum | `momentum` |
| 2 | `mom_10` | 10-bar log momentum | `momentum` |
| 3 | `mom_20` | 20-bar log momentum | `momentum` |
| 4 | `accel_10` | momentum-of-momentum (Δ of 10-bar mom) | `acceleration` |
| 5 | `vol_10` | 10-bar realized vol (std of log returns) | `realized_vol` |
| 6 | `vol_20` | 20-bar realized vol | `realized_vol` |
| 7 | `rsi_14` | RSI(14), scaled to `0..1` (÷100) | `rsi` |
| 8 | `sma_ratio_20` | close / SMA(20) − 1 | `sma_ratio` |
| 9 | `range_pct` | (high − low) / close of the current bar | `range_pct` |
| 10 | `volume_z_20` | z-score of current volume over 20-bar window | `volume_z` |

These mix **multi-horizon returns/momentum**, **realized volatility**,
**oscillator** (RSI), **trend** (SMA ratio), **bar geometry** (range) and
**volume** — a deliberately redundant, correlated set. Redundancy is the point:
correlated coordinates are exactly what makes the cloud of `x_t` collapse onto a
lower-dimensional surface. See `WARMUP = 41` in the pipeline (40 trailing bars +
current, for the longest 20-bar lookback chained through `acceleration`).

Every indicator is **causal**: each function in `indicators.py` takes a trailing
array whose last element is the current bar and looks only backwards. The
`StandardScaler` is fit inside `pipeline.fit()` on TRAIN bars only — see
[[No-Lookahead]].

## The manifold hypothesis

The set `{x_t}` does not fill `R^11` uniformly. Markets revisit a small number of
recurring *states* (quiet uptrend, choppy range, volatile selloff, crash), so the
data concentrate near a low-dimensional **manifold** `M ⊂ R^11`. Two consequences:

1. A smooth map `φ: R^11 → R^k` (`k = 2` or `3`) can represent state with little
   loss — this is the **embedding** and *is* the visualization (`embedding` field
   on `ManifoldState`, rendered by deck.gl, see [[Architecture]]).
2. Points far from `M` are **anomalies** — regimes the model has never learned to
   price. Mannofold de-grosses there rather than trusting a signal.

## Embedding φ

`φ` is learned by an `Embedder` (`mannofold/manifold/embedding.py`). Baseline is
**PCA(3)** — deterministic, with a true linear `transform`. UMAP / Parametric
UMAP are drop-in alternatives behind `make_embedder(...)`. Full treatment in
[[Manifold Embedding]]; the choice rationale is [[ADR-0003-pca-baseline-umap-swappable]].

## Regimes

After embedding, the train cloud is clustered into **regimes** (`KMeans` baseline,
`HDBSCAN` optional) in `mannofold/manifold/model.py`. Each `Regime` carries a
colour, size and `mean_fwd_return`, and is auto-labelled (e.g. `high-vol bear`).
Regime id `-1` (`ANOMALY_REGIME`) is the off-manifold / crash bucket. Details in
[[Regime Detection]].

## Geometry → signals: three mechanisms

All three read **only** the frozen `ManifoldState` for the current bar — never the
raw future. They are realized in `mannofold/manifold/model.py::transform_online`
and consumed by `mannofold/signals/strategy.py`.

1. **Where you sit — neighbourhood forward return.** A `NearestNeighbors` (kNN,
   `k = 25`) index over the *train* embedding holds each train point's realized
   forward return (`fwd_horizon = 10` bars). For a new point we average the
   neighbours' forward returns → `fwd_return_mean` / `fwd_return_std`. This is the
   `expected_return` signal. "States that historically looked like this went up."

2. **How fast you're moving — trajectory velocity.** The engine keeps the last
   `velocity_lookback = 5` embeddings and stores `Δembedding` as
   `ManifoldState.velocity`. The strategy turns the neighbourhood Sharpe
   (`fwd_mean / fwd_std`) into a `momentum` signal via `tanh`. Movement *along the
   return gradient* of the manifold is bullish.

3. **How far you are from the manifold — anomaly.** kNN mean distance, normalised
   by the train 95th percentile (`_dist_ref`), gives `anomaly_score ∈ [0,1]`. High
   anomaly ⇒ off-manifold ⇒ de-gross (it multiplies the target weight down in both
   `ManifoldStrategy.target` and `VolTargetRiskSizer`). At `anomaly ≥ 0.95` the
   point is forced into `ANOMALY_REGIME`.

These feed [[Signal Geometry]] (the signal maths) and [[Risk Model]] (sizing).

## One core, two clocks

The same engine step (`mannofold/engine/engine.py`) drives **backtest** and
**paper**; only the [[Glossary#data feed|data feed]] and clock differ. The
golden equivalence test in `tests/test_correctness.py` proves the two are
bit-identical. See [[Architecture]] and [[No-Lookahead]].
