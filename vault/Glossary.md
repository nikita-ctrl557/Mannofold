---
title: Glossary
tags: [reference, glossary]
---

# Glossary

Back to [[Home]] · Related: [[Mannofold Theory]] · [[Manifold Embedding]] ·
[[Regime Detection]] · [[Signal Geometry]] · [[Risk Model]] · [[Architecture]] ·
[[No-Lookahead]]

Definitions of the core Mannofold vocabulary, each pointing at the authoritative
note and code.

## manifold

The low-dimensional surface `M ⊂ R^D` near which market-state vectors `x_t`
concentrate. Mannofold's central hypothesis: markets revisit a few recurring
states, so `{x_t}` does not fill `R^11` uniformly. See [[Mannofold Theory]].

## market state `x_t`

The high-dimensional feature vector for one bar — `D = 11` features
(`ret_1`, `mom_5/10/20`, `accel_10`, `vol_10/20`, `rsi_14`, `sma_ratio_20`,
`range_pct`, `volume_z_20`). Built by `RollingFeaturePipeline`
(`mannofold/features/pipeline.py`). Carried as `FeatureVector`
(`contracts/models.py`). See [[Mannofold Theory]].

## embedding / φ

The learned map `φ: R^D → R^k` (`k = 2` or `3`) from market state to the
visualization/working coordinate. Baseline PCA, optionally UMAP / Parametric
UMAP. The `embedding` field on `ManifoldState`. See [[Manifold Embedding]] and
[[ADR-0003-pca-baseline-umap-swappable]].

## regime

A cluster of the embedded train cloud — a recurring market mode (e.g. quiet
uptrend, choppy range, selloff). `Regime` model carries id, label, colour, size,
`mean_fwd_return`. KMeans baseline, HDBSCAN optional. See [[Regime Detection]].

## ANOMALY_REGIME (`-1`)

Sentinel regime id (`contracts/models.py`) for off-manifold / crash states.
Reached via HDBSCAN noise or the distance override (`anomaly_score >= 0.95`).
Drives de-grossing, not a directional bet. See [[Regime Detection]].

## anomaly score

Distance-from-manifold in `[0,1]`: kNN mean distance ÷ train 95th-percentile
`_dist_ref`. High = the model has not learned to price this state → de-gross.
`ManifoldState.anomaly_score`. See [[Signal Geometry]], [[Risk Model]].

## density

Local neighbourhood density `1 / (1 + mean_dist)`; higher = more typical state.
Used for marker radius in the manifold map. See [[Architecture]].

## neighbourhood forward return

The mean realized forward return of a new point's kNN neighbours in the train
embedding (`k = 25`, `fwd_horizon = 10` bars). The basis of `expected_return`.
"States that historically looked like this went up." See [[Signal Geometry]].

## trajectory velocity

`Δembedding` over the last `velocity_lookback = 5` embeddings
(`ManifoldState.velocity`); how fast/where the state is moving on the manifold.
See [[Mannofold Theory]], [[Signal Geometry]].

## confidence

`max(0, regime_prob * (1 - anomaly_score)) ∈ [0,1]`; gates entries and scales the
target. High only when squarely inside a known regime and near the manifold.
See [[Signal Geometry]].

## walk-forward refit

Periodically re-fitting the model (scaler → φ → regimes → kNN) on a trailing
window of **past** bars (`refit_every`, capped at `max_train`), never on future
data. Cluster ids are stable only within one fitted model, so they are versioned
per refit. See [[Architecture]], [[No-Lookahead]].

## lookahead bias

Using information from bar *t+1…* in a decision at bar *t*. Structurally guarded
(scaler inside pipeline, fit-on-train-only, frozen `transform`,
`approximate_predict`, NaN-tail forward returns) and enforced by
`test_no_lookahead`. See [[No-Lookahead]].

## fit-on-train-only

The rule that `fit()` (scaler, embedder, clusterer, kNN) sees only past TRAIN
bars; new bars are scored with the frozen model via `transform` /
`transform_online`. See [[No-Lookahead]], [[Manifold Embedding]].

## vol targeting

Scaling exposure by `target_vol / realized_vol` so realized portfolio volatility
stays near a target regardless of regime. `VolTargetRiskSizer`
(`mannofold/signals/risk.py`). See [[Risk Model]].

## de-grossing

Cutting gross exposure as anomaly rises (`weight *= 1 - anomaly`), applied in both
the strategy and the sizer. See [[Risk Model]], [[Signal Geometry]].

## rebalance band / hysteresis

A two-level band (wider when adding same-side risk, tighter when trimming/flipping)
plus a min-trade floor that suppresses churn. Mirrored by the strategy's
entry/exit thresholds. See [[Risk Model]], [[Signal Geometry]].

## data feed

The `DataFeed` Protocol — the only thing that drives the clock. Backtest
(`HistoricalReplayFeed`) and paper (`LiveReplayFeed` / `AlpacaFeed`) differ only
here. See [[Architecture]], [[ADR-0006-alpaca-paper-feed]].

## backtest ≡ paper (golden equivalence)

The invariant (and test) that the same series produces bit-identical
`StepResult`s through the historical and live-replay feeds — proving the single
online step. See [[No-Lookahead]], [[Architecture]].

## StepResult

The unit of the event stream: bar, features, manifold, signals, target, order,
fill, portfolio for one engine step (`contracts/models.py`). Persisted and
streamed over the WebSocket. See [[Architecture]].

## micro-batch Parquet

The persistence pattern: immutable Parquet parts in a partitioned folder, queried
through a DuckDB glob view — never streaming rows into one file. `LocalStateStore`
(`mannofold/persist/store.py`). See [[Architecture]],
[[ADR-0004-supabase-optional-exporter]].
