---
title: Signal Geometry
tags: [theory, signals, strategy]
---

# Signal Geometry — geometry → `SignalSet` → `TargetPosition`

Back to [[Home]] · Related: [[Mannofold Theory]] · [[Manifold Embedding]] ·
[[Regime Detection]] · [[Risk Model]] · [[No-Lookahead]] · [[Glossary]]

This note traces how the four trading signals are derived from manifold geometry
and shaped into a desired exposure. The geometric inputs are computed in
`mannofold/manifold/model.py::transform_online`; the signal maths and target
shaping live in `mannofold/signals/strategy.py` (`ManifoldStrategy`). Sizing into
a concrete order is [[Risk Model]].

## Inputs: the frozen `ManifoldState`

Every signal reads **only** the `ManifoldState` for the current bar
(`mannofold/contracts/models.py`) — never a future bar (see [[No-Lookahead]]).
The geometrically-derived fields:

| Field | How it is computed (`transform_online`) | Meaning |
|-------|------------------------------------------|---------|
| `fwd_return_mean` | mean of the kNN neighbours' realized forward returns (`k = 25`, `fwd_horizon = 10`), NaN neighbours dropped | "where you sit" — neighbourhood forward return |
| `fwd_return_std` | std of the same neighbour forward returns | dispersion of that estimate |
| `anomaly_score` | kNN mean distance ÷ train 95th-pct `_dist_ref`, clipped to `[0,2]`, ÷2 → `[0,1]` | "how far off-manifold" |
| `density` | `1 / (1 + mean_dist)` | local density (higher = more typical) |
| `regime_id` / `regime_prob` | nearest-centroid softmax (KMeans) or `approximate_predict` strength (HDBSCAN) | which regime + membership |
| `velocity` | `Δembedding` over `velocity_lookback = 5` steps, attached by the engine | "how fast you're moving" |

## `signals()` → `SignalSet`

```text
sharpe      = fwd_return_mean / (fwd_return_std + eps)   # neighbourhood Sharpe
momentum    = tanh(sharpe)
expected_return = fwd_return_mean
anomaly     = anomaly_score
confidence  = max(0, regime_prob * (1 - anomaly_score))
regime_id   = state.regime_id
```

- **`expected_return`** is the raw neighbourhood forward-return mean — the
  directional view.
- **`momentum`** is the `tanh` of the neighbourhood *Sharpe* (return per unit of
  its spread): a noisy neighbourhood (high `fwd_return_std`) shrinks momentum.
- **`anomaly`** passes through untouched and drives de-grossing.
- **`confidence ∈ [0,1]`** multiplies regime membership by `(1 − anomaly)`: you
  are confident only when the state is squarely inside a known regime *and* near
  the manifold.

## `target()` → `TargetPosition`

Raw conviction is mapped from `expected_return` through a `tanh` gain, then gated
and smoothed (all per-symbol, deterministic instance state → no lookahead):

```text
raw     = tanh(gain * expected_return)        # gain = 60
desired = raw * confidence * (1 - anomaly)    # confidence + anomaly de-gross
gated   = _gate(desired, prev, signals)       # hysteresis + regime/conf gate
smoothed= prev + smoothing * (gated - prev)   # EMA, smoothing = 0.35
if |smoothed| < deadband (0.04): smoothed = 0 # no-trade dead-band
target_weight = clip(smoothed, -1, 1)
```

### The `_gate` (overtrading control)

- **Off-manifold / unknown regime** (`regime_id == ANOMALY_REGIME` or
  `anomaly > anomaly_gate = 0.6`): return `0.0` — never add risk, only decay to
  flat. This is the geometric kill-switch ([[Regime Detection]]).
- **Holding, same side**: hold unless conviction decays below the *lower*
  `exit_threshold = 0.05` (hysteresis).
- **Flat or flipping sign** (fresh entry): require the *higher*
  `entry_threshold = 0.15` **and** `confidence >= confidence_floor = 0.25`.

Entry harder than exit (hysteresis) + EMA smoothing + dead-band together damp
bar-to-bar churn while still letting risk come off promptly. The same
hysteresis philosophy appears one layer down in the sizer's rebalance band —
[[Risk Model]].

## How the three [[Mannofold Theory]] mechanisms map here

1. **Where you sit** → `expected_return` (kNN forward return) → `raw` conviction.
2. **How fast you move** → `momentum` (`tanh` of neighbourhood Sharpe).
3. **How far off-manifold** → `anomaly` → confidence shrink + `(1 − anomaly)`
   de-gross + the hard gate.
