---
title: Risk Model
tags: [theory, risk, sizing]
---

# Risk Model — vol targeting, anomaly de-grossing, rebalance band

Back to [[Home]] · Related: [[Signal Geometry]] · [[Mannofold Theory]] ·
[[Regime Detection]] · [[Architecture]] · [[Glossary]]

The strategy emits a *desired* `target_weight ∈ [-1, 1]` ([[Signal Geometry]]).
The **risk sizer** turns that into a concrete `Order | None` given the live
portfolio. Code: `mannofold/signals/risk.py` (`VolTargetRiskSizer`). It is a
**pure function of its inputs** — no hidden state — invoked by the engine each
bar (`mannofold/engine/engine.py`).

## `size(target, portfolio, price, anomaly, volatility)`

```text
scale   = target_vol / (volatility + eps)        # target_vol = 0.01 (per bar)
weight  = clip(target_weight * scale, -max_leverage, +max_leverage)  # max_lev = 1.0
weight *= (1 - anomaly)                           # extra off-manifold de-gross
desired_value = weight * equity
delta_value   = desired_value - current_value     # current_qty * price
... rebalance-band test ... -> Order(side, qty) or None
```

### 1. Volatility targeting

`scale = target_vol / realized_vol` shrinks exposure when the market is volatile
and expands it when calm, so *realized* portfolio vol stays near `target_vol`
regardless of regime. `volatility` is the engine's `realized_vol(closes,
vol_window=20)` over the trailing window (`mannofold/features/indicators.py`).
Exposure is then capped at `max_leverage` (default `1.0` → no leverage).

### 2. Anomaly de-grossing

`weight *= (1 - anomaly)` cuts gross exposure as the state moves off the manifold
([[Mannofold Theory]], [[Regime Detection]]). This is the **second** de-gross:
the strategy already shrank the target by `(1 − anomaly)` in
`ManifoldStrategy.target` ([[Signal Geometry]]); the sizer applies it again on the
vol-scaled weight. At full anomaly (`anomaly → 1`) the target collapses to flat —
the system refuses to bet in a regime it has never learned to price.

### 3. Rebalance band (hysteresis)

To curb churn the sizer only trades when the dollar gap clears a band:

| Condition | Band used |
|-----------|-----------|
| **Adding** same-side risk (`|desired| > |current|`, same sign, holding) | `rebalance_band * hold_band_mult` (wider; `2.0×`) |
| **Trimming** toward flat, or **flipping** sign | `rebalance_band` (tighter base; default `0.02` via `EngineConfig`) |

A wider band when adding keeps a position from being nudged by small target
wobble; the tighter band when reducing means **risk can always come off
promptly**. A `min_trade_frac` (1% of equity) floor suppresses dust orders. If
`|delta_value|` clears neither bar, `size` returns `None` (no order).

This mirrors the strategy's entry/exit hysteresis one layer up ([[Signal
Geometry]]): two independent anti-churn mechanisms, neither of which can leak the
future because both depend only on the current target + current book.

## Order accounting

When an order fires, the engine (`engine.py`) applies it at the bar `close`
(`fill.price = price`), charges `commission = qty * price * commission_bps / 1e4`
(`commission_bps = 1.0`), and updates `cash` / `positions`. The resulting
`PortfolioState` (equity, gross/net exposure, period return, drawdown-from-peak)
is recorded on the `StepResult` — see [[Architecture]] and the metrics in
`mannofold/engine/metrics.py`.
