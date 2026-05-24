"""Index long-bias strategy — USE-CASE specialist for broad index instruments.

Designed for assets with upward secular drift (e.g. SPY).  The strategy is
structurally long: it holds a small baseline positive tilt at all times, scales
up toward full gross when trend conviction is high, and only de-grosses (rather
than shorting) when conditions deteriorate.  Shorts are heavily capped.

Heavy EMA smoothing across both the raw Sharpe and the final weight ensures
low turnover.  The strategy goes flat on ANOMALY_REGIME or when anomaly_score
exceeds ~0.6.

Weight formula
--------------
  sharpe_ema   <- EMA of (fwd_return_mean / (fwd_return_std + eps))
  conviction   = tanh(GAIN * tanh(sharpe_ema))
  raw_weight   = BASELINE + conviction * (1 - BASELINE)
  confidence   = regime_prob * (1 - anomaly_score)
  pre_weight   = clamp(raw_weight, MIN_WEIGHT, 1.0) * confidence
  weight_ema   <- EMA of pre_weight
  final_weight = 0 if |weight_ema| < DEADBAND else weight_ema
"""

from __future__ import annotations

import math

from mannofold.contracts.interfaces import Strategy
from mannofold.contracts.models import (
    ANOMALY_REGIME,
    ManifoldState,
    SignalSet,
    TargetPosition,
)

NAME = "index_long_bias"
DESCRIPTION = (
    "Structurally long-biased index follower with secular-drift baseline, "
    "tanh conviction scaling, capped shorts (-0.1), and heavy EMA smoothing "
    "for low turnover.  Flat on anomaly or ANOMALY_REGIME."
)

_EPS = 1e-9
_GAIN = 3.0           # tanh gain applied to tanh(sharpe)
_BASELINE = 0.15      # always-on small positive tilt for secular drift
_MIN_WEIGHT = -0.1    # short cap — de-gross rather than short aggressively
_ANOMALY_GATE = 0.6   # anomaly_score threshold -> flat
_DEADBAND = 0.04      # |weight| below this collapses to zero
_SHARPE_EMA_ALPHA = 0.10   # slow EMA on Sharpe (low turnover)
_WEIGHT_EMA_ALPHA = 0.15   # slow EMA on final weight (smooths rebalances)


class IndexLongBiasStrategy:
    """Long-biased index follower with per-symbol EMA state.

    No lookahead: EMA state is updated strictly from past observations.
    """

    def __init__(
        self,
        gain: float = _GAIN,
        baseline: float = _BASELINE,
        min_weight: float = _MIN_WEIGHT,
        anomaly_gate: float = _ANOMALY_GATE,
        deadband: float = _DEADBAND,
        sharpe_alpha: float = _SHARPE_EMA_ALPHA,
        weight_alpha: float = _WEIGHT_EMA_ALPHA,
    ) -> None:
        self._gain = gain
        self._baseline = baseline
        self._min_weight = min_weight
        self._anomaly_gate = anomaly_gate
        self._deadband = deadband
        self._sharpe_alpha = sharpe_alpha
        self._weight_alpha = weight_alpha
        # per-symbol EMA state
        self._sharpe_ema: dict[str, float] = {}
        self._weight_ema: dict[str, float] = {}

    def signals(self, state: ManifoldState) -> SignalSet:
        sharpe = state.fwd_return_mean / (state.fwd_return_std + _EPS)
        # update per-symbol Sharpe EMA (no lookahead — uses current bar only)
        prev = self._sharpe_ema.get(state.symbol, sharpe)
        sharpe_ema = prev + self._sharpe_alpha * (sharpe - prev)
        self._sharpe_ema[state.symbol] = sharpe_ema

        confidence = max(0.0, state.regime_prob * (1.0 - state.anomaly_score))
        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=sharpe_ema,
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        sym = signals.symbol

        # Hard gates: anomalous regime or high anomaly -> flat, reset EMA
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > self._anomaly_gate:
            self._weight_ema[sym] = 0.0
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        # Trend conviction: tanh-shaped, bounded in (-1, 1)
        conviction = math.tanh(self._gain * math.tanh(signals.momentum))

        # Baseline positive tilt + conviction-scaled incremental exposure
        raw_weight = self._baseline + conviction * (1.0 - self._baseline)

        # Clamp: never more than -0.1 short, never more than 1.0 long
        raw_weight = max(self._min_weight, min(1.0, raw_weight))

        # Scale by confidence (regime_prob * (1 - anomaly_score))
        pre_weight = raw_weight * signals.confidence

        # Heavy EMA smoothing for low turnover
        prev_w = self._weight_ema.get(sym, pre_weight)
        weight = prev_w + self._weight_alpha * (pre_weight - prev_w)
        self._weight_ema[sym] = weight

        # Dead-band: suppress negligible weights
        if abs(weight) < self._deadband:
            weight = 0.0

        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=weight)


def build() -> Strategy:
    """Return a fresh IndexLongBiasStrategy instance."""
    return IndexLongBiasStrategy()
