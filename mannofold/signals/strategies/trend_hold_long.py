"""Trend-hold-long strategy: persistent long exposure in established uptrends.

Identifies an established up-trend (positive drift sustained) and HOLDS a long
position through it with heavy per-symbol EMA smoothing (low turnover). Stays
invested to capture the many up-bars in a trending market. Reduces/exits only
when drift clearly turns negative.

weight_raw = clamp(tanh(gain * tanh(sharpe)), -0.15, 1.0)
smoothed   = alpha * weight_raw + (1 - alpha) * prev_ema
target     = smoothed * confidence
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

NAME = "trend_hold_long"
DESCRIPTION = (
    "Persistent long exposure in established uptrends. Heavy EMA smoothing "
    "(alpha~0.10) for low turnover; long-biased (shorts capped at -0.15). "
    "Flat on anomalous regimes or high anomaly score."
)

_EPS = 1e-9
# High gain so moderate positive Sharpe -> near-full commitment.
_GAIN = 80.0
# Anomaly gate: go flat if anomaly_score exceeds this.
_ANOMALY_GATE = 0.6
# Dead-band: zero out tiny positions to avoid noise trading.
_DEADBAND = 0.04
# Heavy smoothing alpha: very slow-moving to stay invested through uptrend.
_EMA_ALPHA = 0.10
# Long bias: cap short exposure at this magnitude.
_MAX_SHORT = -0.15


class TrendHoldLongStrategy:
    """Hold longs through established uptrends with heavy EMA smoothing.

    Signal: sharpe = fwd_return_mean / (fwd_return_std + eps)
    Raw weight = clamp(tanh(gain * tanh(sharpe)), _MAX_SHORT, 1.0)
    EMA-smoothed then multiplied by confidence = regime_prob * (1 - anomaly_score).
    """

    def __init__(
        self,
        gain: float = _GAIN,
        anomaly_gate: float = _ANOMALY_GATE,
        deadband: float = _DEADBAND,
        ema_alpha: float = _EMA_ALPHA,
        max_short: float = _MAX_SHORT,
    ):
        self._gain = gain
        self._anomaly_gate = anomaly_gate
        self._deadband = deadband
        self._ema_alpha = ema_alpha
        self._max_short = max_short
        # Per-symbol EMA state for raw weight smoothing.
        self._ema: dict[str, float] = {}

    def signals(self, state: ManifoldState) -> SignalSet:
        std = state.fwd_return_std + _EPS
        sharpe = state.fwd_return_mean / std
        momentum = math.tanh(self._gain * math.tanh(sharpe))
        confidence = max(0.0, state.regime_prob * (1.0 - state.anomaly_score))
        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=momentum,
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        sym = signals.symbol

        # Hard gates: anomalous regime or high anomaly score -> flat, decay EMA.
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > self._anomaly_gate:
            self._ema[sym] = (1.0 - self._ema_alpha) * self._ema.get(sym, 0.0)
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        # Raw weight with long bias applied before smoothing.
        raw = max(self._max_short, min(1.0, signals.momentum))

        # Heavy EMA smoothing: very slow to respond -> stays invested in uptrend.
        prev = self._ema.get(sym, 0.0)
        smoothed = self._ema_alpha * raw + (1.0 - self._ema_alpha) * prev
        self._ema[sym] = smoothed

        # Scale by confidence.
        weight = smoothed * signals.confidence

        # Dead-band: avoid noise trades near zero.
        if abs(weight) < self._deadband:
            weight = 0.0

        # Final hard clip.
        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=weight)


def build() -> Strategy:
    return TrendHoldLongStrategy()
