"""Long-bias trend-following strategy variant.

Trend-follows the neighbourhood drift (Sharpe-based) but is LONG-BIASED:
allows full long exposure while capping short exposure to a small magnitude.
Suited to upward-drifting markets.
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

NAME = "long_bias_trend"
DESCRIPTION = "Trend-follow neighbourhood drift, long-biased: full long, capped short exposure."

_EPS = 1e-9
_GAIN = 60.0          # tanh gain applied to Sharpe proxy
_ANOMALY_GATE = 0.6   # anomaly_score threshold above which we go flat
_DEADBAND = 0.04      # |weight| below this collapses to zero
_MAX_SHORT = -0.2     # long-bias cap: short exposure bounded to this magnitude


class LongBiasTrendStrategy:
    """Trend-follow the manifold drift with a long bias.

    Signal construction:
      sharpe  = fwd_return_mean / (fwd_return_std + eps)
      confidence = regime_prob * (1 - anomaly_score)
      base    = tanh(gain * tanh(sharpe)) * confidence

    Long bias: weights < _MAX_SHORT are clamped to _MAX_SHORT,
    so the strategy runs asymmetric: up to +1 long, capped at -0.2 short.
    """

    def __init__(
        self,
        gain: float = _GAIN,
        anomaly_gate: float = _ANOMALY_GATE,
        deadband: float = _DEADBAND,
        max_short: float = _MAX_SHORT,
    ):
        self._gain = gain
        self._anomaly_gate = anomaly_gate
        self._deadband = deadband
        self._max_short = max_short  # negative value, e.g. -0.2

    def signals(self, state: ManifoldState) -> SignalSet:
        sharpe = state.fwd_return_mean / (state.fwd_return_std + _EPS)
        confidence = max(0.0, state.regime_prob * (1.0 - state.anomaly_score))
        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=math.tanh(sharpe),
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        # Hard gates: anomalous regime or high anomaly score -> flat.
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > self._anomaly_gate:
            return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        sharpe = signals.expected_return / (abs(signals.momentum) + _EPS)
        base = math.tanh(self._gain * math.tanh(sharpe)) * signals.confidence

        # Dead-band: collapse negligible weights to flat.
        if abs(base) < self._deadband:
            base = 0.0

        # Clamp to [-1, 1] then apply long bias (floor short exposure).
        weight = max(-1.0, min(1.0, base))
        weight = max(self._max_short, weight)  # long-bias: no more than 20% short

        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=weight)


def build() -> Strategy:
    return LongBiasTrendStrategy()
