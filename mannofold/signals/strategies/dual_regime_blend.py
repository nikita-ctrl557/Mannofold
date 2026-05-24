"""Dual-regime blend strategy: soft regime assignment via continuous blending.

When regime_prob is HIGH (confident assignment) -> trend-follow (+tanh(sharpe)).
When regime_prob is LOW (ambiguous, between regimes) -> mean-revert (-tanh(sharpe)).
Blend continuously: mix = 2*regime_prob - 1 in [-1, 1].
dir_signal = mix * tanh(sharpe).
target_weight = tanh(gain * dir_signal) * (1 - anomaly_score).
Flat on ANOMALY_REGIME or anomaly > 0.6; dead-band |w| < 0.04 -> 0.
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

NAME = "dual_regime_blend"
DESCRIPTION = (
    "Soft regime blend: trend-follow when regime_prob is high (confident), "
    "mean-revert when regime_prob is low (ambiguous); blended continuously via "
    "mix=2*regime_prob-1; flat on anomaly regime or anomaly>0.6."
)

_GAIN = 2.5
_ANOMALY_THRESH = 0.6
_DEAD_BAND = 0.04
_EPS = 1e-9


class DualRegimeBlendStrategy:
    """Continuously blends trend-following and mean-reversion by regime confidence.

    Core math:
      sharpe     = fwd_return_mean / (fwd_return_std + eps)
      mix        = 2 * regime_prob - 1        # in [-1, 1]
                   +1 -> fully trend-following (confident regime)
                   -1 -> fully mean-reverting  (ambiguous, between regimes)
      dir_signal = mix * tanh(sharpe)
      raw_weight = tanh(gain * dir_signal) * (1 - anomaly_score)

    Gates:
      - Flat on ANOMALY_REGIME or anomaly_score > _ANOMALY_THRESH.
      - Dead-band: |raw_weight| < _DEAD_BAND -> 0.
    """

    def signals(self, state: ManifoldState) -> SignalSet:
        sharpe = state.fwd_return_mean / (state.fwd_return_std + _EPS)

        # mix in [-1, 1]: +1 = fully trend, -1 = fully mean-revert
        mix = 2.0 * state.regime_prob - 1.0

        # Direction signal: blend of trend-follow and mean-reversion
        dir_signal = mix * math.tanh(sharpe)

        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=dir_signal,
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=state.regime_prob,
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        # Hard gates: anomalous regime or high anomaly score -> flat
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_THRESH:
            return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        # Scale by (1 - anomaly_score) to reduce exposure near anomaly threshold
        anomaly_scale = 1.0 - signals.anomaly

        raw = math.tanh(_GAIN * signals.momentum) * anomaly_scale

        # Dead-band: suppress small noisy weights
        if abs(raw) < _DEAD_BAND:
            raw = 0.0

        raw = max(-1.0, min(1.0, raw))
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=raw)


def build() -> Strategy:
    """Return a fresh DualRegimeBlendStrategy instance."""
    return DualRegimeBlendStrategy()
