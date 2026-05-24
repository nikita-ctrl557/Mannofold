"""Long-horizon compounder strategy: low-turnover, persistent, compounding.

Designed for 10y/15y horizons where minimising friction and maximising
holding period is paramount. Uses a very heavy per-symbol EMA (alpha~0.1)
and wide hysteresis dead-band to hold positions through noise and capture
multi-year compounding.

Conviction = tanh(gain * tanh(sharpe)) * confidence
confidence = regime_prob * (1 - anomaly_score)
smoothed   = alpha * raw + (1 - alpha) * prev_ema   (alpha ~ 0.1)
dead-band  : |smoothed| < 0.04 -> 0
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

NAME = "longhorizon_compounder"
DESCRIPTION = (
    "Low-turnover trend compounder for 10y/15y horizons: wide hysteresis dead-band, "
    "very heavy per-symbol EMA smoothing (alpha~0.1), conviction = "
    "tanh(gain*tanh(sharpe)) * regime_prob*(1-anomaly_score). "
    "Flat on ANOMALY_REGIME or anomaly>0.6; dead-band |w|<0.04->0."
)

_EPS = 1e-9
# Moderate gain — enough conviction without over-trading at low Sharpe.
_GAIN = 3.0
# Hard anomaly gate: go flat above this score.
_ANOMALY_GATE = 0.6
# Wide dead-band: kills small noisy weights; reduces turnover significantly.
_DEADBAND = 0.04
# Very heavy smoothing — slow-moving, persistent positions.
_EMA_ALPHA = 0.10  # new_ema = alpha * raw + (1 - alpha) * prev_ema


class LongHorizonCompounder:
    """Low-turnover trend compounder optimised for multi-decade compounding."""

    def __init__(
        self,
        gain: float = _GAIN,
        anomaly_gate: float = _ANOMALY_GATE,
        deadband: float = _DEADBAND,
        ema_alpha: float = _EMA_ALPHA,
    ):
        self._gain = gain
        self._anomaly_gate = anomaly_gate
        self._deadband = deadband
        self._ema_alpha = ema_alpha
        # Per-symbol EMA state — persists across steps, no lookahead.
        self._ema: dict[str, float] = {}

    def signals(self, state: ManifoldState) -> SignalSet:
        std = state.fwd_return_std + _EPS
        sharpe = state.fwd_return_mean / std

        # Double-tanh conviction: inner squash Sharpe, outer applies gain.
        momentum = math.tanh(self._gain * math.tanh(sharpe))

        # Confidence: regime stability discounted by anomaly proximity.
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

        # Hard gates: anomalous regime or high anomaly -> go flat, decay EMA.
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > self._anomaly_gate:
            self._ema[sym] = (1.0 - self._ema_alpha) * self._ema.get(sym, 0.0)
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        # Raw weight: conviction scaled by confidence.
        raw_weight = signals.momentum * signals.confidence

        # Very heavy EMA smoothing — positions change slowly, reducing turnover.
        prev_ema = self._ema.get(sym, 0.0)
        smoothed = self._ema_alpha * raw_weight + (1.0 - self._ema_alpha) * prev_ema
        self._ema[sym] = smoothed

        # Wide dead-band: flat near zero to avoid noise trades.
        weight = 0.0 if abs(smoothed) < self._deadband else smoothed

        # Hard clip to [-1, 1].
        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=weight)


def build() -> Strategy:
    return LongHorizonCompounder()
