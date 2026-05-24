"""Aggressive-smoothed strategy: high-gain trend conviction with heavy EMA smoothing.

Directionally aggressive (gain~4-5) so it commits hard to high-conviction regimes,
but applies a heavy per-symbol EMA (alpha~0.15) to target weights to cut turnover
and whipsaw.  Aggressive in direction, smooth in time.

weight_raw = tanh(gain * tanh(sharpe)) * confidence
ema        = alpha * weight_raw + (1 - alpha) * prev_ema
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

NAME = "aggressive_smoothed"
DESCRIPTION = (
    "High-gain trend conviction (gain~4.5) with heavy per-symbol EMA smoothing "
    "(alpha~0.15) to reduce turnover while committing hard to high-conviction regimes."
)

_EPS = 1e-9
# High gain drives near-binary conviction once Sharpe is meaningful.
_GAIN = 4.5
# Anomaly gate: go flat if anomaly_score exceeds this.
_ANOMALY_GATE = 0.6
# Dead-band: zero out tiny positions to avoid noise trading.
_DEADBAND = 0.04
# Heavy smoothing alpha: low value = slow-moving, low-turnover targets.
_EMA_ALPHA = 0.15  # new_ema = alpha * raw + (1 - alpha) * prev_ema


class AggressiveSmoothedStrategy:
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
        # Per-symbol EMA state for target weight smoothing.
        self._ema: dict[str, float] = {}

    def signals(self, state: ManifoldState) -> SignalSet:
        # Sharpe-like ratio: mean / std (neighbourhood forward-return estimate).
        std = state.fwd_return_std + _EPS
        sharpe = state.fwd_return_mean / std

        # Double-tanh: inner squash Sharpe to (-1,1), outer applies high gain.
        momentum = math.tanh(self._gain * math.tanh(sharpe))

        # Confidence: regime stability attenuated by anomaly proximity.
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

        # Hard gates: anomalous regime or high anomaly score -> flat.
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > self._anomaly_gate:
            # Decay EMA toward zero to avoid stale state on re-entry.
            self._ema[sym] = (1.0 - self._ema_alpha) * self._ema.get(sym, 0.0)
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        # Raw weight: aggressive momentum scaled by confidence.
        raw_weight = signals.momentum * signals.confidence

        # Heavy EMA smoothing to cut turnover and whipsaw.
        prev_ema = self._ema.get(sym, 0.0)
        smoothed = self._ema_alpha * raw_weight + (1.0 - self._ema_alpha) * prev_ema
        self._ema[sym] = smoothed

        # Dead-band: avoid noise trades near zero.
        weight = 0.0 if abs(smoothed) < self._deadband else smoothed

        # Hard clip to [-1, 1].
        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=weight)


def build() -> Strategy:
    return AggressiveSmoothedStrategy()
