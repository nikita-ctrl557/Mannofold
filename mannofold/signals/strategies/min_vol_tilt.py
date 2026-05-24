"""Minimum-volatility tilt strategy.

Prefer LOW-dispersion manifold states: size grows as neighbourhood
forward-return std SHRINKS.  Direction is sign(fwd_return_mean); magnitude
is tanh(gain / (k * fwd_return_std + eps)), so tighter distributions produce
larger conviction.  Gated by confidence = regime_prob * (1 - anomaly_score).
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

NAME = "min_vol_tilt"
DESCRIPTION = (
    "Min-vol tilt: go in the direction of mean return sized INVERSELY to "
    "forward-return std via tanh(gain / (k*std + eps)), gated by confidence."
)

_EPS = 1e-9
# Gain applied inside tanh: controls how fast conviction saturates.
_GAIN = 1.0
# Softness around std=0: prevents explosive magnitude when std is tiny.
_K = 50.0
# Clamp on the raw tanh argument to guard against edge-case near-zero std.
_ARG_CLAMP = 8.0
# Anomaly score above which we go flat.
_ANOMALY_GATE = 0.6
# Dead-band: positions smaller than this are zeroed out.
_DEADBAND = 0.04


class MinVolTiltStrategy:
    def __init__(
        self,
        gain: float = _GAIN,
        k: float = _K,
        anomaly_gate: float = _ANOMALY_GATE,
        deadband: float = _DEADBAND,
    ):
        self._gain = gain
        self._k = k
        self._anomaly_gate = anomaly_gate
        self._deadband = deadband

    def signals(self, state: ManifoldState) -> SignalSet:
        # Direction: sign of expected neighbourhood return.
        direction = math.copysign(1.0, state.fwd_return_mean) if state.fwd_return_mean != 0.0 else 0.0

        # Magnitude: tanh(gain / (k*std + eps)) — bigger when std is small.
        raw_arg = self._gain / (self._k * state.fwd_return_std + _EPS)
        raw_arg = max(-_ARG_CLAMP, min(_ARG_CLAMP, raw_arg))
        magnitude = math.tanh(raw_arg)

        momentum = direction * magnitude

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
        # Hard gates: anomalous regime or elevated anomaly score -> flat.
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > self._anomaly_gate:
            return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        weight = signals.momentum * signals.confidence

        # Dead-band: avoid noise trading near zero.
        if abs(weight) < self._deadband:
            weight = 0.0

        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=weight)


def build() -> Strategy:
    return MinVolTiltStrategy()
