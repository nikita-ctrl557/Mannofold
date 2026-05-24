"""Vol-scaled carry strategy variant.

Harvests neighbourhood drift (carry), sizing INVERSELY to dispersion:
prefer states where the forward-return distribution is tight (low fwd_return_std)
and the mean is positive/clear.

Raw weight: fwd_return_mean / (fwd_return_std**2 + eps) — inverse-variance /
Kelly-like sizing. Squashed via tanh and multiplied by confidence.
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

NAME = "vol_scaled_carry"
DESCRIPTION = "Inverse-variance carry: harvest neighbourhood drift sized by 1/variance with tanh squash."

_EPS = 1e-9
# Clamp on weight_raw to avoid tanh saturation / blowups when std is tiny.
_RAW_CLAMP = 10.0
# tanh gain applied to weight_raw before confidence scaling.
_GAIN = 3.0
# Anomaly score threshold above which the strategy goes flat.
_ANOMALY_GATE = 0.6
# Dead-band: positions smaller than this are zeroed out.
_DEADBAND = 0.04


class VolScaledCarryStrategy:
    def __init__(
        self,
        gain: float = _GAIN,
        anomaly_gate: float = _ANOMALY_GATE,
        deadband: float = _DEADBAND,
    ):
        self._gain = gain
        self._anomaly_gate = anomaly_gate
        self._deadband = deadband

    def signals(self, state: ManifoldState) -> SignalSet:
        # Inverse-variance sizing: high mean + low variance -> strong signal.
        variance = state.fwd_return_std ** 2 + _EPS
        weight_raw = state.fwd_return_mean / variance
        # Clamp defensively before tanh to avoid extreme values.
        weight_raw = max(-_RAW_CLAMP, min(_RAW_CLAMP, weight_raw))
        momentum = math.tanh(self._gain * weight_raw)

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
        # Hard gates: anomalous regime or high anomaly score -> flat.
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > self._anomaly_gate:
            return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        # momentum already holds tanh(gain * weight_raw); scale by confidence.
        weight = signals.momentum * signals.confidence

        # Dead-band: avoid noise trading near zero.
        if abs(weight) < self._deadband:
            weight = 0.0

        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=weight)


def build() -> Strategy:
    return VolScaledCarryStrategy()
