"""Trend vol-target strategy.

Follow the drift DIRECTION but scale position size to a constant risk target:
  direction  = sign(fwd_return_mean)
  magnitude  = tanh(gain * target_risk / (fwd_return_std + eps))
  raw_weight = direction * magnitude * confidence

Volatile neighbourhoods receive smaller size; calm ones receive larger size,
implementing vol-targeting at the signal level.  Flat when anomalous.
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

NAME = "trend_vol_target"
DESCRIPTION = (
    "Follow drift direction sized to a constant risk target via "
    "tanh(gain * target_risk / vol); flat on anomaly."
)

# Tunable knobs
_TARGET_RISK = 0.02       # desired risk per unit — scales the tanh argument
_GAIN = 2.0               # amplifier inside tanh
_EPS = 1e-9               # numerical floor for std
_ANOMALY_THRESH = 0.6     # anomaly_score above this -> flat
_DEAD_BAND = 0.04         # collapse |weight| below this to 0


class TrendVolTargetStrategy:
    """Vol-targeting trend strategy built on manifold neighbourhood statistics."""

    def __init__(
        self,
        target_risk: float = _TARGET_RISK,
        gain: float = _GAIN,
        anomaly_thresh: float = _ANOMALY_THRESH,
        dead_band: float = _DEAD_BAND,
    ) -> None:
        self._target_risk = target_risk
        self._gain = gain
        self._anomaly_thresh = anomaly_thresh
        self._dead_band = dead_band

    # ------------------------------------------------------------------ #
    #  signals()                                                           #
    # ------------------------------------------------------------------ #
    def signals(self, state: ManifoldState) -> SignalSet:
        # Direction: sign of neighbourhood drift
        direction = math.copysign(1.0, state.fwd_return_mean) if state.fwd_return_mean != 0.0 else 0.0

        # Magnitude: tanh(gain * target_risk / vol) — vol-targeting at signal level
        magnitude = math.tanh(self._gain * self._target_risk / (state.fwd_return_std + _EPS))

        # Combine into a momentum-like scalar: direction * magnitude
        momentum = direction * magnitude

        # Confidence: regime certainty × non-anomalousness
        confidence = max(0.0, min(1.0, state.regime_prob * (1.0 - state.anomaly_score)))

        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=momentum,
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    # ------------------------------------------------------------------ #
    #  target()                                                            #
    # ------------------------------------------------------------------ #
    def target(self, signals: SignalSet) -> TargetPosition:
        # Hard gates: anomalous regime or high anomaly score -> flat
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > self._anomaly_thresh:
            return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        # Raw weight: vol-targeted direction scaled by confidence
        raw = signals.momentum * signals.confidence

        # Dead-band: suppress small, noisy weights
        if abs(raw) < self._dead_band:
            raw = 0.0

        # Clamp to [-1, 1]
        raw = max(-1.0, min(1.0, raw))

        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=raw)


def build() -> Strategy:
    """Return a fresh TrendVolTargetStrategy instance."""
    return TrendVolTargetStrategy()
