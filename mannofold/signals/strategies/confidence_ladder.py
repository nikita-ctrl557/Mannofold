"""Confidence-ladder strategy: laddered exposure by regime confidence.

Direction from sign(tanh(sharpe)); magnitude stepped by confidence tiers
(confidence = regime_prob * (1 - anomaly_score)), then scaled by a
double-tanh amplification of the Sharpe signal.
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

NAME = "confidence_ladder"
DESCRIPTION = (
    "Laddered exposure by confidence (regime_prob*(1-anomaly_score)): "
    "four discrete position-size steps scaled by double-tanh Sharpe conviction."
)

_EPS = 1e-9
_GAIN = 3.0               # amplifier inside outer tanh(gain * tanh(sharpe))
_ANOMALY_CUTOFF = 0.60    # flat when anomaly_score > this
_DEAD_BAND = 0.04         # |weight| below this -> zero


def _confidence_step(confidence: float) -> float:
    """Map confidence to a discrete ladder rung: 0, 0.33, 0.66, or 1.0."""
    if confidence < 0.30:
        return 0.0
    if confidence < 0.60:
        return 0.33
    if confidence < 0.85:
        return 0.66
    return 1.0


class ConfidenceLadderStrategy:
    """Laddered position sizing driven by regime confidence and Sharpe direction."""

    def __init__(self, gain: float = _GAIN):
        self._gain = gain

    # ------------------------------------------------------------------ #
    #  signals()                                                          #
    # ------------------------------------------------------------------ #
    def signals(self, state: ManifoldState) -> SignalSet:
        sharpe = state.fwd_return_mean / (state.fwd_return_std + _EPS)
        momentum = math.tanh(sharpe)
        conviction = math.tanh(self._gain * math.tanh(sharpe))
        confidence = max(0.0, min(1.0, state.regime_prob * (1.0 - state.anomaly_score)))
        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=momentum,
            expected_return=conviction,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    # ------------------------------------------------------------------ #
    #  target()                                                           #
    # ------------------------------------------------------------------ #
    def target(self, signals: SignalSet) -> TargetPosition:
        zero = TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        # Hard gates: anomalous regime or high anomaly score -> flat
        if signals.regime_id == ANOMALY_REGIME:
            return zero
        if signals.anomaly > _ANOMALY_CUTOFF:
            return zero

        conviction = signals.expected_return  # carried from signals()
        confidence = signals.confidence

        # Ladder step: zero step -> flat
        step = _confidence_step(confidence)
        if step == 0.0:
            return zero

        # Direction from sign of conviction (tanh(sharpe))
        direction = 1.0 if conviction >= 0.0 else -1.0

        # Final weight: direction * step * |tanh(gain * tanh(sharpe))|
        weight = direction * step * abs(conviction)

        # Dead-band filter
        if abs(weight) < _DEAD_BAND:
            return zero

        # Clamp to [-1, 1]
        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=weight)


def build() -> Strategy:
    """Return a fresh ConfidenceLadderStrategy with default parameters."""
    return ConfidenceLadderStrategy()
