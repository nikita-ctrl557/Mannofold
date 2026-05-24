"""Confidence-breakout strategy: few but high-conviction trades only.

Stays flat unless BOTH conviction (from neighbourhood Sharpe) AND confidence
(regime stability × non-anomalousness) exceed their respective thresholds.
When the gate opens, the strategy commits near-fully in the signal direction.
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

NAME = "confidence_breakout"
DESCRIPTION = "Stay flat; commit hard only when conviction AND confidence both exceed thresholds."

_EPS = 1e-9
_GAIN = 3.0              # amplifier inside outer tanh(gain * tanh(sharpe))
_ENTRY_THRESHOLD = 0.30  # minimum |conviction| to open a position
_CONFIDENCE_FLOOR = 0.30 # minimum confidence (regime_prob*(1-anomaly)) to trade
_TARGET_MAG = 0.90       # magnitude we aim for when gate clears


class ConfidenceBreakoutStrategy:
    """High-conviction breakout: flat until both gates open, then near-full size."""

    def __init__(
        self,
        gain: float = _GAIN,
        entry_threshold: float = _ENTRY_THRESHOLD,
        confidence_floor: float = _CONFIDENCE_FLOOR,
        target_mag: float = _TARGET_MAG,
    ):
        self._gain = gain
        self._entry_threshold = entry_threshold
        self._confidence_floor = confidence_floor
        self._target_mag = target_mag

    # ------------------------------------------------------------------ #
    #  signals()                                                          #
    # ------------------------------------------------------------------ #
    def signals(self, state: ManifoldState) -> SignalSet:
        sharpe = state.fwd_return_mean / (state.fwd_return_std + _EPS)
        momentum = math.tanh(sharpe)
        # conviction: double-tanh amplification of Sharpe, bounded to [-1, 1]
        conviction = math.tanh(self._gain * math.tanh(sharpe))
        # confidence: regime certainty × non-anomalousness
        confidence = max(0.0, min(1.0, state.regime_prob * (1.0 - state.anomaly_score)))
        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=momentum,
            expected_return=conviction,   # reuse field to carry conviction into target()
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    # ------------------------------------------------------------------ #
    #  target()                                                           #
    # ------------------------------------------------------------------ #
    def target(self, signals: SignalSet) -> TargetPosition:
        zero = TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        # Hard gate: anomalous regime -> always flat
        if signals.regime_id == ANOMALY_REGIME:
            return zero

        conviction = signals.expected_return  # carried from signals()
        confidence = signals.confidence

        # Selective gate: both conviction magnitude AND confidence must clear thresholds
        if abs(conviction) < self._entry_threshold or confidence < self._confidence_floor:
            return zero

        # Direction comes from conviction sign; size toward target_mag
        direction = 1.0 if conviction > 0 else -1.0
        weight = direction * self._target_mag * min(1.0, confidence)
        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=weight)


def build() -> Strategy:
    """Return a fresh ConfidenceBreakoutStrategy with default parameters."""
    return ConfidenceBreakoutStrategy()
