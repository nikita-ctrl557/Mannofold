"""Decisive-breakout strategy: near-binary conviction via steep tanh.

Uses a high-gain tanh (gain=6) on neighbourhood Sharpe so once |sharpe| clears
a small threshold the engine commits to near-full long or short.  Tiny signals
stay flat via a dead-band.  Confidence gates the final weight.
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

NAME = "decisive_breakout"
DESCRIPTION = (
    "Near-binary conviction: steep tanh(6*sharpe) commits near-fully once |sharpe| "
    "clears a small threshold; tiny signals stay flat via dead-band."
)

_EPS = 1e-9
_GAIN = 6.0           # steep tanh amplifier — decisive commitment
_DEAD_BAND = 0.04     # |weight| below this -> flat, no dithering
_ANOMALY_GATE = 0.6   # anomaly_score above this -> flat


class DecisiveBreakoutStrategy:
    """Steep-tanh breakout: decisive directional bets, flat otherwise."""

    def __init__(
        self,
        gain: float = _GAIN,
        dead_band: float = _DEAD_BAND,
        anomaly_gate: float = _ANOMALY_GATE,
    ):
        self._gain = gain
        self._dead_band = dead_band
        self._anomaly_gate = anomaly_gate

    # ------------------------------------------------------------------ #
    #  signals()                                                           #
    # ------------------------------------------------------------------ #
    def signals(self, state: ManifoldState) -> SignalSet:
        sharpe = state.fwd_return_mean / (state.fwd_return_std + _EPS)
        # Steep tanh — once |sharpe| clears ~0.2 we get near-binary output
        conviction = math.tanh(self._gain * sharpe)
        confidence = max(0.0, min(1.0, state.regime_prob * (1.0 - state.anomaly_score)))
        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=math.tanh(sharpe),
            expected_return=conviction,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    # ------------------------------------------------------------------ #
    #  target()                                                            #
    # ------------------------------------------------------------------ #
    def target(self, signals: SignalSet) -> TargetPosition:
        zero = TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        # Hard gate: anomalous regime -> flat
        if signals.regime_id == ANOMALY_REGIME:
            return zero

        # Hard gate: anomaly score too high -> flat
        if signals.anomaly > self._anomaly_gate:
            return zero

        conviction = signals.expected_return  # tanh(gain*sharpe) from signals()
        confidence = signals.confidence

        # Core weight: decisive conviction scaled by confidence
        weight = conviction * confidence

        # Dead-band: suppress noise / dithering
        if abs(weight) < self._dead_band:
            return zero

        # Clamp to [-1, 1]
        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=weight)


def build() -> Strategy:
    """Return a fresh DecisiveBreakoutStrategy with default parameters."""
    return DecisiveBreakoutStrategy()
