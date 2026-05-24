"""Confidence-floor steady strategy: fixed modest size above a confidence floor.

Only acts when confidence = regime_prob * (1 - anomaly_score) exceeds a floor
(~0.45). Below the floor the strategy is flat, so monthly returns stay steady
rather than chasing marginal or noisy signals. Position size is intentionally
modest and fixed to avoid large bets.
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

NAME = "confidence_floor_steady"
DESCRIPTION = (
    "Month-over-month consistency: flat below a confidence floor (~0.45), "
    "fixed modest size (0.4 * sign) above it — avoids large bets so returns "
    "stay steady. weight = sign(sharpe)*0.4*tanh(gain*|tanh(sharpe)|)."
)

_EPS = 1e-9
_CONFIDENCE_FLOOR = 0.45   # minimum confidence to enter a position
_FIXED_SIZE = 0.40         # cap on position weight magnitude
_GAIN = 3.0                # inner amplifier: tanh(gain * |tanh(sharpe)|)
_ANOMALY_CUTOFF = 0.60     # flat when anomaly_score exceeds this
_DEAD_BAND = 0.04          # |weight| below this -> zero (avoid rounding noise)


class ConfidenceFloorSteadyStrategy:
    """Fixed-size strategy that only trades above a confidence floor."""

    def __init__(
        self,
        confidence_floor: float = _CONFIDENCE_FLOOR,
        fixed_size: float = _FIXED_SIZE,
        gain: float = _GAIN,
        anomaly_cutoff: float = _ANOMALY_CUTOFF,
        dead_band: float = _DEAD_BAND,
    ) -> None:
        self._floor = confidence_floor
        self._size = fixed_size
        self._gain = gain
        self._anomaly_cutoff = anomaly_cutoff
        self._dead_band = dead_band

    # ------------------------------------------------------------------ #
    #  signals()                                                           #
    # ------------------------------------------------------------------ #
    def signals(self, state: ManifoldState) -> SignalSet:
        sharpe = state.fwd_return_mean / (state.fwd_return_std + _EPS)
        momentum = math.tanh(sharpe)
        # Conviction: direction-preserving, magnitude from double-tanh
        conviction = math.tanh(self._gain * abs(math.tanh(sharpe)))
        # Confidence fuses regime stability with inverse anomaly score
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
    #  target()                                                            #
    # ------------------------------------------------------------------ #
    def target(self, signals: SignalSet) -> TargetPosition:
        zero = TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        # Hard gates: anomalous regime or elevated anomaly score -> flat
        if signals.regime_id == ANOMALY_REGIME:
            return zero
        if signals.anomaly > self._anomaly_cutoff:
            return zero

        # Confidence floor: do nothing in uncertain / low-probability regimes
        if signals.confidence < self._floor:
            return zero

        # Direction from sign of momentum (tanh(sharpe)), size is fixed-modest
        direction = 1.0 if signals.momentum >= 0.0 else -1.0
        # weight = sign(sharpe) * fixed_size * tanh(gain * |tanh(sharpe)|)
        weight = direction * self._size * signals.expected_return

        # Dead-band filter: avoid nano-positions
        if abs(weight) < self._dead_band:
            return zero

        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=weight)


def build() -> Strategy:
    """Return a fresh ConfidenceFloorSteadyStrategy with default parameters."""
    return ConfidenceFloorSteadyStrategy()
