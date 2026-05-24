"""Short-window sniper strategy: fast-reacting, decisive, few high-quality entries.

Specialist for 3-month (≈60-bar) windows. Minimal smoothing, steep tanh response,
no EMA. Only enters when |sharpe| clears a modest threshold AND confidence is
decent; then commits hard via tanh(5*sharpe)*confidence. Flat on anomaly.
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

NAME = "shortwindow_sniper"
DESCRIPTION = (
    "Short-window specialist: fast, decisive entries via steep tanh(5*sharpe)*confidence. "
    "No EMA smoothing. Acts only when |sharpe| clears a threshold and confidence is decent. "
    "Flat on anomalous regimes or high anomaly scores."
)

_EPS = 1e-9
_GAIN = 5.0           # steep tanh gain — decisive commitment
_SHARPE_THRESHOLD = 0.20  # minimum |sharpe| to open a position
_CONFIDENCE_FLOOR = 0.25  # minimum confidence to trade
_ANOMALY_GATE = 0.60  # anomaly_score above this -> flat
_DEAD_BAND = 0.04     # |weight| below this -> zero, suppress noise


class ShortWindowSniperStrategy:
    """Fast, decisive sniper for short 3-month windows.

    No smoothing: reacts immediately to each bar's manifold state.
    Steep tanh(5*sharpe) combined with confidence gating produces
    few but high-conviction trades.
    """

    def __init__(
        self,
        gain: float = _GAIN,
        sharpe_threshold: float = _SHARPE_THRESHOLD,
        confidence_floor: float = _CONFIDENCE_FLOOR,
        anomaly_gate: float = _ANOMALY_GATE,
        dead_band: float = _DEAD_BAND,
    ):
        self._gain = gain
        self._sharpe_threshold = sharpe_threshold
        self._confidence_floor = confidence_floor
        self._anomaly_gate = anomaly_gate
        self._dead_band = dead_band

    # ------------------------------------------------------------------ #
    #  signals()                                                           #
    # ------------------------------------------------------------------ #
    def signals(self, state: ManifoldState) -> SignalSet:
        sharpe = state.fwd_return_mean / (state.fwd_return_std + _EPS)
        # Steep tanh — no double-tanh, no EMA; raw decisive response
        conviction = math.tanh(self._gain * sharpe)
        # Confidence: regime certainty × non-anomalousness
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

        # Hard gate: high anomaly score -> flat
        if signals.anomaly > self._anomaly_gate:
            return zero

        conviction = signals.expected_return  # tanh(gain*sharpe) from signals()
        confidence = signals.confidence

        # Dual threshold: need both a meaningful sharpe signal and decent confidence
        # Recover sharpe direction from conviction sign using atanh approximation:
        # just use conviction as proxy — if |conviction| is low, |sharpe| was low
        if abs(conviction) < math.tanh(self._gain * self._sharpe_threshold):
            return zero

        if confidence < self._confidence_floor:
            return zero

        # Core weight: decisive conviction scaled by confidence (no smoothing)
        weight = conviction * confidence

        # Dead-band: suppress noise / micro-dithering
        if abs(weight) < self._dead_band:
            return zero

        # Clamp to [-1, 1]
        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=weight)


def build() -> Strategy:
    """Return a fresh ShortWindowSniperStrategy with default parameters."""
    return ShortWindowSniperStrategy()
