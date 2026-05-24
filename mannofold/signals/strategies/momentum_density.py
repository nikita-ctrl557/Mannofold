"""Momentum-density strategy: streak-based conviction gated by manifold density.

Combines momentum_persistence (streak-scaled trend) with the density typicality
gate from density_gated. Persistent trends are only sized up in TYPICAL manifold
states (high density). Flat on ANOMALY_REGIME or high anomaly scores.
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

NAME = "momentum_density"
DESCRIPTION = (
    "Streak-based momentum conviction multiplied by a density typicality gate "
    "(logistic of manifold density) so persistent trends are only sized up in "
    "typical states; flat on anomaly."
)

# Tunable knobs
_STREAK_SCALE = 6.0      # k: streak length at which tanh saturates to ~0.76
_ANOMALY_THRESH = 0.6    # anomaly_score above this -> flat immediately
_DEAD_BAND = 0.04        # collapse |weight| below this to 0

# Density gate logistic parameters (from density_gated)
_DENSITY_MID = 1.0       # density at which gate = 0.5
_DENSITY_SCALE = 2.0     # steepness of sigmoid
_DENSITY_CLAMP = 50.0    # defensive upper clamp


def _density_gate(density: float) -> float:
    """Smooth gate in [0, 1]: low density -> ~0, high density -> ~1."""
    d = max(0.0, min(density, _DENSITY_CLAMP))
    return 1.0 / (1.0 + math.exp(-_DENSITY_SCALE * (d - _DENSITY_MID)))


class MomentumDensityStrategy:
    """Per-symbol streak momentum gated by manifold density typicality."""

    def __init__(self) -> None:
        # per-symbol: signed streak counter (positive=bullish, negative=bearish)
        self._streak: dict[str, int] = {}

    def signals(self, state: ManifoldState) -> SignalSet:
        sym = state.symbol

        # Update per-symbol streak based on sign of fwd_return_mean
        drift = state.fwd_return_mean
        prev_streak = self._streak.get(sym, 0)

        if drift > 0:
            new_streak = prev_streak + 1 if prev_streak > 0 else 1
        elif drift < 0:
            new_streak = prev_streak - 1 if prev_streak < 0 else -1
        else:
            new_streak = 0

        self._streak[sym] = new_streak

        # Streak conviction: saturating tanh scaled by streak length
        streak_mag = math.tanh(abs(new_streak) / _STREAK_SCALE)
        direction = 1.0 if new_streak > 0 else (-1.0 if new_streak < 0 else 0.0)

        # Density gate: logistic gate so only typical manifold states get full size
        gate = _density_gate(state.density)

        # Conviction = direction * tanh(streak/k) * density_gate
        conviction = direction * streak_mag * gate

        # Confidence: regime certainty × non-anomalousness
        confidence = state.regime_prob * (1.0 - state.anomaly_score)
        confidence = max(0.0, min(1.0, confidence))

        return SignalSet(
            ts=state.ts,
            symbol=sym,
            momentum=conviction,
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        # Hard flat on anomalous regime or high anomaly score
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_THRESH:
            return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        # Raw weight: conviction (momentum) * confidence
        raw = signals.momentum * signals.confidence

        # Dead-band: suppress small, noisy weights
        if abs(raw) < _DEAD_BAND:
            raw = 0.0

        raw = max(-1.0, min(1.0, raw))
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=raw)


def build() -> Strategy:
    """Return a fresh MomentumDensityStrategy instance."""
    return MomentumDensityStrategy()
