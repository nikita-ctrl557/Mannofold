"""Momentum-persistence strategy: size by length of consecutive same-sign drift.

Tracks a per-symbol streak counter of bars where fwd_return_mean has the same
sign. Conviction grows with a saturating tanh(streak/k) so a persistent trend
earns progressively larger size while a sign flip resets the counter immediately.
Flattens on anomalous regimes or high anomaly scores.
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

NAME = "momentum_persistence"
DESCRIPTION = (
    "Per-symbol streak-based sizing: conviction = tanh(streak/k) * confidence; "
    "flat on anomaly."
)

# Tunable knobs
_STREAK_SCALE = 6.0      # k: streak length at which tanh saturates to ~0.76
_ANOMALY_THRESH = 0.6    # anomaly_score above this -> flat immediately
_DEAD_BAND = 0.04        # collapse |weight| below this to 0


class MomentumPersistenceStrategy:
    """Streak-based momentum persistence with per-symbol state."""

    def __init__(self) -> None:
        # per-symbol: current streak count (positive = bullish streak, negative = bearish)
        self._streak: dict[str, int] = {}

    # ------------------------------------------------------------------ #
    #  signals()                                                          #
    # ------------------------------------------------------------------ #
    def signals(self, state: ManifoldState) -> SignalSet:
        sym = state.symbol

        # Update streak counter based on sign of fwd_return_mean
        drift = state.fwd_return_mean
        prev_streak = self._streak.get(sym, 0)

        if drift > 0:
            # Bullish bar: increment if already positive, else reset to +1
            new_streak = prev_streak + 1 if prev_streak > 0 else 1
        elif drift < 0:
            # Bearish bar: decrement if already negative, else reset to -1
            new_streak = prev_streak - 1 if prev_streak < 0 else -1
        else:
            # Exactly zero drift: reset streak
            new_streak = 0

        self._streak[sym] = new_streak

        # Conviction from streak length via saturating tanh
        streak_conviction = math.tanh(abs(new_streak) / _STREAK_SCALE)
        # Direction from streak sign
        direction = 1.0 if new_streak > 0 else (-1.0 if new_streak < 0 else 0.0)
        # Store signed momentum for target()
        momentum = direction * streak_conviction

        # Confidence: regime certainty × non-anomalousness
        confidence = state.regime_prob * (1.0 - state.anomaly_score)
        confidence = max(0.0, min(1.0, confidence))

        return SignalSet(
            ts=state.ts,
            symbol=sym,
            momentum=momentum,
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    # ------------------------------------------------------------------ #
    #  target()                                                           #
    # ------------------------------------------------------------------ #
    def target(self, signals: SignalSet) -> TargetPosition:
        # Flat unconditionally on anomalous regime or high anomaly score
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_THRESH:
            return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        # Raw weight: streak conviction (already in signals.momentum) * confidence
        raw = signals.momentum * signals.confidence

        # Dead-band: suppress small, noisy weights
        if abs(raw) < _DEAD_BAND:
            raw = 0.0

        # Clamp to [-1, 1]
        raw = max(-1.0, min(1.0, raw))

        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=raw)


def build() -> Strategy:
    """Return a fresh MomentumPersistenceStrategy instance."""
    return MomentumPersistenceStrategy()
