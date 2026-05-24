"""Momentum-velocity strategy: lean into trajectory velocity via neighbourhood Sharpe.

Computes a Sharpe from manifold neighbourhood forward-return stats, maps it through
tanh for momentum, then scales the target weight by regime confidence. Flattens when
the regime is anomalous or anomaly_score is high.
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

NAME = "momentum_velocity"
DESCRIPTION = "Neighbourhood-Sharpe momentum scaled by regime confidence; flat on anomaly."

# Tunable knobs
_GAIN = 2.5          # amplifier inside the outer tanh
_ANOMALY_THRESH = 0.6  # anomaly_score above this -> flat
_DEAD_BAND = 0.04    # collapse |weight| below this to 0


class MomentumVelocityStrategy:
    """Velocity-momentum strategy built on manifold neighbourhood statistics."""

    # ------------------------------------------------------------------ #
    #  signals()                                                          #
    # ------------------------------------------------------------------ #
    def signals(self, state: ManifoldState) -> SignalSet:
        # Neighbourhood Sharpe ratio
        sharpe = state.fwd_return_mean / (state.fwd_return_std + 1e-9)

        # Momentum = tanh of Sharpe (bounded, smooth)
        momentum = math.tanh(sharpe)

        # Confidence = regime certainty × non-anomalousness
        confidence = state.regime_prob * (1.0 - state.anomaly_score)
        # Clamp to [0, 1] for safety
        confidence = max(0.0, min(1.0, confidence))

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
    #  target()                                                           #
    # ------------------------------------------------------------------ #
    def target(self, signals: SignalSet) -> TargetPosition:
        # Flat on anomalous regime or high anomaly score
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_THRESH:
            return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        # Raw weight: tanh of gain-scaled momentum, tempered by confidence
        raw = math.tanh(_GAIN * signals.momentum) * signals.confidence

        # Dead-band: suppress small, noisy weights
        if abs(raw) < _DEAD_BAND:
            raw = 0.0

        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=raw)


def build() -> Strategy:
    """Return a fresh MomentumVelocityStrategy instance."""
    return MomentumVelocityStrategy()
