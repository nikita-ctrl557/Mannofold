"""Dual-signal vote strategy: take a position only when drift and carry AGREE.

Two sub-signals derived from manifold forward-return statistics:
  drift  = tanh(mu / (sigma + eps))          — Sharpe-like direction
  carry  = tanh(clamp(mu / (sigma^2 + eps))) — risk-adjusted carry ratio

Consensus rule: if sign(drift) == sign(carry) -> trade; else flat.
Position size = tanh(gain * sign(drift) * min(|drift|, |carry|)) * confidence.
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

NAME = "dual_signal_vote"
DESCRIPTION = (
    "Positions only when drift and carry sub-signals agree in sign (consensus vote); "
    "size = tanh(gain * min(|drift|,|carry|)) * confidence."
)

_EPS             = 1e-9
_CARRY_CLAMP     = 10.0   # clamp carry_raw before tanh to avoid runaway
_GAIN            = 3.0    # gain applied before outer tanh on consensus weight
_ANOMALY_THRESH  = 0.6    # anomaly_score above this -> flat
_DEAD_BAND       = 0.04   # |weight| below this -> 0


class DualSignalVoteStrategy:
    """Take a position only when drift and carry sub-signals vote the same direction."""

    # ------------------------------------------------------------------ #
    #  signals()                                                           #
    # ------------------------------------------------------------------ #
    def signals(self, state: ManifoldState) -> SignalSet:
        mu  = state.fwd_return_mean
        sig = state.fwd_return_std

        # (a) Drift: Sharpe-like direction
        drift = math.tanh(mu / (sig + _EPS))

        # (b) Carry: mean / variance, clamped then tanh
        carry_raw = mu / (sig ** 2 + _EPS)
        carry_clamped = max(-_CARRY_CLAMP, min(_CARRY_CLAMP, carry_raw))
        carry = math.tanh(carry_clamped)

        # Consensus signal: use drift as the canonical sub-signal slot;
        # store carry in expected_return so target() can recover it.
        confidence = max(0.0, min(1.0, state.regime_prob * (1.0 - state.anomaly_score)))

        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=drift,           # drift sub-signal
            expected_return=carry,    # carry sub-signal
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    # ------------------------------------------------------------------ #
    #  target()                                                            #
    # ------------------------------------------------------------------ #
    def target(self, signals: SignalSet) -> TargetPosition:
        flat = TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        # Hard gates: anomalous regime or high anomaly score -> flat
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_THRESH:
            return flat

        drift = signals.momentum          # sub-signal (a)
        carry = signals.expected_return   # sub-signal (b)

        # Consensus vote: both signals must agree in sign
        if drift * carry <= 0.0:
            return flat

        # Consensus weight: sign of drift * min magnitude, squashed by tanh
        consensus_mag = min(abs(drift), abs(carry))
        direction     = 1.0 if drift > 0.0 else -1.0
        raw = math.tanh(_GAIN * consensus_mag) * direction * signals.confidence

        # Dead-band: suppress small noisy weights
        if abs(raw) < _DEAD_BAND:
            return flat

        # Clamp to [-1, 1]
        raw = max(-1.0, min(1.0, raw))

        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=raw)


def build() -> Strategy:
    """Return a fresh DualSignalVoteStrategy instance."""
    return DualSignalVoteStrategy()
