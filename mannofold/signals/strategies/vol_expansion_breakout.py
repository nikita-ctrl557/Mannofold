"""Vol-expansion breakout strategy.

When current fwd_return_std exceeds a per-symbol EMA of std (vol expanding ->
breakout), take a position with FULL conviction in the drift direction.
When vol is contracting / normal, take only a small fractional position.
Direction = sign(fwd_return_mean); magnitude = tanh(GAIN * tanh(sharpe)).
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

NAME = "vol_expansion_breakout"
DESCRIPTION = (
    "Vol-expansion breakout: full conviction when current std > EMA of std "
    "(breakout regime); fractional sizing when vol is contracting/normal."
)

# Tunable knobs
_GAIN = 3.0              # amplifier inside tanh(gain * tanh(sharpe))
_ANOMALY_THRESH = 0.6    # anomaly_score above this -> flat
_DEAD_BAND = 0.04        # collapse |weight| below this to 0
_EXPANSION_MULT = 1.2    # ratio current_std / ema_std must exceed for breakout
_EMA_ALPHA = 0.1         # smoothing for per-symbol EMA of fwd_return_std
_FRACTION_NORMAL = 0.25  # fraction of full conviction when vol is NOT expanding
_MIN_EMA_OBS = 5         # number of updates before EMA is considered reliable


class VolExpansionBreakoutStrategy:
    """Vol-expansion breakout with per-symbol EMA state."""

    def __init__(self) -> None:
        # per-symbol EMA of fwd_return_std
        self._ema_std: dict[str, float] = {}
        # per-symbol count of EMA updates
        self._ema_count: dict[str, int] = {}

    # ------------------------------------------------------------------ #
    #  signals()                                                          #
    # ------------------------------------------------------------------ #
    def signals(self, state: ManifoldState) -> SignalSet:
        confidence = state.regime_prob * (1.0 - state.anomaly_score)
        confidence = max(0.0, min(1.0, confidence))

        sym = state.symbol
        current_std = state.fwd_return_std

        # Update EMA of std (past-only — update AFTER reading the ratio)
        ema = self._ema_std.get(sym, current_std)
        count = self._ema_count.get(sym, 0)

        # Vol-expansion indicator stored in momentum field as a multiplier [0..1+]
        # Use raw std/ema ratio so target() can decide full vs fractional conviction
        std_ratio = current_std / (ema + 1e-9)

        # Sharpe for direction + magnitude
        sharpe = state.fwd_return_mean / (current_std + 1e-9)

        # Update EMA for NEXT call (no lookahead)
        if count == 0:
            new_ema = current_std
        else:
            new_ema = _EMA_ALPHA * current_std + (1.0 - _EMA_ALPHA) * ema
        self._ema_std[sym] = new_ema
        self._ema_count[sym] = count + 1

        return SignalSet(
            ts=state.ts,
            symbol=sym,
            momentum=sharpe,           # Sharpe: sign = direction, magnitude = strength
            expected_return=std_ratio, # repurpose: std_ratio for breakout detection
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    # ------------------------------------------------------------------ #
    #  target()                                                           #
    # ------------------------------------------------------------------ #
    def target(self, signals: SignalSet) -> TargetPosition:
        sym = signals.symbol

        # Flat on anomalous regime or high anomaly score
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_THRESH:
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        sharpe = signals.momentum      # fwd_return_mean / fwd_return_std
        std_ratio = signals.expected_return  # current_std / ema_std

        # Conviction magnitude: tanh(gain * tanh(sharpe))
        conviction = math.tanh(_GAIN * math.tanh(sharpe))

        # Check whether EMA is reliable (enough observations)
        count = self._ema_count.get(sym, 0)
        ema_reliable = count >= _MIN_EMA_OBS

        # Vol-expanding breakout: full conviction; otherwise fractional
        if ema_reliable and std_ratio > _EXPANSION_MULT:
            raw = conviction * signals.confidence
        else:
            raw = conviction * signals.confidence * _FRACTION_NORMAL

        # Dead-band
        if abs(raw) < _DEAD_BAND:
            raw = 0.0

        # Clamp to [-1, 1]
        raw = max(-1.0, min(1.0, raw))

        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=raw)


def build() -> Strategy:
    """Return a fresh VolExpansionBreakoutStrategy instance."""
    return VolExpansionBreakoutStrategy()
