"""Regime-Shift Rider: extra conviction on fresh regime transitions.

When the regime_id changes for a symbol, this strategy enters with boosted
confidence in the new regime's drift direction. The boost decays back to
baseline as the regime persists, tracking bars-since-transition per symbol.
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

NAME = "regime_shift_rider"
DESCRIPTION = (
    "Rides regime transitions with extra conviction: boosts sizing immediately"
    " after a regime change, decaying back to baseline as the new regime persists."
)

# Tunable knobs
_GAIN = 3.0               # amplifier inside tanh(gain * sharpe)
_ANOMALY_THRESH = 0.6     # anomaly_score above this -> flat
_DEAD_BAND = 0.04         # collapse |weight| below this to zero
_BOOST = 1.5              # multiplier applied right after a regime transition
_BOOST_DECAY_BARS = 8     # bars over which boost linearly decays from _BOOST -> 1.0
_ENTRY_THRESH = 0.05      # minimum |weight| to open a position
_EXIT_THRESH = 0.03       # exit if conviction decays below this


class RegimeShiftRider:
    """Per-symbol stateful strategy that rides fresh regime transitions."""

    def __init__(self) -> None:
        # per-symbol last observed regime_id (None = not yet seen)
        self._prev_regime: dict[str, int] = {}
        # per-symbol bars elapsed since last regime transition (0 = just changed)
        self._bars_since_shift: dict[str, int] = {}
        # per-symbol current signed stance (+1, -1, 0)
        self._stance: dict[str, int] = {}

    # ------------------------------------------------------------------ #
    #  signals()                                                           #
    # ------------------------------------------------------------------ #
    def signals(self, state: ManifoldState) -> SignalSet:
        sym = state.symbol
        prev = self._prev_regime.get(sym)

        # Detect transition: first bar OR regime_id changed
        if prev is None or state.regime_id != prev:
            self._bars_since_shift[sym] = 0
        else:
            self._bars_since_shift[sym] = self._bars_since_shift.get(sym, 0) + 1

        self._prev_regime[sym] = state.regime_id

        confidence = state.regime_prob * (1.0 - state.anomaly_score)
        confidence = max(0.0, min(1.0, confidence))

        sharpe = state.fwd_return_mean / (state.fwd_return_std + 1e-9)

        return SignalSet(
            ts=state.ts,
            symbol=sym,
            momentum=sharpe,
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    # ------------------------------------------------------------------ #
    #  target()                                                            #
    # ------------------------------------------------------------------ #
    def target(self, signals: SignalSet) -> TargetPosition:
        sym = signals.symbol

        # Flat on anomalous regime or high anomaly
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_THRESH:
            self._stance[sym] = 0
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        # Compute transition boost: linearly decay from _BOOST -> 1.0 over _BOOST_DECAY_BARS
        bars_since = self._bars_since_shift.get(sym, _BOOST_DECAY_BARS)
        if bars_since < _BOOST_DECAY_BARS:
            t = bars_since / _BOOST_DECAY_BARS  # 0..1
            boost = _BOOST + (_BOOST - 1.0) * (-t)  # linear: _BOOST at t=0, 1.0 at t=1
            # Simpler: boost = _BOOST * (1 - t) + 1.0 * t
            boost = _BOOST * (1.0 - t) + 1.0 * t
        else:
            boost = 1.0

        # Base weight from tanh of Sharpe scaled by confidence, then boosted
        raw = math.tanh(_GAIN * signals.momentum) * signals.confidence * boost

        # Dead-band
        if abs(raw) < _DEAD_BAND:
            raw = 0.0

        current_stance = self._stance.get(sym, 0)
        desired_sign = 1 if raw > 0 else (-1 if raw < 0 else 0)

        if current_stance == 0:
            if abs(raw) >= _ENTRY_THRESH:
                self._stance[sym] = desired_sign
            else:
                return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)
        else:
            if desired_sign != 0 and desired_sign != current_stance and abs(raw) >= _ENTRY_THRESH:
                self._stance[sym] = desired_sign
            elif abs(raw) < _EXIT_THRESH:
                self._stance[sym] = 0
                return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        weight = abs(raw) * self._stance.get(sym, 0)
        # Clamp to [-1, 1]
        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=weight)


def build() -> Strategy:
    """Return a fresh RegimeShiftRider instance."""
    return RegimeShiftRider()
