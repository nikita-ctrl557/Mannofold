"""Signal-stoploss strategy: trend-follow conviction with a trailing peak stop.

Conviction = tanh(GAIN * tanh(sharpe)).  While in a position, tracks the peak
absolute conviction*confidence reached since entry.  If the current value falls
below TRAIL_FRAC of that peak the position is closed (trailing stop proxy) and
re-entry requires clearing ENTRY_THRESH afresh.  No lookahead; all state is
per-symbol and deterministic.
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

NAME = "signal_stoploss"
DESCRIPTION = (
    "Trend-follow conviction with trailing peak-stop: exits when |conviction| "
    "drops below a fraction of the peak conviction seen since entry."
)

# ---- tunable knobs --------------------------------------------------------
_GAIN           = 3.0   # outer gain: tanh(GAIN * tanh(sharpe))
_ANOMALY_THRESH = 0.6   # anomaly_score above this -> flat immediately
_DEAD_BAND      = 0.04  # collapse |weight| below this to 0
_ENTRY_THRESH   = 0.10  # minimum |weight| to open a fresh position
_EXIT_THRESH    = 0.05  # secondary floor: weight too small -> exit regardless
_TRAIL_FRAC     = 0.45  # exit when |weight| < TRAIL_FRAC * peak_weight_since_entry


class SignalStoplossStrategy:
    """Trend-follow with trailing conviction stop, per-symbol state."""

    def __init__(self) -> None:
        # per-symbol tracking
        self._in_position: dict[str, bool]  = {}   # currently holding?
        self._stance:      dict[str, int]   = {}   # +1 / -1 / 0
        self._peak_w:      dict[str, float] = {}   # peak |weight| since entry

    # ---------------------------------------------------------------------- #
    #  signals()                                                              #
    # ---------------------------------------------------------------------- #
    def signals(self, state: ManifoldState) -> SignalSet:
        confidence = state.regime_prob * (1.0 - state.anomaly_score)
        confidence = max(0.0, min(1.0, confidence))

        sharpe = state.fwd_return_mean / (state.fwd_return_std + 1e-9)

        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=sharpe,
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    # ---------------------------------------------------------------------- #
    #  target()                                                               #
    # ---------------------------------------------------------------------- #
    def target(self, signals: SignalSet) -> TargetPosition:
        sym = signals.symbol

        def _flat() -> TargetPosition:
            self._in_position[sym] = False
            self._stance[sym]      = 0
            self._peak_w[sym]      = 0.0
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        # --- unconditional flat guards ---
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_THRESH:
            return _flat()

        # --- conviction = tanh(GAIN * tanh(sharpe)) ---
        conviction = math.tanh(_GAIN * math.tanh(signals.momentum))
        # gate by confidence
        weight = conviction * signals.confidence

        # dead-band: treat tiny signals as zero
        if abs(weight) < _DEAD_BAND:
            weight = 0.0

        # clamp to [-1, 1]
        weight = max(-1.0, min(1.0, weight))

        abs_w  = abs(weight)
        in_pos = self._in_position.get(sym, False)

        if not in_pos:
            # --- out of position: need fresh entry signal ---
            if abs_w >= _ENTRY_THRESH:
                self._in_position[sym] = True
                self._stance[sym]      = 1 if weight > 0 else -1
                self._peak_w[sym]      = abs_w
                return TargetPosition(ts=signals.ts, symbol=sym, target_weight=weight)
            # below entry threshold — stay flat
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        # --- in position ---
        # update running peak
        peak = self._peak_w.get(sym, abs_w)
        if abs_w > peak:
            self._peak_w[sym] = abs_w
            peak = abs_w

        # secondary floor exit
        if abs_w < _EXIT_THRESH:
            return _flat()

        # trailing stop: conviction has decayed too far from peak
        if peak > 0.0 and abs_w < _TRAIL_FRAC * peak:
            return _flat()

        # sign reversal with enough conviction: flip stance immediately
        current_stance = self._stance.get(sym, 0)
        new_stance = 1 if weight > 0 else (-1 if weight < 0 else 0)
        if new_stance != 0 and new_stance != current_stance:
            if abs_w >= _ENTRY_THRESH:
                # flip directly
                self._stance[sym] = new_stance
                self._peak_w[sym] = abs_w
                return TargetPosition(ts=signals.ts, symbol=sym, target_weight=weight)
            # weak opposite signal — exit, re-entry later
            return _flat()

        # hold: emit current weight (signed by stance)
        signed_w = abs_w * current_stance
        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=signed_w)


def build() -> Strategy:
    """Return a fresh SignalStoplossStrategy instance."""
    return SignalStoplossStrategy()
