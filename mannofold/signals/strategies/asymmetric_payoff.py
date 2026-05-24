"""Asymmetric-payoff strategy: high win-rate with positive expectancy.

Takes modest trend positions, but AGGRESSIVELY de-grosses at the first sign of
trouble — if anomaly rises or per-symbol conviction drops from its recent peak,
cut to flat immediately (keeps losing bars few and small). Re-enters only on a
clean signal above the entry threshold.

weight = tanh(GAIN * tanh(sharpe)) * cut_factor * confidence
  cut_factor drops fast when anomaly rises OR conviction falls below TRAIL_FRAC
  of the per-symbol peak seen since entry.
  confidence = regime_prob * (1 - anomaly_score)

Flat on ANOMALY_REGIME or anomaly > ~0.6.
Dead-band: |w| < 0.04 -> 0.
Per-symbol state, no lookahead. build() takes no args.
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

NAME = "asymmetric_payoff"
DESCRIPTION = (
    "High win-rate trend strategy: take modest positions with aggressive "
    "de-grossing when anomaly rises or per-symbol conviction drops from its "
    "recent peak, keeping losses small and re-entering only on clean signals."
)

# ---- tunable knobs --------------------------------------------------------
_GAIN           = 3.0    # outer gain inside double-tanh
_ANOMALY_THRESH = 0.60   # hard flat at or above this anomaly score
_ANOMALY_RAMP   = 0.30   # cut_factor starts declining above this level
_DEAD_BAND      = 0.04   # suppress |weight| below this to 0
_ENTRY_THRESH   = 0.08   # minimum |weight| to open / re-enter a position
_TRAIL_FRAC     = 0.45   # exit if |weight| < TRAIL_FRAC * peak since entry


class AsymmetricPayoffStrategy:
    """Trend follow with aggressive cut on anomaly or conviction drawdown."""

    def __init__(self) -> None:
        self._in_position: dict[str, bool]  = {}
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

        # --- hard flat guards ---
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_THRESH:
            return _flat()

        # --- cut_factor: linear decay from 1 at _ANOMALY_RAMP to 0 at _ANOMALY_THRESH ---
        if signals.anomaly <= _ANOMALY_RAMP:
            cut_factor = 1.0
        else:
            span = max(_ANOMALY_THRESH - _ANOMALY_RAMP, 1e-9)
            cut_factor = max(0.0, 1.0 - (signals.anomaly - _ANOMALY_RAMP) / span)

        # --- conviction: tanh(GAIN * tanh(sharpe)) ---
        conviction = math.tanh(_GAIN * math.tanh(signals.momentum))

        # --- final weight ---
        weight = conviction * cut_factor * signals.confidence

        # dead-band
        if abs(weight) < _DEAD_BAND:
            weight = 0.0
        weight = max(-1.0, min(1.0, weight))
        abs_w  = abs(weight)
        in_pos = self._in_position.get(sym, False)

        if not in_pos:
            if abs_w >= _ENTRY_THRESH:
                self._in_position[sym] = True
                self._stance[sym]      = 1 if weight > 0 else -1
                self._peak_w[sym]      = abs_w
                return TargetPosition(ts=signals.ts, symbol=sym, target_weight=weight)
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        # --- in position: update peak, check trailing stop ---
        peak = self._peak_w.get(sym, abs_w)
        if abs_w > peak:
            self._peak_w[sym] = abs_w
            peak = abs_w

        # trailing stop on conviction drawdown
        if peak > 0.0 and abs_w < _TRAIL_FRAC * peak:
            return _flat()

        # too small after dead-band
        if abs_w < _DEAD_BAND:
            return _flat()

        # sign flip: flip only with sufficient conviction, else cut
        current_stance = self._stance.get(sym, 0)
        new_stance = 1 if weight > 0 else (-1 if weight < 0 else 0)
        if new_stance != 0 and new_stance != current_stance:
            if abs_w >= _ENTRY_THRESH:
                self._stance[sym] = new_stance
                self._peak_w[sym] = abs_w
                return TargetPosition(ts=signals.ts, symbol=sym, target_weight=weight)
            return _flat()

        # hold: emit current weight signed by recorded stance
        signed_w = abs_w * current_stance
        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=signed_w)


def build() -> Strategy:
    """Return a fresh AsymmetricPayoffStrategy instance."""
    return AsymmetricPayoffStrategy()
