"""Quick-profit scalper: small frequent positions with rapid profit-taking.

Enters small positions aligned with the regime drift (Sharpe signal), capping
weight at ~0.5. Tracks bars-in-position per symbol and applies a hold_decay
that ramps down quickly (take-profit / reduce-exposure effect). Re-enters on a
fresh signal once the previous position has been scaled to near-zero and reset.
Flat on ANOMALY_REGIME or high anomaly score. Only enters when the expected
return is positive on the intended side (positive-drift filter).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from mannofold.contracts.interfaces import Strategy
from mannofold.contracts.models import (
    ANOMALY_REGIME,
    ManifoldState,
    SignalSet,
    TargetPosition,
)

NAME = "quick_profit_scalper"
DESCRIPTION = (
    "Small-sized positions aligned with regime drift; hold_decay drives rapid "
    "profit-taking after a few bars. Positive-drift filter + high-confidence "
    "entries => high win-rate with positive expectancy."
)

# ---- tunable knobs --------------------------------------------------------
_GAIN = 2.5            # amplifier inside outer tanh(gain * |tanh(sharpe)|)
_MAX_WEIGHT = 0.5      # cap on absolute weight
_ANOMALY_THRESH = 0.6  # anomaly_score above this -> flat immediately
_DEAD_BAND = 0.04      # collapse |weight| below this to 0
_CONF_MIN = 0.25       # minimum confidence to enter (regime_prob*(1-anomaly))

# Hold-decay: after _DECAY_START bars, multiply accumulated weight by _DECAY_RATE per bar.
_DECAY_START = 1       # bars of full-size before decay kicks in
_DECAY_RATE = 0.65     # per-bar multiplier once decay starts (aggressive quick-exit)

# Re-entry: reset bars_held counter once position has decayed to near-zero.
_REENTRY_THRESH = 0.035  # weight below this triggers hold-clock reset
_MAX_HOLD = 6            # hard cap — force flat & reset after this many bars
# -------------------------------------------------------------------------


class QuickProfitScalper:
    """Small-size scalper with per-symbol hold-decay profit-taking."""

    def __init__(self) -> None:
        # per-symbol: (bars_held, last_sign)
        self._bars_held: dict[str, int] = {}
        self._last_sign: dict[str, int] = {}

    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    def target(self, signals: SignalSet) -> TargetPosition:
        sym = signals.symbol
        bars_held = self._bars_held.get(sym, 0)
        last_sign = self._last_sign.get(sym, 0)

        def _flat() -> TargetPosition:
            self._bars_held[sym] = 0
            self._last_sign[sym] = 0
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        # --- flat conditions ---
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_THRESH:
            return _flat()

        if bars_held >= _MAX_HOLD:
            return _flat()

        confidence = signals.confidence

        # Require minimum confidence to trade
        if confidence < _CONF_MIN:
            return _flat()

        sharpe = signals.momentum

        # Only trade when the expected return agrees with the sharpe direction
        # (positive-drift filter: avoids entering against the neighbourhood mean)
        if sharpe > 0 and signals.expected_return <= 0:
            return _flat()
        if sharpe < 0 and signals.expected_return >= 0:
            return _flat()

        # --- base signal: sign(sharpe)*min(0.5, tanh(gain*|tanh(sharpe)|))*confidence ---
        sign_s = 1 if sharpe >= 0 else -1
        inner = math.tanh(_GAIN * abs(math.tanh(sharpe)))
        base_w = sign_s * min(_MAX_WEIGHT, inner) * confidence

        if abs(base_w) < _DEAD_BAND:
            return _flat()

        new_sign = 1 if base_w > 0 else -1

        # --- direction flip: reset hold clock ---
        if new_sign != last_sign:
            bars_held = 0
            self._last_sign[sym] = new_sign

        # --- hold-decay -------------------------------------------------------
        if bars_held < _DECAY_START:
            hold_decay = 1.0
        else:
            hold_decay = _DECAY_RATE ** (bars_held - _DECAY_START + 1)

        weight = base_w * hold_decay

        # Increment bars counter
        self._bars_held[sym] = bars_held + 1

        # --- dead-band + re-entry reset ---------------------------------------
        if abs(weight) < _DEAD_BAND:
            weight = 0.0

        if abs(weight) < _REENTRY_THRESH:
            self._bars_held[sym] = 0
            self._last_sign[sym] = 0

        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=weight)


def build() -> Strategy:
    """Return a fresh QuickProfitScalper instance."""
    return QuickProfitScalper()
