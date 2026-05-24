"""Quick-profit scalper: small frequent positions with rapid profit-taking.

Enters small positions (capped at 0.5) aligned with the regime drift.
Uses light EMA smoothing on the raw weight signal to reduce churn/commission
while still responding to regime changes.  Tracks bars-in-position per symbol
and applies a hold_decay multiplier that ramps down after a grace period —
mimicking a take-profit that peels off exposure quickly so most bars are
flat-or-exiting (low risk) rather than deeply committed.

weight = sign(sharpe) * min(0.5, tanh(gain*|tanh(sharpe)|)) * hold_decay * confidence
hold_decay = decay_rate^max(0, bars_held - grace)
confidence = regime_prob * (1 - anomaly_score)

Flat on ANOMALY_REGIME or anomaly_score > 0.6.  Dead-band: |w| < 0.04 -> 0.
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

NAME = "quick_profit_scalper"
DESCRIPTION = (
    "Small frequent positions (|w|<=0.5) with hold_decay profit-taking; "
    "EMA smoothing reduces churn; flat on anomaly; per-symbol state."
)

_EPS = 1e-9

# Signal gain: tanh(gain * |tanh(sharpe)|) maps Sharpe -> (0,1)
_GAIN = 3.0
# Hard cap on |weight|
_MAX_WEIGHT = 0.5
# Anomaly gate
_ANOMALY_THRESH = 0.6
# Dead-band
_DEAD_BAND = 0.04

# EMA smoothing on raw signal: alpha high -> quick response, alpha low -> smooth/low turnover
# Use moderate alpha to balance responsiveness and commission reduction
_EMA_ALPHA = 0.25

# Hold-decay: grace bars before decay starts, then decay_rate per bar
_GRACE_BARS = 2       # hold full-size for this many bars
_DECAY_RATE = 0.55    # aggressive per-bar factor after grace (profit-taking)

# Reset clock once decayed position weight falls below this
_RESET_THRESH = 0.05


class QuickProfitScalper:
    """Small-size scalper with EMA signal smoothing and hold-decay profit-taking."""

    def __init__(self) -> None:
        self._ema: dict[str, float] = {}       # per-symbol EMA of raw weight
        self._bars_held: dict[str, int] = {}   # bars since last direction
        self._last_sign: dict[str, int] = {}   # last committed direction

    # ------------------------------------------------------------------
    def signals(self, state: ManifoldState) -> SignalSet:
        sharpe = state.fwd_return_mean / (state.fwd_return_std + _EPS)
        confidence = state.regime_prob * (1.0 - state.anomaly_score)
        confidence = max(0.0, min(1.0, confidence))
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
        prev_ema = self._ema.get(sym, 0.0)
        bars_held = self._bars_held.get(sym, 0)
        last_sign = self._last_sign.get(sym, 0)

        def _flat(decay_ema: bool = True) -> TargetPosition:
            if decay_ema:
                self._ema[sym] = (1.0 - _EMA_ALPHA) * prev_ema
            self._bars_held[sym] = 0
            self._last_sign[sym] = 0
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        # Hard flat conditions
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_THRESH:
            return _flat()

        sharpe = signals.momentum
        confidence = signals.confidence

        # Compute raw signal: sign(sharpe) * min(0.5, tanh(gain*|tanh(sharpe)|)) * confidence
        sign_s = 1.0 if sharpe >= 0 else -1.0
        inner = math.tanh(_GAIN * abs(math.tanh(sharpe)))
        raw = sign_s * min(_MAX_WEIGHT, inner) * confidence

        # EMA smooth to cut churn
        smoothed = _EMA_ALPHA * raw + (1.0 - _EMA_ALPHA) * prev_ema
        self._ema[sym] = smoothed

        if abs(smoothed) < _DEAD_BAND:
            return _flat(decay_ema=False)

        new_sign = 1 if smoothed > 0 else -1

        # Direction flip resets bars_held
        if new_sign != last_sign:
            bars_held = 0
            self._last_sign[sym] = new_sign

        # Hold-decay: full size for grace bars, then decay
        extra = max(0, bars_held - _GRACE_BARS)
        hold_decay = _DECAY_RATE ** extra if extra > 0 else 1.0

        weight = smoothed * hold_decay

        # Increment bars held
        self._bars_held[sym] = bars_held + 1

        # Dead-band after decay
        if abs(weight) < _DEAD_BAND:
            weight = 0.0

        # When fully decayed, reset so next fresh signal starts at bar 0
        if abs(weight) < _RESET_THRESH:
            self._bars_held[sym] = 0
            self._last_sign[sym] = 0

        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=weight)


def build() -> Strategy:
    """Return a fresh QuickProfitScalper instance."""
    return QuickProfitScalper()
