"""Quick-profit scalper: small frequent positions with rapid profit-taking.

Enters small positions (|weight| capped at 0.5) aligned with regime drift.
Heavy EMA smoothing (alpha~0.10) limits turnover and commission drag.
Tracks bars-in-position and applies a hold_decay after a short grace period —
simulating taking profits and reducing exposure after holding a few bars.
When decay brings exposure below a reset threshold the position clock resets,
allowing a fresh re-entry on the next confirming signal.

weight = sign(sharpe)*min(0.5, tanh(gain*|tanh(sharpe)|))*hold_decay*confidence
hold_decay = decay_rate^max(0, bars_held - grace)
confidence = regime_prob * (1 - anomaly_score)

Flat on ANOMALY_REGIME or anomaly_score > 0.6.  Dead-band |w|<0.04 -> 0.
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
    "Small positions capped at 0.5; EMA-smoothed signal with hold_decay profit-taking; "
    "per-symbol state; flat on anomaly."
)

_EPS = 1e-9

# Signal gain inside outer tanh
_GAIN = 3.0
# Hard weight cap
_MAX_WEIGHT = 0.5
# Anomaly gate
_ANOMALY_THRESH = 0.6
# Dead-band
_DEAD_BAND = 0.04

# Heavy EMA smoothing to keep turnover low (like trend_hold_long alpha=0.10)
_EMA_ALPHA = 0.10

# Hold-decay: grace period before decay kicks in, then aggressive decay
_GRACE_BARS = 3        # hold full EMA weight for this many bars
_DECAY_RATE = 0.70     # per-bar decay after grace (aggressive take-profit)

# Reset hold clock when decayed weight falls below this threshold
_RESET_THRESH = 0.04


class QuickProfitScalper:
    """Small-size scalper with EMA smoothing and hold-decay profit-taking."""

    def __init__(self) -> None:
        self._ema: dict[str, float] = {}       # per-symbol EMA of raw signal
        self._bars_held: dict[str, int] = {}   # consecutive bars in current direction
        self._last_sign: dict[str, int] = {}   # last direction (+1/-1/0)

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

        def _flat() -> TargetPosition:
            # Decay EMA toward zero; reset hold clock
            self._ema[sym] = (1.0 - _EMA_ALPHA) * prev_ema
            self._bars_held[sym] = 0
            self._last_sign[sym] = 0
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        # Hard gates
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_THRESH:
            return _flat()

        sharpe = signals.momentum
        confidence = signals.confidence

        # Raw signal: sign(sharpe) * min(0.5, tanh(gain*|tanh(sharpe)|)) * confidence
        sign_s = 1.0 if sharpe >= 0 else -1.0
        inner = math.tanh(_GAIN * abs(math.tanh(sharpe)))
        raw = sign_s * min(_MAX_WEIGHT, inner) * confidence

        # Heavy EMA smoothing to reduce churn
        smoothed = _EMA_ALPHA * raw + (1.0 - _EMA_ALPHA) * prev_ema
        self._ema[sym] = smoothed

        if abs(smoothed) < _DEAD_BAND:
            self._bars_held[sym] = 0
            self._last_sign[sym] = 0
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        new_sign = 1 if smoothed > 0 else -1

        # Reset hold clock on direction flip
        if new_sign != last_sign:
            bars_held = 0
            self._last_sign[sym] = new_sign

        # Apply hold_decay: grace bars at full size, then decay per bar
        extra = max(0, bars_held - _GRACE_BARS)
        hold_decay = (_DECAY_RATE ** extra) if extra > 0 else 1.0

        weight = smoothed * hold_decay

        # Increment bars counter
        self._bars_held[sym] = bars_held + 1

        # Dead-band after decay
        if abs(weight) < _DEAD_BAND:
            weight = 0.0

        # When decayed to near-zero, reset clock to allow fresh re-entry
        if abs(weight) < _RESET_THRESH:
            self._bars_held[sym] = 0
            self._last_sign[sym] = 0

        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=weight)


def build() -> Strategy:
    """Return a fresh QuickProfitScalper instance."""
    return QuickProfitScalper()
