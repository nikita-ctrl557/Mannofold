"""Multi-horizon trend strategy: combine FAST and SLOW EMAs of the drift signal.

Maintains two per-symbol exponential moving averages of fwd_return_mean:
  - FAST EMA (alpha ~ 0.3): recent short-horizon trend
  - SLOW EMA (alpha ~ 0.05): longer-horizon structural trend

When both EMAs agree in sign -> strong trend, full conviction in that direction.
When they disagree -> near flat (sign-agreement factor collapses weight).

weight = sign_agreement * tanh(GAIN * tanh(sharpe)) * confidence
confidence = regime_prob * (1 - anomaly_score)

Flat on ANOMALY_REGIME or anomaly_score > 0.6.
Dead-band: |weight| < 0.04 -> 0.
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

NAME = "multi_horizon_trend"
DESCRIPTION = (
    "Dual-EMA (fast/slow) drift-signal trend strategy: full conviction when "
    "both horizons agree in direction, near-flat when they disagree."
)

# Tunable knobs
_FAST_ALPHA = 0.3        # fast EMA decay (short horizon)
_SLOW_ALPHA = 0.05       # slow EMA decay (long horizon)
_GAIN = 2.5              # amplifier inside outer tanh
_ANOMALY_THRESH = 0.6    # anomaly_score above this -> flat
_DEAD_BAND = 0.04        # collapse |weight| below this to 0


class MultiHorizonTrendStrategy:
    """Per-symbol dual-EMA trend strategy."""

    def __init__(self) -> None:
        # per-symbol EMA state: (fast_ema, slow_ema, initialised)
        self._fast: dict[str, float] = {}
        self._slow: dict[str, float] = {}
        self._init: dict[str, bool] = {}

    # ------------------------------------------------------------------ #
    #  signals()                                                          #
    # ------------------------------------------------------------------ #
    def signals(self, state: ManifoldState) -> SignalSet:
        sym = state.symbol
        drift = state.fwd_return_mean

        # Bootstrap EMAs on first observation
        if not self._init.get(sym, False):
            self._fast[sym] = drift
            self._slow[sym] = drift
            self._init[sym] = True
        else:
            self._fast[sym] = _FAST_ALPHA * drift + (1.0 - _FAST_ALPHA) * self._fast[sym]
            self._slow[sym] = _SLOW_ALPHA * drift + (1.0 - _SLOW_ALPHA) * self._slow[sym]

        fast = self._fast[sym]
        slow = self._slow[sym]

        # Sign-agreement factor: +1 if both agree, -1 if they disagree (then we scale by
        # a soft blended signal), 0 if either is zero.
        # Use product of signs: > 0 means agree, < 0 means disagree.
        sign_product = fast * slow
        if sign_product > 0:
            sign_agreement = 1.0
        elif sign_product < 0:
            sign_agreement = -1.0   # will pull weight toward zero via blend below
        else:
            sign_agreement = 0.0

        # Blended drift for Sharpe: average of fast and slow (weights both horizons)
        blended_drift = 0.5 * (fast + slow)
        sharpe = blended_drift / (state.fwd_return_std + 1e-9)

        # Confidence = regime certainty × non-anomalousness
        confidence = state.regime_prob * (1.0 - state.anomaly_score)
        confidence = max(0.0, min(1.0, confidence))

        # sign_agreement is re-derived in target() from the live EMA state.
        return SignalSet(
            ts=state.ts,
            symbol=sym,
            momentum=sharpe,
            expected_return=blended_drift,
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

        # Recover EMA states for sign-agreement check
        fast = self._fast.get(sym, 0.0)
        slow = self._slow.get(sym, 0.0)

        sign_product = fast * slow
        if sign_product > 0:
            sign_agreement = 1.0
        elif sign_product < 0:
            # EMAs disagree: use soft blend that reduces conviction strongly
            sign_agreement = 0.1
        else:
            sign_agreement = 0.5

        # Raw weight: sign_agreement * tanh(GAIN * tanh(sharpe)) * confidence
        inner = math.tanh(signals.momentum)          # tanh(sharpe)
        raw = sign_agreement * math.tanh(_GAIN * inner) * signals.confidence

        # Dead-band
        if abs(raw) < _DEAD_BAND:
            raw = 0.0

        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=raw)


def build() -> Strategy:
    """Return a fresh MultiHorizonTrendStrategy instance."""
    return MultiHorizonTrendStrategy()
