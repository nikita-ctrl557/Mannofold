"""Acceleration-trend strategy: trade the *acceleration* of drift.

Keeps a per-symbol EMA of fwd_return_mean (the drift). Acceleration is
defined as current_drift - ema_drift. When drift and acceleration share
the same sign the trend is strengthening → add conviction. When they
oppose the trend is decelerating → trim toward flat.
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

NAME = "accel_trend"
DESCRIPTION = (
    "Drift-acceleration strategy: boosts conviction when trend is strengthening, "
    "trims when decelerating; flat on anomaly."
)

# Tunable knobs
_EMA_ALPHA = 0.15       # smoothing factor for per-symbol drift EMA
_GAIN = 2.5             # amplifier inside outer tanh
_ANOMALY_THRESH = 0.6   # anomaly_score above this → flat
_DEAD_BAND = 0.04       # collapse |weight| below this to 0
_ACCEL_BOOST_MIN = 0.3  # accel_boost when trend is decelerating
_ACCEL_BOOST_MAX = 1.5  # accel_boost when trend is accelerating


class AccelTrendStrategy:
    """Acceleration-trend strategy with per-symbol EMA state."""

    def __init__(self) -> None:
        # per-symbol EMA of fwd_return_mean
        self._ema_drift: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    #  signals()                                                          #
    # ------------------------------------------------------------------ #
    def signals(self, state: ManifoldState) -> SignalSet:
        sym = state.symbol
        drift = state.fwd_return_mean

        # Update EMA (initialise on first observation)
        prev_ema = self._ema_drift.get(sym, drift)
        ema = prev_ema + _EMA_ALPHA * (drift - prev_ema)
        self._ema_drift[sym] = ema

        # Acceleration: positive when drift is growing, negative when shrinking
        accel = drift - ema

        # accel_boost: higher when drift and accel agree, lower when they oppose
        if drift * accel > 0:
            # trend accelerating → boost toward max
            boost = _ACCEL_BOOST_MAX
        elif drift * accel < 0:
            # trend decelerating → trim toward min
            boost = _ACCEL_BOOST_MIN
        else:
            boost = (_ACCEL_BOOST_MIN + _ACCEL_BOOST_MAX) / 2.0

        # Neighbourhood Sharpe drives the base momentum signal
        sharpe = drift / (state.fwd_return_std + 1e-9)

        # momentum encodes direction + accel_boost as a single scalar
        # sign(drift) * |tanh(sharpe)| * boost  (stored in SignalSet.momentum)
        momentum = math.copysign(abs(math.tanh(sharpe)) * boost, sharpe)

        confidence = state.regime_prob * (1.0 - state.anomaly_score)
        confidence = max(0.0, min(1.0, confidence))

        return SignalSet(
            ts=state.ts,
            symbol=sym,
            momentum=momentum,
            expected_return=drift,
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

        # weight = sign(drift) * tanh(gain * |momentum|) * confidence
        # signals.momentum already carries sign(drift) * |tanh(sharpe)| * boost
        raw = math.tanh(_GAIN * signals.momentum) * signals.confidence

        # Dead-band
        if abs(raw) < _DEAD_BAND:
            raw = 0.0

        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=raw)


def build() -> Strategy:
    """Return a fresh AccelTrendStrategy instance."""
    return AccelTrendStrategy()
