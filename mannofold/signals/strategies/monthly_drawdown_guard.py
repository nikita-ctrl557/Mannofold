"""Monthly drawdown guard strategy.

Month-over-month CONSISTENCY — protect against losing months.

Trend conviction = tanh(GAIN * tanh(sharpe)).  Maintains a per-symbol
synthetic equity proxy that compounds by (prev_weight * realized drift)
each step, tracking its running peak.  As the proxy drawdown deepens
within the month, exposure is throttled toward zero (protecting the month)
and restored as the proxy recovers.  The month-peak is reset at the start
of each new calendar month.
"""

from __future__ import annotations

import math
from datetime import datetime

from mannofold.contracts.interfaces import Strategy
from mannofold.contracts.models import (
    ANOMALY_REGIME,
    ManifoldState,
    SignalSet,
    TargetPosition,
)

NAME = "monthly_drawdown_guard"
DESCRIPTION = (
    "Trend conviction throttled by a per-symbol synthetic equity proxy whose "
    "peak resets every calendar month; exposure collapses as the monthly proxy "
    "drawdown deepens, restoring as it recovers.  Flat on anomalous regimes."
)

# Tunable knobs
_GAIN = 3.0             # outer tanh amplifier: tanh(GAIN * tanh(sharpe))
_ANOMALY_THRESH = 0.6   # anomaly_score above this -> flat immediately
_DEAD_BAND = 0.04       # collapse |weight| below this to 0
_MAXDD_FLOOR = -0.15    # monthly proxy drawdown at which risk_factor reaches 0


class MonthlyDrawdownGuardStrategy:
    """Trend-follow with per-symbol monthly synthetic drawdown throttle."""

    def __init__(self) -> None:
        # per-symbol synthetic equity proxy state
        self._equity: dict[str, float] = {}          # current proxy equity level
        self._month_peak: dict[str, float] = {}      # running peak THIS month
        self._prev_weight: dict[str, float] = {}     # weight used LAST step
        self._current_month: dict[str, int] = {}     # (year*12+month) sentinel

    # ------------------------------------------------------------------ #
    #  signals()                                                          #
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    #  target()                                                           #
    # ------------------------------------------------------------------ #
    def target(self, signals: SignalSet) -> TargetPosition:
        sym = signals.symbol
        ts: datetime = signals.ts

        # --- initialise per-symbol state on first visit ---
        if sym not in self._equity:
            self._equity[sym] = 1.0
            self._month_peak[sym] = 1.0
            self._prev_weight[sym] = 0.0
            self._current_month[sym] = ts.year * 12 + ts.month

        # --- month boundary: reset peak each new calendar month ---
        month_key = ts.year * 12 + ts.month
        if month_key != self._current_month[sym]:
            self._month_peak[sym] = self._equity[sym]
            self._current_month[sym] = month_key

        # --- update synthetic equity proxy (strictly past-only) ---
        # realized drift proxy: the signal observed NOW was the drift that
        # the PREVIOUS weight was exposed to
        drift = signals.expected_return
        prev_w = self._prev_weight[sym]
        self._equity[sym] *= 1.0 + prev_w * drift
        if self._equity[sym] <= 0.0:
            self._equity[sym] = 1e-8

        # update this month's running peak
        if self._equity[sym] > self._month_peak[sym]:
            self._month_peak[sym] = self._equity[sym]

        # monthly proxy drawdown (0 at peak, negative when below)
        dd = self._equity[sym] / self._month_peak[sym] - 1.0

        # --- flat unconditionally on anomalous regime or high anomaly ---
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_THRESH:
            self._prev_weight[sym] = 0.0
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        # --- trend conviction ---
        sharpe = signals.momentum
        conviction = math.tanh(_GAIN * math.tanh(sharpe))

        # --- monthly drawdown throttle ---
        # risk_factor = clamp(1 + dd / |maxdd_floor|, 0, 1)
        # = 1 when dd=0, = 0 when dd <= _MAXDD_FLOOR
        risk_factor = max(0.0, min(1.0, 1.0 + dd / abs(_MAXDD_FLOOR)))

        # --- final weight ---
        raw = conviction * risk_factor * signals.confidence

        # dead-band
        if abs(raw) < _DEAD_BAND:
            raw = 0.0

        self._prev_weight[sym] = raw
        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=raw)


def build() -> Strategy:
    """Return a fresh MonthlyDrawdownGuardStrategy instance."""
    return MonthlyDrawdownGuardStrategy()
