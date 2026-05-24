"""Profit-lock monthly strategy: month-over-month consistency.

Builds a synthetic equity proxy per symbol from weight × realized drift.
Once monthly gains breach a threshold the exposure is reduced (profit locked)
and the high-water mark reset, so accumulated gains are not given back.

Conviction = tanh(GAIN * tanh(sharpe)) * confidence
confidence = regime_prob * (1 - anomaly_score)

Flat on ANOMALY_REGIME or anomaly_score > ANOMALY_THRESH.
Dead-band: |weight| < DEAD_BAND -> 0.
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

NAME = "profit_lock_monthly"
DESCRIPTION = (
    "Month-over-month consistency via synthetic equity proxy: locks in gains once "
    "the per-symbol proxy equity rises beyond a threshold, reducing exposure and "
    "resetting the high-water mark to preserve green months."
)

# ---- tunable knobs --------------------------------------------------------
_GAIN            = 3.0   # outer gain: tanh(GAIN * tanh(sharpe))
_ANOMALY_THRESH  = 0.6   # anomaly_score above this -> flat immediately
_DEAD_BAND       = 0.04  # collapse |weight| below this to 0
_LOCK_THRESH     = 0.02  # +2% synthetic gain triggers profit lock
_LOCK_SCALE      = 0.40  # reduce weight to this fraction of base when locking
_UNLOCK_DRAWDOWN = 0.005 # re-open full exposure after equity drops this far from lock


class ProfitLockMonthlyStrategy:
    """Trend-follow conviction with per-symbol synthetic equity profit-lock."""

    def __init__(self) -> None:
        # per-symbol synthetic equity proxy
        self._equity:     dict[str, float] = {}   # current synthetic equity (starts 1.0)
        self._hwm:        dict[str, float] = {}   # high-water mark since last reset
        self._locked:     dict[str, bool]  = {}   # currently in profit-lock mode?
        self._lock_level: dict[str, float] = {}   # equity level when locked

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

        # initialise per-symbol state lazily
        equity      = self._equity.get(sym, 1.0)
        hwm         = self._hwm.get(sym, 1.0)
        locked      = self._locked.get(sym, False)
        lock_level  = self._lock_level.get(sym, 1.0)

        def _flat() -> TargetPosition:
            # On anomaly: reset equity tracking to avoid stale state
            self._equity[sym]      = 1.0
            self._hwm[sym]         = 1.0
            self._locked[sym]      = False
            self._lock_level[sym]  = 1.0
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        # --- unconditional flat guards ---
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_THRESH:
            return _flat()

        # --- base conviction = tanh(GAIN * tanh(sharpe)) * confidence ---
        conviction = math.tanh(_GAIN * math.tanh(signals.momentum))
        base_weight = conviction * signals.confidence

        # dead-band
        if abs(base_weight) < _DEAD_BAND:
            base_weight = 0.0

        # clamp
        base_weight = max(-1.0, min(1.0, base_weight))

        # --- update synthetic equity proxy using expected return proxy ---
        # drift = base_weight * fwd_return_mean (past-only: mean is from train nbhd)
        drift = base_weight * signals.expected_return
        equity = equity * (1.0 + drift)
        # guard against blow-up
        equity = max(equity, 1e-6)

        # update high-water mark
        if equity > hwm:
            hwm = equity

        # --- profit-lock logic ---
        gain_from_hwm_base = (hwm - 1.0)   # total gain since last reset

        if not locked and gain_from_hwm_base >= _LOCK_THRESH:
            # Trigger profit lock: reduce exposure
            locked     = True
            lock_level = equity

        if locked:
            # Check if equity has pulled back enough from lock level to re-open
            drawdown_from_lock = (lock_level - equity) / (lock_level + 1e-9)
            if drawdown_from_lock >= _UNLOCK_DRAWDOWN:
                # Reset: new cycle begins
                locked    = False
                hwm       = equity   # reset HWM to current
                lock_level = equity
            else:
                # Still locked: scale down exposure
                base_weight = base_weight * _LOCK_SCALE

        # persist state
        self._equity[sym]      = equity
        self._hwm[sym]         = hwm
        self._locked[sym]      = locked
        self._lock_level[sym]  = lock_level

        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=base_weight)


def build() -> Strategy:
    """Return a fresh ProfitLockMonthlyStrategy instance."""
    return ProfitLockMonthlyStrategy()
