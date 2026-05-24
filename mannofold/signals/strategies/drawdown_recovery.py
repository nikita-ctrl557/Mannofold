"""Drawdown-recovery strategy: re-risk gradually after losses.

After a per-symbol drawdown, exposure is scaled DOWN immediately, then
restored GRADUALLY as the synthetic equity proxy recovers toward its
peak — avoiding full-size re-entry into continued weakness and smoothing
month-over-month consistency.

exposure_factor ramps from ~0.2 (deep drawdown) to 1.0 (at or near peak).
Recovery is not instant: the factor moves toward its target value by a
small step each bar, ensuring gradual re-risking rather than a snap-back.
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

NAME = "drawdown_recovery"
DESCRIPTION = (
    "Re-risks gradually after losses: exposure_factor ramps from ~0.2 at peak "
    "drawdown back to 1.0 as the synthetic equity proxy recovers, smoothing "
    "month-over-month consistency and avoiding full re-entry into continued weakness."
)

# ---- tunable knobs ----
_GAIN = 3.0              # outer tanh amplifier
_ANOMALY_THRESH = 0.6    # anomaly_score above this -> flat
_DEAD_BAND = 0.04        # collapse |weight| below this to 0
_MAXDD_FLOOR = -0.25     # drawdown at which target_exposure_factor reaches _MIN_EXPOSURE
_MIN_EXPOSURE = 0.2      # minimum exposure factor at max drawdown
_RECOVERY_RATE = 0.04    # fraction of gap closed per bar (gradual re-risk)


class DrawdownRecoveryStrategy:
    """Trend strategy with gradual re-risking after drawdowns."""

    def __init__(self) -> None:
        # per-symbol synthetic equity proxy state
        self._equity: dict[str, float] = {}          # current proxy equity level
        self._peak: dict[str, float] = {}            # running peak of proxy equity
        self._prev_weight: dict[str, float] = {}     # weight used LAST step
        self._exposure_factor: dict[str, float] = {} # current smoothed exposure factor

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

        # --- initialise per-symbol state on first visit ---
        if sym not in self._equity:
            self._equity[sym] = 1.0
            self._peak[sym] = 1.0
            self._prev_weight[sym] = 0.0
            self._exposure_factor[sym] = 1.0

        # --- update synthetic equity proxy (strictly past-only) ---
        # drift is the now-observed neighbourhood drift that the PREVIOUS
        # weight was exposed to
        drift = signals.expected_return
        prev_w = self._prev_weight[sym]
        self._equity[sym] *= 1.0 + prev_w * drift
        if self._equity[sym] <= 0.0:
            self._equity[sym] = 1e-8

        # update running peak
        if self._equity[sym] > self._peak[sym]:
            self._peak[sym] = self._equity[sym]

        # proxy drawdown (0 at peak, negative when below)
        dd = self._equity[sym] / self._peak[sym] - 1.0

        # --- target exposure factor based on current drawdown ---
        # linearly ramps: 1.0 at dd=0, _MIN_EXPOSURE at dd=_MAXDD_FLOOR
        dd_ratio = max(0.0, min(1.0, dd / _MAXDD_FLOOR))  # 0 at peak, 1 at max dd
        target_exposure = 1.0 - dd_ratio * (1.0 - _MIN_EXPOSURE)

        # --- gradual re-risk: smoothly move exposure toward target ---
        # This prevents instant snap-back when proxy recovers
        current = self._exposure_factor[sym]
        self._exposure_factor[sym] = current + _RECOVERY_RATE * (target_exposure - current)
        exposure_factor = self._exposure_factor[sym]

        # --- flat unconditionally on anomalous regime or high anomaly ---
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_THRESH:
            self._prev_weight[sym] = 0.0
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        # --- trend conviction ---
        sharpe = signals.momentum
        conviction = math.tanh(_GAIN * math.tanh(sharpe))

        # --- final weight ---
        # weight = tanh(GAIN*tanh(sharpe)) * exposure_factor * confidence
        raw = conviction * exposure_factor * signals.confidence

        # dead-band
        if abs(raw) < _DEAD_BAND:
            raw = 0.0

        self._prev_weight[sym] = raw
        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=raw)


def build() -> Strategy:
    """Return a fresh DrawdownRecoveryStrategy instance."""
    return DrawdownRecoveryStrategy()
