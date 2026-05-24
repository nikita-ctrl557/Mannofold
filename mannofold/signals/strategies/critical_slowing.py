"""Critical Slowing Down: early-warning signals before a regime tipping point.

Rising lag-1 autocorrelation AND rising variance in fwd_return_mean are
hallmarks of "critical slowing down" — the system loses resilience before a
regime transition. When both are rising, de-risk to near flat. When they are
low/stable, the current regime is robust and we trade the drift at full size.
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

NAME = "critical_slowing"
DESCRIPTION = (
    "Dynamical-systems early-warning strategy: detects critical slowing down "
    "(rising lag-1 autocorrelation + rising variance) ahead of regime transitions "
    "and de-risks accordingly; trades full drift when the regime is robust."
)

# Tunable knobs
_GAIN = 3.0            # amplifier inside tanh(gain * sharpe)
_ANOMALY_THRESH = 0.6  # anomaly_score above this -> flat
_DEAD_BAND = 0.04      # collapse |weight| below this to zero
_EMA_FAST = 0.15       # fast EMA alpha for level estimates
_EMA_SLOW = 0.05       # slow EMA alpha for trend estimates (rising = slow rising)
_EPS = 1e-9            # numerical guard


class _SymbolState:
    """Per-symbol rolling statistics for critical slowing down detection."""

    __slots__ = (
        "prev_x",
        "ema_mean", "ema_sq", "ema_var",
        "ema_autocorr",
        "ema_var_trend", "ema_autocorr_trend",
        "_warning",
    )

    def __init__(self) -> None:
        self.prev_x: float | None = None
        self.ema_mean: float = 0.0
        self.ema_sq: float = 0.0
        self.ema_var: float = 0.0        # EMA of (x - mean)^2
        self.ema_autocorr: float = 0.0   # EMA of lag-1 autocorrelation proxy
        self.ema_var_trend: float = 0.0  # slow EMA of ema_var  (tracks level)
        self.ema_autocorr_trend: float = 0.0  # slow EMA of ema_autocorr
        self._warning: float = 0.0

    def update(self, x: float) -> tuple[float, float, float, float]:
        """Update statistics with new observation x.

        Returns (autocorr_level, var_level, autocorr_trend, var_trend).
        trend > level  =>  rising  =>  warning.
        """
        af = _EMA_FAST
        # Running mean and variance via EMA
        self.ema_mean = af * x + (1 - af) * self.ema_mean
        deviation = x - self.ema_mean
        self.ema_sq = af * deviation * deviation + (1 - af) * self.ema_sq
        self.ema_var = self.ema_sq  # convenience alias

        # Lag-1 autocorrelation proxy: EMA of (x_t - mean)(x_{t-1} - mean)
        # normalised by variance so it lives in [-1, 1]
        if self.prev_x is not None:
            prev_dev = self.prev_x - self.ema_mean
            cov = af * (deviation * prev_dev) + (1 - af) * self.ema_autocorr * (self.ema_var + _EPS)
            self.ema_autocorr = cov / (self.ema_var + _EPS)
        self.prev_x = x

        # Slow trend EMAs — when ema_var > ema_var_trend the level is rising
        as_ = _EMA_SLOW
        self.ema_var_trend = as_ * self.ema_var + (1 - as_) * self.ema_var_trend
        self.ema_autocorr_trend = as_ * self.ema_autocorr + (1 - as_) * self.ema_autocorr_trend

        return (
            self.ema_autocorr,
            self.ema_var,
            self.ema_autocorr_trend,
            self.ema_var_trend,
        )


class CriticalSlowingStrategy:
    """Trades regime drift at full size; de-risks when critical slowing is detected."""

    def __init__(self) -> None:
        self._sym_state: dict[str, _SymbolState] = {}

    # ------------------------------------------------------------------ #
    #  signals()                                                           #
    # ------------------------------------------------------------------ #
    def signals(self, state: ManifoldState) -> SignalSet:
        sym = state.symbol
        if sym not in self._sym_state:
            self._sym_state[sym] = _SymbolState()

        ss = self._sym_state[sym]
        autocorr, var_level, autocorr_trend, var_trend = ss.update(state.fwd_return_mean)

        # Rising = current level > slow-trend level (normalised difference)
        autocorr_rise = max(0.0, autocorr - autocorr_trend) / (abs(autocorr_trend) + _EPS)
        var_rise = max(0.0, var_level - var_trend) / (var_trend + _EPS)

        # Warning: both must be rising; clamp to [0, 1]
        warning = min(1.0, max(0.0, autocorr_rise + var_rise))

        confidence = state.regime_prob * (1.0 - state.anomaly_score)
        confidence = max(0.0, min(1.0, confidence))

        sharpe = state.fwd_return_mean / (state.fwd_return_std + _EPS)

        # Pack warning into momentum field so target() can read it via a side-channel;
        # store it directly on the object instead to keep SignalSet fields clean.
        self._sym_state[sym]._warning = warning  # type: ignore[attr-defined]

        return SignalSet(
            ts=state.ts,
            symbol=sym,
            momentum=sharpe,
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    # ------------------------------------------------------------------ #
    #  target()                                                            #
    # ------------------------------------------------------------------ #
    def target(self, signals: SignalSet) -> TargetPosition:
        sym = signals.symbol

        # Flat on anomalous regime or high anomaly
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_THRESH:
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        ss = self._sym_state.get(sym)
        warning = getattr(ss, "_warning", 0.0) if ss is not None else 0.0

        # target_weight = tanh(gain * tanh(sharpe)) * (1 - warning) * confidence
        raw = (
            math.tanh(_GAIN * math.tanh(signals.momentum))
            * (1.0 - warning)
            * signals.confidence
        )

        # Dead-band
        if abs(raw) < _DEAD_BAND:
            raw = 0.0

        weight = max(-1.0, min(1.0, raw))
        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=weight)


def build() -> Strategy:
    """Return a fresh CriticalSlowingStrategy instance."""
    return CriticalSlowingStrategy()
