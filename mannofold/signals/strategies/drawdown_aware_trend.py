"""Drawdown-aware trend-following strategy.

Trend conviction = tanh(GAIN * tanh(sharpe)).  Maintains a per-symbol
synthetic 'signal-equity' proxy that compounds by (prev_weight * drift)
each step, tracks its running peak, and computes a proxy drawdown.
Exposure is throttled as the proxy drawdown deepens.
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

NAME = "drawdown_aware_trend"
DESCRIPTION = (
    "Trend-follow with a synthetic equity proxy that throttles exposure "
    "as the per-symbol drawdown deepens; flat on anomalous regimes."
)

# Tunable knobs
_GAIN = 3.0            # outer tanh amplifier: tanh(GAIN * tanh(sharpe))
_ANOMALY_THRESH = 0.6  # anomaly_score above this -> flat immediately
_DEAD_BAND = 0.04      # collapse |weight| below this to 0
_MAXDD_FLOOR = -0.20   # drawdown at which risk_factor reaches 0  (e.g. -20 %)


class DrawdownAwareTrendStrategy:
    """Trend-follow with per-symbol synthetic drawdown throttle."""

    def __init__(self) -> None:
        # per-symbol synthetic equity proxy state
        self._equity: dict[str, float] = {}   # current proxy equity level
        self._peak: dict[str, float] = {}     # running peak of proxy equity
        self._prev_weight: dict[str, float] = {}  # weight used LAST step

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

        # --- update synthetic equity proxy (strictly past-only) ---
        # realized drift proxy: the signal we observed NOW is the drift
        # that the PREVIOUS weight was exposed to
        drift = signals.expected_return          # now-observed neighbourhood drift
        prev_w = self._prev_weight[sym]
        self._equity[sym] *= 1.0 + prev_w * drift
        # guard against equity going non-positive
        if self._equity[sym] <= 0.0:
            self._equity[sym] = 1e-8

        # update running peak
        if self._equity[sym] > self._peak[sym]:
            self._peak[sym] = self._equity[sym]

        # proxy drawdown (0 at peak, negative when below)
        dd = self._equity[sym] / self._peak[sym] - 1.0

        # --- flat unconditionally on anomalous regime or high anomaly ---
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_THRESH:
            self._prev_weight[sym] = 0.0
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        # --- trend conviction ---
        sharpe = signals.momentum
        conviction = math.tanh(_GAIN * math.tanh(sharpe))

        # --- drawdown throttle ---
        # risk_factor = clamp(1 + dd / |maxdd_floor|, 0, 1)
        # = 1 when dd=0, = 0 when dd <= maxdd_floor
        risk_factor = max(0.0, min(1.0, 1.0 + dd / abs(_MAXDD_FLOOR)))

        # --- final weight ---
        raw = conviction * risk_factor * signals.confidence

        # dead-band
        if abs(raw) < _DEAD_BAND:
            raw = 0.0

        self._prev_weight[sym] = raw
        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=raw)


def build() -> Strategy:
    """Return a fresh DrawdownAwareTrendStrategy instance."""
    return DrawdownAwareTrendStrategy()
