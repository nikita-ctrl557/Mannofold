"""Steady monthly volatility strategy.

Targets CONSISTENT positive months with low monthly return variance while
remaining net-profitable.  Core idea: vol-control — scale trend exposure
INVERSELY to a per-symbol EMA of fwd_return_std so that position risk is
roughly constant from month to month.  Heavy EMA smoothing on the final
target suppresses turnover and whipsaw, keeping monthly P&L smooth.

Signal:   sharpe  = fwd_return_mean / (fwd_return_std + eps)
          base    = tanh(gain * tanh(sharpe))
Vol ctrl: ema_std tracked per symbol (slow EMA of fwd_return_std)
          vol_scale = clamp(target_risk / (ema_std + eps), 0, 1)
Weight:   base * vol_scale * confidence   [raw]
          smoothed via heavy per-symbol EMA to minimise turnover variance
          confidence = regime_prob * (1 - anomaly_score)
Gates:    flat on ANOMALY_REGIME or anomaly_score > 0.6
          dead-band |w| < 0.04 -> 0
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

NAME = "steady_monthly_vol"
DESCRIPTION = (
    "Month-over-month consistency strategy: vol-control scales trend exposure "
    "inversely to a per-symbol EMA of fwd_return_std so risk is roughly constant "
    "each month; heavy EMA smoothing on the target minimises turnover variance."
)

_EPS = 1e-9
# Moderate-high gain for sufficient conviction once Sharpe is meaningful.
_GAIN = 3.0
# Anomaly gate: go flat above this threshold.
_ANOMALY_GATE = 0.6
# Dead-band: zero tiny positions to avoid noise trading.
_DEADBAND = 0.04
# Slow EMA for fwd_return_std tracking (long memory → stable vol estimate).
_STD_EMA_ALPHA = 0.10
# Heavy smoothing on the final target weight (slow → low monthly variance).
_WEIGHT_EMA_ALPHA = 0.15
# Target per-period risk level; positions are sized so ema_std * weight ≈ target_risk.
_TARGET_RISK = 0.02


class SteadyMonthlyVolStrategy:
    def __init__(
        self,
        gain: float = _GAIN,
        anomaly_gate: float = _ANOMALY_GATE,
        deadband: float = _DEADBAND,
        std_ema_alpha: float = _STD_EMA_ALPHA,
        weight_ema_alpha: float = _WEIGHT_EMA_ALPHA,
        target_risk: float = _TARGET_RISK,
    ):
        self._gain = gain
        self._anomaly_gate = anomaly_gate
        self._deadband = deadband
        self._std_ema_alpha = std_ema_alpha
        self._weight_ema_alpha = weight_ema_alpha
        self._target_risk = target_risk
        # Per-symbol EMA of fwd_return_std for vol-control denominator.
        self._ema_std: dict[str, float] = {}
        # Per-symbol EMA of the final target weight for smoothing.
        self._ema_weight: dict[str, float] = {}

    def signals(self, state: ManifoldState) -> SignalSet:
        sym = state.symbol

        # --- Update per-symbol vol EMA (no lookahead) ---
        prev_std = self._ema_std.get(sym, state.fwd_return_std)
        new_std = self._std_ema_alpha * state.fwd_return_std + (1.0 - self._std_ema_alpha) * prev_std
        self._ema_std[sym] = new_std

        # --- Sharpe-based trend signal ---
        sharpe = state.fwd_return_mean / (state.fwd_return_std + _EPS)
        base = math.tanh(self._gain * math.tanh(sharpe))

        # --- Vol-control scale: inversely proportional to smoothed std ---
        vol_scale = min(1.0, max(0.0, self._target_risk / (new_std + _EPS)))

        # --- Confidence: regime stability attenuated by anomaly ---
        confidence = max(0.0, state.regime_prob * (1.0 - state.anomaly_score))

        # Pack vol-controlled raw signal into momentum field.
        raw_signal = base * vol_scale

        return SignalSet(
            ts=state.ts,
            symbol=sym,
            momentum=raw_signal,
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        sym = signals.symbol

        # Hard gates: anomalous regime or high anomaly score -> flat, decay EMA.
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > self._anomaly_gate:
            self._ema_weight[sym] = (1.0 - self._weight_ema_alpha) * self._ema_weight.get(sym, 0.0)
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        # Raw weight: vol-controlled signal scaled by confidence.
        raw_weight = signals.momentum * signals.confidence

        # Heavy EMA smoothing: minimises month-to-month target variance.
        prev_ema = self._ema_weight.get(sym, 0.0)
        smoothed = self._weight_ema_alpha * raw_weight + (1.0 - self._weight_ema_alpha) * prev_ema
        self._ema_weight[sym] = smoothed

        # Dead-band: suppress noise trades near zero.
        weight = 0.0 if abs(smoothed) < self._deadband else smoothed

        # Hard clip to [-1, 1].
        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=weight)


def build() -> Strategy:
    return SteadyMonthlyVolStrategy()
