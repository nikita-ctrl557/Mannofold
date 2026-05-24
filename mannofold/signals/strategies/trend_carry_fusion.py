"""Trend-Carry Fusion strategy.

Combines the DIRECTION signal from long_bias_trend (highest return, +355%)
with the MAGNITUDE sizing from kelly_capped carry.

Direction: sign of the trend drift (Sharpe proxy = fwd_return_mean / (fwd_return_std + eps))
Magnitude: Kelly inverse-variance |kelly| = |fwd_return_mean| / (fwd_return_std^2 + eps),
           clamped and squashed via tanh(gain * |kelly|).
Target weight: direction * tanh(gain * |kelly|) * confidence
Confidence: regime_prob * (1 - anomaly_score)

Flat on ANOMALY_REGIME or anomaly_score > 0.6.
Dead-band: |weight| < 0.04 collapses to zero.
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

NAME = "trend_carry_fusion"
DESCRIPTION = (
    "Fuses long_bias_trend direction (Sharpe sign) with kelly_capped magnitude "
    "(inverse-variance Kelly sizing, tanh-squashed), scaled by regime confidence."
)

_EPS = 1e-9
_GAIN = 2.0           # tanh gain applied to clamped Kelly magnitude
_KELLY_CAP = 3.0      # hard cap on |kelly_raw| before squash
_ANOMALY_GATE = 0.6   # anomaly_score threshold above which we go flat
_DEADBAND = 0.04      # |weight| below this collapses to zero


class TrendCarryFusionStrategy:
    """Fusion of trend direction and Kelly-carry magnitude.

    Step 1 – Direction (from long_bias_trend):
        sharpe     = fwd_return_mean / (fwd_return_std + eps)
        direction  = sign(sharpe)  (+1 / -1 / 0)

    Step 2 – Magnitude (from kelly_capped):
        kelly_raw  = |fwd_return_mean| / (fwd_return_std^2 + eps)
        kelly_clmp = clamp(kelly_raw, 0, _KELLY_CAP)
        magnitude  = tanh(_GAIN * kelly_clmp)

    Step 3 – Confidence gate:
        confidence = regime_prob * (1 - anomaly_score)

    Step 4 – Combined weight:
        weight = direction * magnitude * confidence

    Step 5 – Risk gates and dead-band applied.
    """

    def __init__(
        self,
        gain: float = _GAIN,
        kelly_cap: float = _KELLY_CAP,
        anomaly_gate: float = _ANOMALY_GATE,
        deadband: float = _DEADBAND,
    ):
        self._gain = gain
        self._kelly_cap = kelly_cap
        self._anomaly_gate = anomaly_gate
        self._deadband = deadband

    def signals(self, state: ManifoldState) -> SignalSet:
        sharpe = state.fwd_return_mean / (state.fwd_return_std + _EPS)
        variance = state.fwd_return_std ** 2 + _EPS
        kelly_raw = abs(state.fwd_return_mean) / variance
        kelly_clamped = min(self._kelly_cap, kelly_raw)
        magnitude = math.tanh(self._gain * kelly_clamped)
        direction = math.copysign(1.0, sharpe) if sharpe != 0.0 else 0.0
        momentum = direction * magnitude
        confidence = max(0.0, state.regime_prob * (1.0 - state.anomaly_score))
        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=momentum,
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        # Hard gates: anomalous regime or high anomaly score -> flat.
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > self._anomaly_gate:
            return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        weight = signals.momentum * signals.confidence

        # Dead-band: collapse negligible weights to flat.
        if abs(weight) < self._deadband:
            weight = 0.0

        # Hard clip to [-1, 1].
        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=weight)


def build() -> Strategy:
    return TrendCarryFusionStrategy()
