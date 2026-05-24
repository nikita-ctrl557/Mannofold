"""Kelly-smoothed strategy: fractional-Kelly sizing gated by confidence, then heavy EMA.

Combines kelly_capped (inverse-variance sizing with half-Kelly and tanh squash) with
aggressive_smoothed (heavy per-symbol EMA to cut turnover and raise Sharpe).

Signal: kelly = mu / (sigma^2 + eps), apply half-Kelly, clamp, tanh squash.
Gate:   confidence = regime_prob * (1 - anomaly_score); scale raw weight by confidence.
Smooth: heavy EMA (alpha~0.2) per symbol — slower than kelly_capped (0.25) to further
        reduce turnover and realised vol, targeting a Sharpe improvement.
Flat:   ANOMALY_REGIME or anomaly_score > 0.6; dead-band |w| < 0.04 -> 0.
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

NAME = "kelly_smoothed"
DESCRIPTION = (
    "Fractional-Kelly inverse-variance sizing (half-Kelly, clamped, tanh) gated by "
    "confidence=regime_prob*(1-anomaly_score), then heavy per-symbol EMA smoothing "
    "(alpha~0.2) to reduce turnover and lift Sharpe vs kelly_capped."
)

_EPS = 1e-9
# Fractional Kelly multiplier — half-Kelly is prudent.
_KELLY_FRACTION = 0.5
# Hard cap on |kelly * fraction| before tanh squash.
_KELLY_CAP = 3.0
# tanh gain applied after capping.
_GAIN = 2.0
# Anomaly gate: go flat if anomaly_score exceeds this.
_ANOMALY_GATE = 0.6
# Dead-band: zero out tiny positions.
_DEADBAND = 0.04
# Heavier EMA than kelly_capped (0.25) -> slower, less turnover.
_EMA_ALPHA = 0.20


class KellySmoothedStrategy:
    def __init__(
        self,
        kelly_fraction: float = _KELLY_FRACTION,
        kelly_cap: float = _KELLY_CAP,
        gain: float = _GAIN,
        anomaly_gate: float = _ANOMALY_GATE,
        deadband: float = _DEADBAND,
        ema_alpha: float = _EMA_ALPHA,
    ):
        self._kelly_fraction = kelly_fraction
        self._kelly_cap = kelly_cap
        self._gain = gain
        self._anomaly_gate = anomaly_gate
        self._deadband = deadband
        self._ema_alpha = ema_alpha
        # Per-symbol EMA state.
        self._ema: dict[str, float] = {}

    def signals(self, state: ManifoldState) -> SignalSet:
        # Inverse-variance (Kelly) sizing: mu / sigma^2.
        variance = state.fwd_return_std ** 2 + _EPS
        kelly_raw = state.fwd_return_mean / variance

        # Half-Kelly then hard-cap to control leverage.
        kelly_scaled = self._kelly_fraction * kelly_raw
        kelly_clamped = max(-self._kelly_cap, min(self._kelly_cap, kelly_scaled))

        # Squash to (-1, 1).
        momentum = math.tanh(self._gain * kelly_clamped)

        # Confidence: regime stability attenuated by anomaly proximity.
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
        sym = signals.symbol

        # Hard gates: anomalous regime or high anomaly score -> flat.
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > self._anomaly_gate:
            # Decay EMA toward zero to avoid stale state on re-entry.
            self._ema[sym] = (1.0 - self._ema_alpha) * self._ema.get(sym, 0.0)
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        # Raw weight: kelly-sized momentum gated by confidence.
        raw_weight = signals.momentum * signals.confidence

        # Heavy EMA smoothing to cut turnover and realised vol.
        prev_ema = self._ema.get(sym, 0.0)
        smoothed = self._ema_alpha * raw_weight + (1.0 - self._ema_alpha) * prev_ema
        self._ema[sym] = smoothed

        # Dead-band: avoid noise trades near zero.
        weight = 0.0 if abs(smoothed) < self._deadband else smoothed

        # Hard clip to [-1, 1].
        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=weight)


def build() -> Strategy:
    return KellySmoothedStrategy()
