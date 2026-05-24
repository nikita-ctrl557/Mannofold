"""Kelly-capped strategy: fractional-Kelly inverse-variance sizing with strict risk caps.

Refinement of vol_scaled_carry: applies a Kelly fraction and a hard cap on the raw
Kelly ratio BEFORE squashing, then EMA-smooths per-symbol target weights to reduce
turnover and realised volatility — raising Sharpe vs the base strategy.

Kelly criterion: f* = mu / sigma^2
Apply fraction k_f and cap |f*| at cap -> tanh squash -> confidence scale -> EMA -> deadband.
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

NAME = "kelly_capped"
DESCRIPTION = "Fractional-Kelly inverse-variance sizing with hard cap, EMA smoothing, and strict risk gates for improved Sharpe."

_EPS = 1e-9
# Fractional Kelly multiplier (half-Kelly is a common prudent choice).
_KELLY_FRACTION = 0.5
# Hard cap on |kelly * fraction| before tanh squash — prevents extreme leverage.
_KELLY_CAP = 3.0
# tanh gain: steeper than vol_scaled_carry but applied after capping.
_GAIN = 2.0
# Anomaly gate: go flat if anomaly_score exceeds this.
_ANOMALY_GATE = 0.6
# Dead-band: zero out tiny positions to avoid noise trading.
_DEADBAND = 0.04
# EMA decay per step (higher = smoother, slower to react).
_EMA_ALPHA = 0.25  # new_ema = alpha * raw + (1 - alpha) * prev_ema


class KellycappedStrategy:
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
        # Per-symbol EMA state for target weight smoothing.
        self._ema: dict[str, float] = {}

    def signals(self, state: ManifoldState) -> SignalSet:
        # Raw Kelly ratio: mu / sigma^2 (inverse-variance sizing).
        variance = state.fwd_return_std ** 2 + _EPS
        kelly_raw = state.fwd_return_mean / variance

        # Apply Kelly fraction then hard-cap to control leverage.
        kelly_scaled = self._kelly_fraction * kelly_raw
        kelly_clamped = max(-self._kelly_cap, min(self._kelly_cap, kelly_scaled))

        # Squash to (-1, 1) with gain.
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

        # Raw weight: tanh(gain * clamped_kelly) * confidence.
        raw_weight = signals.momentum * signals.confidence

        # EMA smoothing to cut turnover and realised vol.
        prev_ema = self._ema.get(sym, 0.0)
        smoothed = self._ema_alpha * raw_weight + (1.0 - self._ema_alpha) * prev_ema
        self._ema[sym] = smoothed

        # Dead-band: avoid noise trades near zero.
        weight = 0.0 if abs(smoothed) < self._deadband else smoothed

        # Hard clip to [-1, 1].
        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=weight)


def build() -> Strategy:
    return KellycappedStrategy()
