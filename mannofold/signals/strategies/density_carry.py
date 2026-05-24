"""Density-carry strategy — evolutionary cross of density_gated and vol_scaled_carry.

Harvests inverse-variance carry (vol_scaled_carry) but gates the entire position
by a manifold-density typicality factor (density_gated).  The density gate collapses
positions in sparse / off-manifold regions, trading the raw carry return for lower
realised volatility and a better Sharpe than either parent.

Confidence fuses regime stability and low anomaly (like both parents) PLUS the
density gate is applied to the confidence itself, so the position scales
continuously from zero (off-manifold) to full (dense / typical regions).
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

NAME = "density_carry"
DESCRIPTION = (
    "Inverse-variance carry sized by manifold density gate for improved Sharpe."
)

_EPS = 1e-9

# --- Inverse-variance carry parameters (from vol_scaled_carry) ---
_RAW_CLAMP = 10.0   # clamp on weight_raw before tanh to prevent saturation
_GAIN = 3.0         # tanh gain applied to clamped carry weight

# --- Density gate parameters (from density_gated) ---
_DENSITY_MID = 1.0    # density value at which gate = 0.5
_DENSITY_SCALE = 2.0  # logistic steepness
_DENSITY_CLAMP = 50.0 # defensive upper clamp (density can be unbounded)

# --- Common thresholds ---
_ANOMALY_GATE = 0.6  # anomaly_score above this -> flat
_DEADBAND = 0.04     # |weight| below this -> 0 (avoid noise trades)


def _density_gate(density: float) -> float:
    """Smooth sigmoid gate in [0, 1]: low density -> ~0, high density -> ~1."""
    d = max(0.0, min(density, _DENSITY_CLAMP))
    return 1.0 / (1.0 + math.exp(-_DENSITY_SCALE * (d - _DENSITY_MID)))


class DensityCarryStrategy:
    """Density-gated inverse-variance carry strategy."""

    def __init__(
        self,
        gain: float = _GAIN,
        anomaly_gate: float = _ANOMALY_GATE,
        deadband: float = _DEADBAND,
    ):
        self._gain = gain
        self._anomaly_gate = anomaly_gate
        self._deadband = deadband

    def signals(self, state: ManifoldState) -> SignalSet:
        # Inverse-variance carry term (vol_scaled_carry core).
        variance = state.fwd_return_std ** 2 + _EPS
        weight_raw = state.fwd_return_mean / variance
        weight_raw = max(-_RAW_CLAMP, min(_RAW_CLAMP, weight_raw))
        momentum = math.tanh(self._gain * weight_raw)

        # Density gate (density_gated core): collapses in sparse regions.
        gate = _density_gate(state.density)

        # Confidence: regime stability * low anomaly * density typicality.
        confidence = max(0.0, state.regime_prob * (1.0 - state.anomaly_score) * gate)

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

        # momentum = tanh(gain * carry_raw); scale by density-fused confidence.
        weight = signals.momentum * signals.confidence

        # Dead-band: collapse negligible weights to flat.
        if abs(weight) < self._deadband:
            weight = 0.0

        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=weight)


def build() -> Strategy:
    return DensityCarryStrategy()
