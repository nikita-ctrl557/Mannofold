"""Kelly-half-density strategy: half-Kelly inverse-variance sizing gated by manifold density.

A lean cross of kelly_capped and density_gated:
  - Half-Kelly inverse-variance sizing: kelly = mu / (sigma^2 + eps), clamped to [-cap, cap]
  - Base signal: tanh(gain * 0.5 * kelly)
  - Density typicality gate: logistic function of local neighbourhood density in [0, 1]
  - Confidence gate: regime_prob * (1 - anomaly_score)
  - Final weight: base * density_gate * confidence -> dead-band -> clamp
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

NAME = "kelly_half_density"
DESCRIPTION = (
    "Half-Kelly inverse-variance sizing multiplied by a logistic density typicality gate "
    "and a regime-confidence weight; flat in anomalous regimes or low-density regions."
)

_EPS = 1e-9
# Half-Kelly: fraction applied before cap (0.5 is the classic prudent choice).
_KELLY_FRACTION = 0.5
# Hard cap on |kelly * fraction| before tanh squash.
_KELLY_CAP = 3.0
# tanh gain applied after kelly fraction.
_GAIN = 2.0
# Anomaly gate threshold.
_ANOMALY_GATE = 0.6
# Dead-band: collapse tiny weights to zero to avoid noise trading.
_DEADBAND = 0.04
# Density gate logistic parameters (density=0 -> gate~0, high density -> gate->1).
_DENSITY_MID = 1.0    # density value at which gate = 0.5
_DENSITY_SCALE = 2.0  # steepness of the sigmoid
_DENSITY_CLAMP = 50.0 # defensive upper clamp (density can be unbounded)


def _density_gate(density: float) -> float:
    """Smooth gate in [0, 1]: low density -> ~0, high density -> ~1."""
    d = max(0.0, min(density, _DENSITY_CLAMP))
    return 1.0 / (1.0 + math.exp(-_DENSITY_SCALE * (d - _DENSITY_MID)))


class KellyHalfDensityStrategy:
    """Half-Kelly sizing gated by manifold density typicality and regime confidence."""

    def signals(self, state: ManifoldState) -> SignalSet:
        # Half-Kelly inverse-variance ratio: mu / sigma^2, fraction already 0.5.
        variance = state.fwd_return_std ** 2 + _EPS
        kelly_raw = state.fwd_return_mean / variance
        kelly_scaled = _KELLY_FRACTION * kelly_raw
        kelly_clamped = max(-_KELLY_CAP, min(_KELLY_CAP, kelly_scaled))

        # Base signal: tanh squash with gain applied to half-Kelly.
        base = math.tanh(_GAIN * kelly_clamped)

        # Density typicality gate: logistic of neighbourhood density.
        gate = _density_gate(state.density)

        # Confidence: regime stability attenuated by anomaly proximity.
        confidence = max(0.0, state.regime_prob * (1.0 - state.anomaly_score))

        # Embed density gate into momentum; confidence stored separately.
        momentum = base * gate

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
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_GATE:
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        # weight = base_density_signal * confidence
        weight = signals.momentum * signals.confidence

        # Dead-band: collapse negligible weights to flat to avoid noise trades.
        if abs(weight) < _DEADBAND:
            weight = 0.0

        # Hard clip to [-1, 1].
        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=weight)


def build() -> Strategy:
    return KellyHalfDensityStrategy()
