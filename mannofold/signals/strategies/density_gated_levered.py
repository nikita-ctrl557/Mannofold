"""Density-gated levered strategy.

Extends density_gated by applying leverage in the highest-conviction states:
when density gate AND confidence are both high, the base signal is scaled up
(×1.5–2.0), then clamped to [-1, 1]. In marginal states the strategy stays
conservative, matching density_gated behaviour. Flat on ANOMALY_REGIME or
anomaly > 0.6.
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

NAME = "density_gated_levered"
DESCRIPTION = (
    "Density-gated strategy with conviction-proportional leverage: "
    "scales up position size in high-density, high-confidence states "
    "to lift returns while preserving risk-adjusted edge."
)

_EPS = 1e-9
# Density gate sigmoid parameters (identical to density_gated).
_DENSITY_MID = 1.0
_DENSITY_SCALE = 2.0
_DENSITY_CLAMP = 50.0

# Leverage thresholds — applied to the final weight before clamping.
_HIGH_DENSITY_THRESH = 0.70   # density_gate >= this -> eligible for leverage
_HIGH_CONF_THRESH = 0.55      # confidence >= this -> apply max leverage
_LOW_CONF_THRESH = 0.30       # confidence < this -> stay at 1× (no leverage)
_MAX_LEVERAGE = 1.8           # multiplier in highest-conviction bucket
_MID_LEVERAGE = 1.35          # multiplier in mid-conviction bucket


def _density_gate(density: float) -> float:
    """Smooth gate in [0, 1]: low density -> ~0, high density -> ~1."""
    d = max(0.0, min(density, _DENSITY_CLAMP))
    return 1.0 / (1.0 + math.exp(-_DENSITY_SCALE * (d - _DENSITY_MID)))


class DensityGatedLeveredStrategy:
    """Gate conviction on manifold density; apply leverage in high-conviction states."""

    def __init__(
        self,
        gain: float = 60.0,
        anomaly_gate: float = 0.6,
        deadband: float = 0.04,
    ):
        self._gain = gain
        self._anomaly_gate = anomaly_gate
        self._deadband = deadband

    def signals(self, state: ManifoldState) -> SignalSet:
        sharpe = state.fwd_return_mean / (state.fwd_return_std + _EPS)
        gate = _density_gate(state.density)
        confidence = max(0.0, state.regime_prob * (1.0 - state.anomaly_score) * gate)
        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=math.tanh(sharpe),
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        # Hard gates: anomalous regime or high anomaly -> flat.
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > self._anomaly_gate:
            return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        # Recover density gate value from confidence + regime_prob + (1-anomaly).
        # confidence = regime_prob * (1 - anomaly) * gate => gate = conf / denom
        denom = max(signals.confidence, _EPS)  # gate is already embedded
        sharpe_proxy = signals.expected_return / (signals.momentum if abs(signals.momentum) > _EPS else 1.0)
        base = math.tanh(self._gain * math.tanh(sharpe_proxy))

        # Base weight (same formula as density_gated).
        weight = base * signals.confidence

        # Determine the leverage multiplier based on density gate and confidence.
        # Re-derive a density_gate proxy: confidence embeds gate; compare against thresholds.
        # Use confidence directly as proxy for combined density+conviction quality.
        if signals.confidence >= _HIGH_DENSITY_THRESH and signals.confidence >= _HIGH_CONF_THRESH:
            leverage = _MAX_LEVERAGE
        elif signals.confidence >= _LOW_CONF_THRESH:
            # Linearly interpolate leverage between 1.0 and MAX in the middle band.
            t = (signals.confidence - _LOW_CONF_THRESH) / (_HIGH_CONF_THRESH - _LOW_CONF_THRESH + _EPS)
            t = min(1.0, t)
            leverage = 1.0 + t * (_MID_LEVERAGE - 1.0)
        else:
            leverage = 1.0

        weight *= leverage

        # Dead-band: collapse negligible weights to flat.
        if abs(weight) < self._deadband:
            weight = 0.0

        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=weight)


def build() -> Strategy:
    return DensityGatedLeveredStrategy()
