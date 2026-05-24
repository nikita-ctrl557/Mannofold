"""Density-gated strategy variant.

Only trusts neighbourhood statistics in HIGH-DENSITY (typical) regions of the
manifold. In sparse / off-manifold regions the density gate collapses the
position toward flat, regardless of the apparent expected return.
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

NAME = "density_gated"
DESCRIPTION = "Trust signals only in high-density manifold regions; low density -> flat."

_EPS = 1e-9
# Logistic scale parameters for the density gate.
_DENSITY_MID = 1.0    # density value at which gate = 0.5
_DENSITY_SCALE = 2.0  # steepness of the sigmoid
_DENSITY_CLAMP = 50.0 # defensive upper clamp (density can be unbounded)


def _density_gate(density: float) -> float:
    """Smooth gate in [0, 1]: low density -> ~0, high density -> ~1."""
    d = max(0.0, min(density, _DENSITY_CLAMP))
    return 1.0 / (1.0 + math.exp(-_DENSITY_SCALE * (d - _DENSITY_MID)))


class DensityGatedStrategy:
    """Gate conviction on manifold density so only typical states trade."""

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
        # Confidence fuses regime stability, low anomaly, and density gate.
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
        # Hard gates: anomalous regime or high anomaly score -> flat.
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > self._anomaly_gate:
            return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        # sharpe = fwd_return_mean / (fwd_return_std + eps); base = tanh(gain*tanh(sharpe))
        # momentum == tanh(sharpe), so invert to recover the Sharpe proxy.
        sharpe_proxy = signals.expected_return / (signals.momentum if abs(signals.momentum) > _EPS else 1.0)
        base = math.tanh(self._gain * math.tanh(sharpe_proxy))

        # target_weight = base * density_gate * confidence
        # (density_gate is already embedded in confidence from signals())
        weight = base * signals.confidence

        # Dead-band: collapse negligible weights to flat.
        if abs(weight) < self._deadband:
            weight = 0.0

        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=weight)


def build() -> Strategy:
    return DensityGatedStrategy()
