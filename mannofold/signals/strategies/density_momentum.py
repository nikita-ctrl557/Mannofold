"""Density-momentum strategy: neighbourhood momentum gated by manifold density.

Follows neighbourhood Sharpe momentum (base = tanh(gain * tanh(sharpe))) but
multiplies by BOTH a density typicality gate in [0,1] AND confidence derived
from regime_prob * (1 - anomaly_score).  Bets are only taken where the manifold
neighbourhood is densely sampled (reliable statistics).  Flat on ANOMALY_REGIME
or anomaly_score above the threshold.
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

NAME = "density_momentum"
DESCRIPTION = (
    "Neighbourhood-Sharpe momentum scaled by density typicality gate and regime confidence; flat on anomaly."
)

_EPS = 1e-9

# Momentum gain: amplifier inside the outer tanh
_GAIN = 2.5

# Density gate sigmoid parameters
_DENSITY_MID = 1.0     # density value at which gate = 0.5
_DENSITY_SCALE = 2.0   # steepness of the logistic
_DENSITY_CLAMP = 50.0  # defensive upper clamp (density can be unbounded)

# Thresholds
_ANOMALY_THRESH = 0.6  # anomaly_score above this -> flat
_DEAD_BAND = 0.04      # collapse |weight| below this to 0


def _density_gate(density: float) -> float:
    """Smooth logistic gate in [0, 1]: low density -> ~0, high density -> ~1."""
    d = max(0.0, min(density, _DENSITY_CLAMP))
    return 1.0 / (1.0 + math.exp(-_DENSITY_SCALE * (d - _DENSITY_MID)))


class DensityMomentumStrategy:
    """Momentum bets restricted to densely sampled manifold neighbourhoods."""

    def __init__(
        self,
        gain: float = _GAIN,
        anomaly_gate: float = _ANOMALY_THRESH,
        deadband: float = _DEAD_BAND,
    ) -> None:
        self._gain = gain
        self._anomaly_gate = anomaly_gate
        self._deadband = deadband

    def signals(self, state: ManifoldState) -> SignalSet:
        sharpe = state.fwd_return_mean / (state.fwd_return_std + _EPS)
        momentum = math.tanh(sharpe)

        # Density gate: rewards typical (dense) manifold regions
        gate = _density_gate(state.density)

        # Confidence fuses regime stability, low anomaly, and density gate
        confidence = max(0.0, min(1.0, state.regime_prob * (1.0 - state.anomaly_score) * gate))

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
        # Hard gates: anomalous regime or high anomaly score -> flat
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > self._anomaly_gate:
            return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        # base = tanh(gain * tanh(sharpe)); momentum == tanh(sharpe)
        base = math.tanh(self._gain * signals.momentum)

        # Full weight: base * confidence (confidence already encodes density gate)
        weight = base * signals.confidence

        # Dead-band: suppress small, noisy weights
        if abs(weight) < self._deadband:
            weight = 0.0

        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=weight)


def build() -> Strategy:
    """Return a fresh DensityMomentumStrategy instance."""
    return DensityMomentumStrategy()
