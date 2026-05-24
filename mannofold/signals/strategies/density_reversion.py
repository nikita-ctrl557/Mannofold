"""Density-reversion strategy variant.

In HIGH-density (typical) manifold regions, FOLLOW the neighbourhood drift:
the crowd is right when the market is in a well-trodden state.

In LOW-density (atypical) regions, FADE the drift: extreme / rare states tend
to mean-revert back toward the manifold core.

The direction factor is a smooth logistic of density, so the transition between
follow and fade is continuous rather than binary.

target_weight = dir_factor * tanh(gain * tanh(sharpe)) * confidence

where:
  dir_factor  = 2*sigmoid(k*(density - d0)) - 1   in [-1, 1]
  confidence  = regime_prob * (1 - anomaly_score)
  sharpe      = fwd_return_mean / (fwd_return_std + eps)
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

NAME = "density_reversion"
DESCRIPTION = (
    "Follow drift in high-density (typical) states; fade it in low-density "
    "(atypical) states via a smooth logistic direction factor."
)

_EPS = 1e-9

# Outer gain applied to tanh(sharpe).
_GAIN = 3.0

# Logistic parameters for the direction factor.
# d0: density pivot where dir_factor == 0 (neutral between follow and fade).
# k:  steepness — higher means sharper transition.
_DENSITY_PIVOT = 1.0
_DENSITY_K = 2.0

# Hard gates.
_ANOMALY_GATE = 0.6
_DEADBAND = 0.04
_DENSITY_CLAMP = 50.0


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _dir_factor(density: float) -> float:
    """Smooth direction factor in [-1, 1].

    density >> d0  ->  +1  (follow drift)
    density  ~ d0  ->   0  (neutral)
    density << d0  ->  -1  (fade drift)
    """
    d = max(0.0, min(density, _DENSITY_CLAMP))
    return 2.0 * _sigmoid(_DENSITY_K * (d - _DENSITY_PIVOT)) - 1.0


class DensityReversionStrategy:
    """Blend follow/fade signal based on manifold density.

    signals() is always called by the engine immediately before target(), so
    stashing density on self involves no lookahead.
    """

    def __init__(
        self,
        gain: float = _GAIN,
        anomaly_gate: float = _ANOMALY_GATE,
        deadband: float = _DEADBAND,
        density_pivot: float = _DENSITY_PIVOT,
        density_k: float = _DENSITY_K,
    ) -> None:
        self._gain = gain
        self._anomaly_gate = anomaly_gate
        self._deadband = deadband
        self._density_pivot = density_pivot
        self._density_k = density_k
        self._last_density: float = 0.0

    def signals(self, state: ManifoldState) -> SignalSet:
        # Stash density so target() can use it without lookahead.
        self._last_density = state.density
        sharpe = state.fwd_return_mean / (state.fwd_return_std + _EPS)
        momentum = math.tanh(sharpe)
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

        # Density-conditioned direction: +1 follow, -1 fade.
        dir_f = _dir_factor(self._last_density)

        # base signal: tanh(gain * tanh(sharpe)); momentum == tanh(sharpe).
        base = math.tanh(self._gain * signals.momentum)

        weight = dir_f * base * signals.confidence

        # Dead-band: suppress micro positions.
        if abs(weight) < self._deadband:
            weight = 0.0

        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=weight)


def build() -> Strategy:
    return DensityReversionStrategy()
