"""Geodesic-flow manifold strategy.

A state moving along a GEODESIC (straight, low-curvature path) on the manifold
is in a stable, persistent regime — trade with it.  Path curvature is estimated
from the change in velocity DIRECTION between consecutive steps: a low turning
angle means the trajectory is nearly geodesic (trustworthy); high curvature
means the manifold is bending and the regime may be turning.

straightness  = cosine-similarity of consecutive velocity vectors, clamped [0,1]
target_weight = sign(fwd_return_mean) · tanh(gain · |tanh(sharpe)|)
                · straightness · confidence
confidence    = regime_prob · (1 − anomaly_score)
"""

from __future__ import annotations

import math
from typing import Dict, List

from mannofold.contracts.interfaces import Strategy
from mannofold.contracts.models import (
    ANOMALY_REGIME,
    ManifoldState,
    SignalSet,
    TargetPosition,
)

NAME = "geodesic_flow"
DESCRIPTION = (
    "Scales conviction by manifold path straightness: near-geodesic trajectories "
    "(low curvature) earn full weight; high-curvature regime turns are faded."
)

_EPS = 1e-9
_GAIN = 60.0
_ANOMALY_GATE = 0.6
_DEADBAND = 0.04


def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(v: List[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors; returns 0 on degenerate input."""
    if not a or not b or len(a) != len(b):
        return 0.0
    na, nb = _norm(a), _norm(b)
    if na < _EPS or nb < _EPS:
        return 0.0
    return _dot(a, b) / (na * nb)


class _GeodesicSignalSet(SignalSet):
    """SignalSet subclass that carries path-straightness for target()."""

    straightness: float = 0.0

    model_config = {"extra": "allow"}


class GeodesicFlowStrategy:
    """Manifold strategy that gates conviction on geodesic straightness."""

    def __init__(
        self,
        gain: float = _GAIN,
        anomaly_gate: float = _ANOMALY_GATE,
        deadband: float = _DEADBAND,
    ) -> None:
        self._gain = gain
        self._anomaly_gate = anomaly_gate
        self._deadband = deadband
        # Per-symbol previous velocity (tangent vector).
        self._prev_velocity: Dict[str, List[float]] = {}

    def signals(self, state: ManifoldState) -> _GeodesicSignalSet:
        """Compute signals; estimate straightness from consecutive velocities."""
        sharpe = state.fwd_return_mean / (state.fwd_return_std + _EPS)
        momentum = math.tanh(sharpe)
        confidence = max(0.0, state.regime_prob * (1.0 - state.anomaly_score))

        prev = self._prev_velocity.get(state.symbol)
        cur = state.velocity

        if prev is not None and len(cur) > 0 and len(prev) == len(cur):
            # cosine similarity in [-1,1]; clamp to [0,1] so anti-parallel
            # trajectories (reversal) also read as low-straightness.
            cos_sim = _cosine_similarity(prev, cur)
            straightness = max(0.0, cos_sim)
        else:
            # No previous velocity available — treat as uncertain (mid-range).
            straightness = 0.5

        # Update per-symbol previous velocity (no lookahead — current step only).
        if len(cur) > 0:
            self._prev_velocity[state.symbol] = list(cur)

        return _GeodesicSignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=momentum,
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
            straightness=straightness,
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        """Compute target weight, scaled by geodesic straightness."""
        # Hard-off: anomalous regime or high anomaly score.
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > self._anomaly_gate:
            return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        sharpe_tanh = signals.momentum  # tanh(sharpe) from signals()
        confidence = signals.confidence
        straightness = getattr(signals, "straightness", 0.5)

        # Directional component: sign(fwd_return_mean) · tanh(gain · |tanh(sharpe)|)
        direction = math.copysign(1.0, sharpe_tanh) if abs(sharpe_tanh) > _EPS else 0.0
        magnitude = math.tanh(self._gain * abs(sharpe_tanh))

        w = direction * magnitude * straightness * confidence

        # Dead-band.
        if abs(w) < self._deadband:
            w = 0.0

        w = max(-1.0, min(1.0, w))
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=w)


def build() -> Strategy:
    """Construct and return a GeodesicFlowStrategy with default parameters."""
    return GeodesicFlowStrategy()
