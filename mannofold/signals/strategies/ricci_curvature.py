"""Ricci-curvature manifold strategy.

Ollivier-Ricci curvature perspective: NEGATIVE curvature flags fragility and
imminent transitions; POSITIVE curvature flags robustness.

We approximate local manifold curvature from two complementary signals:

  density_change   — how much local density is rising/falling (proxy for whether
                     the state is moving toward or away from a typical region).
  velocity_alignment — cosine similarity between consecutive velocity vectors,
                       clamped to [-1, 1].  High alignment (near +1) means the
                       state is travelling in a straight, robust path; low / negative
                       alignment means the velocity is turning sharply, indicating
                       a fragile, negatively-curved neighbourhood.

  curvature_proxy = density_change + velocity_alignment   (per-symbol)

A positively-curved neighbourhood (high density, straight path) supports trading
the drift at full conviction.  A negatively-curved neighbourhood (density dropping
AND velocity turning) de-risks the position via the curvature multiplier.

  confidence      = regime_prob * (1 − anomaly_score)
  raw_weight      = tanh(gain * tanh(sharpe)) * clamp(0.5 + curvature_proxy, 0, 1) * confidence
  target_weight   = raw_weight with |w| < deadband → 0, then clamp to [-1, 1]
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from mannofold.contracts.interfaces import Strategy
from mannofold.contracts.models import (
    ANOMALY_REGIME,
    ManifoldState,
    SignalSet,
    TargetPosition,
)

NAME = "ricci_curvature"
DESCRIPTION = (
    "Ollivier-Ricci curvature proxy: positive curvature (stable density + straight "
    "velocity path) earns full conviction; negative curvature (dropping density + "
    "sharp velocity turn) de-risks the position."
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
    """Cosine similarity in [-1, 1]; returns 0 on degenerate input."""
    if not a or not b or len(a) != len(b):
        return 0.0
    na, nb = _norm(a), _norm(b)
    if na < _EPS or nb < _EPS:
        return 0.0
    return _dot(a, b) / (na * nb)


class _RicciSignalSet(SignalSet):
    """SignalSet that carries curvature_proxy for target()."""

    curvature_proxy: float = 0.0

    model_config = {"extra": "allow"}


class RicciCurvatureStrategy:
    """Manifold strategy gated by an Ollivier-Ricci curvature proxy."""

    def __init__(
        self,
        gain: float = _GAIN,
        anomaly_gate: float = _ANOMALY_GATE,
        deadband: float = _DEADBAND,
    ) -> None:
        self._gain = gain
        self._anomaly_gate = anomaly_gate
        self._deadband = deadband
        # Per-symbol previous state: (prev_velocity, prev_density)
        self._prev: Dict[str, Tuple[List[float], float]] = {}

    def signals(self, state: ManifoldState) -> _RicciSignalSet:
        sharpe = state.fwd_return_mean / (state.fwd_return_std + _EPS)
        momentum = math.tanh(sharpe)
        confidence = max(0.0, state.regime_prob * (1.0 - state.anomaly_score))

        cur_vel = state.velocity
        cur_density = state.density

        prev_entry = self._prev.get(state.symbol)

        if prev_entry is not None:
            prev_vel, prev_density = prev_entry

            # velocity_alignment: cosine similarity clamped to [-1, 1]
            if len(cur_vel) > 0 and len(prev_vel) == len(cur_vel):
                velocity_alignment = _cosine_similarity(prev_vel, cur_vel)
            else:
                velocity_alignment = 0.0

            # density_change: normalised by a soft scale to stay in [-1, 1] range
            density_scale = max(abs(cur_density), abs(prev_density), _EPS)
            density_change = (cur_density - prev_density) / density_scale
            density_change = max(-1.0, min(1.0, density_change))

        else:
            # No history: neutral curvature estimate
            velocity_alignment = 0.0
            density_change = 0.0

        # curvature_proxy in roughly [-2, 2]; used as offset from 0.5
        curvature_proxy = density_change + velocity_alignment

        # Update per-symbol state (no lookahead)
        if len(cur_vel) > 0:
            self._prev[state.symbol] = (list(cur_vel), cur_density)
        else:
            self._prev[state.symbol] = ([], cur_density)

        return _RicciSignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=momentum,
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
            curvature_proxy=curvature_proxy,
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        # Hard-off: anomalous regime or high anomaly score
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > self._anomaly_gate:
            return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        sharpe_tanh = signals.momentum  # tanh(sharpe) already computed
        confidence = signals.confidence
        curvature_proxy: float = getattr(signals, "curvature_proxy", 0.0)

        # Curvature multiplier: clamp(0.5 + curvature_proxy, 0, 1)
        # positive curvature → multiplier approaching 1.0
        # negative curvature → multiplier approaching 0.0
        curvature_mult = max(0.0, min(1.0, 0.5 + curvature_proxy * 0.5))

        w = math.tanh(self._gain * sharpe_tanh) * curvature_mult * confidence

        # Dead-band
        if abs(w) < self._deadband:
            w = 0.0

        w = max(-1.0, min(1.0, w))
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=w)


def build() -> Strategy:
    """Construct and return a RicciCurvatureStrategy with default parameters."""
    return RicciCurvatureStrategy()
