"""Density-aggressive strategy: high-gain trend conviction gated by manifold density.

Combines:
- ``density_gated`` (#1): typicality gate — only trusts signals in dense/typical
  manifold regions; sparse/off-manifold -> flat.
- ``aggressive_smoothed`` (#2): high-gain double-tanh conviction with heavy
  per-symbol EMA smoothing to cut turnover and whipsaw.

CONCEPT: aggressive where the manifold neighbourhood is dense and reliable,
quiet elsewhere.  confidence = regime_prob * (1 - anomaly_score); density factor
is a smooth sigmoid gate in [0, 1].  Flat on ANOMALY_REGIME or anomaly > ~0.6.
Dead-band |w| < 0.04 -> 0; per-symbol EMA, no lookahead; build() takes no args.
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

NAME = "density_aggressive"
DESCRIPTION = (
    "High-gain trend conviction (gain~4) with heavy per-symbol EMA smoothing "
    "(alpha~0.2), gated by manifold density typicality. Aggressive where density "
    "is high/reliable, quiet in sparse or anomalous regions."
)

_EPS = 1e-9
_GAIN = 4.0          # double-tanh outer gain: drives near-binary conviction
_ANOMALY_GATE = 0.6  # flat if anomaly_score exceeds this
_DEADBAND = 0.04     # zero out tiny weights
_EMA_ALPHA = 0.2     # EMA smoothing: new = alpha*raw + (1-alpha)*prev

# Density gate logistic parameters (from density_gated)
_DENSITY_MID = 1.0    # density value at which gate = 0.5
_DENSITY_SCALE = 2.0  # steepness of sigmoid
_DENSITY_CLAMP = 50.0 # defensive upper clamp


def _density_gate(density: float) -> float:
    """Smooth sigmoid gate in [0, 1]: low density -> ~0, high density -> ~1."""
    d = max(0.0, min(density, _DENSITY_CLAMP))
    return 1.0 / (1.0 + math.exp(-_DENSITY_SCALE * (d - _DENSITY_MID)))


class DensityAggressiveStrategy:
    """Aggressive trend following gated by manifold neighbourhood density."""

    def __init__(
        self,
        gain: float = _GAIN,
        anomaly_gate: float = _ANOMALY_GATE,
        deadband: float = _DEADBAND,
        ema_alpha: float = _EMA_ALPHA,
    ):
        self._gain = gain
        self._anomaly_gate = anomaly_gate
        self._deadband = deadband
        self._ema_alpha = ema_alpha
        self._ema: dict[str, float] = {}

    def signals(self, state: ManifoldState) -> SignalSet:
        std = state.fwd_return_std + _EPS
        sharpe = state.fwd_return_mean / std

        # High-gain double-tanh: inner squashes Sharpe, outer amplifies conviction
        momentum = math.tanh(self._gain * math.tanh(sharpe))

        # Density typicality gate [0,1]: dense/typical regions -> ~1, sparse -> ~0
        gate = _density_gate(state.density)

        # Confidence: regime stability * (1 - anomaly) * density gate
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
        sym = signals.symbol

        # Hard gates: anomalous regime or high anomaly score -> flat
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > self._anomaly_gate:
            self._ema[sym] = (1.0 - self._ema_alpha) * self._ema.get(sym, 0.0)
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        # Raw weight: high-gain momentum scaled by density-gated confidence
        raw_weight = signals.momentum * signals.confidence

        # Heavy EMA smoothing (alpha~0.2) to cut turnover
        prev_ema = self._ema.get(sym, 0.0)
        smoothed = self._ema_alpha * raw_weight + (1.0 - self._ema_alpha) * prev_ema
        self._ema[sym] = smoothed

        weight = 0.0 if abs(smoothed) < self._deadband else smoothed
        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=weight)


def build() -> Strategy:
    return DensityAggressiveStrategy()
