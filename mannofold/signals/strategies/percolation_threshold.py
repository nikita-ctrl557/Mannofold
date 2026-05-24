"""Percolation-threshold strategy.

Statistical-physics framing: below a critical occupation probability p_c there
is no spanning cluster (no coherent trend); above p_c a giant connected
component emerges abruptly — a real, tradeable trend.

Occupation probability:  p = regime_prob * density_norm
Order parameter (∝ (p-p_c)^β):  clamp(((p-p_c)/(1-p_c))**0.5, 0, 1)
Target weight = tanh(gain*tanh(sharpe)) * order_param * (1-anomaly_score)
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

NAME = "percolation_threshold"
DESCRIPTION = (
    "Percolation / critical-phase-transition strategy: exposure is zero below the "
    "critical occupation probability p_c (~0.55) and grows as the order parameter "
    "(p-p_c)^beta above it, capturing only coherent-cluster (trending) regimes."
)

_EPS = 1e-9
_P_C = 0.55          # critical percolation threshold (2-D square lattice ≈ 0.593)
_BETA = 0.5          # mean-field order-parameter exponent (β = 0.5)
_GAIN = 60.0         # tanh amplification on the Sharpe proxy
_ANOMALY_GATE = 0.6  # hard-flat above this anomaly_score
_DEADBAND = 0.04     # |weight| below this collapses to 0
_DENSITY_CAP = 5.0   # cap raw density before normalising to [0,1]


class PercolationThresholdStrategy:
    """Size positions only when the percolation order parameter is non-zero.

    weight = tanh(gain * tanh(sharpe))
             * clamp(((p - p_c) / (1 - p_c)) ** beta, 0, 1)
             * (1 - anomaly_score)

    Flat when:
      - regime_id == ANOMALY_REGIME
      - anomaly_score > _ANOMALY_GATE
      - p <= p_c  (below the percolation threshold)
      - |weight| < _DEADBAND  (dead-band suppression)
    """

    def __init__(
        self,
        p_c: float = _P_C,
        beta: float = _BETA,
        gain: float = _GAIN,
        anomaly_gate: float = _ANOMALY_GATE,
        deadband: float = _DEADBAND,
        density_cap: float = _DENSITY_CAP,
    ) -> None:
        self._p_c = p_c
        self._beta = beta
        self._gain = gain
        self._anomaly_gate = anomaly_gate
        self._deadband = deadband
        self._density_cap = density_cap

    def signals(self, state: ManifoldState) -> SignalSet:
        # Normalise density to [0, 1] with a soft cap.
        density_norm = min(1.0, max(0.0, state.density / self._density_cap))
        # Occupation probability: both regime coherence and local density must be high.
        p = state.regime_prob * density_norm
        # Order parameter: zero below p_c, grows as (p-p_c)^beta above.
        if p > self._p_c:
            raw = ((p - self._p_c) / (1.0 - self._p_c)) ** self._beta
            order_param = min(1.0, max(0.0, raw))
        else:
            order_param = 0.0
        confidence = order_param * max(0.0, 1.0 - state.anomaly_score)
        sharpe = state.fwd_return_mean / (state.fwd_return_std + _EPS)
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
        flat = TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        # Hard gates.
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > self._anomaly_gate:
            return flat
        # Below percolation threshold: no spanning cluster.
        if signals.confidence <= 0.0:
            return flat

        # Reconstruct sharpe proxy via atanh(momentum).
        mom = max(-1.0 + _EPS, min(1.0 - _EPS, signals.momentum))
        sharpe_proxy = math.atanh(mom)

        base = math.tanh(self._gain * math.tanh(sharpe_proxy))
        weight = base * signals.confidence

        # Dead-band.
        if abs(weight) < self._deadband:
            return flat

        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=weight)


def build() -> Strategy:
    return PercolationThresholdStrategy()
