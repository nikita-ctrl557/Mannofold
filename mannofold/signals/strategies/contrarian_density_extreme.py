"""Contrarian-density-extreme strategy: fade only in rare low-density extreme states.

Stays completely flat in normal/typical (high-density) states. Only acts when
density is LOW (atypical dislocation) AND |Sharpe| is large — then takes a
CONTRARIAN position expecting mean reversion of the dislocation.

weight = -sign(sharpe) * tanh(gain * tanh(sharpe)) * low_density_gate * confidence

where:
  sharpe           = fwd_return_mean / (fwd_return_std + eps)
  low_density_gate = inverted logistic gate: ~1 when density is in its low tail
  confidence       = regime_prob * (1 - anomaly_score)
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

NAME = "contrarian_density_extreme"
DESCRIPTION = (
    "Fade only in rare low-density (atypical/extreme) manifold states where the "
    "neighbourhood Sharpe is large; expects dislocation reversion. Flat in normal "
    "high-density regions."
)

_EPS = 1e-9
# Low-density gate: inverted logistic — low density -> ~1, high density -> ~0
_LOW_DENSITY_MID = 0.6    # density below this is considered 'atypical'
_LOW_DENSITY_SCALE = 4.0  # steepness of the sigmoid
# Contrarian gain applied inside tanh(gain * tanh(sharpe))
_GAIN = 3.0
# Absolute Sharpe threshold — only trade when |sharpe| is meaningful
_SHARPE_THR = 0.5
# Dead-band: collapse micro positions to flat
_DEADBAND = 0.04
# Anomaly score hard gate
_ANOMALY_GATE = 0.7


def _low_density_gate(density: float) -> float:
    """Smooth gate in [0, 1]: LOW density -> ~1, HIGH density -> ~0.

    Inverse of the density_gated strategy: we want to act only when the state
    is atypical (sparse neighbourhood), not in the well-populated core.
    """
    d = max(0.0, density)
    return 1.0 / (1.0 + math.exp(_LOW_DENSITY_SCALE * (d - _LOW_DENSITY_MID)))


class ContrarianDensityExtremeStrategy:
    """Contrarian fade activated only in low-density, extreme-Sharpe states.

    target_weight = -sign(sharpe) * tanh(gain * tanh(sharpe))
                    * low_density_gate(density) * confidence

    Goes flat when:
      - regime_id == ANOMALY_REGIME
      - anomaly_score > _ANOMALY_GATE
      - density is high (typical state — low_density_gate -> 0)
      - |sharpe| < _SHARPE_THR (not extreme enough)
      - |weight| < _DEADBAND (micro-position dead-band)
    """

    def signals(self, state: ManifoldState) -> SignalSet:
        sharpe = state.fwd_return_mean / (state.fwd_return_std + _EPS)
        # Embed low-density gate into confidence: only atypical states pass through
        gate = _low_density_gate(state.density)
        confidence = max(0.0, min(1.0,
            state.regime_prob * (1.0 - state.anomaly_score) * gate
        ))
        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=sharpe,           # raw Sharpe carried into target()
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        zero = TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        # Hard gate: anomalous regime or excessive anomaly score -> flat
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_GATE:
            return zero

        sharpe = signals.momentum
        abs_sharpe = abs(sharpe)

        # Selectivity: only fade when Sharpe magnitude is meaningful
        if abs_sharpe < _SHARPE_THR:
            return zero

        # Contrarian weight: -sign(sharpe) * tanh(gain * tanh(sharpe))
        # confidence already encodes low_density_gate * regime_prob * (1 - anomaly)
        raw_weight = -math.copysign(1.0, sharpe) * math.tanh(_GAIN * math.tanh(sharpe))
        weight = raw_weight * signals.confidence

        # Dead-band: suppress micro positions
        if abs(weight) < _DEADBAND:
            return zero

        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=weight)


def build() -> Strategy:
    """Return a fresh ContrarianDensityExtremeStrategy with default parameters."""
    return ContrarianDensityExtremeStrategy()
