"""Capital-preservation strategy: stay flat unless conditions are clearly favorable AND calm.

Prioritises NOT losing over capturing upside. Takes only small positions when
density is high (typical manifold region), anomaly is low (on-manifold),
regime confidence is high, and neighbourhood drift is positive. Even then the
maximum weight is capped tightly to avoid large drawdowns.

Weight formula:
    confidence   = regime_prob * (1 - anomaly_score)
    strict_gate  = density_gate * anomaly_gate * regime_gate   (each a sigmoid)
    raw_weight   = tanh(GAIN * tanh(sharpe))
    weight       = clamp(raw_weight, -0.3, 0.6) * strict_gate * confidence
    dead-band    = if |weight| < 0.04 -> 0
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

NAME = "capital_preservation"
DESCRIPTION = (
    "Defensive strategy: stay flat unless density is high, anomaly is low, "
    "regime confidence is high, and drift is positive. Caps max weight at 0.6 "
    "to prioritise avoiding negative months over capturing upside."
)

# --- Tunable knobs ---
_GAIN = 2.5          # amplifier inside double-tanh: tanh(GAIN * tanh(sharpe))
_MAX_LONG = 0.6      # hard cap on long weight
_MAX_SHORT = -0.3    # hard cap on short weight (shallow shorts only)
_DEAD_BAND = 0.04    # collapse |weight| below this to exactly 0

# Strict gate parameters — calibrated against density in [0.18, 0.77], mean ~0.54
_GATE_K = 8.0           # sigmoid steepness
_DENSITY_MID = 0.55     # gate opens above this density (above average = "high density")
_ANOMALY_CUTOFF = 0.50  # hard flat when anomaly >= this
_ANOMALY_SAFE = 0.35    # gate half-open at this anomaly (inverted sigmoid)
_REGIME_PROB_MIN = 0.55 # gate closes below this regime probability


def _sigmoid(x: float) -> float:
    """Numerically stable logistic sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _strict_gate(density: float, anomaly: float, regime_prob: float) -> float:
    """Product of three soft gates; near 1 only when all three conditions pass."""
    density_gate = _sigmoid(_GATE_K * (density - _DENSITY_MID))
    anomaly_gate = _sigmoid(_GATE_K * (_ANOMALY_SAFE - anomaly))  # inverted: low anomaly -> 1
    regime_gate = _sigmoid(_GATE_K * (regime_prob - _REGIME_PROB_MIN))
    return density_gate * anomaly_gate * regime_gate


class CapitalPreservationStrategy:
    """Flat unless manifold is dense, calm, and regime-confident; small positions only."""

    def __init__(self) -> None:
        self._last_density: float = 0.0

    def signals(self, state: ManifoldState) -> SignalSet:
        # Cache density for use in target() (SignalSet has no density field)
        self._last_density = state.density
        sharpe = state.fwd_return_mean / (state.fwd_return_std + 1e-9)
        # Confidence fuses regime stability and low anomaly
        confidence = max(0.0, min(1.0, state.regime_prob * (1.0 - state.anomaly_score)))
        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=math.tanh(_GAIN * math.tanh(sharpe)),
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        # Hard flat: anomalous regime sentinel or anomaly too high
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly >= _ANOMALY_CUTOFF:
            return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        # Recover regime_prob estimate from confidence = regime_prob * (1 - anomaly)
        regime_prob_est = min(1.0, signals.confidence / max(1.0 - signals.anomaly, 1e-9))

        gate = _strict_gate(
            density=self._last_density,
            anomaly=signals.anomaly,
            regime_prob=regime_prob_est,
        )

        # Clamp raw momentum to asymmetric range; light on shorts for preservation
        raw = max(_MAX_SHORT, min(_MAX_LONG, signals.momentum))
        weight = raw * gate * signals.confidence

        # Dead-band: suppress small noisy weights
        if abs(weight) < _DEAD_BAND:
            weight = 0.0

        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=weight)


def build() -> Strategy:
    """Return a fresh CapitalPreservationStrategy instance."""
    return CapitalPreservationStrategy()
