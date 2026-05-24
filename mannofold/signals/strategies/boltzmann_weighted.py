"""Boltzmann-weighted strategy — statistical mechanics perspective.

Treats long/short tendencies as micro-states with energies derived from the
signal, using the canonical Boltzmann distribution p_i ∝ exp(−E_i / kT):

    E_long  = −fwd_return_mean    (low energy = favourable drift up)
    E_short = +fwd_return_mean    (low energy = favourable drift down)
    T       =  fwd_return_std     (temperature = volatility / thermal agitation)

Boltzmann weights:
    w_long  ∝ exp(−E_long  / (k·T)) = exp( fwd_return_mean / (k·T))
    w_short ∝ exp(−E_short / (k·T)) = exp(−fwd_return_mean / (k·T))

Net signed tendency (partition function normalised):
    net = (w_long − w_short) / (w_long + w_short) = tanh(mu / (k·T + eps))

At low T (cool, decisive) this saturates to ±1; at high T (hot, uncertain)
it flattens toward 0 — physically exact Boltzmann behaviour.

Target weight:
    weight = tanh(gain · tanh(mu / (k·T + eps))) · confidence
    confidence = regime_prob · (1 − anomaly_score)

Flat on ANOMALY_REGIME or anomaly > 0.6. Dead-band |w| < 0.04 → 0.
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

NAME = "boltzmann_weighted"
DESCRIPTION = (
    "Boltzmann distribution over long/short micro-states: "
    "E_long = -mu, E_short = +mu, T = sigma. "
    "Net tendency = tanh(mu / (k*T + eps)) — decisive at low T, cautious at high T. "
    "weight = tanh(gain * tanh(mu/(k*T+eps))) * confidence, "
    "confidence = regime_prob*(1-anomaly_score). "
    "Flat on ANOMALY_REGIME or anomaly > 0.6; dead-band |w| < 0.04 -> 0."
)

_EPS = 1e-9
_K = 1.0              # Boltzmann constant (dimensionless; absorb into gain)
_GAIN = 50.0          # tanh amplification on the Boltzmann net tendency
_ANOMALY_CUTOFF = 0.6 # hard gate on anomaly score
_DEADBAND = 0.04      # |weight| below this -> flat (avoids churn)


class BoltzmannWeightedStrategy:
    """Trade proportional to the Boltzmann-weighted long/short partition function.

    signals():
        - momentum = tanh(mu / (k*T)) encodes the Boltzmann net tendency.
        - confidence = regime_prob * (1 - anomaly_score).
        - expected_return carries fwd_return_mean for downstream inspection.

    target():
        - Flat when regime_id == ANOMALY_REGIME or anomaly > _ANOMALY_CUTOFF.
        - weight = tanh(gain * momentum) * confidence.
        - Dead-band |weight| < _DEADBAND -> 0.
        - Clamped to [-1, 1].
    """

    def __init__(
        self,
        k: float = _K,
        gain: float = _GAIN,
        anomaly_cutoff: float = _ANOMALY_CUTOFF,
        deadband: float = _DEADBAND,
    ) -> None:
        self._k = k
        self._gain = gain
        self._anomaly_cutoff = anomaly_cutoff
        self._deadband = deadband

    def signals(self, state: ManifoldState) -> SignalSet:
        # Boltzmann net tendency: tanh(mu / (k*T + eps))
        kT = self._k * state.fwd_return_std + _EPS
        boltzmann_tendency = math.tanh(state.fwd_return_mean / kT)
        confidence = max(0.0, state.regime_prob * (1.0 - state.anomaly_score))
        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=boltzmann_tendency,
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        flat = TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        # Hard gates: anomalous regime or high anomaly score -> flat.
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > self._anomaly_cutoff:
            return flat

        # Amplify the Boltzmann tendency and scale by confidence.
        weight = math.tanh(self._gain * signals.momentum) * signals.confidence

        # Dead-band suppression.
        if abs(weight) < self._deadband:
            return flat

        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=weight)


def build() -> Strategy:
    """Construct a BoltzmannWeightedStrategy with default hyperparameters."""
    return BoltzmannWeightedStrategy()
