"""Hamiltonian momentum strategy.

Classical mechanics framing: treat the manifold state as a particle in phase space.
Position = drift level (fwd_return_mean); momentum p = velocity vector.
Kinetic energy KE = 0.5 * |velocity|^2.

Conservation of momentum principle: a particle with high KE and aligned drift will
continue on its current trajectory. Conviction grows with KE and decays as the
particle approaches a turning point (velocity decelerating toward zero).

target_weight = sign(fwd_return_mean) * tanh(gain * (|tanh(sharpe)| + kappa * KE))
                * confidence

where confidence = regime_prob * (1 - anomaly_score).
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

NAME = "hamiltonian_momentum"
DESCRIPTION = (
    "Classical-mechanics Hamiltonian strategy: treats the manifold as a phase-space "
    "particle. Kinetic energy KE = 0.5*|velocity|^2 amplifies conviction when the "
    "trajectory carries momentum aligned with drift; flattens near turning points."
)

_EPS = 1e-9

# Outer gain: maps (|tanh(sharpe)| + kappa*KE) through tanh to a weight.
_GAIN = 2.5

# Kinetic-energy coupling: scales how much KE adds to conviction.
_KAPPA = 0.5

# Anomaly threshold above which we go flat.
_ANOMALY_GATE = 0.6

# Dead-band: collapse |w| < this to 0.
_DEADBAND = 0.04


def _kinetic_energy(velocity: list[float]) -> float:
    """KE = 0.5 * |velocity|^2; returns 0.0 for empty or zero vector."""
    if not velocity:
        return 0.0
    v2 = sum(v * v for v in velocity)
    return 0.5 * v2


class _HamiltonianSignalSet(SignalSet):
    """SignalSet that carries kinetic energy and drift sign for target()."""

    kinetic_energy: float = 0.0
    drift_sign: float = 0.0

    model_config = {"extra": "allow"}


class HamiltonianMomentumStrategy:
    """Phase-space momentum strategy grounded in classical Hamiltonian mechanics."""

    def __init__(
        self,
        gain: float = _GAIN,
        kappa: float = _KAPPA,
        anomaly_gate: float = _ANOMALY_GATE,
        deadband: float = _DEADBAND,
    ) -> None:
        self._gain = gain
        self._kappa = kappa
        self._anomaly_gate = anomaly_gate
        self._deadband = deadband

    # ------------------------------------------------------------------ #
    #  signals()                                                           #
    # ------------------------------------------------------------------ #
    def signals(self, state: ManifoldState) -> SignalSet:
        sharpe = state.fwd_return_mean / (state.fwd_return_std + _EPS)
        momentum = math.tanh(sharpe)
        confidence = max(0.0, state.regime_prob * (1.0 - state.anomaly_score))
        ke = _kinetic_energy(state.velocity)
        drift_sign = math.copysign(1.0, state.fwd_return_mean) if state.fwd_return_mean != 0.0 else 0.0
        return _HamiltonianSignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=momentum,
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
            kinetic_energy=ke,
            drift_sign=drift_sign,
        )

    # ------------------------------------------------------------------ #
    #  target()                                                            #
    # ------------------------------------------------------------------ #
    def target(self, signals: SignalSet) -> TargetPosition:
        # Hard-off: anomalous regime or high anomaly score.
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > self._anomaly_gate:
            return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        ke = getattr(signals, "kinetic_energy", 0.0)
        drift_sign = getattr(signals, "drift_sign", 0.0)

        # |tanh(sharpe)| captures absolute directional conviction from the neighbourhood.
        abs_momentum = abs(signals.momentum)

        # Hamiltonian amplification: KE adds conviction when trajectory has energy.
        hamiltonian_input = abs_momentum + self._kappa * ke

        # Direction from drift sign; magnitude from Hamiltonian conviction * confidence.
        magnitude = math.tanh(self._gain * hamiltonian_input) * signals.confidence
        w = drift_sign * magnitude

        # Dead-band: suppress small, noisy weights.
        if abs(w) < self._deadband:
            w = 0.0

        w = max(-1.0, min(1.0, w))
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=w)


def build() -> Strategy:
    """Construct and return a HamiltonianMomentumStrategy with default parameters."""
    return HamiltonianMomentumStrategy()
